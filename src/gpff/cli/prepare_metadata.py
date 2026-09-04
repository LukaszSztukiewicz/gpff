"""Compute and store the per-sample evaluation metadata (``gpff_prepare_metadata``).

Walks a sampled-molecule database and writes the per-sample arrays into the
evaluation metadata folder, one file per quantity. The database is only read.
Reduce the arrays to metrics with ``gpff_calculate_metrics``.

    <run>/evaluation/metadata/   one <key>.npy per numeric array, smiles.txt

--targets selects which arrays to compute (default: all). The structure arrays
come from an RDKit pass and a slower PoseBusters pass; the property arrays are
predicted and need their models:

    rdkit_valid, ring_counts, elongation, nfe, smiles   (RDKit pass)
    posebusters_valid                                                (PoseBusters)
    gap        needs --gap_model_path
    energy_U0  needs --energy_u0_model_path and --atomrefs_path (the single-atom
               reference energies, e.g. assets/qm9_energy_U0_atomrefs_eV.npy)

Only the requested arrays are (re)written; any other file already in the folder
is left as it is. Computing one array of a pass runs the whole pass (the RDKit
pass yields its five arrays together), but only the targeted ones are saved. The
run stops before doing any work if a requested target misses an input.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from ase.db import connect
from schnetpack.data import ASEAtomsData, AtomsLoader
import schnetpack.properties as structure_properties
from schnetpack.nn import scatter_add
from schnetpack.transform import CastTo32, MatScipyNeighborList
from schnetpack.utils import load_model
from tqdm import tqdm

# gpff/__init__.py installs the matred -> gpff alias so torch.load can unpickle
# bundled checkpoints saved under the old package name.
import gpff  # noqa: F401
from gpff import keys as keys_module
from gpff.analysis import analyze_with_rdkit, get_posebusters_validity

# the RDKit pass yields these five arrays in one walk; the PoseBusters pass is
# separate and much slower. The property arrays are predicted, one model each.
RDKIT_KEYS = ("rdkit_valid", "ring_counts", "elongation", "nfe", "smiles")
POSEBUSTERS_KEYS = ("posebusters_valid",)
PROPERTY_KEYS = ("gap", "energy_U0")
# every per-sample array, in a stable order; smiles is the only text one
METADATA_KEYS = (*RDKIT_KEYS, *POSEBUSTERS_KEYS, *PROPERTY_KEYS)

# the argument holding the predictor of each predicted-property array
PROPERTY_MODEL_ARGS = {"gap": "gap_model_path", "energy_U0": "energy_u0_model_path"}

# arguments that name files; each is checked to exist when given
PATH_ARGS = ("samples_db_path", "gap_model_path", "energy_u0_model_path", "atomrefs_path")


def _path_exists_errors(args, names) -> list[str]:
    """One 'not found' line per given path argument that names a missing file."""
    errors = []
    for name in names:
        path = getattr(args, name)
        if path is not None and not Path(path).exists():
            errors.append(f"--{name} not found: {path}")
    return errors


def _selected(requested: list[str], known) -> list[str]:
    """Resolve an --xyz selection, expanding the "all" sentinel to `known` in order."""
    return [name for name in known if "all" in requested or name in requested]


def missing_inputs(targets, property_model_paths, atomrefs_path) -> list[str]:
    """Why the requested targets cannot be prepared from the given assets.

    A property array is predicted, so it needs its model (and, for energy_U0,
    --atomrefs_path); the structure arrays need nothing but the database.
    """
    missing = []
    for key in targets:
        if key not in PROPERTY_KEYS:
            continue
        if property_model_paths.get(key) is None:
            missing.append(f"target {key} needs --{PROPERTY_MODEL_ARGS[key]} to predict it")
        if key == "energy_U0" and atomrefs_path is None:
            missing.append("target energy_U0 needs --atomrefs_path for the atom reference energies")
    return missing


def count_molecules(db_path: str | Path) -> int:
    """The number of molecules in a database."""
    with connect(db_path) as db:
        return db.count()


def load_atomrefs(path: str | Path) -> np.ndarray:
    """Load the single-atom reference energies (one per atomic number) from a .npy file."""
    return np.asarray(np.load(path), dtype=float)


def write_array(metadata_dir: Path, key: str, values) -> None:
    """Write one per-sample array to the metadata folder (smiles as text, else .npy)."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    if key == "smiles":
        # one SMILES per line; a molecule RDKit rejected contributes an empty one
        (metadata_dir / "smiles.txt").write_text("\n".join(values) + "\n")
    else:
        np.save(metadata_dir / f"{key}.npy", np.asarray(values))


def _copy_db_to_workdir(db_path: str, data_workdir: str | None) -> str:
    """Copy the database into a local workdir once, to keep reads off a slow share."""
    if data_workdir is None:
        return db_path

    source_path = Path(db_path)
    workdir_path = Path(data_workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)

    source_hash = hashlib.sha1(str(source_path.resolve()).encode("utf-8")).hexdigest()[:12]
    local_path = workdir_path / f"{source_path.stem}_{source_hash}{source_path.suffix}"

    if (
        not local_path.exists()
        or local_path.stat().st_size != source_path.stat().st_size
        or local_path.stat().st_mtime < source_path.stat().st_mtime
    ):
        shutil.copy2(source_path, local_path)

    return str(local_path)


def compute_structure_arrays(
    db_path: str,
    run_rdkit: bool,
    run_posebusters: bool,
    timeout: int = 5,
    data_workdir: str | None = None,
    verbose: bool = False,
) -> dict:
    """Walk the database and return the per-structure arrays of the passes that run.

    run_rdkit yields rdkit_valid, ring_counts, elongation, nfe and smiles in one
    walk; run_posebusters yields posebusters_valid from a separate, slower one.
    """
    read_db_path = _copy_db_to_workdir(db_path, data_workdir)
    n_molecules = count_molecules(read_db_path)
    rdkit_valid = np.zeros(n_molecules, dtype=bool)
    posebusters_valid = np.zeros(n_molecules, dtype=bool)
    ring_counts = np.zeros((n_molecules, 4))
    elongation = np.zeros(n_molecules)
    nfe = np.zeros(n_molecules)
    smiles = []
    passes = [name for name, run in (("rdkit", run_rdkit), ("posebusters", run_posebusters)) if run]

    with connect(read_db_path) as read_db:
        for i in tqdm(
            range(n_molecules),
            desc=f"Computing structure metadata ({', '.join(passes)})",
            disable=not verbose,
        ):
            atoms_row = read_db.get(i + 1)
            atoms = atoms_row.toatoms()
            if run_rdkit:
                result = analyze_with_rdkit(atoms, timeout=timeout)
                rdkit_valid[i] = result["rdkit_valid"]
                ring_counts[i] = result["ring_counts"]
                elongation[i] = result["elongation"]
                smiles.append(result["smiles"])
                nfe[i] = atoms_row.data.get(keys_module.nfe, 0)
            if run_posebusters:
                posebusters_valid[i] = get_posebusters_validity(atoms)

    arrays = {}
    if run_rdkit:
        arrays.update(
            rdkit_valid=rdkit_valid,
            ring_counts=ring_counts,
            elongation=elongation,
            nfe=nfe,
            smiles=smiles,
        )
    if run_posebusters:
        arrays["posebusters_valid"] = posebusters_valid
    return arrays


_PAINN_LEGACY_DEFAULTS = {
    "epsilon": 1e-8,
    "return_vector_representation": False,
}


def _patch_legacy_painn_attrs(model: torch.nn.Module) -> None:
    """Backfill instance attributes that newer schnetpack PaiNN forward() expects.

    Bundled property-predictor checkpoints were saved before schnetpack's PaiNN
    (and PaiNNMixing) started reading attributes like ``self.epsilon`` and
    ``self.return_vector_representation`` in forward(). Set the schnetpack
    defaults on any submodule whose instance is missing them so unpickled
    legacy instances still run.
    """
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name not in {"PaiNN", "PaiNNMixing", "PaiNNInteraction"}:
            continue
        for attr, default in _PAINN_LEGACY_DEFAULTS.items():
            if attr not in module.__dict__:
                setattr(module, attr, default)


def _ensure_spk_dataset_db_metadata(db_path: str) -> None:
    """Backfill the schnetpack dataset metadata an ASEAtomsData load expects."""
    with connect(db_path) as db:
        metadata = db.metadata.copy()
        metadata_updated = False

        if "_property_unit_dict" not in metadata:
            metadata["_property_unit_dict"] = {}
            metadata_updated = True

        if "_distance_unit" not in metadata:
            metadata["_distance_unit"] = "Ang"
            metadata_updated = True

        if metadata_updated:
            db.metadata = metadata


def predict_property(
    samples_db_path: str,
    property_name: str,
    model_path: str,
    atomrefs: np.ndarray | None = None,
    device: str = "cpu",
    cutoff: float = 5.0,
    batch_size: int = 64,
    num_workers: int = 6,
    data_workdir: str | None = None,
    progress: bool = True,
) -> np.ndarray:
    """Predict one property (gap, energy_U0) for every molecule, as a float array.

    energy_U0 additionally needs `atomrefs`, the single-atom reference energies
    (one value per atomic number), which are subtracted from the raw prediction.
    """
    if property_name == "energy_U0" and atomrefs is None:
        raise ValueError("Predicting energy_U0 needs atomrefs, the single-atom reference energies.")

    model = load_model(model_path, device=device)
    _patch_legacy_painn_attrs(model)
    model.to(device)
    model.eval()

    if property_name == "energy_U0":
        atomrefs = torch.as_tensor(atomrefs, dtype=torch.float32)

    local_samples_db_path = _copy_db_to_workdir(samples_db_path, data_workdir)
    _ensure_spk_dataset_db_metadata(local_samples_db_path)
    dataset = ASEAtomsData(
        datapath=local_samples_db_path,
        load_properties=[],
        load_structure=True,
        transforms=[MatScipyNeighborList(cutoff=cutoff), CastTo32()],
        subset_idx=None,
        distance_unit="Ang",
    )
    dataloader = AtomsLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    property_values = []
    for batch in tqdm(dataloader, desc=f"Predicting {property_name}", disable=not progress):
        batch = {key: value.to(device) for key, value in batch.items()}
        predictions = model(batch)
        values = predictions[property_name].detach().cpu()
        if property_name == "energy_U0":
            batch_cpu = {key: value.detach().cpu() for key, value in batch.items()}
            z = batch_cpu[structure_properties.Z]
            idx_m = batch_cpu[structure_properties.idx_m]
            atomref_per_atom = atomrefs[z]
            values = values - scatter_add(atomref_per_atom, idx_m, dim_size=int(idx_m[-1]) + 1)
        property_values.extend(values.tolist())

    return np.asarray(property_values, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples_db_path", required=True, type=Path)
    parser.add_argument("--targets", nargs="+", choices=["all", *METADATA_KEYS], default=["all"],
                        metavar="TARGET",
                        help=f"Arrays to compute: {', '.join(METADATA_KEYS)}. Default: all.")
    parser.add_argument("--evaluation_dir", type=Path, default=None,
                        help="Where metadata/ goes. Default: an evaluation folder "
                             "next to the samples DB.")
    parser.add_argument("--gap_model_path", type=Path, default=None,
                        help="Local path to the gap predictor checkpoint.")
    parser.add_argument("--energy_u0_model_path", type=Path, default=None,
                        help="Local path to the energy_U0 predictor checkpoint.")
    parser.add_argument("--atomrefs_path", type=Path, default=None,
                        help="Single-atom reference energies (.npy, one per atomic "
                             "number) for energy_U0, e.g. assets/qm9_energy_U0_atomrefs_eV.npy.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_workdir", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # PoseBusters logs one INFO line per molecule; on the progress bars' stream
    # that leaves them unreadable.
    logging.getLogger("posebusters").setLevel(logging.WARNING)

    errors = _path_exists_errors(args, PATH_ARGS)
    if errors:
        for line in errors:
            print(f"ERROR: {line}", file=sys.stderr)
        return 1

    targets = _selected(args.targets, METADATA_KEYS)
    evaluation_dir = args.evaluation_dir or (args.samples_db_path.parent / "evaluation")
    metadata_dir = evaluation_dir / "metadata"
    verbose = not args.quiet

    property_model_paths = {
        key: str(getattr(args, model_arg))
        for key, model_arg in PROPERTY_MODEL_ARGS.items()
        if getattr(args, model_arg) is not None
    }
    missing = missing_inputs(targets, property_model_paths, args.atomrefs_path)
    if missing:
        print("ERROR: missing inputs for the requested targets:", file=sys.stderr)
        for line in missing:
            print(f"  {line}", file=sys.stderr)
        return 1

    print(f"Preparing    : {args.samples_db_path}")
    print(f"Targets      : {', '.join(targets)}")
    print(f"Metadata dir : {metadata_dir}")

    samples_db_path = str(args.samples_db_path)
    data_workdir = str(args.data_workdir) if args.data_workdir else None
    atomrefs = load_atomrefs(args.atomrefs_path) if args.atomrefs_path is not None else None

    # compute every requested array (a pass yields its whole group), then write
    # only the targeted ones so untargeted files on disk stay as they are
    arrays: dict = {}
    run_rdkit = bool(set(targets) & set(RDKIT_KEYS))
    run_posebusters = "posebusters_valid" in targets
    if run_rdkit or run_posebusters:
        arrays.update(
            compute_structure_arrays(
                samples_db_path, run_rdkit, run_posebusters,
                data_workdir=data_workdir, verbose=verbose,
            )
        )
    for property_name in PROPERTY_KEYS:
        if property_name in targets:
            arrays[property_name] = predict_property(
                samples_db_path,
                property_name,
                model_path=property_model_paths[property_name],
                atomrefs=atomrefs,
                device=args.device,
                cutoff=args.cutoff,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                data_workdir=data_workdir,
                progress=verbose,
            )

    for key in targets:
        write_array(metadata_dir, key, arrays[key])
    return 0


if __name__ == "__main__":
    sys.exit(main())
