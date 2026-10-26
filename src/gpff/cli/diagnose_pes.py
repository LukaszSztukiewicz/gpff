"""Probe the pseudo-PES of a trained model (``gpff_diagnose_pes``).

Runs the checks of :mod:`gpff.analysis.pes` on clean data geometries, which the
pseudo-PES labels put at the bottom of the surface, and prints one row per
molecule and a summary. Nothing is written unless ``--output_json`` is given,
and the database is only read.

What a trained model should show, per molecule:

    residual force       near zero: a data geometry is a stationary point
    jacobian asymmetry   near zero only for experiment=gpff_conservative; a
                         direct force head is asymmetric by construction and
                         so has no potential and no valid Hessian
    zero modes           six, the translations and rotations the learned energy
                         is invariant under (five if the molecule is linear)
    negative modes       none: the geometry is a minimum, not a saddle
    relaxed rmsd         below the perturbed rmsd: following the forces carries
                         a displaced geometry back

Read the whole report together. A direct force model can pass the residual and
relaxation checks while failing the other two, which is the point of the
comparison; and a model that fails the residual check has not learned the small
sigma end of the surface, whatever the rest says.

With --modes_dir the softest non-zero normal modes are written out as .xyz
trajectories: the structural changes the surface holds cheapest, which read as
the modifications the data manifold permits. They are only meaningful once the
curvature check passes, since a Hessian needs a potential behind it.

    gpff_diagnose_pes --model_path runs/<project>/<date>/best_model \\
      --datapath /path/to/qm9_filtered.db \\
      --split_path runs/<project>/<date>/split.npz
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from ase import Atoms
from ase.io import write as ase_write
from schnetpack import properties
from schnetpack.data import ASEAtomsData, AtomsLoader
from schnetpack.utils import load_model

# gpff/__init__.py installs the matred -> gpff alias so torch.load can unpickle
# bundled checkpoints saved under the old package name.
import gpff  # noqa: F401
from gpff.analysis.pes import diagnose, force_jacobian, mode_trajectory, normal_modes
from gpff.utils.neighbors import AllToAllNeighborList

log = logging.getLogger(__name__)

DTYPES = {"float32": torch.float32, "float64": torch.float64}

# the row, in print order; every key but n_atoms is a float
COLUMNS = (
    "n_atoms",
    "residual_force_max",
    "residual_force_rms",
    "jacobian_asymmetry",
    "eigenvalue_min",
    "n_zero_modes",
    "n_negative_modes",
    "perturbed_rmsd",
    "relaxed_rmsd",
    "relax_steps",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model_path", type=Path, required=True,
                        help="Checkpoint to probe, e.g. a run's best_model.")
    parser.add_argument("--datapath", type=Path, required=True,
                        help="ASE database of clean geometries, e.g. the QM9 database.")
    parser.add_argument("--split_path", type=Path, default=None,
                        help="Split file of the run that trained the model. Without it "
                             "the molecules are taken from the start of the database, "
                             "which the model has most likely trained on.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="Which split to draw from (default: test). Needs --split_path.")
    parser.add_argument("--n_molecules", type=int, default=16,
                        help="How many molecules to probe (default: 16). Each one costs "
                             "3 n_atoms backward passes for its Jacobian.")
    parser.add_argument("--perturbation", type=float, default=0.1,
                        help="Standard deviation in Angstrom of the displacement the "
                             "relaxation starts from (default: 0.1). Zero relaxes from "
                             "the data geometry itself.")
    parser.add_argument("--step_size", type=float, default=0.2,
                        help="Relaxation step size (default: 0.2); the labels make every "
                             "value below 0.5 converge on the exact surface.")
    parser.add_argument("--n_steps", type=int, default=200,
                        help="Maximum relaxation steps (default: 200).")
    parser.add_argument("--force_tol", type=float, default=1e-3,
                        help="Relaxation stops below this largest per-atom force (default: 1e-3).")
    parser.add_argument("--zero_tol", type=float, default=1e-3,
                        help="Zero-mode tolerance, relative to the largest eigenvalue "
                             "(default: 1e-3).")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float64",
                        help="Precision of the model and the geometries (default: float64, "
                             "which the Hessian and the asymmetry are worth).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (default: cuda when available).")
    parser.add_argument("--seed", type=int, default=999,
                        help="Seed of the relaxation's displacement (default: 999).")
    parser.add_argument("--output_json", type=Path, default=None,
                        help="Also write the rows and the summary to this JSON file.")
    parser.add_argument("--modes_dir", type=Path, default=None,
                        help="Also write the softest non-zero normal modes into this "
                             "directory, one molecule_<i>_mode_<k>.xyz trajectory each. "
                             "The modes are the structural changes the surface holds "
                             "cheapest, so they read as the modifications it permits.")
    parser.add_argument("--n_modes", type=int, default=3,
                        help="How many of the softest non-zero modes to write "
                             "(default: 3). Needs --modes_dir.")
    parser.add_argument("--mode_amplitude", type=float, default=0.5,
                        help="Largest per-atom displacement of a written mode, in "
                             "Angstrom (default: 0.5).")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def write_modes(
    directory: Path,
    molecule_index: int,
    positions: torch.Tensor,
    numbers: torch.Tensor,
    jacobian: torch.Tensor,
    n_modes: int,
    amplitude: float,
    n_zero_modes: int = 6,
) -> None:
    """
    Write the softest non-zero modes of one molecule as .xyz trajectories.

    The first ``n_zero_modes`` eigenvalues are the translations and rotations
    the learned energy is invariant under, so the modes written start past
    them.
    """
    eigenvalues, eigenvectors = normal_modes(jacobian)
    numbers = numbers.detach().cpu().numpy()
    directory.mkdir(parents=True, exist_ok=True)

    for offset in range(n_modes):
        index = n_zero_modes + offset
        if index >= eigenvalues.numel():
            break
        frames = mode_trajectory(
            positions, eigenvectors[:, index], amplitude=amplitude
        )
        ase_write(
            directory / f"molecule_{molecule_index}_mode_{offset}.xyz",
            [
                Atoms(numbers=numbers, positions=frame.detach().cpu().numpy())
                for frame in frames
            ],
            format="extxyz",
        )


def load_indices(split_path: Optional[Path], split: str) -> Optional[np.ndarray]:
    """
    The dataset indices of one split, or None to take the database in order.
    """
    if split_path is None:
        return None
    if not split_path.exists():
        raise FileNotFoundError(f"No split file at {split_path}.")
    with np.load(split_path) as splits:
        key = f"{split}_idx"
        if key not in splits:
            raise KeyError(
                f"{split_path} holds {sorted(splits)}, not {key}."
            )
        return splits[key]


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Reduce the per-molecule rows to the worst case of each check.

    The worst case, not the mean: one molecule that is not a minimum of the
    surface is the finding, and an average would hide it.
    """
    return {
        "n_molecules": len(rows),
        "residual_force_max": max(row["residual_force_max"] for row in rows),
        "jacobian_asymmetry_max": max(row["jacobian_asymmetry"] for row in rows),
        "eigenvalue_min": min(row["eigenvalue_min"] for row in rows),
        "n_zero_modes_min": min(row["n_zero_modes"] for row in rows),
        "n_zero_modes_max": max(row["n_zero_modes"] for row in rows),
        "n_negative_modes_max": max(row["n_negative_modes"] for row in rows),
        "molecules_with_negative_modes": sum(
            row["n_negative_modes"] > 0 for row in rows
        ),
        "relaxed_rmsd_max": max(row["relaxed_rmsd"] for row in rows),
        "relaxed_worse_than_perturbed": sum(
            row["relaxed_rmsd"] > row["perturbed_rmsd"] for row in rows
        ),
    }


def format_table(rows: List[Dict[str, float]]) -> str:
    """
    Render the rows as a fixed-width table, one molecule per line.
    """
    widths = {column: max(len(column), 11) for column in COLUMNS}
    lines = ["  ".join(column.rjust(widths[column]) for column in COLUMNS)]
    for row in rows:
        lines.append(
            "  ".join(
                (
                    f"{row[column]:d}"
                    if isinstance(row[column], int)
                    else f"{row[column]:.3e}"
                ).rjust(widths[column])
                for column in COLUMNS
            )
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )
    dtype = DTYPES[args.dtype]

    if not args.model_path.exists():
        raise FileNotFoundError(f"No model checkpoint at {args.model_path}.")
    if not args.datapath.exists():
        raise FileNotFoundError(f"No database at {args.datapath}.")
    if args.n_molecules < 1:
        raise ValueError(f"--n_molecules must be positive, got {args.n_molecules}.")

    indices = load_indices(args.split_path, args.split)
    if indices is not None:
        if len(indices) < args.n_molecules:
            raise ValueError(
                f"The {args.split} split holds {len(indices)} molecules, fewer than the "
                f"{args.n_molecules} requested."
            )
        indices = indices[: args.n_molecules]
        log.info(f"Probing {len(indices)} molecules of the {args.split} split.")
    else:
        log.info(
            f"Probing the first {args.n_molecules} molecules of the database; pass "
            f"--split_path to draw held-out ones instead."
        )

    log.info(f"Loading model from: {args.model_path}")
    # load_model does not set the mode, and sampling-free probing is inference
    model = load_model(str(args.model_path), device=args.device)
    model = model.to(device=args.device, dtype=dtype)
    model.eval()

    # no CastTo32: the geometries keep the database's precision and are cast once
    dataset = ASEAtomsData(
        datapath=str(args.datapath),
        load_properties=[],
        load_structure=True,
        transforms=[AllToAllNeighborList()],
        subset_idx=None if indices is None else indices.tolist(),
        distance_unit="Ang",
    )
    loader = AtomsLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    generator = torch.Generator().manual_seed(args.seed)
    rows: List[Dict[str, float]] = []
    for batch in loader:
        batch = {
            key: (
                value.to(device=args.device, dtype=dtype)
                if value.is_floating_point()
                else value.to(device=args.device)
            )
            for key, value in batch.items()
        }
        # taken once here when the modes are wanted too, since the Jacobian is
        # the expensive part of the report
        jacobian = force_jacobian(model, batch) if args.modes_dir else None
        rows.append(
            diagnose(
                model,
                batch,
                perturbation=args.perturbation,
                zero_tol=args.zero_tol,
                generator=generator,
                jacobian=jacobian,
                n_steps=args.n_steps,
                step_size=args.step_size,
                force_tol=args.force_tol,
            )
        )
        if jacobian is not None:
            write_modes(
                args.modes_dir,
                len(rows) - 1,
                batch[properties.R],
                batch[properties.Z],
                jacobian,
                n_modes=args.n_modes,
                amplitude=args.mode_amplitude,
            )
        if len(rows) >= args.n_molecules:
            break

    summary = summarize(rows)
    print(format_table(rows))
    print()
    for key, value in summary.items():
        print(f"{key:<32} {value:.3e}" if isinstance(value, float) else f"{key:<32} {value:d}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"molecules": rows, "summary": summary}, indent=2)
        )
        log.info(f"Wrote {args.output_json}")
    if args.modes_dir is not None:
        log.info(f"Wrote up to {args.n_modes} modes per molecule into {args.modes_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
