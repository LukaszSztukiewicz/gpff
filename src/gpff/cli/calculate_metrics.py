"""Reduce the cached evaluation metadata to metrics (``gpff_calculate_metrics``).

Reads the per-sample arrays that ``gpff_prepare_metadata`` wrote and reduces
them to one value per metric, merged into results.json (metrics not computed
this time are kept):

    <run>/evaluation/results.json
    <run>/evaluation/<metric>.json         per-environment breakdown of
                                           bond_length_w1 / bond_angle_w1,
                                           when computed

--metrics selects the metrics (default: all of them):

    rdkit_validity          fraction of molecules RDKit accepts
    posebusters_validity    fraction of molecules PoseBusters accepts
    uniqueness              fraction of the valid molecules with a new SMILES
    novelty                 fraction of the valid molecules not in the reference
    mean_tanimoto_similarity   mean pairwise fingerprint similarity (stereo-blind)
    js_divergence_energy_U0 / _gap / _elongation
                            distance to the reference distribution
    mean_strain             mean energy a relaxation released, over the valid
                            molecules (kcal/mol; needs a database relaxed by
                            gpff_relax, which writes the strain of every row)
    bond_angle_w1           the same for the bond angles, in degrees, each
                            angle against the reference angles of its own
                            environment (see --reference_bond_angles_path)
    bond_length_w1          Wasserstein distance (in Angstroms) of the bond
                            lengths to the reference, each bond compared with
                            the reference bonds of its own chemical environment,
                            weighted by the reference's environment shares

Every metric needs its per-sample arrays already prepared (a missing one is
reported when the metric is computed). The reference inputs below are checked up
front, and the run stops before doing any work if a requested metric misses one.
The reference data all comes from files (no reference database):

    --reference_energy_U0_path / _gap_path / _elongation_path
                            the js_divergence_* reference distributions
                            (assets/qm9_<property>_values.npy)
    --reference_bond_angles_path
                            bond_angle_w1: the reference angle table
                            (assets/qm9_bond_angles.npz)
    --reference_bond_lengths_path
                            bond_length_w1: the reference bond table
                            (assets/qm9_bond_lengths.npz)
    --reference_smiles_path  novelty: the reference SMILES, one per line
                            (assets/qm9_smiles_values.txt)
    --split_path            novelty with --novelty_reference train_val (default):
                            novelty is measured against the model's train+val
                            splits. With --novelty_reference full it is measured
                            against every reference SMILES and no split is needed.

The metadata folder is found from --evaluation_dir, or from --samples_db_path
(an evaluation folder next to it). The samples DB itself is not read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

import gpff  # noqa: F401
from gpff.analysis import (
    calculate_bond_length_w1,
    calculate_js_divergence,
    calculate_mean_tanimoto_similarity,
    check_novelty,
    check_uniqueness,
)

# splits a sampled molecule has to be absent from to count as novel: the model
# was fitted on train and model selection used val, test was never seen
NOVELTY_SPLITS = ("train", "val")

# every metric, in the order it is reported. What each one needs is spelled out
# in compute_metrics (below) and missing_reference_inputs; there is no table.
# A js_divergence_<property> metric compares <property> to a reference
# distribution; novelty compares the molecules to a set of known SMILES.
METRICS = (
    "rdkit_validity",
    "posebusters_validity",
    "uniqueness",
    "novelty",
    "mean_tanimoto_similarity",
    "js_divergence_energy_U0",
    "js_divergence_gap",
    "js_divergence_elongation",
    "bond_length_w1",
    "bond_angle_w1",
    "mean_strain",
)

# the metrics comparing a table of measurements (one row per bond or angle) with
# a reference table: the metadata array it reads, the argument naming the
# reference, and the table's measured column
TABLE_METRICS = {
    "bond_length_w1": ("bond_lengths", "reference_bond_lengths_path", "length"),
    "bond_angle_w1": ("bond_angles", "reference_bond_angles_path", "angle"),
}
_JS_DIVERGENCE_PREFIX = "js_divergence_"


def js_divergence_property(metric: str) -> str | None:
    """The property a js_divergence_* metric compares, e.g. 'energy_U0'; else None."""
    if metric.startswith(_JS_DIVERGENCE_PREFIX):
        return metric[len(_JS_DIVERGENCE_PREFIX):]
    return None


# the properties whose reference distribution the js_divergence_* metrics need
REFERENCE_DISTRIBUTION_PROPERTIES = tuple(
    sorted({prop for name in METRICS if (prop := js_divergence_property(name))})
)


def _reference_value_arg(property_name: str) -> str:
    return f"reference_{property_name}_path"


def _dedup(lines: list[str]) -> list[str]:
    """The lines with duplicates removed, first occurrence kept."""
    seen: set[str] = set()
    return [line for line in lines if not (line in seen or seen.add(line))]


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


def load_metadata(metadata_dir: str | Path) -> dict:
    """Load whatever per-sample arrays a metadata folder holds, keyed by filename.

    smiles.txt is read as a list of strings (one per line); every .npy is loaded
    as saved, every .npz (the per-bond table) as a dict of its arrays. A folder
    that was never prepared yields an empty dict.
    """
    arrays: dict = {}
    for path in sorted(Path(metadata_dir).glob("*")):
        if path.suffix == ".txt":
            arrays[path.stem] = path.read_text().splitlines()
        elif path.suffix == ".npy":
            arrays[path.stem] = np.load(path)
        elif path.suffix == ".npz":
            arrays[path.stem] = load_bond_table(path)
    return arrays


def load_bond_table(path: str | Path) -> dict[str, np.ndarray]:
    """Load a measurement table (see gpff.analysis.collect_bonds) from a .npz file."""
    with np.load(path) as table:
        return dict(table)


def load_split_indices(split_path: str | Path, splits=NOVELTY_SPLITS) -> np.ndarray:
    """Load the row indices of the given splits from a split file."""
    with np.load(split_path) as split:
        missing = [name for name in splits if f"{name}_idx" not in split]
        if missing:
            raise KeyError(f"No {', '.join(f'{name}_idx' for name in missing)} in {split_path}.")
        return np.concatenate([np.asarray(split[f"{name}_idx"]) for name in splits])


def load_reference_smiles(smiles_path, split_path=None, splits=NOVELTY_SPLITS) -> set[str]:
    """Load the SMILES of the known molecules, for novelty.

    They come from `smiles_path` — one SMILES per line, one line per molecule of
    the reference database, in database order. With `split_path`, only the
    molecules of `splits` count as known, so novelty is measured against the
    data the model saw rather than the whole database.
    """
    values = Path(smiles_path).read_text().splitlines()
    if split_path is not None:
        indices = load_split_indices(split_path, splits)
        values = [values[index] for index in indices]
    return {smiles for smiles in values if smiles}


def missing_reference_inputs(
    metrics,
    reference_value_paths,
    reference_smiles_path,
    split_path,
    novelty_reference,
    reference_table_paths=None,
) -> list[str]:
    """The reference inputs the requested metrics need but were not given.

    js_divergence_* needs its reference distribution file, a TABLE_METRICS one
    its reference table, novelty the reference SMILES and — against train+val —
    a split file. The per-sample arrays are not checked here; a missing one is
    reported by compute_metrics.
    """
    missing = []
    for name in metrics:
        prop = js_divergence_property(name)
        if prop is not None and reference_value_paths.get(prop) is None:
            missing.append(f"{name} needs --reference_{prop}_path for the reference {prop} distribution")
        if name in TABLE_METRICS and (reference_table_paths or {}).get(name) is None:
            array, argument, _ = TABLE_METRICS[name]
            missing.append(f"{name} needs --{argument} for the reference {array} table")
        if name == "novelty":
            if reference_smiles_path is None:
                missing.append(f"{name} needs --reference_smiles_path")
            if novelty_reference == "train_val" and split_path is None:
                missing.append(
                    f"{name} needs --split_path with --novelty_reference train_val "
                    "(or --novelty_reference full to measure against the whole reference set)"
                )
    return _dedup(missing)


def load_reference_distributions(metrics, reference_value_paths) -> dict:
    """Load the reference distribution file each requested js_divergence_* metric needs."""
    distributions: dict[str, np.ndarray] = {}
    for name in metrics:
        prop = js_divergence_property(name)
        if prop is not None and prop not in distributions:
            distributions[prop] = np.asarray(np.load(reference_value_paths[prop]), dtype=float)
    return distributions


def build_reference_smiles(
    metrics, reference_smiles_path, split_path, novelty_reference, verbose=False
) -> set[str] | None:
    """The set of known SMILES novelty is measured against, or None if unused.

    train_val restricts the reference to the model's train+val splits (needs
    split_path); full keeps every reference molecule.
    """
    if "novelty" not in metrics:
        return None

    split = split_path if novelty_reference == "train_val" else None
    known = load_reference_smiles(smiles_path=reference_smiles_path, split_path=split)
    if verbose:
        against = (
            f"{'+'.join(NOVELTY_SPLITS)} of {split_path}"
            if novelty_reference == "train_val"
            else f"full {reference_smiles_path}"
        )
        print(f"Novelty vs   : {len(known)} SMILES ({against})")
    return known


def _table_w1(metric: str, reference_tables: dict, arrays: dict) -> dict:
    """calculate_bond_length_w1 of one TABLE_METRICS metric, per label."""
    array, _, value = TABLE_METRICS[metric]
    return calculate_bond_length_w1(reference_tables[metric], arrays[array], value=value)


def compute_metrics(
    arrays,
    metrics,
    reference_distributions,
    reference_smiles,
    progress=False,
    reference_tables=None,
) -> dict:
    """Reduce the per-sample arrays to one value per metric, in order.

    `arrays` is the dict from load_metadata; each metric reads the ones it needs
    (a missing one raises KeyError). Validity is over all molecules; uniqueness,
    novelty, the Tanimoto similarity, the JS divergences and the TABLE_METRICS
    over the valid ones only. A metric with no valid molecules to reduce is np.nan;
    n_molecules is always reported.
    """
    # a measurement table (a dict) has one row per bond or angle, not per sample
    lengths = {len(values) for values in arrays.values() if not isinstance(values, dict)}
    if len(lengths) != 1:
        raise ValueError("The per-sample arrays differ in length.")

    results = {"n_molecules": float(lengths.pop())}
    bar = tqdm(metrics, desc="Computing metrics", disable=not progress)
    for name in bar:
        bar.set_postfix_str(name)

        if name == "rdkit_validity":
            results[name] = float(arrays["rdkit_valid"].mean())
            continue
        if name == "posebusters_validity":
            results[name] = float(arrays["posebusters_valid"].mean())
            continue
        if name in TABLE_METRICS:
            # the tables only hold the valid molecules' rows
            results[name] = _table_w1(name, reference_tables, arrays)["w1"]
            continue

        valid = arrays["rdkit_valid"]
        if name == "mean_strain":
            strain = arrays["strain"][valid]
            results[name] = float(np.nanmean(strain)) if strain.size else float("nan")
            continue

        prop = js_divergence_property(name)
        if prop is not None:
            observed = arrays[prop][valid]
            results[name] = (
                calculate_js_divergence(reference_distributions[prop], observed)
                if observed.size
                else float("nan")
            )
            continue

        # the remaining metrics all reduce the valid molecules' SMILES
        valid_smiles = [s for s, is_valid in zip(arrays["smiles"], valid) if is_valid]
        if name == "mean_tanimoto_similarity":
            results[name] = calculate_mean_tanimoto_similarity(valid_smiles, progress=progress)
        elif not valid_smiles:
            results[name] = float("nan")
        elif name == "uniqueness":
            results[name] = float(check_uniqueness(valid_smiles).mean())
        elif name == "novelty":
            results[name] = float(check_novelty(valid_smiles, reference_smiles).mean())
        else:
            raise ValueError(f"Unknown metric {name}; known are {', '.join(METRICS)}.")
    return results


def merge_results(results_path, values: dict) -> dict:
    """Merge `values` into results.json, keeping metrics not computed this time."""
    results_path = Path(results_path)
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results.update(values)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics", nargs="+", choices=["all", *METRICS], default=["all"],
                        metavar="METRIC",
                        help=f"Metrics to compute: {', '.join(METRICS)}. Default: all.")
    parser.add_argument("--samples_db_path", type=Path, default=None,
                        help="Samples DB the metadata belongs to. Locates the "
                             "evaluation folder next to it.")
    parser.add_argument("--evaluation_dir", type=Path, default=None,
                        help="Evaluation folder holding metadata/ and results.json. "
                             "Default: an evaluation folder next to the samples DB.")
    for property_name in REFERENCE_DISTRIBUTION_PROPERTIES:
        parser.add_argument(f"--{_reference_value_arg(property_name)}", type=Path, default=None,
                            help=f"Reference {property_name} distribution (.npy) for "
                                 f"js_divergence_{property_name}.")
    parser.add_argument("--reference_bond_angles_path", type=Path, default=None,
                        help="Reference angle table (.npz, assets/qm9_bond_angles.npz) "
                             "for bond_angle_w1.")
    parser.add_argument("--reference_bond_lengths_path", type=Path, default=None,
                        help="Reference bond table (.npz, assets/qm9_bond_lengths.npz) "
                             "for bond_length_w1.")
    parser.add_argument("--reference_smiles_path", type=Path, default=None,
                        help="Text file with one reference SMILES per line "
                             "(assets/qm9_smiles_values.txt), for novelty.")
    parser.add_argument("--novelty_reference", choices=["train_val", "full"], default="train_val",
                        help="What novelty is measured against: the train+val splits "
                             "of --split_path (default), or the full reference set.")
    parser.add_argument("--split_path", type=Path, default=None,
                        help="Split file of the model that produced the samples; its "
                             "train+val splits are the novelty reference with "
                             "--novelty_reference train_val (the default).")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def resolve_evaluation_dir(args) -> Path | None:
    """The evaluation folder from --evaluation_dir or --samples_db_path (or None)."""
    if args.evaluation_dir is not None:
        return args.evaluation_dir
    if args.samples_db_path is not None:
        return args.samples_db_path.parent / "evaluation"
    return None


def main() -> int:
    args = parse_args()

    path_args = (
        "samples_db_path",
        *(argument for _, argument, _ in TABLE_METRICS.values()),
        "reference_smiles_path",
        "split_path",
        *(_reference_value_arg(name) for name in REFERENCE_DISTRIBUTION_PROPERTIES),
    )
    errors = _path_exists_errors(args, path_args)
    if errors:
        for line in errors:
            print(f"ERROR: {line}", file=sys.stderr)
        return 1

    evaluation_dir = resolve_evaluation_dir(args)
    if evaluation_dir is None:
        print("ERROR: pass --evaluation_dir or --samples_db_path.", file=sys.stderr)
        return 1
    metadata_dir = evaluation_dir / "metadata"
    results_path = evaluation_dir / "results.json"
    verbose = not args.quiet

    metrics = _selected(args.metrics, METRICS)
    reference_value_paths = {
        name: getattr(args, _reference_value_arg(name))
        for name in REFERENCE_DISTRIBUTION_PROPERTIES
    }
    missing = missing_reference_inputs(
        metrics,
        reference_value_paths,
        args.reference_smiles_path,
        args.split_path,
        args.novelty_reference,
        {name: getattr(args, argument) for name, (_, argument, _) in TABLE_METRICS.items()},
    )
    if missing:
        print("ERROR: missing reference inputs for the requested metrics:", file=sys.stderr)
        for line in missing:
            print(f"  {line}", file=sys.stderr)
        return 1

    print(f"Metadata dir : {metadata_dir}")
    print(f"Metrics      : {', '.join(metrics)}")
    print(f"Results file : {results_path}")

    arrays = load_metadata(metadata_dir)
    reference_distributions = load_reference_distributions(metrics, reference_value_paths)
    reference_smiles = build_reference_smiles(
        metrics, args.reference_smiles_path, args.split_path, args.novelty_reference, verbose=verbose
    )
    reference_tables = {
        name: load_bond_table(getattr(args, argument))
        for name, (_, argument, _) in TABLE_METRICS.items()
        if name in metrics
    }
    try:
        values = compute_metrics(
            arrays,
            metrics,
            reference_distributions,
            reference_smiles,
            progress=verbose,
            reference_tables=reference_tables,
        )
    except KeyError as missing_array:
        print(
            f"ERROR: a requested metric needs the {missing_array} array; "
            "run gpff_prepare_metadata for it first.",
            file=sys.stderr,
        )
        return 1
    merge_results(results_path, values)
    for name in reference_tables:
        breakdown_path = evaluation_dir / f"{name}.json"
        breakdown = _table_w1(name, reference_tables, arrays)
        breakdown_path.write_text(json.dumps(breakdown, indent=2) + "\n")
        print(f"Breakdown    : {breakdown_path}")

    print("Metrics:")
    for name, value in values.items():
        print(f"  {name:28s} {value:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
