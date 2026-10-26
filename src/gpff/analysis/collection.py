"""Reductions over a sample set: one aggregate value from per-sample arrays/lists.

These are the reusable building blocks the evaluation is made of; which metric
needs which of them, and how they are wired together, lives in the
``gpff_calculate_metrics`` script.
"""
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from tqdm import tqdm

__all__ = [
    "calculate_angle_w1",
    "calculate_bond_length_w1",
    "calculate_js_divergence",
    "calculate_mean_tanimoto_similarity",
    "check_novelty",
    "check_uniqueness",
    "collect_angles",
    "collect_bonds",
]


def check_uniqueness(smiles_list: list[str]) -> np.ndarray:
    """Mark the first occurrence of each SMILES; empty SMILES are never unique."""
    unique = np.zeros(len(smiles_list), dtype=bool)
    seen: set[str] = set()
    for i, smiles in enumerate(smiles_list):
        if smiles and smiles not in seen:
            seen.add(smiles)
            unique[i] = True
    return unique


def check_novelty(smiles_list: list[str], reference_smiles: set[str]) -> np.ndarray:
    """Mark the molecules whose SMILES does not occur in the reference set.

    Empty SMILES (invalid molecules) are never novel.
    """
    return np.array(
        [bool(smiles) and smiles not in reference_smiles for smiles in smiles_list],
        dtype=bool,
    )


def calculate_mean_tanimoto_similarity(
    smiles_list: list[str],
    max_path: int = 5,
    n_bits: int = 2048,
    progress: bool = False,
) -> float:
    """Mean pairwise Tanimoto similarity of RDKit fingerprints (internal diversity).

    Uses the topological, path-based RDKit fingerprint over bond paths up to
    `max_path` bonds long. It encodes the molecular graph only, not
    stereochemistry. Empty or unparseable SMILES are skipped. Returns np.nan if
    fewer than two molecules remain. The pairwise pass is quadratic in the
    number of molecules; `progress` shows a bar for it.
    """
    generator = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=max_path, fpSize=n_bits)
    fingerprints = []
    for smiles in smiles_list:
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fingerprints.append(generator.GetFingerprint(mol))

    if len(fingerprints) < 2:
        return float("nan")

    # sum over the upper triangle: each molecule against all later ones
    similarity_sum = 0.0
    n_pairs = 0
    for i, fingerprint in enumerate(
        tqdm(fingerprints[:-1], desc="Tanimoto similarity", disable=not progress, leave=False)
    ):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints[i + 1:])
        similarity_sum += sum(similarities)
        n_pairs += len(similarities)

    return similarity_sum / n_pairs


def calculate_js_divergence(
    target: np.ndarray,
    observed: np.ndarray,
    bins: int = 100,
) -> float:
    """Compute the Jensen-Shannon divergence between two one-dimensional distributions."""
    min_val = min(np.min(target), np.min(observed))
    max_val = max(np.max(target), np.max(observed))
    common_bins = np.linspace(min_val, max_val, bins)

    p, _ = np.histogram(target, bins=common_bins, density=True)
    q, _ = np.histogram(observed, bins=common_bins, density=True)

    js_distance = jensenshannon(p, q, base=2)
    return float(js_distance**2)


def _collect(per_molecule: list[dict], labels: tuple[str, ...], value: str) -> dict[str, np.ndarray]:
    """Stack per-molecule label and value lists into one table, one entry per row.

    The table is a dict of equally long arrays: molecule (the molecule's
    position in the input list), one array per label column, and the measured
    value. Invalid molecules contribute nothing.
    """
    counts = [len(record[value]) for record in per_molecule]
    table = {"molecule": np.repeat(np.arange(len(per_molecule)), counts)}
    for label in labels:
        table[label] = np.array(
            [entry for record in per_molecule for entry in record[label]], dtype=str
        )
    table[value] = np.array(
        [entry for record in per_molecule for entry in record[value]], dtype=float
    )
    return table


def collect_bonds(per_molecule_bonds: list[dict]) -> dict[str, np.ndarray]:
    """Stack the describe_bonds lists of many molecules into one bond table.

    Its columns are molecule, bond_type, environment and length.
    """
    return _collect(per_molecule_bonds, ("bond_type", "environment"), "length")


def collect_angles(per_molecule_angles: list[dict]) -> dict[str, np.ndarray]:
    """Stack the describe_angles lists of many molecules into one angle table.

    Its columns are molecule, angle_type, environment and angle.
    """
    return _collect(per_molecule_angles, ("angle_type", "environment"), "angle")


def _group_lengths(table: dict, by: str, value: str = "length") -> dict[str, np.ndarray]:
    """The measured values of a table, grouped by one of its label columns."""
    labels = np.asarray(table[by])
    lengths = np.asarray(table[value], dtype=float)
    order = np.argsort(labels, kind="stable")
    unique, starts = np.unique(labels[order], return_index=True)
    return dict(zip(unique.tolist(), np.split(lengths[order], starts[1:])))


def _iqr(values: np.ndarray) -> float:
    upper, lower = np.percentile(values, [75, 25])
    return float(upper - lower)


def calculate_bond_length_w1(
    reference: dict,
    observed: dict,
    by: str = "environment",
    min_count: int = 20,
    value: str = "length",
) -> dict:
    """1-D Wasserstein distance of the bond lengths to a reference, per bond label.

    A bond is only compared with the reference bonds of its own label (by
    "environment" by default, or "bond_type"), so the lengths of one label
    cannot stand in for those of another. The labels with at least `min_count`
    bonds on both sides are averaged, weighted by their share of the reference
    bonds: how often the observed set uses each label does not enter, so a
    different mix of molecules alone does not raise the distance.

    Args:
        reference: bond table of the reference set (see collect_bonds)
        observed: bond table of the set to compare
        by: the label column to group by, "environment" or "bond_type"
        min_count: fewest bonds a label needs on each side to be compared
        value: the measured column, "length" for a bond table

    Returns:
        Dictionary with
        - w1: the weighted mean distance in Angstroms (np.nan if no label is compared)
        - reference_weight_compared: share of the reference bonds whose label was compared
        - observed_fraction_compared: share of the observed bonds whose label was compared
        - labels: per label, by reference count: n_reference, n_observed,
          weight (share of the reference bonds), and, if compared, w1,
          median_shift (observed minus reference, in Angstroms) and
          spread_ratio (observed over reference interquartile range);
          np.nan where not compared
    """
    reference_groups = _group_lengths(reference, by, value)
    observed_groups = _group_lengths(observed, by, value)
    n_reference_total = sum(len(lengths) for lengths in reference_groups.values())
    n_observed_total = sum(len(lengths) for lengths in observed_groups.values())

    rows = {}
    for label in sorted(
        reference_groups.keys() | observed_groups.keys(),
        key=lambda label: (-len(reference_groups.get(label, ())), label),
    ):
        reference_lengths = reference_groups.get(label, np.empty(0))
        observed_lengths = observed_groups.get(label, np.empty(0))
        row = {
            "n_reference": len(reference_lengths),
            "n_observed": len(observed_lengths),
            "weight": len(reference_lengths) / n_reference_total if n_reference_total else np.nan,
            "w1": np.nan,
            "median_shift": np.nan,
            "spread_ratio": np.nan,
        }
        if min(len(reference_lengths), len(observed_lengths)) >= min_count:
            reference_iqr = _iqr(reference_lengths)
            row.update(
                w1=float(wasserstein_distance(reference_lengths, observed_lengths)),
                median_shift=float(np.median(observed_lengths) - np.median(reference_lengths)),
                spread_ratio=_iqr(observed_lengths) / reference_iqr if reference_iqr > 0 else np.nan,
            )
        rows[label] = row

    compared = [row for row in rows.values() if not np.isnan(row["w1"])]
    n_reference_compared = sum(row["n_reference"] for row in compared)
    n_observed_compared = sum(row["n_observed"] for row in compared)
    return {
        "w1": (
            sum(row["n_reference"] * row["w1"] for row in compared) / n_reference_compared
            if compared
            else float("nan")
        ),
        "reference_weight_compared": (
            n_reference_compared / n_reference_total if n_reference_total else float("nan")
        ),
        "observed_fraction_compared": (
            n_observed_compared / n_observed_total if n_observed_total else float("nan")
        ),
        "labels": rows,
    }


def calculate_angle_w1(
    reference: dict,
    observed: dict,
    by: str = "environment",
    min_count: int = 20,
) -> dict:
    """1-D Wasserstein distance of the bond angles to a reference, per angle label.

    calculate_bond_length_w1 on an angle table (see collect_angles): the labels
    are angle_type or environment and every distance is in degrees.
    """
    return calculate_bond_length_w1(reference, observed, by, min_count, value="angle")
