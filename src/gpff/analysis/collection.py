"""Reductions over a sample set: one aggregate value from per-sample arrays/lists.

These are the reusable building blocks the evaluation is made of; which metric
needs which of them, and how they are wired together, lives in the
``gpff_calculate_metrics`` script.
"""
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

__all__ = [
    "calculate_js_divergence",
    "calculate_mean_tanimoto_similarity",
    "check_novelty",
    "check_uniqueness",
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
