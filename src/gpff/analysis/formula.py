"""
Distances between the distributions of whole molecular formulas.

The element shares of ``gpff.analysis.composition`` pool every atom of every
molecule, so a model can match them and still put its nitrogen in the wrong
molecules: three N in a few molecules and none in the rest has the shares of one
N in every third. The formula of a molecule, C7H11NO, is the joint count of all
its elements, and the distribution of formulas is a much stricter test: it also
catches a wrong number of hydrogens for the heavy atoms, which is what turns a
molecule into a radical.

Everything works on *count tables*, ``{symbol: atoms of that element per
molecule}`` as ``gpff_prepare_metadata`` stores them (``element_counts.npz``, see
``collect_element_counts``), or on the ``(n_molecules, n_elements)`` matrices
``count_matrix`` makes of them.

What a number means is only clear next to two baselines, which is why both are
here:

    oracle      a model that draws, for every test molecule, a random training
                molecule of the same size. It is wrong only by sampling noise
                (and any difference between the splits), so its distance is the
                floor: a perfect model does not score 0, because two finite
                samples of the same distribution are never identical, and over
                hundreds of formulas the floor of the total variation is large.
    independent atoms
                a model that has the set's pooled element shares but draws every
                atom independently. Matching the shares, and nothing else, buys
                exactly this distance.

Sample i of a run is generated from test molecule i and has its number of atoms,
so the reference is that test molecule's formula and both distributions have the
same sizes by construction.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from ase.data import chemical_symbols


__all__ = [
    "QM9_VALENCE",
    "composition_strata",
    "count_distribution",
    "count_matrix",
    "count_w1",
    "distances",
    "element_shares",
    "formula_distances",
    "formula_distribution",
    "formula_feasible",
    "formula_ids",
    "hellinger",
    "hill_formula",
    "independent_atoms",
    "jensen_shannon",
    "oracle_floor",
    "resample_by_size",
    "standardized_validity",
    "symbols_of",
    "total_variation",
]

# the valence of every element of QM9, for formula_feasible
QM9_VALENCE: Dict[str, int] = {"H": 1, "C": 4, "N": 3, "O": 2, "F": 1}


def symbols_of(*tables: dict) -> list:
    """The symbols of every count table, once each, in the order of the atomic numbers."""
    symbols = set()
    for table in tables:
        symbols.update(table)
    return sorted(symbols, key=chemical_symbols.index)


def count_matrix(table: dict, symbols: Sequence[str], n_rows: Optional[int] = None) -> np.ndarray:
    """The count table as a matrix, one row per molecule and one column per symbol.

    A symbol the table lacks is a column of zeros, so tables of different
    elements can be put side by side.
    """
    if n_rows is None:
        n_rows = len(next(iter(table.values()))) if table else 0
    columns = [
        np.asarray(table[symbol], dtype=np.int64) if symbol in table
        else np.zeros(n_rows, dtype=np.int64)
        for symbol in symbols
    ]
    if not columns:
        return np.zeros((n_rows, 0), dtype=np.int64)
    return np.stack(columns, axis=1)


def hill_formula(counts: Sequence[int], symbols: Sequence[str]) -> str:
    """The formula in Hill order: C, then H, then the rest by name (all by name without C)."""
    parts = {symbol: int(count) for symbol, count in zip(symbols, counts) if count > 0}
    if "C" in parts:
        order = [s for s in ("C", "H") if s in parts] + sorted(
            s for s in parts if s not in ("C", "H")
        )
    else:
        order = sorted(parts)
    return "".join(s + (str(parts[s]) if parts[s] > 1 else "") for s in order)


def formula_ids(*matrices: np.ndarray) -> Tuple[list, np.ndarray]:
    """One id per molecule, the same id for the same formula in every matrix.

    Returns the id arrays, one per matrix, and the distinct formulas: row k is
    the count vector of the formulas with id k.
    """
    stacked = np.concatenate(matrices, axis=0)
    rows, inverse = np.unique(stacked, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    cuts = np.cumsum([len(matrix) for matrix in matrices])[:-1]
    return np.split(inverse, cuts), rows


def formula_distribution(ids: np.ndarray, n_formulas: int) -> np.ndarray:
    """The share of the molecules with each formula id (see formula_ids)."""
    if len(ids) == 0:
        return np.zeros(n_formulas)
    return np.bincount(ids, minlength=n_formulas) / len(ids)


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """Half the L1 distance: the share of the mass that would have to move.

    For formulas it is the smallest share of the molecules whose formula would
    have to change to make one set's formulas those of the other.
    """
    return float(0.5 * np.abs(p - q).sum())


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """The Jensen-Shannon divergence in bits, between 0 and 1.

    Finite where one distribution has formulas the other lacks, unlike the KL
    divergence. Like the distance of ``calculate_js_divergence``: the divergence,
    not its square root.
    """
    m = 0.5 * (p + q)

    def kl(a: np.ndarray) -> float:
        keep = a > 0
        return float((a[keep] * np.log2(a[keep] / m[keep])).sum())

    return 0.5 * kl(p) + 0.5 * kl(q)


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """The Hellinger distance, between 0 and 1."""
    return float(np.sqrt(0.5 * ((np.sqrt(p) - np.sqrt(q)) ** 2).sum()))


def distances(p: np.ndarray, q: np.ndarray) -> Dict[str, float]:
    """tv, js and hellinger of two distributions over the same categories."""
    return {
        "tv": total_variation(p, q),
        "js": jensen_shannon(p, q),
        "hellinger": hellinger(p, q),
    }


def formula_distances(
    generated: np.ndarray, reference: np.ndarray
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """The distances between the formula distributions of two sets of molecules.

    Both matrices need the same columns (see count_matrix). Returns the
    distances, the two distributions over the distinct formulas, and those
    formulas (rows of counts).
    """
    (generated_ids, reference_ids), rows = formula_ids(generated, reference)
    p = formula_distribution(generated_ids, len(rows))
    q = formula_distribution(reference_ids, len(rows))
    return distances(p, q), p, q, rows


def element_shares(counts: np.ndarray) -> np.ndarray:
    """The share of all atoms each column (element) has."""
    total = counts.sum()
    return counts.sum(axis=0) / total if total else np.zeros(counts.shape[1])


def resample_by_size(
    pool: np.ndarray, sizes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """For every size, the counts of a random molecule of the pool with that many atoms.

    A size the pool has no molecule of takes the nearest size it has.
    """
    pool_sizes = pool.sum(axis=1)
    if not len(pool_sizes):
        raise ValueError("The pool to resample from is empty.")
    available = np.unique(pool_sizes)
    drawn = np.empty((len(sizes), pool.shape[1]), dtype=pool.dtype)
    for size in np.unique(sizes):
        nearest = available[np.abs(available - size).argmin()]
        rows = np.flatnonzero(pool_sizes == nearest)
        targets = np.flatnonzero(sizes == size)
        drawn[targets] = pool[rng.choice(rows, size=len(targets))]
    return drawn


def independent_atoms(
    shares: np.ndarray, sizes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Molecules of the given sizes whose atoms are drawn independently with these shares."""
    shares = np.asarray(shares, dtype=float)
    shares = shares / shares.sum()
    drawn = np.zeros((len(sizes), len(shares)), dtype=np.int64)
    for size in np.unique(sizes):
        targets = np.flatnonzero(sizes == size)
        drawn[targets] = rng.multinomial(int(size), shares, size=len(targets))
    return drawn


def oracle_floor(
    pool: np.ndarray,
    reference: np.ndarray,
    n_draws: int = 20,
    seed: int = 0,
) -> Dict[str, Tuple[float, float]]:
    """What a perfect model scores: ``{metric: (mean, sd)}`` of the distances.

    Each draw gives every reference molecule a random molecule of the pool with
    the same size, and measures the formula distance of the draws to the
    reference. The pool is training molecules, so this is the distance of two
    finite samples of the data distribution: the lowest a model that generates
    the formulas can expect.
    """
    rng = np.random.default_rng(seed)
    sizes = reference.sum(axis=1)
    values: Dict[str, list] = {"tv": [], "js": [], "hellinger": []}
    for _ in range(n_draws):
        drawn = resample_by_size(pool, sizes, rng)
        measured, _, _, _ = formula_distances(drawn, reference)
        for name, value in measured.items():
            values[name].append(value)
    return {name: (float(np.mean(v)), float(np.std(v))) for name, v in values.items()}


def formula_feasible(
    counts: np.ndarray,
    symbols: Sequence[str],
    valence: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """Whether a neutral closed-shell molecule can have each formula.

    Two conditions every such molecule meets, whatever its geometry:

    * the valences add up to an even number, as every bond has two ends. For
      the elements of QM9 this is the nitrogen rule: N + H + F is even;
    * the degree of unsaturation (2C + 2 + N - H - F) / 2 is not negative: there
      are not more H and F than the heavy atoms can hold.

    A formula that fails either is a radical (or has no molecule at all) however
    well the atoms are placed, so a model that generates such formulas loses
    validity in its type head, not in its geometry. Only meaningful for elements
    of fixed valence parity: the default is QM9's, and an element not in
    ``valence`` (an undecided type that leaked into a sample, say) makes the
    formula infeasible.
    """
    valence = QM9_VALENCE if valence is None else valence
    symbols = list(symbols)
    known = np.array([symbol in valence for symbol in symbols], dtype=bool)
    weights = np.array([valence.get(symbol, 0) for symbol in symbols], dtype=np.int64)

    def column(symbol: str) -> np.ndarray:
        if symbol in symbols:
            return counts[:, symbols.index(symbol)]
        return np.zeros(len(counts), dtype=counts.dtype)

    unknown_atoms = counts[:, ~known].sum(axis=1)
    even = (counts @ weights) % 2 == 0
    twice_unsaturation = (
        2 * column("C") + 2 + column("N") - column("H") - column("F")
    )
    return (unknown_atoms == 0) & even & (twice_unsaturation >= 0)


def composition_strata(
    counts: np.ndarray,
    symbols: Sequence[str],
    elements: Sequence[str] = ("N", "O"),
    cap: int = 3,
) -> np.ndarray:
    """One stratum per molecule from its number of atoms of the given elements.

    Each count is capped at ``cap``, so there are (cap + 1) ** len(elements)
    strata, 16 for N and O.
    """
    symbols = list(symbols)
    strata = np.zeros(len(counts), dtype=np.int64)
    for symbol in elements:
        column = (
            counts[:, symbols.index(symbol)]
            if symbol in symbols
            else np.zeros(len(counts), dtype=np.int64)
        )
        strata = strata * (cap + 1) + np.minimum(column, cap)
    return strata


def standardized_validity(
    valid: np.ndarray, strata: np.ndarray, reference_strata: np.ndarray
) -> float:
    """The validity the set would have with the reference's mix of strata.

    A model that leaves out its hard molecules, nitrogen for one, scores a higher
    validity for it. This is the validity within each stratum, weighted by the
    reference's share of that stratum instead of the model's own. A stratum the
    reference has and the set has no molecule of counts as 0 valid: a model that
    cannot generate a kind of molecule does not get credit for the rest.
    """
    n_strata = int(max(strata.max(initial=0), reference_strata.max(initial=0))) + 1
    total = np.bincount(strata, minlength=n_strata)
    good = np.bincount(strata[valid], minlength=n_strata)
    rate = np.divide(good, total, out=np.zeros(n_strata), where=total > 0)
    weight = np.bincount(reference_strata, minlength=n_strata) / len(reference_strata)
    return float((weight * rate).sum())


def count_distribution(counts: np.ndarray, symbols: Sequence[str], symbol: str, cap: int = 3) -> np.ndarray:
    """The share of the molecules with 0, 1, ... cap - 1 and cap or more atoms of an element."""
    symbols = list(symbols)
    column = (
        counts[:, symbols.index(symbol)]
        if symbol in symbols
        else np.zeros(len(counts), dtype=np.int64)
    )
    return np.bincount(np.minimum(column, cap), minlength=cap + 1) / max(len(column), 1)


def count_w1(a: np.ndarray, b: np.ndarray) -> float:
    """The Wasserstein-1 distance of two distributions of atoms per molecule, in atoms."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    top = int(max(a.max(initial=0), b.max(initial=0))) + 1
    cdf_a = np.cumsum(np.bincount(a, minlength=top)) / max(len(a), 1)
    cdf_b = np.cumsum(np.bincount(b, minlength=top)) / max(len(b), 1)
    return float(np.abs(cdf_a - cdf_b).sum())
