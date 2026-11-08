"""Paired relaxation outcomes and displacement diagnostics, with fixed atom order."""
from __future__ import annotations

import numpy as np

from gpff.analysis.structure import analyze_with_rdkit


def kabsch_rmsd(before, after) -> float:
    """RMSD in Angstroms after translation and proper rotation; no atom matching."""
    a, b = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("Positions must have the same (n_atoms, 3) shape.")
    if not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    return float(np.sqrt(np.square(a @ rotation - b).sum(1).mean()))


def internal_displacement(displacement, positions) -> np.ndarray:
    """Remove translation and the best infinitesimal rotation from an atomic step."""
    r = np.asarray(positions, dtype=float)
    v = np.asarray(displacement, dtype=float)
    r, v = r - r.mean(0), v - v.mean(0)
    cross = np.stack([np.cross(axis, r) for axis in np.eye(3)], axis=-1)
    omega, *_ = np.linalg.lstsq(cross.reshape(-1, 3), v.reshape(-1), rcond=None)
    return v - cross @ omega


def compare_pair(before, after, data: dict) -> dict:
    """Analyze a source and relaxed row; their atomic numbers and order must agree.

    Canonical isomeric SMILES changes include stereochemical changes. Strain
    is taken from the relaxation's own energies, never recomputed with another
    method. Raw and aligned RMSD remain available even for invalid molecules.
    """
    if not np.array_equal(before.numbers, after.numbers):
        raise ValueError("Paired structures must have the same atoms in the same order.")
    a, b = (analyze_with_rdkit(atoms, remove_hs=True) for atoms in (before, after))
    energies = [data.get(key, np.nan) for key in ("energy_before", "energy_after")]
    return {
        "valid_before": bool(a["rdkit_valid"]),
        "valid_after": bool(b["rdkit_valid"]),
        "failure_before": a["failure"],
        "failure_after": b["failure"],
        "smiles_before": a["canonical_smiles"],
        "smiles_after": b["canonical_smiles"],
        "relaxed": bool(data["relaxed"]),
        "converged": bool(data["converged"]),
        "strain": float(energies[0] - energies[1]),
        "rmsd": float(np.sqrt(np.square(before.positions - after.positions).sum(1).mean()))
        if len(before) else float("nan"),
        "aligned_rmsd": kabsch_rmsd(before.positions, after.positions),
        "bond_orders": bool(data.get("bond_orders", False)),
    }


def finite_summary(values) -> dict:
    """Finite count, mean, median and p90; None for an empty statistic (valid JSON)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "n": len(values),
        "mean": float(values.mean()) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "p90": float(np.percentile(values, 90)) if len(values) else None,
    }


def summarize_pairs(records: list[dict]) -> dict:
    """Counts over every pair and strain/displacement statistics in explicit cohorts."""
    get = lambda key: np.array([r[key] for r in records])
    before, after = get("valid_before").astype(bool), get("valid_after").astype(bool)
    relaxed, converged = get("relaxed").astype(bool), get("converged").astype(bool)
    both = before & after
    same = both & (get("smiles_before") == get("smiles_after"))
    cohorts = {
        "all": np.ones(len(records), dtype=bool),
        "initially_valid": before,
        "finally_valid": after,
        "stayed_valid": both,
        "same_identity": same,
        "gained": ~before & after,
        "lost": before & ~after,
    }
    return {
        "n_structures": len(records),
        "valid_before": int(before.sum()),
        "valid_after": int(after.sum()),
        "gained": int(cohorts["gained"].sum()),
        "lost": int(cohorts["lost"].sum()),
        "smiles_changed": int((both & ~same).sum()),
        "smiles_compared": int(both.sum()),
        "relaxed": int(relaxed.sum()),
        "converged": int((relaxed & converged).sum()),
        "uff_connectivity_only": int((relaxed & ~get("bond_orders").astype(bool)).sum()),
        "cohorts": {
            name: {
                "n": int(mask.sum()),
                "relaxed": int((mask & relaxed).sum()),
                "converged": int((mask & relaxed & converged).sum()),
                # Failed setups keep their geometry and do not contribute a
                # spurious zero strain or displacement to relaxation statistics.
                **{key: finite_summary(get(key)[mask & relaxed])
                   for key in ("strain", "rmsd", "aligned_rmsd")},
                "converged_strain": finite_summary(get("strain")[mask & relaxed & converged]),
            }
            for name, mask in cohorts.items()
        },
    }
