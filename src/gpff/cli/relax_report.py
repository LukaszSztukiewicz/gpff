"""Compare a source database with its relaxed copy (gpff_relax_report).

Writes summary.json, summary.md, pairs.npz and provenance.json. No reference
distribution is needed: the rows must describe the same atoms in input order.
For distribution scores, evaluate separately against a reference relaxed with
the same method. Strain cohorts here explicitly distinguish validity before
and after relaxation; mean_strain in gpff_calculate_metrics uses validity after.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.db import connect
from tqdm import tqdm

from gpff.analysis.relaxation_diagnostics import compare_pair, summarize_pairs
from gpff.cli import _md5, record_provenance


def compare_databases(samples_db_path, relaxed_db_path, verbose=True):
    samples_db_path, relaxed_db_path = Path(samples_db_path), Path(relaxed_db_path)
    if not samples_db_path.is_file() or not relaxed_db_path.is_file():
        raise FileNotFoundError("Both source and relaxed databases must exist.")
    with connect(samples_db_path) as source, connect(relaxed_db_path) as target:
        if source.count() != target.count():
            raise ValueError("Source and relaxed databases have different row counts.")
        expected = target.metadata.get("relaxation_source_md5")
        if expected is not None and expected != _md5(samples_db_path):
            raise ValueError("The relaxed database was made from a different source database.")
        records = []
        for before, after in tqdm(
            zip(source.select(), target.select()), total=source.count(),
            desc="Comparing relaxation", disable=not verbose,
        ):
            data = dict(after.data)
            if not {"relaxed", "converged", "energy_before", "energy_after"} <= data.keys():
                raise ValueError(f"Relaxation metadata is missing from row {after.id}.")
            if data.get("relaxation_source_id", before.id) != before.id:
                raise ValueError(f"Source row order differs at relaxed row {after.id}.")
            records.append({
                **compare_pair(before.toatoms(), after.toatoms(), data),
                "source_id": before.id,
                "relaxed_id": after.id,
            })
        summary = summarize_pairs(records)
        summary["method"] = target.metadata.get("relaxed_with", "unknown")
        summary["settings"] = target.metadata.get("relaxation_settings", {})
        # bond_orders only describes UFF; do not label MACE/xTB as fallback UFF.
        if summary["method"] != "uff":
            summary.pop("uff_connectivity_only")
        summary["pairing"] = "source_md5_and_atom_order" if expected else "atom_order_only"
    return records, summary


def markdown(summary):
    lines = [
        f"# Paired relaxation ({summary['method']})", "",
        f"{summary['n_structures']} structures; validity before {summary['valid_before']}, "
        f"after {summary['valid_after']}; gained {summary['gained']}, lost {summary['lost']}.",
        f"Canonical isomeric SMILES changed in {summary['smiles_changed']} of "
        f"{summary['smiles_compared']} pairs valid on both sides.", "",
        "Strain is in kcal/mol; aligned RMSD is in Angstroms. Statistics include "
        "successful setups with finite values; converged counts are separate.", "",
        "| cohort | pairs | relaxed | converged | strain median | strain p90 | aligned RMSD median |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    fmt = lambda v: "-" if v is None else f"{v:.5g}"
    for name, row in summary["cohorts"].items():
        lines.append(
            f"| {name} | {row['n']} | {row['relaxed']} | {row['converged']} | "
            f"{fmt(row['strain']['median'])} | {fmt(row['strain']['p90'])} | "
            f"{fmt(row['aligned_rmsd']['median'])} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples_db_path", type=Path, required=True)
    parser.add_argument("--relaxed_db_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    records, summary = compare_databases(
        args.samples_db_path, args.relaxed_db_path, verbose=not args.quiet
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "pairs.npz",
        **{key: np.array([record[key] for record in records]) for key in (records[0] if records else [])},
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report = markdown(summary)
    (args.output_dir / "summary.md").write_text(report)
    record_provenance(args.output_dir, "results", ["paired_relaxation"], inputs={
        "samples_db_path": args.samples_db_path, "relaxed_db_path": args.relaxed_db_path,
    })
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
