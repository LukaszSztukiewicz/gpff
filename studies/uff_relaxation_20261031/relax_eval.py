"""RDKit validity and bond strain of sample sets, before and after relaxation with UFF, GFN2-xTB or
a relaxation computed elsewhere (e.g. MACE-OFF on the GPU, mace_relax.py).

    python relax_eval.py <out_dir> <reference_name>=<reference.db> <name>=<samples.db> ...
                         [--method uff|xtb|precomputed] [--relaxed_dir DIR] [--xtb_command xtb]
                         [--n_jobs 16] [--limit N] [--reuse] [--records_only]

The first set is the reference (QM9 test molecules, DFT geometries); its mean length per bond type
is the target for the bond-length error of every set, itself included. Per molecule:
- RDKit validity (gpff.analysis.analyze_with_rdkit) before and after relaxation.
- relaxation with
  - uff (gpff.analysis.relax_with_uff): valid molecules with their bond orders, invalid ones with
    the connectivity only, which is all RDKit can read from them. Bonds never change.
  - xtb (gpff.analysis.relax_with_xtb): every molecule, neutral and closed shell (each sample has
    the composition of a QM9 molecule). Bonds can form and break. Needs XTBPATH set.
  - precomputed: the relaxed structures in <relaxed_dir>/<name>.db, row i relaxing row i, with
    data energy_before / energy_after (kcal/mol), converged and relaxed.
- strain: energy released by the relaxation, kcal/mol (valid molecules only).
- bond-length error: mean |length - reference mean of its bond type| over the molecule's bonds,
  before and after relaxation on the same bonds (valid molecules only); summary.json also has it
  per bond type, over all bonds of that type, with the mean signed deviation (bias).
Writes <out_dir>/<name>.pkl (per molecule), <name>_relaxed.db (relaxed structures), summary.json
and summary.md. --reuse loads the sets whose <name>.pkl exists instead of recomputing them, and
--records_only stops before the summary, for computing sets in separate jobs.
"""
import argparse
import json
import os
import pickle
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
from ase.db import connect

from gpff.analysis import (
    analyze_with_rdkit,
    get_bond_lengths,
    mol_from_atoms,
    relax_with_uff,
    relax_with_xtb,
)


def evaluate(item, method, xtb_command):
    atoms, relaxation = item
    before = analyze_with_rdkit(atoms)
    valid = before["rdkit_valid"]
    if method == "uff":
        relaxation = relax_with_uff(atoms, bond_orders=valid)
    elif method == "xtb":
        relaxation = relax_with_xtb(atoms, xtb_command=xtb_command)
    relaxed = relaxation["atoms"]
    after = analyze_with_rdkit(relaxed) if relaxation["relaxed"] else before
    bonds_before = bonds_after = None
    if valid and relaxation["relaxed"]:
        mol = mol_from_atoms(atoms)
        bonds_before = get_bond_lengths(mol, atoms.positions)
        bonds_after = get_bond_lengths(mol, relaxed.positions)
    return {
        "valid_before": valid,
        "valid_after": after["rdkit_valid"],
        "smiles_before": before["canonical_smiles"],
        "smiles_after": after["canonical_smiles"],
        "relaxed": relaxation["relaxed"],
        "converged": relaxation["converged"],
        "energy_before": relaxation["energy_before"],
        "energy_after": relaxation["energy_after"],
        "bonds_before": bonds_before,
        "bonds_after": bonds_after,
        "relaxed_atoms": relaxed,
    }


def load_relaxations(path, limit):
    """Relaxations computed elsewhere, as the dictionaries relax_with_uff returns."""
    with connect(path) as db:
        return [
            {
                "atoms": row.toatoms(),
                "relaxed": bool(row.data["relaxed"]),
                "converged": bool(row.data["converged"]),
                "energy_before": row.data["energy_before"],
                "energy_after": row.data["energy_after"],
            }
            for row in db.select(limit=limit)
        ]


def run_set(path, out_dir, name, method, xtb_command, n_jobs, limit, reuse, relaxed_dir=None):
    records_path = os.path.join(out_dir, f"{name}.pkl")
    if reuse and os.path.exists(records_path):
        with open(records_path, "rb") as f:
            return pickle.load(f)
    with connect(path) as db:
        rows = [(row.toatoms(), row.data.get("nfe", np.nan)) for row in db.select(limit=limit)]
    relaxations = (
        load_relaxations(os.path.join(relaxed_dir, f"{name}.db"), limit)
        if method == "precomputed" else [None] * len(rows)
    )
    if len(relaxations) != len(rows):
        raise ValueError(f"{name}: {len(relaxations)} relaxed structures for {len(rows)} samples")
    with Pool(n_jobs) as pool:
        records = pool.map(
            partial(evaluate, method=method, xtb_command=xtb_command),
            [(atoms, relaxation) for (atoms, _), relaxation in zip(rows, relaxations)],
            chunksize=4,
        )

    relaxed_db = os.path.join(out_dir, f"{name}_relaxed.db")
    if os.path.exists(relaxed_db):
        os.remove(relaxed_db)
    with connect(relaxed_db) as db:
        for record in records:
            db.write(record.pop("relaxed_atoms"), data={k: record[k] for k in ["valid_before", "valid_after", "relaxed"]})
    for record, (_, nfe) in zip(records, rows):
        record["nfe"] = nfe
    with open(records_path, "wb") as f:
        pickle.dump(records, f)
    return records


def reference_lengths(records):
    """Mean length per bond type over the reference set's valid molecules."""
    lengths = {}
    for record in records:
        for label, values in (record["bonds_before"] or {}).items():
            lengths.setdefault(label, []).append(values)
    return {label: float(np.concatenate(values).mean()) for label, values in lengths.items()}


def bond_error(bonds, reference):
    deviations = [np.abs(values - reference[label]) for label, values in bonds.items() if label in reference]
    return float(np.concatenate(deviations).mean())


def bond_type_errors(records, reference):
    """Per bond type, over all bonds of that type in the valid molecules: count, mean absolute and
    mean signed deviation from the reference mean, before and after relaxation."""
    deviations = {}
    for record in records:
        if record["bonds_before"] is None:
            continue
        for label, before in record["bonds_before"].items():
            if label in reference:
                entry = deviations.setdefault(label, ([], []))
                entry[0].append(before - reference[label])
                entry[1].append(record["bonds_after"][label] - reference[label])
    table = {}
    for label, (before, after) in deviations.items():
        before, after = np.concatenate(before), np.concatenate(after)
        table[label] = {
            "n": int(len(before)),
            "mad_before": float(np.abs(before).mean()), "mad_after": float(np.abs(after).mean()),
            "bias_before": float(before.mean()), "bias_after": float(after.mean()),
        }
    return dict(sorted(table.items(), key=lambda item: -item[1]["n"]))


def summarize(records, reference):
    get = lambda key: np.array([r[key] for r in records])
    valid_before, valid_after, relaxed = get("valid_before"), get("valid_after"), get("relaxed")
    # strain and bond errors on the valid molecules that could be relaxed
    strained = valid_before & relaxed
    strain = (get("energy_before") - get("energy_after"))[strained]
    errors = np.array([
        [bond_error(r["bonds_before"], reference), bond_error(r["bonds_after"], reference)]
        for r in records if r["bonds_before"] is not None
    ])
    both = valid_before & valid_after
    smiles_changed = sum(r["smiles_before"] != r["smiles_after"] for r, b in zip(records, both) if b)
    return {
        "n": len(records),
        "nfe": float(np.nanmean(get("nfe"))),
        "valid_before": float(valid_before.mean()),
        "valid_after": float(valid_after.mean()),
        "gained": int((~valid_before & valid_after).sum()),
        "lost": int((valid_before & ~valid_after).sum()),
        "smiles_changed": int(smiles_changed),
        "relaxed_valid": float(relaxed[valid_before].mean()),
        "relaxed_invalid": float(relaxed[~valid_before].mean()) if (~valid_before).any() else float("nan"),
        "converged": float(get("converged")[relaxed].mean()),
        "strain_median": float(np.median(strain)),
        "strain_p90": float(np.percentile(strain, 90)),
        "bond_error_before": float(errors[:, 0].mean()),
        "bond_error_after": float(errors[:, 1].mean()),
        "bond_types": bond_type_errors(records, reference),
    }


def markdown(summaries, method):
    lines = [
        f"| set | NFE | valid % | valid % after {method} | gained / lost | strain median (p90), kcal/mol "
        f"| bond error, mA | bond error after {method}, mA |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        lines.append(
            f"| {name} | {s['nfe']:.0f} | {100 * s['valid_before']:.1f} | {100 * s['valid_after']:.1f} "
            f"| +{s['gained']} / -{s['lost']} | {s['strain_median']:.1f} ({s['strain_p90']:.1f}) "
            f"| {1000 * s['bond_error_before']:.1f} | {1000 * s['bond_error_after']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir")
    parser.add_argument("sets", nargs="+", help="name=path/to/samples.db, the reference first")
    parser.add_argument("--method", choices=["uff", "xtb", "precomputed"], default="uff")
    parser.add_argument("--relaxed_dir", default=None, help="precomputed: folder with <name>.db")
    parser.add_argument("--xtb_command", default="xtb")
    parser.add_argument("--n_jobs", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reuse", action="store_true", help="load sets whose <name>.pkl exists")
    parser.add_argument("--records_only", action="store_true", help="compute the sets, skip the summary")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sets = dict(item.split("=", 1) for item in args.sets)
    records = {}
    for name, path in sets.items():
        start = time.time()
        records[name] = run_set(
            path, args.out_dir, name, args.method, args.xtb_command, args.n_jobs, args.limit, args.reuse,
            args.relaxed_dir,
        )
        print(f"done {name} in {time.time() - start:.0f} s", flush=True)
    if args.records_only:
        return

    reference = reference_lengths(next(iter(records.values())))
    summaries = {name: summarize(r, reference) for name, r in records.items()}
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"method": args.method, "reference_lengths": reference, "sets": summaries}, f, indent=1)
    table = markdown(summaries, args.method)
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(table)
    print(table)


if __name__ == "__main__":
    main()
