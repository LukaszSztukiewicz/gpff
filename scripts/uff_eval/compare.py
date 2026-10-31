"""UFF and xTB results side by side, from the records of relax_eval.py.

    python compare.py <uff_dir> <xtb_dir> <reference> <name> ...

Sets whose records are missing in either folder are skipped. Bond errors use the reference set's
mean length per bond type, as in relax_eval.py.
"""
import os
import pickle
import sys

from relax_eval import reference_lengths, summarize

uff_dir, xtb_dir, names = sys.argv[1], sys.argv[2], sys.argv[3:]


def load(folder, name):
    path = os.path.join(folder, f"{name}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


reference = {"uff": load(uff_dir, names[0]), "xtb": load(xtb_dir, names[0])}
reference = {method: reference_lengths(records) for method, records in reference.items()}
print("| set | NFE | valid % | after UFF | after xTB | xTB +rescued / -broken | bond error, mA | after UFF | after xTB "
      "| strain UFF, kcal/mol | strain xTB, kcal/mol |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for name in names:
    records = {"uff": load(uff_dir, name), "xtb": load(xtb_dir, name)}
    if None in records.values():
        print(f"| {name} | (xTB still running) |")
        continue
    u, x = (summarize(records[m], reference[m]) for m in ["uff", "xtb"])
    nfe = "-" if u["nfe"] != u["nfe"] else f"{u['nfe']:.0f}"
    print(f"| {name} | {nfe} | {100 * u['valid_before']:.1f} | {100 * u['valid_after']:.1f} | {100 * x['valid_after']:.1f} "
          f"| +{x['gained']} / -{x['lost']} | {1000 * u['bond_error_before']:.1f} | {1000 * u['bond_error_after']:.1f} "
          f"| {1000 * x['bond_error_after']:.1f} | {u['strain_median']:.1f} | {x['strain_median']:.1f} |")
