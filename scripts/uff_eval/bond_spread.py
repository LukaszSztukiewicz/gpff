"""Spread of bond lengths per bond type in a set's records (relax_eval.py), before relaxation.

    python bond_spread.py <records.pkl> [label ...]
"""
import pickle
import sys

import numpy as np

records = pickle.load(open(sys.argv[1], "rb"))
labels = sys.argv[2:] or ["C-H", "C-C", "C-N", "C-O", "C=O", "C:C"]
print("| bond | count | mean, A | std, mA | mean abs deviation, mA | min, A | max, A |")
print("|---|---|---|---|---|---|---|")
for label in labels:
    values = np.concatenate([r["bonds_before"][label] for r in records if r["bonds_before"] and label in r["bonds_before"]])
    print(f"| {label} | {len(values)} | {values.mean():.3f} | {1000 * values.std():.1f} "
          f"| {1000 * np.abs(values - values.mean()).mean():.1f} | {values.min():.3f} | {values.max():.3f} |")
