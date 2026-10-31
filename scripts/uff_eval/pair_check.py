"""Check that row i of each samples database has the composition of row i of the reference.

    python pair_check.py <reference.db> <samples.db> ...
"""
import sys

from ase.db import connect

reference = [row.formula for row in connect(sys.argv[1]).select()]
for path in sys.argv[2:]:
    formulas = [row.formula for row in connect(path).select()]
    same = sum(a == b for a, b in zip(formulas, reference))
    print(f"{same}/{len(formulas)} rows pair with the reference: {path}")
