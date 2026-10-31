"""Quantiles of the per-step largest atomic shift from refine_dd.py --trace files.

    python trace_stats.py <trace.npy> ...
"""
import sys

import numpy as np

for path in sys.argv[1:]:
    trace = np.load(path)
    print(f"== {path}  ({trace.shape[1]} molecules); largest atomic shift per step, A")
    print("  step     p50      p90      p99      max")
    for step in [0, 1, 2, 4, 9, 19, 49, 99, 149, 199]:
        if step < len(trace):
            q = np.percentile(trace[step], [50, 90, 99, 100])
            print(f"  {step + 1:4d}  " + "  ".join(f"{v:7.4f}" for v in q))
