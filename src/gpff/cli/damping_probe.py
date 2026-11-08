"""Probe DD oscillation and rigid-body drift (gpff_damping_probe).

Writes traces.npz, summary.json and provenance.json. Each eta starts from the
same saved geometries. Uses the saved GPFF force model, with no noise added.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase.db import connect
from schnetpack.utils import load_model

from gpff.analysis.damping import probe_damping
from gpff.analysis.relaxation_diagnostics import finite_summary
from gpff.cli import record_provenance
from gpff.sample.base import Calculator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--samples_db_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--etas", type=float, nargs="+", default=[1.0, 0.5, 1.5])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()
    if not args.samples_db_path.is_file():
        raise FileNotFoundError(args.samples_db_path)
    if args.limit < 1:
        parser.error("--limit must be positive")
    with connect(args.samples_db_path) as source:
        structures = [row.toatoms() for row in source.select(limit=args.limit)]
    calculator = Calculator(load_model(str(args.model_path), device="cpu"), device=args.device, dtype=args.dtype)
    probes = probe_damping(structures, calculator, args.etas, args.steps, args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "traces.npz", etas=args.etas, **{
        key: np.stack([probe[key] for probe in probes]) for key in probes[0] if key != "eta"
    })
    summary = {
        "settings": {"etas": args.etas, "steps": args.steps, "batch_size": args.batch_size,
                     "n_structures": len(structures), "device": args.device, "dtype": args.dtype},
        "probes": [
            {"eta": probe["eta"],
             "rmsd": finite_summary(probe["rmsd"]),
             "aligned_rmsd": finite_summary(probe["aligned_rmsd"]),
             "steps": [{key: finite_summary(probe[key][step]) for key in
                        ("max_shift", "step_cosine", "sigma", "translation", "internal_max_shift")}
                       for step in range(args.steps)]}
            for probe in probes
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    record_provenance(args.output_dir, "results", ["damping_probe"], inputs={
        "samples_db_path": args.samples_db_path, "model_path": args.model_path,
    })
    print(f"Saved {len(probes)} damping probes of {len(structures)} structures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
