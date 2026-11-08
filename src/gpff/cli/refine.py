"""Continue saved structures with deterministic DD (gpff_refine).

Preserves source row data and key-value pairs, adds dd_nfe / dd_converged /
dd_failed, and increments total nfe. Stale physical-relaxation energies and
status are removed. --trace_path saves a compressed NPZ with max_shift in
Angstroms (steps x molecules), dd_nfe and dd_converged; stopped steps are NaN.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from ase.db import connect
from schnetpack.utils import load_model

from gpff.sample.base import Calculator
from gpff.sample.refinement import refine_structures
from gpff.cli import _md5


def refine_database(samples_db_path, output_db_path, calculator, *, model_path=None,
                    f_max=0.05, max_steps=200, batch_size=128, trace_path=None):
    source_path, target_path = Path(samples_db_path), Path(output_db_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if target_path.exists():
        raise FileExistsError(target_path)
    if trace_path is not None:
        trace_path = Path(trace_path)
        if trace_path.exists():
            raise FileExistsError(trace_path)
        if trace_path.resolve() == target_path.resolve():
            raise ValueError("Trace and output database must be different files.")
    with connect(source_path) as source:
        rows, metadata = list(source.select()), source.metadata
    results, shifts = refine_structures(
        [row.toatoms() for row in rows], calculator, f_max, max_steps, batch_size,
        trace=trace_path is not None,
    )
    stale = {"energy_before", "energy_after", "strain", "relaxed", "converged",
             "steps", "bond_orders", "relaxation_source_id"}
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(target_path) as target:
        for row, result in zip(rows, results):
            data = {key: value for key, value in row.data.items() if key not in stale}
            nfe = int(data.get("nfe", 0)) + result["dd_nfe"]
            target.write(
                result["atoms"], **row.key_value_pairs,
                data={**data, **{k: v for k, v in result.items() if k != "atoms"},
                      "nfe": nfe, "refinement_source_id": row.id},
            )
        target.metadata = {
            **{k: v for k, v in metadata.items() if k != "relaxed_with" and not k.startswith("relaxation_")},
            "refinement": {
                "source_path": str(source_path.resolve()), "source_md5": _md5(source_path),
                "model_path": str(Path(model_path).resolve()) if model_path is not None else None,
                "model_md5": _md5(model_path) if model_path is not None else None,
                "f_max": f_max, "max_steps": max_steps, "batch_size": batch_size,
                "device": str(calculator.device), "dtype": str(calculator.dtype),
            },
        }
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        # A file handle preserves the exact requested filename.
        with trace_path.open("wb") as file:
            np.savez_compressed(file, max_shift=shifts,
                                dd_nfe=np.array([r["dd_nfe"] for r in results]),
                                dd_converged=np.array([r["dd_converged"] for r in results]))
    return {"n_structures": len(results),
            "converged": sum(r["dd_converged"] for r in results),
            "failed": sum(r["dd_failed"] for r in results),
            "additional_nfe": sum(r["dd_nfe"] for r in results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--samples_db_path", type=Path, required=True)
    parser.add_argument("--output_db_path", type=Path, required=True)
    parser.add_argument("--f_max", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--trace_path", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()
    calculator = Calculator(load_model(str(args.model_path), device="cpu"), device=args.device, dtype=args.dtype)
    print(refine_database(
        args.samples_db_path, args.output_db_path, calculator, model_path=args.model_path,
        f_max=args.f_max, max_steps=args.max_steps, batch_size=args.batch_size, trace_path=args.trace_path,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
