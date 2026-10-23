"""Relax every structure of a database into a new one (``gpff_relax``).

    gpff_relax --samples_db_path samples.db --output_db_path relaxed.db --method uff

The output holds the relaxed structures in the input's order, with the input's
row data and key-value pairs kept, plus

    energy_before, energy_after  the method's energies in kcal/mol (nan if the
                                 structure could not be relaxed)
    steps                        minimizer steps taken (method mace only)
    strain                       energy_before - energy_after, the energy the
                                 relaxation released
    relaxed, converged           whether the method ran, and whether it
                                 converged within its iteration limit

A structure the method cannot set up is written unchanged with relaxed=False, so
the rows still line up with the input and with a split file.

Relaxation is how a geometry is told apart from its chemistry: the bond lengths
and angles of a relaxed structure are the ones its bond graph asks for, so what
the relaxation had to move (strain, and the bond_length_w1 / bond_angle_w1 of
the structure before against after) measures the geometry a model got wrong
rather than the molecule it chose. Run gpff_prepare_metadata and
gpff_calculate_metrics on both databases to compare them.

Methods:
    uff   RDKit's UFF on the bonds perceived from the geometry; never changes
          the bond graph, needs no extra software, and is the cheapest
    xtb   GFN2-xTB, by running the xtb program (--xtb_command, XTBPATH); takes
          no bonds, so they can form and break
    mace  a MACE model, many molecules at once on the GPU (--mace_model_path);
          by far the fastest per structure and the most accurate of the three
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ase.db import connect
from tqdm import tqdm

import gpff  # noqa: F401
from gpff.analysis import relax_with_uff, relax_with_xtb

METHODS = ("uff", "xtb", "mace")


def _relax_one(atoms, method: str, xtb_command: str) -> dict:
    """One structure through the single-structure methods."""
    if method == "uff":
        # without bond orders RDKit also reads a geometry no valid molecule fits
        return relax_with_uff(atoms, bond_orders=False)
    return relax_with_xtb(atoms, xtb_command=xtb_command)


def relax_database(
    samples_db_path: str | Path,
    output_db_path: str | Path,
    method: str = "uff",
    xtb_command: str = "xtb",
    mace_model_path: str | Path | None = None,
    batch_size: int = 256,
    n_workers: int = 1,
    verbose: bool = True,
) -> dict:
    """Relax every row of a database into a new one; returns the run's counts.

    uff and xtb relax one structure per task in a pool of n_workers processes,
    or in this process when n_workers is 1; mace relaxes batch_size molecules at
    a time on the GPU, so n_workers does not apply to it. The output database is
    overwritten.
    """
    output_db_path = Path(output_db_path)
    if output_db_path.exists():
        output_db_path.unlink()

    with connect(samples_db_path) as source:
        rows = [
            (row.toatoms(), dict(row.data), dict(row.key_value_pairs)) for row in source.select()
        ]
        metadata = source.metadata
    structures = [atoms for atoms, _, _ in rows]

    if method == "mace":
        if mace_model_path is None:
            raise ValueError("method mace needs mace_model_path")
        from gpff.analysis.mace_relaxation import relax_with_mace

        results = relax_with_mace(
            structures, mace_model_path, batch_size=batch_size, progress=verbose
        )
    else:
        results = [None] * len(structures)
        with tqdm(
            total=len(structures), desc=f"Relaxing with {method}", disable=not verbose
        ) as progress:
            if n_workers <= 1:
                for i, atoms in enumerate(structures):
                    results[i] = _relax_one(atoms, method, xtb_command)
                    progress.update(1)
            else:
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futures = {
                        pool.submit(_relax_one, atoms, method, xtb_command): i
                        for i, atoms in enumerate(structures)
                    }
                    for future in as_completed(futures):
                        results[futures[future]] = future.result()
                        progress.update(1)

    counts = {"n_structures": len(rows), "relaxed": 0, "converged": 0}
    with connect(output_db_path) as target:
        for (_, data, key_value_pairs), result in zip(rows, results):
            counts["relaxed"] += bool(result["relaxed"])
            counts["converged"] += bool(result["converged"])
            target.write(
                result["atoms"],
                data={
                    **data,
                    "energy_before": result["energy_before"],
                    "energy_after": result["energy_after"],
                    "strain": result["energy_before"] - result["energy_after"],
                    "relaxed": bool(result["relaxed"]),
                    "converged": bool(result["converged"]),
                    # only the batched minimizer counts its steps
                    **({"steps": result["steps"]} if "steps" in result else {}),
                },
                **key_value_pairs,
            )
        target.metadata = {**metadata, "relaxed_with": method}
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples_db_path", type=Path, required=True)
    parser.add_argument("--output_db_path", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, default="uff")
    parser.add_argument("--xtb_command", default="xtb", help="xtb: the executable to run.")
    parser.add_argument("--mace_model_path", type=Path, default=None,
                        help="mace: the model to relax with.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="mace: molecules relaxed at once on the GPU.")
    parser.add_argument("--n_workers", type=int, default=4,
                        help="uff and xtb: processes relaxing structures in parallel.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.samples_db_path.exists():
        print(f"ERROR: --samples_db_path not found: {args.samples_db_path}", file=sys.stderr)
        return 1
    if args.method == "mace" and args.mace_model_path is None:
        print("ERROR: --method mace needs --mace_model_path", file=sys.stderr)
        return 1

    print(f"Relaxing     : {args.samples_db_path}")
    print(f"Method       : {args.method}")
    print(f"Output       : {args.output_db_path}")
    counts = relax_database(
        args.samples_db_path,
        args.output_db_path,
        method=args.method,
        xtb_command=args.xtb_command,
        mace_model_path=args.mace_model_path,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        verbose=not args.quiet,
    )
    n = counts["n_structures"]
    print(f"Relaxed      : {counts['relaxed']} of {n}")
    print(f"Converged    : {counts['converged']} of {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
