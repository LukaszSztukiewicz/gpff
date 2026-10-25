"""Evaluate the learned pseudo-force field on held-out molecules (``gpff_landscape_report``).

Writes one JSON report about the first --n_molecules of the test split:

    denoising_error  e(sigma) on a log grid, against the aligned (GPFF) and the
                     unaligned (score model) targets, overall and per molecule
                     size, the perfect floor 1/(N-1) at small sigma, and the
                     integral over log(sigma); no sampler involved
    fixed_point      how far one denoising step moves a clean molecule (Å)
    jacobian         asymmetry, kept rigid motions and internal eigenvalues of
                     dD/dx at the clean molecules (the first --n_jacobian)
    relaxation       noise-free relaxation x <- D(x) from x0 + sigma * eps:
                     validity, same molecule (SMILES) as x0, RMSD to x0
    regeneration     stochastic Heun from x0 + sigma * eps down to 0. The start
                     is an exact sample at sigma, so a perfect field keeps the
                     test set's own validity (reference_validity) at every sigma

gpff.analysis.landscape lists the values of a perfect landscape. For
time-agnostic force models (GPFF); pass --relax_sigmas or --regen_sigmas
without values to skip that section.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from pytorch_lightning import seed_everything
from schnetpack import properties
from schnetpack.nn import scatter_add
from schnetpack.utils import load_model

from gpff.analysis import analyze_with_rdkit
from gpff.analysis.landscape import (
    center_per_molecule,
    denoise,
    denoiser_jacobians,
    denoising_errors,
    jacobian_statistics,
    mean_free_noise,
)
from gpff.data.datasets.qm9 import QM9Valid
from gpff.sample import Calculator, DirectDenoisingSampler, EDMNoiseSchedule, StochasticHeunSampler
from gpff.utils.alignment import align

# the lowest sigma of the EDM schedule; regeneration has to start above it
EDM_SIGMA_MIN = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--split_path", type=Path, required=True,
                        help="Split file of the run that trained the model; its test split is used.")
    parser.add_argument("--datapath", type=Path, required=True, help="The filtered QM9 database.")
    parser.add_argument("--output", type=Path, default=Path("landscape_report.json"))
    parser.add_argument("--n_molecules", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=np.geomspace(0.005, 30, 25).round(4).tolist(),
                        help="Noise levels of the denoising error curve. Default: 25 from 0.005 to 30.")
    parser.add_argument("--n_jacobian", type=int, default=200,
                        help="Number of molecules whose Jacobian is analysed (0 skips it).")
    parser.add_argument("--relax_sigmas", type=float, nargs="*", default=[0.1, 0.3, 1.0, 3.0])
    parser.add_argument("--relax_steps", type=int, default=100)
    parser.add_argument("--f_max", type=float, default=0.01,
                        help="Relaxation stops once every atom's |F| is below f_max (a step of f_max/2 Å).")
    parser.add_argument("--regen_sigmas", type=float, nargs="*", default=[0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    parser.add_argument("--regen_steps", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_workdir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.sigmas = sorted(args.sigmas)
    return args


def _to_atoms(numbers: torch.Tensor, positions: torch.Tensor, sizes: list[int]) -> list[Atoms]:
    return [
        Atoms(numbers=z.cpu().numpy(), positions=x.detach().cpu().numpy())
        for z, x in zip(numbers.split(sizes), positions.split(sizes))
    ]


def _outcomes(sampler, batch, x_start, x_0, references, sizes) -> list[dict]:
    """Run ``sampler`` from ``x_start``; compare each result with its x0."""
    final = sampler.denoise({**batch, properties.R: x_start})
    outcomes = []
    for atoms, x_ref, reference in zip(
        _to_atoms(final[properties.Z], final[properties.R], sizes), x_0.cpu().split(sizes), references
    ):
        analysis = analyze_with_rdkit(atoms)
        numbers = torch.from_numpy(atoms.numbers)
        aligned, _ = align(torch.from_numpy(atoms.positions), x_ref.double(), numbers, numbers)
        outcomes.append({
            "valid": analysis["rdkit_valid"],
            "same_molecule": analysis["rdkit_valid"]
            and analysis["canonical_smiles"] == reference["canonical_smiles"],
            "rmsd": ((aligned - x_ref.double()) ** 2).sum(-1).mean().sqrt().item(),
        })
    return outcomes


def _summarize_outcomes(outcomes: dict[float, list[dict]], n: int) -> dict:
    return {
        "sigma": list(outcomes),
        **{
            key: [float(np.mean([o[key] for o in per_sigma[:n]])) for per_sigma in outcomes.values()]
            for key in ("valid", "same_molecule", "rmsd")
        },
    }


def build_report(args, records: dict) -> dict:
    n = min(len(records["n_atoms"]), args.n_molecules)
    n_atoms = np.array(records["n_atoms"][:n])
    errors = {name: np.array(records[f"e_{name}"][:n]) for name in ("aligned", "unaligned")}
    log_sigmas = np.log(args.sigmas)
    jacobian = records["jacobian"]

    report = {
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "n_molecules": n,
        "reference_validity": float(np.mean(records["reference_valid"][:n])),
        "denoising_error": {
            "sigma": args.sigmas,
            "floor": float(np.mean(1 / (n_atoms - 1))),
            **{name: e.mean(0).tolist() for name, e in errors.items()},
            "log_sigma_integral": {
                name: float(np.trapezoid(e.mean(0), log_sigmas)) for name, e in errors.items()
            },
            "by_n_atoms": {
                str(size): {
                    "count": int((n_atoms == size).sum()),
                    "floor": 1 / (size - 1),
                    **{name: e[n_atoms == size].mean(0).tolist() for name, e in errors.items()},
                }
                for size in np.unique(n_atoms).tolist()
            },
        },
        "fixed_point": {
            "shift_rms_mean": float(np.mean(records["fixed_point_shift"][:n])),
            "shift_rms_max": float(np.max(records["fixed_point_shift"][:n])),
        },
    }
    if jacobian:
        report["jacobian"] = {
            "n_molecules": len(jacobian),
            **{key: float(np.nanmean([s[key] for s in jacobian])) for key in jacobian[0]},
            "unstable_fraction": float(np.mean([s["internal_max"] > 1 for s in jacobian])),
        }
    if records["relaxation"]:
        report["relaxation"] = _summarize_outcomes(records["relaxation"], n)
    if records["regeneration"]:
        report["regeneration"] = _summarize_outcomes(records["regeneration"], n)
    return report


def print_summary(report: dict) -> None:
    curve = report["denoising_error"]
    print(f"\nDenoising error e(sigma) over {report['n_molecules']} test molecules "
          f"(perfect: -> {curve['floor']:.4f} as sigma -> 0)")
    print(f"{'sigma':>8} {'aligned':>9} {'unaligned':>9}")
    for sigma, aligned, unaligned in zip(curve["sigma"], curve["aligned"], curve["unaligned"]):
        print(f"{sigma:8.4g} {aligned:9.4f} {unaligned:9.4f}")
    integral = curve["log_sigma_integral"]
    print(f"integral over log(sigma): aligned {integral['aligned']:.4f}, "
          f"unaligned {integral['unaligned']:.4f}")
    shift = report["fixed_point"]
    print(f"\nFixed point (perfect: 0): mean shift {shift['shift_rms_mean']:.4f} Å, "
          f"max {shift['shift_rms_max']:.4f} Å")
    if "jacobian" in report:
        jac = report["jacobian"]
        print(f"Jacobian over {jac['n_molecules']} molecules (perfect: 0 everywhere): "
              f"asymmetry {jac['asymmetry']:.4f}, rigid residual {jac['rigid_residual']:.4f}, "
              f"internal mean {jac['internal_mean']:.4f}, max {jac['internal_max']:.4f}, "
              f"unstable {jac['unstable_fraction']:.1%}")
    for section in ("relaxation", "regeneration"):
        if section not in report:
            continue
        print(f"\n{section.capitalize()} (test set validity {report['reference_validity']:.1%})")
        print(f"{'sigma':>8} {'valid':>7} {'same':>7} {'rmsd':>7}")
        outcomes = report[section]
        for row in zip(outcomes["sigma"], outcomes["valid"], outcomes["same_molecule"], outcomes["rmsd"]):
            print("{:8.4g} {:7.1%} {:7.1%} {:7.3f}".format(*row))


def main() -> int:
    args = parse_args()
    for name in ("model_path", "split_path", "datapath"):
        if not getattr(args, name).exists():
            print(f"ERROR: --{name} not found: {getattr(args, name)}", file=sys.stderr)
            return 1
    if any(sigma <= EDM_SIGMA_MIN for sigma in args.regen_sigmas):
        print(f"ERROR: --regen_sigmas must be above {EDM_SIGMA_MIN}.", file=sys.stderr)
        return 1

    # the samplers draw from the global RNGs, the noise of x0 + sigma * eps
    # from its own generator
    seed_everything(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    calculator = Calculator(model=load_model(str(args.model_path), device="cpu"), device=args.device)
    model = calculator.model.eval()
    relaxer = DirectDenoisingSampler(
        calculator=calculator, n_steps=args.relax_steps, stochastic_lambda=0.0, f_max=args.f_max
    )
    regenerators = {
        sigma: StochasticHeunSampler(
            calculator=calculator,
            n_steps=args.regen_steps,
            noise_schedule=EDMNoiseSchedule(n_steps=args.regen_steps, sigma_min=EDM_SIGMA_MIN, sigma_max=sigma),
        )
        for sigma in args.regen_sigmas
    }

    data = QM9Valid(
        datapath=str(args.datapath),
        batch_size=args.batch_size,
        split_file=str(args.split_path),
        load_properties=[],
        num_workers=args.num_workers,
        data_workdir=args.data_workdir,
    )
    data.prepare_data()
    data.setup()

    records = defaultdict(list)
    records["relaxation"] = defaultdict(list)
    records["regeneration"] = defaultdict(list)
    for batch in data.test_dataloader():
        if len(records["n_atoms"]) >= args.n_molecules:
            break
        batch = calculator.update_neighbors(calculator.cast_batch(batch))
        idx_m = batch[properties.idx_m]
        sizes = batch[properties.n_atoms].tolist()
        x_0 = center_per_molecule(batch[properties.R], idx_m)
        noise = mean_free_noise(x_0, idx_m, generator)
        records["n_atoms"] += sizes

        # 1. denoising error curve
        for aligned in (True, False):
            name = "aligned" if aligned else "unaligned"
            records[f"e_{name}"] += denoising_errors(model, batch, x_0, args.sigmas, noise, aligned).tolist()

        # 2. fixed point and Jacobian at the clean molecules
        with torch.no_grad():
            shift = ((denoise(model, batch, x_0) - x_0) ** 2).sum(-1)
        records["fixed_point_shift"] += (scatter_add(shift, idx_m, len(sizes)) / batch[properties.n_atoms]).sqrt().tolist()
        n_jacobian = min(len(sizes), args.n_jacobian - len(records["jacobian"]))
        if n_jacobian > 0:
            jacobians = denoiser_jacobians(model, batch, x_0)
            records["jacobian"] += [
                jacobian_statistics(jac, x) for jac, x in zip(jacobians[:n_jacobian], x_0.split(sizes))
            ]

        # 3. basins and regeneration from x0 + sigma * eps
        references = [analyze_with_rdkit(atoms) for atoms in _to_atoms(batch[properties.Z], x_0, sizes)]
        records["reference_valid"] += [r["rdkit_valid"] for r in references]
        for sigma in args.relax_sigmas:
            records["relaxation"][sigma] += _outcomes(relaxer, batch, x_0 + sigma * noise, x_0, references, sizes)
        for sigma, regenerator in regenerators.items():
            records["regeneration"][sigma] += _outcomes(regenerator, batch, x_0 + sigma * noise, x_0, references, sizes)
        print(f"evaluated {len(records['n_atoms'])} molecules", flush=True)

    report = build_report(args, records)
    print_summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
