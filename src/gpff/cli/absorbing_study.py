"""Reproducible study entry point; execute only on Hydra inside Slurm/container.

    python -m gpff.cli.absorbing_study --help
"""
import argparse
import json
import os
import socket
from pathlib import Path
import time
import hydra
import numpy as np
import torch
from ase import Atoms
from ase.db import connect
from omegaconf import OmegaConf
from schnetpack import properties as P
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.utils import load_model
from gpff import keys, vocab
from gpff.cli import header
from gpff.sample.base import Calculator
from gpff.sample.absorbing import AbsorbingPrior
from gpff.sample.absorbing_study import CoupledAbsorbingSampler, ScheduledAbsorbingSampler
from gpff.sample.direct_denoising import DirectDenoisingSampler
from gpff.sample.priors import GaussianPrior
from gpff.sample.formula import FormulaPrior, build_formula_prior, file_sha256
from gpff.analysis.absorbing_diagnostics import DecisionDiagnostics, snapshot
from gpff.analysis.type_conditionals import conditional_metrics, fixed_geometry_decode


def sample(z, r, index):
    return {P.Z: z, P.R: r, P.n_atoms: torch.tensor([len(z)]), P.idx: torch.tensor([index]),
            P.cell: torch.zeros(1, 3, 3, dtype=r.dtype), P.pbc: torch.zeros(1, 3, dtype=torch.bool)}


def load_rows(args, partition):
    with np.load(args.split) as split:
        indices = split[partition + "_idx"][:args.n_samples]
    if len(indices) < args.n_samples:
        raise ValueError("Requested sample count exceeds the selected split")
    rows = []
    with connect(args.dataset) as db:
        for i in indices:
            atoms = db.get(int(i) + 1).toatoms()
            rows.append(sample(torch.tensor(atoms.numbers), torch.tensor(atoms.positions, dtype=torch.float64), int(i)))
    return rows


def batch_for(calc, rows):
    batch = calc.cast_batch(_atoms_collate_fn(rows))
    batch[keys.converged] = torch.zeros(len(rows), device=calc.device, dtype=torch.bool)
    batch[keys.nfe] = torch.zeros(len(rows), device=calc.device, dtype=torch.int32)
    return batch


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def provenance(args, config):
    paths = {k: v for k in ("model", "dataset", "split", "training_config", "novelty_assets", "formula")
             if (v := getattr(args, k, None))}
    return {"arguments": vars(args), "files": {k: {"path": v, "sha256": file_sha256(v)} for k, v in paths.items()},
            "training_config": OmegaConf.to_container(config, resolve=False),
            "resolved_model": OmegaConf.to_container(config.model, resolve=True),
            "resolved_transform": OmegaConf.to_container(config.generative_transform, resolve=True),
            "snapshot": os.environ.get("APPTAINERENV_PYTHONPATH", os.environ.get("PYTHONPATH")),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__,
            "container": os.environ.get("SIF"), "node": socket.gethostname(),
            "device": torch.cuda.get_device_name() if args.device.startswith("cuda") else args.device}


def conditionals(args, calc, config, out):
    rows = load_rows(args, "val")
    transform = hydra.utils.instantiate(config.generative_transform)
    records, decoded = [], []
    for sigma in args.sigmas:
        for corruption in ("training_transform", "raw_gaussian"):
            corrupted = []
            for row in rows:
                work = snapshot(row)
                # Same noise realization and slots across sigma and corruption.
                rng = torch.Generator().manual_seed(args.seed + int(row[P.idx]))
                noise = torch.randn(work[P.R].shape, generator=rng, dtype=work[P.R].dtype)
                if corruption == "training_transform":
                    work = transform(work, t=torch.tensor(sigma / config.globals.sigma_max), noise=noise)
                else:
                    clean = work[P.R] - work[P.R].mean(0)
                    work[P.R] = clean + sigma * noise
                    work[keys.sigma_curr] = clean.new_full((len(clean), 1), sigma)
                corrupted.append(work)
            for rate in args.mask_rates:
                for start in range(0, len(rows), args.batch_size):
                    subset = [snapshot(row) for row in corrupted[start:start+args.batch_size]]
                    for row in subset:
                        rng = torch.Generator().manual_seed(args.seed + int(row[P.idx]) + 100000)
                        row["true_z"] = row[P.Z].clone()
                        row[P.Z][torch.rand(len(row[P.Z]), generator=rng) < rate] = vocab.MASK_Z
                    batch = batch_for(calc, subset)
                    true_z = batch["true_z"].clone()
                    original = snapshot(batch)
                    pred = calc.calculate(batch, forces=True, x0=True)
                    p = pred["type_probabilities"]
                    masked = original[P.Z] == vocab.MASK_Z
                    rng = torch.Generator(device=calc.device).manual_seed(args.seed + start)
                    for m, row in enumerate(subset):
                        slots = original[P.idx_m] == m
                        selected = slots & masked
                        metrics = conditional_metrics(p, true_z, selected, rng)
                        metrics.update(index=int(row[P.idx]), sigma=sigma, requested_mask_rate=rate,
                            actual_mask_fraction=float(masked[slots].float().mean()), size=int(slots.sum()), corruption=corruption)
                        records.append(metrics)
                    if args.decode and sigma in args.decode_sigmas:
                        for draw in range(args.draws):
                            for mode, parallel in (("generated", False), ("clean", False), ("generated", True)):
                                rng = torch.Generator(device=calc.device).manual_seed(args.seed + start + draw * 100003)
                                z, calls = fixed_geometry_decode(calc, original, true_z, rng, mode=mode, parallel=parallel)
                                for m, row in enumerate(subset):
                                    slots = original[P.idx_m] == m
                                    decoded.append({"index": int(row[P.idx]), "sigma": sigma, "mask_rate": rate,
                                        "corruption": corruption, "draw": draw, "context": mode, "parallel": parallel,
                                        "formula": [int((z[slots] == k).sum()) for k in (1,6,7,8,9)],
                                        "reference_formula": [int((true_z[slots] == k).sum()) for k in (1,6,7,8,9)],
                                        "masked_accuracy": float((z[slots & masked] == true_z[slots & masked]).float().mean()) if bool((slots & masked).any()) else None,
                                        "batch_model_calls": calls})
    for name, values in (("conditionals", records), ("decodings", decoded)):
        with (out / (name + ".jsonl")).open("x") as stream:
            for value in values:
                stream.write(json.dumps(value, allow_nan=False) + "\n")
    # Full predictions for a manageable independent-inspection subset.
    torch.save({"batch": snapshot(original), "probabilities": p.detach().cpu()}, out / "conditional_state.pt")


def generation(args, calc, config, out):
    partition = "val" if args.command == "pilot" else "test"
    rows = load_rows(args, partition)
    kwargs = dict(calculator=calc, n_steps=args.steps, stochastic_lambda=1.0, f_max=0.0,
                  type_seed=args.type_seed, type_schedule=args.type_schedule,
                  remask_kernel=args.kernel, remask_prob=args.remask_strength,
                  remask_mode=args.legacy_remask_mode, recompute_forces=args.recompute,
                  reveal_block_size=args.block_size, reveal_power=args.reveal_power)
    if args.diagnostics:
        kwargs["diagnostics"] = DecisionDiagnostics(out / "diagnostics", steps=sorted(set([args.steps, 3*args.steps//4, args.steps//2, args.steps//4, 1])))
    if args.types_given:
        prior = GaussianPrior(sigma=config.globals.sigma_max)
        sampler = DirectDenoisingSampler(calculator=calc, n_steps=args.steps, stochastic_lambda=1.0)
    elif args.formula:
        prior = FormulaPrior(args.formula, mode=args.formula_mode, seed=args.seed)
        if prior.payload["dataset_sha256"] != file_sha256(args.dataset) or prior.payload["split_sha256"] != file_sha256(args.split):
            raise ValueError("Formula table was fitted to another dataset or training split")
        sampler = DirectDenoisingSampler(calculator=calc, n_steps=args.steps, stochastic_lambda=1.0)
    else:
        prior = AbsorbingPrior(sigma=config.globals.sigma_max)
        sampler = (ScheduledAbsorbingSampler if args.scheduled else CoupledAbsorbingSampler)(**kwargs)
    records = []
    production_calls = 0
    elapsed = time.perf_counter()
    with connect(str(out / "samples.db")) as db:
        for start in range(0, len(rows), args.batch_size):
            originals = rows[start:start+args.batch_size]
            batch = prior.sample_from_batch(_atoms_collate_fn(originals))
            # Fully connected edges stay valid through all coordinate updates.
            # Also validate fresh formula batches through the real calculator.
            batch = calc.update_neighbors(calc.cast_batch(batch))
            with torch.no_grad():
                final = sampler.denoise(batch)
            production_calls += getattr(sampler, "model_calls", int(final[keys.nfe].max()))
            if args.formula:
                prior.verify(final)
            for m, original in enumerate(originals):
                slots = final[P.idx_m] == m
                r, z = final[P.R][slots].numpy(), final[P.Z][slots].numpy()
                reference_z = original[P.Z].tolist()
                record = {"attempt": start+m, "size_source_index": int(original[P.idx]),
                    "size": len(z), "numbers": z.tolist(), "reference_numbers": reference_z,
                    "formula": [int((z == k).sum()) for k in (1,6,7,8,9)],
                    "nonfinite_geometry": not bool(np.isfinite(r).all()),
                    "unresolved_mask": int((z == vocab.MASK_Z).sum()), "absent": int((z == vocab.ABSENT_Z).sum()),
                    "nfe": int(final[keys.nfe][m]),
                    "requested_formula": final["requested_formula"][m].tolist() if args.formula else None}
                db.write(Atoms(numbers=z, positions=r), data=record)
                records.append(record)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - elapsed
    with (out / "attempts.jsonl").open("x") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    write_json(out / "generation.json", {"attempts": len(records), "partition": partition, "wall_seconds": wall,
        "mean_nfe": float(np.mean([r["nfe"] for r in records])), "total_molecule_nfe": sum(r["nfe"] for r in records),
        "production_batched_model_calls": production_calls,
        "diagnostic_calls": sampler.diagnostics.diagnostic_calls if getattr(sampler, "diagnostics", None) else 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("conditionals", "pilot", "generate", "formula"))
    for name in ("dataset", "split", "output"):
        parser.add_argument("--"+name, required=True)
    for name in ("model", "training-config", "novelty-assets", "formula"):
        parser.add_argument("--"+name)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42069)
    parser.add_argument("--type-seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--kernel", choices=("legacy", "random", "saved_confidence"), default="legacy")
    parser.add_argument("--legacy-remask-mode", choices=("random", "confidence"), default="confidence")
    parser.add_argument("--remask-strength", type=float, default=0.0)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--scheduled", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--types-given", action="store_true", help="Explicit sdd256 composition-given control")
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument("--reveal-power", type=float, default=1.0)
    parser.add_argument("--type-schedule", choices=("linear", "cosine"), default="cosine")
    parser.add_argument("--formula-mode", choices=("conditional_size", "joint"), default="conditional_size")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0,0.1,0.25,0.5,1,2,30])
    parser.add_argument("--mask-rates", type=float, nargs="+", default=[0.25,0.75,1])
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--decode-sigmas", type=float, nargs="+", default=[0,0.5,2,30])
    parser.add_argument("--draws", type=int, default=2)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Study execution requires a Slurm allocation")
    if args.command == "formula":
        build_formula_prior(args.dataset, args.split, args.output)
        return
    if not args.model or not args.training_config:
        parser.error("--model and --training-config are required")
    if args.n_samples <= 0 or args.batch_size <= 0 or args.steps <= 0:
        parser.error("Sample counts and steps must be positive")
    if args.formula and args.types_given:
        parser.error("--formula and --types-given are separate generator controls")
    if args.scheduled and (args.formula or args.types_given):
        parser.error("Formula/types-given controls currently require direct denoising")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    config = OmegaConf.load(args.training_config)
    manifest = provenance(args, config)
    write_json(out / "manifest.json", manifest)
    model = load_model(args.model, device="cpu").eval()
    if getattr(model.representation,"sigma_embedding",None) is not None and args.command != "conditionals" and not args.scheduled:
        parser.error("A sigma-conditioned generator requires the declared --scheduled sigma")
    calc = Calculator(model=model, device=args.device)
    with torch.no_grad():
        if args.command == "conditionals":
            conditionals(args, calc, config, out)
        else:
            generation(args, calc, config, out)
    manifest["status"] = "completed"
    write_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
