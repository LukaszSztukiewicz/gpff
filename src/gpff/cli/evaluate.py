"""Loss and force errors of a model on pinned val/test noise (``gpff_evaluate``).

    gpff_evaluate --model_path run/best_model --split_path run/split.npz \\
        --datapath qm9_filtered.db --output run/dev.npz [--compare other/dev.npz]

The inputs are made by the data transforms of the model's training config, the
.hydra/config.yaml next to --model_path; --config takes another run's config
and --experiment a packaged experiment instead. The split's molecules get the
pinned, stratified noise of gpff.data.eval_noise, n_draws per molecule, so two
models evaluated with the same config, split, subset, noise seed and n_draws
see identical inputs and their difference is measured pair-wise. Keep the
noise seeds apart by purpose:

    0   training's val/test sets (data.eval_noise_seed), used to select checkpoints
    1   your own comparisons, on val (the default)
    2   final numbers, on test, looked at once

Only force-field models are scored: one forces output weighted by
forces_weight, as in experiment=gpff, gpff_t_aware and gpff_no_permutation.

One row per (molecule, draw) goes to --output; without --model_path that file is
read instead, to report or compare results computed before. Printed are the
loss as training defines it, the forces MAE training logs, and per noise level
sigma:

    x0 err      RMS over atoms of |x0_pred - x0| = |F_pred - F| / 2, in Å
    noise err   the same error in units of sigma (the error of the predicted noise)
    sigma_hat   median of std(F_pred / 2) / sigma, the adaptive samplers' noise estimate

--compare REF prints this file's loss minus REF's, with a 95% bootstrap
interval over molecules; both files must hold the same inputs (molecules, noise
levels and targets are checked). The interval resamples molecules only: it
tells whether these two sets of weights differ on this split, not whether one
training setup beats another, which needs several training seeds.
"""
import argparse
import os
import sys
from typing import Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from schnetpack import properties

import gpff.cli  # noqa: F401  OmegaConf resolvers used by the configs
from gpff import keys
from gpff.data.eval_noise import pin_noise

SIGMA_BINS = [0, 0.03, 0.1, 0.3, 1, 3, 10, np.inf]


def load_config(args) -> Tuple[DictConfig, str]:
    """The training config whose data transforms make the inputs, and where it came from."""
    if args.experiment:
        with hydra.initialize_config_module("gpff.configs"):
            config = hydra.compose("train", overrides=[f"experiment={args.experiment}"])
        source = f"experiment={args.experiment}"
    else:
        path = args.config or os.path.join(
            os.path.dirname(os.path.abspath(args.model_path)), ".hydra", "config.yaml"
        )
        if not os.path.exists(path):
            sys.exit(f"ERROR: no training config at {path}; pass the run's --config or an --experiment")
        config = OmegaConf.load(path)
        source = os.path.abspath(path)
    # the default resolves through Hydra's runtime, which only gpff_train has
    config.run.work_dir = os.getcwd()
    return config, source


def check_force_field(config: DictConfig, source: str, model) -> None:
    """Exit unless the config trains, and the model predicts, the weighted forces scored here."""
    outputs = config.task.outputs
    output = outputs[0] if len(outputs) == 1 else None
    if not (
        output is not None
        and output.name == keys.forces
        and output.get("target_property") in (None, keys.forces)
        and output.get("weights_key") == keys.forces_weight
        and output.get("loss_weight", 1.0) == 1.0
        and output.get("scale_loss") is None
        and keys.forces in (model.model_outputs or [])
    ):
        sys.exit(
            f"ERROR: {source} does not train the force field scored here: one forces output "
            "weighted by forces_weight, loss weight 1, no scale_loss (e.g. experiment=gpff)"
        )


def evaluate(args, config: DictConfig, source: str) -> dict:
    if not os.path.exists(args.split_path):
        # schnetpack would draw a new split and write it there instead
        sys.exit(f"ERROR: no split file at {args.split_path}")
    model = torch.load(args.model_path, map_location=args.device, weights_only=False)
    check_force_field(config, source, model)
    # training computes its losses without postprocessing
    model.do_postprocessing = False
    model.eval()

    datamodule = hydra.utils.instantiate(
        config.data,
        datapath=args.datapath,
        data_workdir=args.data_workdir,
        split_file=args.split_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_sampler_cls=None,
        eval_noise_seed=None,
    )
    datamodule.setup()
    dataset = datamodule.val_dataset if args.subset == "val" else datamodule.test_dataset
    batches = pin_noise(dataset, args.noise_seed, args.batch_size, args.n_draws, args.num_workers)

    rows = {k: [] for k in (
        "idx", "sigma", "n_atoms", "loss_sum", "abs_sum", "x0_rmsd", "sigma_hat", "target_sq", "weight_sum"
    )}
    with torch.no_grad():
        for batch in batches:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            idx_m, n_atoms = batch[properties.idx_m], batch[properties.n_atoms]

            def per_molecule(x):
                return torch.zeros(len(n_atoms), dtype=x.dtype, device=x.device).index_add_(0, idx_m, x)

            # the model writes its outputs into the dict it gets: keep the targets
            pred = model(dict(batch))[keys.forces].double()
            target = batch[keys.forces].double()
            weights = batch[keys.forces_weight].double()
            error = pred - target
            half = pred / 2
            n = 3 * n_atoms
            rows["idx"].append(batch[properties.idx])
            rows["sigma"].append(per_molecule(batch[keys.sigma_curr][:, 0]) / n_atoms)
            rows["n_atoms"].append(n_atoms)
            rows["loss_sum"].append(per_molecule((weights * error**2).sum(1)))
            rows["abs_sum"].append(per_molecule(error.abs().sum(1)))
            rows["x0_rmsd"].append((per_molecule((error**2).sum(1)) / n_atoms).sqrt() / 2)
            # unbiased std over the molecule's values, as the samplers estimate sigma
            mean = per_molecule(half.sum(1)) / n
            rows["sigma_hat"].append(((per_molecule((half**2).sum(1)) - n * mean**2) / (n - 1)).sqrt())
            # a fingerprint of the targets, for --compare to check the inputs match
            rows["target_sq"].append(per_molecule((target**2).sum(1)))
            rows["weight_sum"].append(per_molecule(weights.sum(1)))

    results = {k: torch.cat(v).cpu().numpy() for k, v in rows.items()}
    results["draw"] = np.arange(len(results["idx"])) % args.n_draws
    return results


def report(r: dict) -> None:
    components = 3 * r["n_atoms"]
    print(f"loss         {r['loss_sum'].sum() / components.sum():.6f}")
    print(f"forces MAE   {r['abs_sum'].sum() / components.sum():.6f}  (per component, as training logs it)")
    print(f"\n{'sigma':>13}  {'share':>6}  {'loss':>8}  {'x0 err/Å':>8}  {'noise err':>9}  {'sigma_hat':>9}")
    bins = np.digitize(r["sigma"], SIGMA_BINS) - 1
    for b in range(len(SIGMA_BINS) - 1):
        m = bins == b
        if m.any():
            print(
                f"{SIGMA_BINS[b]:>5g} - {SIGMA_BINS[b + 1]:<5g}  {m.mean():6.1%}  "
                f"{r['loss_sum'][m].sum() / components[m].sum():8.5f}  {r['x0_rmsd'][m].mean():8.4f}  "
                f"{(r['x0_rmsd'][m] / r['sigma'][m]).mean():9.4f}  {np.median(r['sigma_hat'][m] / r['sigma'][m]):9.3f}"
            )


def check_same_inputs(r: dict, ref: dict) -> None:
    """Exit unless both files hold the same inputs: molecules, noise levels and targets."""
    if "target_sq" not in r or "target_sq" not in ref:
        sys.exit("ERROR: a results file predates the input check; evaluate its model again")
    if not np.array_equal(r["idx"], ref["idx"]):
        problem = "different molecules (split, subset or n_draws)"
    elif not np.allclose(r["sigma"], ref["sigma"], rtol=1e-6):
        problem = "different noise levels (noise seed or the noise sampler's settings)"
    elif not (
        np.allclose(r["target_sq"], ref["target_sq"], rtol=1e-5)
        and np.allclose(r["weight_sum"], ref["weight_sum"], rtol=1e-5)
    ):
        problem = "different targets (the transforms differ, e.g. align_permutation)"
    else:
        return
    sys.exit(
        f"ERROR: the files hold {problem}. Evaluate both models with the same split, subset, "
        "noise seed, n_draws and --config or --experiment"
    )


def compare(r: dict, ref: dict, n_boot: int = 1000) -> None:
    check_same_inputs(r, ref)
    # per molecule over its draws, so that the bootstrap resamples molecules
    _, molecule = np.unique(r["idx"], return_inverse=True)
    a, b, n = (np.bincount(molecule, x) for x in (r["loss_sum"], ref["loss_sum"], 3 * r["n_atoms"]))
    boot = np.random.default_rng(0).integers(0, len(n), (n_boot, len(n)))
    low, high = np.percentile((a[boot].sum(1) - b[boot].sum(1)) / n[boot].sum(1), [2.5, 97.5])
    print(
        f"\nloss - reference  {(a.sum() - b.sum()) / n.sum():+.6f}  (95% CI {low:+.6f} to {high:+.6f}), "
        f"lower on {np.mean(a < b):.1%} of the molecules"
    )
    print(
        "  the interval resamples molecules: it compares these two sets of weights on this split,\n"
        "  not two training setups, which needs several training seeds"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", required=True,
                        help="results (.npz), written with --model_path, read without")
    parser.add_argument("--model_path", help="inference model, e.g. a run's best_model")
    parser.add_argument("--split_path", help="the model's split.npz")
    parser.add_argument("--datapath", help="the database the split indexes")
    parser.add_argument("--data_workdir",
                        help="copy the database here first (node-local disk; reading a shared file system is slow)")
    parser.add_argument("--subset", choices=["val", "test"], default="val")
    parser.add_argument("--noise_seed", type=int, default=1)
    parser.add_argument("--n_draws", type=int, default=4, help="noise draws per molecule")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--config",
                        help="training config whose data transforms make the inputs "
                             "(default: .hydra/config.yaml next to --model_path)")
    inputs.add_argument("--experiment",
                        help="make the inputs with this packaged experiment's transforms instead, e.g. gpff")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compare", help="results of another model on the same inputs")
    args = parser.parse_args()
    if args.model_path and not (args.split_path and args.datapath):
        parser.error("--model_path needs --split_path and --datapath")
    return args


def main() -> int:
    args = parse_args()
    if args.model_path:
        config, source = load_config(args)
        results = evaluate(args, config, source)
        results.update(
            model_path=os.path.abspath(args.model_path),
            config=source,
            split_path=os.path.abspath(args.split_path),
            subset=args.subset,
            noise_seed=args.noise_seed,
            n_draws=args.n_draws,
        )
        np.savez(args.output, **results)
    else:
        results = dict(np.load(args.output))
    print(f"model        {results['model_path']}")
    print(f"config       {results.get('config', 'not recorded')}")
    print(f"inputs       {results['split_path']}  {results['subset']}, noise seed {results['noise_seed']}, "
          f"{len(np.unique(results['idx']))} molecules x {results['n_draws']} draws")
    report(results)
    if args.compare:
        compare(results, dict(np.load(args.compare)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
