"""EDM2 power-function EMA and post-hoc reconstruction (paper, Appendix C).

Independent implementation of the paper's equations; no EDM2 code dependency.
Training time is measured in optimizer updates, not batches or epochs.
"""
import copy
from pathlib import Path
import uuid

import numpy as np
import torch
from pytorch_lightning import Callback


def sigma_rel_to_gamma(sigma_rel):
    """Positive root of Appendix C's cubic, for 0 < sigma_rel < 1/sqrt(12)."""
    sigma = np.asarray(sigma_rel, dtype=np.float64)
    if not np.all(np.isfinite(sigma) & (sigma > 0) & (sigma < 12**-0.5)):
        raise ValueError("sigma_rel must be finite and between 0 and 1/sqrt(12)")
    roots = [np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()
             for s in sigma.flat]
    return np.asarray(roots).reshape(sigma.shape)


def profile_inner_product(t_a, gamma_a, t_b, gamma_b):
    """Analytic integral of two normalized power profiles (Appendix C)."""
    t_a, gamma_a, t_b, gamma_b = np.broadcast_arrays(
        *[np.asarray(x, dtype=np.float64) for x in (t_a, gamma_a, t_b, gamma_b)]
    )
    early, late = np.minimum(t_a, t_b), np.maximum(t_a, t_b)
    late_gamma = np.where(t_a <= t_b, gamma_b, gamma_a)
    return ((gamma_a + 1) * (gamma_b + 1) / (gamma_a + gamma_b + 1)
            * (early / late)**late_gamma / late)


def reconstruction_weights(steps, sigma_rels, target_step, target_sigma_rel):
    """Solve the continuous L2 fit; coefficients can be negative.

    Normalize their sum as in NVIDIA's reference implementation. Returns coefficients
    and relative L2 profile error, which measures fit, not molecular quality.
    """
    steps = np.asarray(steps, dtype=np.float64)
    gammas = sigma_rel_to_gamma(sigma_rels)
    target_gamma = sigma_rel_to_gamma(target_sigma_rel)
    if (steps.ndim != 1 or steps.size == 0 or gammas.shape != steps.shape
            or not np.all(np.isfinite(steps) & (steps > 0) & (steps <= target_step))
            or target_gamma.ndim != 0 or not np.isfinite(target_step)):
        raise ValueError("Expected matching nonempty profiles at 0 < step <= target_step")
    if len(set(zip(steps, gammas))) != len(steps):
        raise ValueError("Duplicate source profiles make reconstruction singular")
    # Rescale time to avoid unnecessarily small entries for long training runs.
    times = steps / target_step
    gram = profile_inner_product(times[:, None], gammas[:, None], times, gammas)
    rhs = profile_inner_product(times, gammas, 1, target_gamma)
    weights = np.linalg.solve(gram, rhs)
    total = weights.sum()
    if not np.all(np.isfinite(weights)) or abs(total) < 1e-12:
        raise ValueError("Unstable post-hoc EMA reconstruction")
    weights /= total
    norm = profile_inner_product(1, target_gamma, 1, target_gamma)
    error = max(0.0, float(weights @ gram @ weights - 2 * weights @ rhs + norm))
    return weights, float(np.sqrt(error / norm))


def _atomic_save(obj, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


class PostHocEMA(Callback):
    """Track EDM2's two source EMAs without swapping GPFF's live model.

    Add alongside the existing validation EMA. Optimizer hooks observe raw
    weights before validation can replace them. Checkpoints carry both full
    precision averages and their clock, independently of snapshot precision.
    Supports one optimizer with ordinary, replicated model parameters.
    """

    def __init__(self, sigma_rels=(0.05, 0.10), every_n_steps=4096,
                 dirpath="posthoc_ema", snapshot_dtype="float32"):
        self.sigma_rels = tuple(float(s) for s in sigma_rels)
        self.gammas = sigma_rel_to_gamma(self.sigma_rels)
        if not self.sigma_rels or len(set(self.sigma_rels)) != len(self.sigma_rels):
            raise ValueError("sigma_rels must be nonempty and distinct")
        if isinstance(every_n_steps, bool) or int(every_n_steps) != every_n_steps or every_n_steps < 1:
            raise ValueError("every_n_steps must be a positive integer")
        if snapshot_dtype not in ("float16", "float32"):
            raise ValueError("snapshot_dtype must be float16 or float32")
        self.every_n_steps = int(every_n_steps)
        self.dirpath = Path(dirpath)
        self.snapshot_dtype = getattr(torch, snapshot_dtype)
        self.num_updates = 0
        self.shadows = []
        self.run_id = uuid.uuid4().hex
        self._handle = None
        self._saved_step = 0
        self._restored = False

    def state_dict(self):
        return dict(sigma_rels=self.sigma_rels, num_updates=self.num_updates,
                    shadows=self.shadows, run_id=self.run_id)

    def load_state_dict(self, state_dict):
        if tuple(state_dict["sigma_rels"]) != self.sigma_rels:
            raise ValueError("Cannot change source sigma_rels when resuming post-hoc EMA")
        self.num_updates = state_dict["num_updates"]
        self.shadows = state_dict["shadows"]
        self.run_id = state_dict["run_id"]
        self._restored = True

    def on_train_start(self, trainer, pl_module):
        if len(trainer.optimizers) != 1:
            raise ValueError("PostHocEMA requires a single optimizer")
        if trainer.global_step and not self._restored:
            raise ValueError("Cannot resume post-hoc EMA without its callback state; start a new run")
        if trainer.is_global_zero and self.dirpath.exists():
            saved = [int(p.stem.removeprefix("step_")) for p in self.dirpath.glob("step_*.pt")]
            if saved and max(saved) > self.num_updates:
                raise ValueError("Snapshots extend beyond this checkpoint; use a new dirpath for this continuation")
        params = dict(pl_module.model.named_parameters())
        if not self.shadows:
            self.shadows = [{name: p.detach().clone() for name, p in params.items()}
                            for _ in self.sigma_rels]
        if len(self.shadows) != len(self.sigma_rels):
            raise ValueError("Wrong number of resumed source averages")
        for shadow in self.shadows:
            if shadow.keys() != params.keys():
                raise ValueError("Model parameters differ from the resumed post-hoc EMA")
            for name, p in params.items():
                if shadow[name].shape != p.shape:
                    raise ValueError(f"Resumed EMA parameter shape mismatch: {name}")
                shadow[name] = shadow[name].to(p)
        # Step post-hooks run only when the optimizer actually executes, so
        # accumulation microbatches and AMP overflow skips are not averaged.
        self._handle = trainer.optimizers[0].register_step_post_hook(
            lambda *_: self._after_step(pl_module.model))

    @torch.no_grad()
    def _after_step(self, model):
        # Lightning may still call optimizer.step when training_step returns
        # None. With no gradients this makes no update. Under accumulation,
        # gradients from preceding microbatches can still produce a real step.
        parameters = list(model.parameters())
        if not any(p.grad is not None for p in parameters):
            return
        self.num_updates += 1
        for gamma, shadow in zip(self.gammas, self.shadows):
            beta = (1 - 1 / self.num_updates)**(float(gamma) + 1)
            torch._foreach_lerp_(list(shadow.values()), parameters, 1 - beta)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.num_updates and self.num_updates % self.every_n_steps == 0:
            self._save_snapshot(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        self._remove_hook()
        self._save_snapshot(trainer, pl_module)

    def on_exception(self, trainer, pl_module, exception):
        self._remove_hook()

    def _remove_hook(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _save_snapshot(self, trainer, pl_module):
        if not trainer.is_global_zero or not self.num_updates or self._saved_step == self.num_updates:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        template_path = self.dirpath / "model.pt"
        if template_path.exists():
            # The small identity file prevents mixing separate training runs.
            if (self.dirpath / "run_id").read_text().strip() != self.run_id:
                raise ValueError("Post-hoc EMA directory belongs to a different training run")
        else:
            model = copy.deepcopy(pl_module.model).cpu()
            model.do_postprocessing = True
            _atomic_save(model, template_path)
            (self.dirpath / "run_id").write_text(self.run_id + "\n")
        payload = self.state_dict()
        payload.update(global_step=trainer.global_step,
            epoch=trainer.current_epoch,
            shadows=[{n: p.detach().to(device="cpu", dtype=self.snapshot_dtype)
                      for n, p in shadow.items()} for shadow in self.shadows],
        )
        payload["buffers"] = {n: b.detach().cpu().clone()
                              for n, b in pl_module.model.named_buffers()}
        _atomic_save(payload, self.dirpath / f"step_{self.num_updates:09d}.pt")
        self._saved_step = self.num_updates


@torch.no_grad()
def reconstruct(dirpath, sigma_rel, step=None):
    """Stream one run's snapshots into an inference model on CPU.

    Only snapshots up to the requested stored endpoint are used. Buffers are
    copied from that endpoint, never averaged (matching EDM2). All inputs must
    be trusted local artifacts, as GPFF inference models are pickled modules.
    """
    directory = Path(dirpath)
    paths = sorted(directory.glob("step_*.pt"))
    if not paths:
        raise ValueError("No post-hoc EMA snapshots found")
    available = {int(p.stem.removeprefix("step_")): p for p in paths}
    step = max(available) if step is None else step
    if step not in available:
        raise ValueError("Target step must match a saved snapshot")
    paths = [available[t] for t in sorted(available) if t <= step]
    endpoint = torch.load(available[step], map_location="cpu", weights_only=True)
    stds = endpoint["sigma_rels"]
    steps = [int(p.stem.removeprefix("step_")) for p in paths for _ in stds]
    weights, error = reconstruction_weights(steps, list(stds) * len(paths), step, sigma_rel)
    if (directory / "run_id").read_text().strip() != endpoint["run_id"]:
        raise ValueError("Snapshot and model template belong to different runs")
    model = torch.load(directory / "model.pt", map_location="cpu", weights_only=False).float()
    parameters = dict(model.named_parameters())
    # Accumulate in float64: reconstruction can subtract large contributions.
    sums = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in parameters.items()}
    for index, path in enumerate(paths):
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        if (snapshot["run_id"] != endpoint["run_id"] or snapshot["sigma_rels"] != stds
                or snapshot["num_updates"] != steps[index * len(stds)]
                or len(snapshot["shadows"]) != len(stds)):
            raise ValueError(f"Incompatible post-hoc EMA snapshot: {path}")
        for j, shadow in enumerate(snapshot["shadows"]):
            if shadow.keys() != parameters.keys():
                raise ValueError(f"Snapshot parameter names differ: {path}")
            for name, value in shadow.items():
                if value.shape != parameters[name].shape:
                    raise ValueError(f"Snapshot shape differs for {name}: {path}")
                sums[name].add_(value.double(), alpha=float(weights[index * len(stds) + j]))
    for name, p in parameters.items():
        p.copy_(sums[name])
    buffers = dict(model.named_buffers())
    if buffers.keys() != endpoint["buffers"].keys():
        raise ValueError("Snapshot buffer names differ from the model template")
    for name, buffer in buffers.items():
        buffer.copy_(endpoint["buffers"][name])
    model.do_postprocessing = True
    model.eval()
    metadata = dict(sigma_rel=float(sigma_rel), step=step, run_id=endpoint["run_id"],
                    relative_profile_error=error, coefficients=weights.tolist(),
                    sources=[str(p) for p in paths], source_sigma_rels=list(stds))
    return model, metadata
