"""Decision-aligned trajectory diagnostics. Probe calls preserve state and RNG."""
from contextlib import contextmanager
import copy
import json
import random
from pathlib import Path
import numpy as np
import torch
from schnetpack import properties as P
from gpff import keys, vocab


def snapshot(batch):
    return {k: v.detach().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
            for k, v in batch.items()}


@contextmanager
def isolated_rng():
    py, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


def probe(calculator, batch, remask=False):
    """Use a cloned input and restore model training flags and all global RNGs."""
    work = snapshot(batch)
    if remask:
        work[P.Z].fill_(vocab.MASK_Z)
    flags = [(m, m.training) for m in calculator.model.modules()]
    buffers = [(b, b.detach().clone()) for b in calculator.model.buffers()]
    try:
        with isolated_rng(), torch.no_grad():
            calculator.model.eval()
            return calculator.calculate(work, forces=True, x0=True, estimate_sigma=True)
    finally:
        for module, flag in flags:
            module.training = flag
        with torch.no_grad():
            for buffer, old in buffers:
                buffer.copy_(old)


def finite_atoms(batch, probabilities, *additional):
    finite = torch.isfinite(batch[P.R]).all(-1) & torch.isfinite(probabilities).all(-1)
    for value in additional:
        if value is None:
            continue
        valid = torch.isfinite(value).reshape(len(value), -1).all(-1)
        if len(value) == len(batch[P.n_atoms]):
            valid = valid[batch[P.idx_m]]
        finite &= valid
    bad = torch.zeros(len(batch[P.n_atoms]), device=finite.device, dtype=torch.long)
    bad.scatter_reduce_(0, batch[P.idx_m], (~finite).long(), reduce="amax", include_self=True)
    return ~bad.bool()[batch[P.idx_m]]


def count_proxy(z, probabilities, idx_m, n_molecules):
    masked = z == vocab.MASK_Z
    mass = probabilities.clone()
    mass[~masked] = (z[~masked, None] == z.new_tensor(vocab.TOKEN_Z)[None, :]).to(mass.dtype)
    counts = mass.new_zeros((n_molecules, mass.shape[-1]))
    counts.index_add_(0, idx_m, mass)
    return counts


class DecisionDiagnostics:
    def __init__(self, output_dir, steps=(256, 192, 128, 64, 16, 1), max_states=32,
                 max_molecules=8, compare_context=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.steps, self.max_states = set(steps), max_states
        self.max_molecules, self.compare_context = max_molecules, compare_context
        self.states, self.diagnostic_calls = 0, 0
        self.snapshot = snapshot

    def wants(self, step):
        return step in self.steps

    def record(self, sampler, batch, pred, step, phase):
        p = pred["type_probabilities"]
        force_sigma = pred.get(keys.sigma_curr)
        if force_sigma is None and keys.forces in pred:
            force_sigma = sampler.calculator._compute_std_per_molecule(batch[P.idx_m], pred[keys.forces]/2)
        finite = finite_atoms(batch, p, pred.get(keys.forces), force_sigma)
        slots = (batch[P.Z] == vocab.MASK_Z) & finite
        row = {"step": step, "phase": phase, "slots": int(slots.sum()),
               "finite_molecules": int(torch.unique(batch[P.idx_m][finite]).numel()),
               "excluded_molecules": len(batch[P.n_atoms]) - int(torch.unique(batch[P.idx_m][finite]).numel()),
               "mask_fraction": float((batch[P.Z] == vocab.MASK_Z).float().mean()),
               "injected_noise_std": (0.0 if hasattr(sampler, "noise_schedule") else sampler.stochastic_lambda * sampler.get_noise_ratio(step)),
               "scheduled_sigma": sampler.noise_schedule.get_sigma(step) if hasattr(sampler, "noise_schedule") else None,
               "entropy": float(-(p[slots] * p[slots].clamp_min(1e-30).log()).sum(-1).mean()) if bool(slots.any()) else None,
               "force_derived_sigma": float(force_sigma[finite].mean()) if force_sigma is not None and bool(finite.any()) else None}
        proxy = count_proxy(batch[P.Z], p, batch[P.idx_m], len(batch[P.n_atoms]))
        good = torch.unique(batch[P.idx_m][finite])
        row["expected_count_proxy_sum"] = proxy[good].sum(0).tolist()
        row["molecule_ids"] = batch.get(P.idx, torch.arange(len(batch[P.n_atoms]), device=p.device)).reshape(-1).tolist()
        good_set = set(good.tolist())
        row["expected_count_proxy_per_molecule"] = [v if i in good_set else None for i,v in enumerate(proxy.cpu().tolist())]
        row["committed_count"] = [int(((batch[P.Z] == z) & finite).sum()) for z in vocab.TOKEN_Z]
        if self.compare_context and phase == "decision_input":
            remasked = probe(sampler.calculator, batch, remask=True)
            self.diagnostic_calls += 1
            q = remasked["type_probabilities"]
            # Intersect at molecule level, then use exactly the original slots.
            paired = finite & finite_atoms(batch, q, remasked.get(keys.forces), remasked.get(keys.sigma_curr))
            slots &= paired
            row["context_slots"] = int(slots.sum())
            row["partial_context_mass"] = p[slots].sum(0).tolist()
            row["remasked_context_mass"] = q[slots].sum(0).tolist()
        self._append(row)
        if self.states < self.max_states:
            keep = batch[P.idx_m] < self.max_molecules
            state = {"step": step, "phase": phase, "positions": batch[P.R][keep].cpu(),
                     "types": batch[P.Z][keep].cpu(), "idx_m": batch[P.idx_m][keep].cpu(),
                     "probabilities": p[keep].cpu(), "finite": finite[keep].cpu(),
                     "expected_count_proxy": proxy[:self.max_molecules].cpu()}
            state["predictions"] = {k: (v[keep] if v.shape[0]==len(keep) else v[:self.max_molecules]).detach().cpu()
                for k,v in pred.items() if isinstance(v,torch.Tensor) and v.ndim>0}
            effective = torch.nan_to_num(p[keep], nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
            effective = effective ** (1 / sampler.type_temperature)
            effective = torch.nan_to_num(effective, nan=0.0, posinf=0.0)
            total = effective.sum(-1, keepdim=True)
            fallback = torch.zeros_like(effective).scatter_(1, effective.argmax(-1)[:,None], 1)
            state["sampling_probabilities"] = torch.where(total > 1e-12, effective / total.clamp_min(1e-30), fallback).cpu()
            state["type_temperature"] = sampler.type_temperature
            if self.compare_context and phase == "decision_input":
                state["remasked_probabilities"] = q[keep].cpu()
                state["context_slots"] = slots[keep].cpu()
            torch.save(state, self.output_dir / f"state_{self.states:05d}.pt")
            self.states += 1

    def probe(self, sampler, batch, step, phase):
        pred = probe(sampler.calculator, batch)
        self.diagnostic_calls += 1
        self.record(sampler, snapshot(batch), pred, step, phase + "_probe")

    def transition(self, batch, step, selected, remasked):
        self._append({"step": step, "phase": "transition", "selected_slots": torch.where(selected)[0].tolist(),
                      "remasked_slots": torch.where(remasked)[0].tolist(),
                      "committed_count": [int((batch[P.Z] == z).sum()) for z in vocab.TOKEN_Z],
                      "diagnostic_calls": self.diagnostic_calls})

    def _append(self, row):
        with (self.output_dir / "decisions.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
