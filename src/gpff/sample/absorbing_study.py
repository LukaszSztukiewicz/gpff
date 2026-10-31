"""Independent inference interventions for the absorbing coupling study.

The v1/v2 sampler remains in absorbing.py. With all switches off this class
uses its update order and random draws. Extra diagnostic calls are isolated.
"""
import torch
from schnetpack import properties as P
from gpff import keys, vocab
from gpff.sample.absorbing import AbsorbingDirectDenoisingSampler, PROBABILITIES_KEY
from gpff.sample.remdm import transition_probabilities
from gpff.sample.schedules import EDMNoiseSchedule


class CoupledAbsorbingSampler(AbsorbingDirectDenoisingSampler):
    def __init__(self, remask_kernel="legacy", recompute_forces=False,
                 reveal_block_size=0, reveal_power=1.0, diagnostics=None, **kwargs):
        super().__init__(**kwargs)
        if remask_kernel not in ("legacy", "random", "saved_confidence"):
            raise ValueError("Unknown remask_kernel")
        if not 0 <= self.remask_prob <= 1:
            raise ValueError("Study remask_prob must be in [0,1]")
        if reveal_block_size < 0 or reveal_power <= 0:
            raise ValueError("Invalid reveal block size or schedule power")
        if reveal_block_size and (not recompute_forces or remask_kernel == "legacy"):
            raise ValueError("Sequential blocks require recompute_forces and a simultaneous kernel")
        self.remask_kernel = remask_kernel
        self.recompute_forces = recompute_forces
        self.reveal_block_size = reveal_block_size
        self.reveal_power = reveal_power
        self.diagnostics = diagnostics

    def _init_state(self, batch):
        batch = super()._init_state(batch)
        # Sampler initializes before cast_batch; a prior can still be float64.
        self._decode_confidence = torch.ones_like(batch[P.Z], dtype=self.calculator.dtype)
        self.last_selected = torch.zeros_like(batch[P.Z], dtype=torch.bool)
        self.last_remasked = self.last_selected.clone()
        self.model_calls = 0
        return batch

    def masking_rate(self, step):
        return super().masking_rate(step) ** self.reveal_power

    def _calculate(self, batch, step, phase, estimate_sigma=True):
        # Snapshot before the model: NeuralNetworkPotential mutates its input.
        captured = self.diagnostics.snapshot(batch) if self.diagnostics and self.diagnostics.wants(step) else None
        pred = self.calculator.calculate(batch, forces=True, x0=True, estimate_sigma=estimate_sigma)
        self.model_calls += 1
        if captured is not None:
            self.diagnostics.record(self, captured, pred, step, phase)
        return pred

    def _probe(self, batch, step, phase):
        if self.diagnostics and self.diagnostics.wants(step):
            self.diagnostics.probe(self, batch, step, phase)

    def _transition(self, batch, probabilities, step):
        before = batch[P.Z].clone()
        if self.remask_kernel == "legacy":
            out = super().unmask_types(batch, probabilities, step)
            self.last_selected = (before == vocab.MASK_Z) & (out[P.Z] != vocab.MASK_Z)
            self.last_remasked = (before != vocab.MASK_Z) & (out[P.Z] == vocab.MASK_Z)
            return out
        masked = before == vocab.MASK_Z
        self._decode_confidence = self._decode_confidence.to(probabilities)
        active = ~batch[keys.converged][batch[P.idx_m]]
        r, s = self.masking_rate(step + 1), self.masking_rate(step)
        a, b = transition_probabilities(r, s, self.remask_prob)
        rng = self._generator(before.device)
        selected = masked & active & (torch.rand(before.shape, device=before.device, generator=rng) < a)
        weight = 1.0
        if self.remask_kernel == "saved_confidence":
            weight = 1 - self._decode_confidence
        remasked = ~masked & active & (torch.rand(before.shape, device=before.device, generator=rng) < b * weight)
        # Both decisions depend on the pre-step state. Freshly decoded slots
        # cannot be remasked. Correction also runs when selected is empty.
        batch[P.Z] = before.clone()
        batch[P.Z][remasked] = vocab.MASK_Z
        pending = selected.clone()
        while bool(pending.any()):
            block = pending.clone()
            if self.reveal_block_size:
                block.zero_()
                for m in torch.unique(batch[P.idx_m][pending]):
                    slots = torch.where(pending & (batch[P.idx_m] == m))[0]
                    order = torch.randperm(len(slots), device=slots.device, generator=rng)
                    block[slots[order[:self.reveal_block_size]]] = True
            indices = self._draw_types(probabilities[block])
            batch[P.Z][block] = vocab.index_to_z(indices).to(before.dtype)
            # Store decoding probabilities, never visible-position logits.
            self._decode_confidence[block] = probabilities[block].gather(1, indices[:, None]).squeeze(1).to(self._decode_confidence.dtype)
            pending &= ~block
            if bool(pending.any()):
                probabilities = self._calculate(batch, step, "sequential_decision")[PROBABILITIES_KEY]
        self.last_selected, self.last_remasked = selected, remasked
        return batch

    def _coordinate_update(self, batch, pred, step):
        return self.update_batch(batch, {P.R: pred[keys.x_0], keys.sigma_curr: pred[keys.sigma_curr]})

    def denoise_step(self, batch, step, conditions=None, **kwargs):
        batch = self.preprocess_step(batch, step, conditions=conditions, **kwargs)
        pred = self._calculate(batch, step, "decision_input")
        if self.recompute_forces:
            batch = self._transition(batch, pred[PROBABILITIES_KEY], step)
            self._probe(batch, step, "after_types_before_geometry")
            # Same noisy coordinates, new types, no second noise injection.
            # Always recompute: two calls/step makes matched-NFE runs explicit.
            pred = self._calculate(batch, step, "type_conditioned_force")
        if self.f_max > 0:
            batch = self.update_convergence(batch, torch.linalg.vector_norm(pred[keys.forces], dim=1), self.f_max, reduce="amax")
        batch = self._coordinate_update(batch, pred, step)
        self._probe(batch, step, "after_geometry")
        if not self.recompute_forces:
            batch = self._transition(batch, pred[PROBABILITIES_KEY], step)
            self._probe(batch, step, "after_types")
        if self.diagnostics and self.diagnostics.wants(step):
            self.diagnostics.transition(batch, step, self.last_selected, self.last_remasked)
        self._log(timestep=step, model_calls=self.model_calls,
                  masked=float((batch[P.Z] == vocab.MASK_Z).float().mean()))
        return self.postprocess_step(batch, step, conditions=conditions, **kwargs)


class ScheduledAbsorbingSampler(CoupledAbsorbingSampler):
    """VE ancestral update from AncestralSampler with absorbing type decisions.

    sigma_curr is the declared EDM schedule value, shared by conditioned and
    unconditioned comparisons. It is never inferred from predicted forces.
    """
    def __init__(self, noise_schedule=None, **kwargs):
        super().__init__(**kwargs)
        self.noise_schedule = noise_schedule or EDMNoiseSchedule(self.n_steps)

    def preprocess_step(self, batch, step, **kwargs):
        sigma = self.noise_schedule.get_sigma(step)
        batch[keys.sigma_curr] = batch[P.R].new_full((len(batch[P.n_atoms]), 1), sigma)
        return batch

    def _calculate(self, batch, step, phase, estimate_sigma=False):
        return super()._calculate(batch, step, phase, estimate_sigma=False)

    def _coordinate_update(self, batch, pred, step):
        sigma = self.noise_schedule.get_sigma(step)
        nxt = self.noise_schedule.get_sigma(step - 1)
        ratio = (nxt / sigma) ** 2
        x = ratio * batch[P.R] + (1 - ratio) * pred[keys.x_0]
        x = x + torch.randn_like(x) * (nxt ** 2 * (1 - ratio)) ** 0.5
        return self.update_batch(batch, {P.R: x})
