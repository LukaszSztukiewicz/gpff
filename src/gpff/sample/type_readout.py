"""
Alternative atom-type readouts for the continuous one-hot samplers.

The samplers of :mod:`gpff.sample.direct_denoising` read the element of an atom
off the noisy one-hot types as ``argmax(h)``, where ``h`` has been set to the
model's clean-type estimate. For a model of ``experiment=gpff_types_categorical``
that estimate is ``s * softmax(logits)``, so the element is the argmax of a
mean-field marginal: a greedy decode, which collapses onto the frequent elements
wherever the model is uncertain.

This module keeps the trajectory of those samplers and only changes how the
element is taken from the marginal:

    argmax   the element of largest probability (what the samplers do today)
    sample   draw the element from the marginal
    commit   draw the element and replace h by the scaled one-hot of the draw,
             so the choice also conditions the following steps

``argmax`` is there as a control: it has to reproduce the plain sampler exactly.

The draws come from a generator of their own, so the positions see the same
noise whatever the readout is and two readouts of one checkpoint are paired
sample by sample.
"""

from typing import Dict, Optional, Sequence

import torch

from schnetpack import properties

from gpff import keys
from gpff.sample.direct_denoising import DirectDenoisingSampler


__all__ = ["TypeReadout", "ReadoutDirectDenoisingSampler"]


class TypeReadout:
    """
    Mixin that gives a sampler a configurable atom-type readout.

    Args:
        type_readout: one of "argmax", "sample", "commit" (see the module
            docstring).
        type_temperature: temperature of the draw. The marginal is raised to
            the power 1/T and renormalized, so T < 1 sharpens it towards the
            argmax and T > 1 flattens it. Ignored by "argmax".
        type_scale: value of the one-hot entries, used by "commit" to rebuild
            h. None takes it from the prediction, which sums to exactly
            type_scale for a categorical model.
        type_seed: seed of the generator the draws come from.
        probabilities_from: how the marginal is taken from the clean-type
            estimate. "normalize" clips the negative entries and normalizes,
            which is exact for a categorical model (its estimate is already
            type_scale times a distribution). "softmax" instead reads the
            estimate as unnormalized scores, divided by type_scale; for a model
            of experiment=gpff_types, whose head regresses the clean types
            without a softmax, neither is exact.
    """

    READOUTS = ("argmax", "sample", "commit")
    PROBABILITIES_FROM = ("normalize", "softmax")

    def __init__(
        self,
        *args,
        type_readout: str = "argmax",
        type_temperature: float = 1.0,
        type_scale: Optional[float] = None,
        type_seed: int = 0,
        probabilities_from: str = "normalize",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if type_readout not in self.READOUTS:
            raise ValueError(
                f"Unknown type_readout {type_readout!r}, expected one of {self.READOUTS}."
            )
        if probabilities_from not in self.PROBABILITIES_FROM:
            raise ValueError(
                f"Unknown probabilities_from {probabilities_from!r}, expected one "
                f"of {self.PROBABILITIES_FROM}."
            )
        if type_temperature <= 0:
            raise ValueError("type_temperature must be positive.")
        self.type_readout = type_readout
        self.type_temperature = type_temperature
        self.type_scale = type_scale
        self.type_seed = type_seed
        self.probabilities_from = probabilities_from
        # one generator per device, so the position noise keeps its own stream
        self._type_generators: Dict[torch.device, torch.Generator] = {}

    def _generator(self, device: torch.device) -> torch.Generator:
        if device not in self._type_generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.type_seed)
            self._type_generators[device] = generator
        return self._type_generators[device]

    def _probabilities(self, h_0: torch.Tensor) -> torch.Tensor:
        """
        The per-atom element marginal of a clean-type estimate, (n_atoms, n_types).
        """
        # A molecule whose positions have just gone to NaN reaches this before
        # check_convergence freezes it, so the estimate can hold NaN or inf.
        # argmax tolerated that -- it returns some index -- but multinomial
        # raises a device-side assert on a distribution that is not finite, so
        # the estimate has to be sanitized here rather than upstream.
        scores = torch.nan_to_num(h_0, nan=0.0, posinf=0.0, neginf=0.0)

        scale = self.type_scale
        if scale is None:
            # exact for a categorical model, whose estimate is s * softmax
            scale = scores.sum(dim=-1, keepdim=True).abs().clamp(min=1e-6)

        if self.probabilities_from == "softmax":
            probabilities = torch.softmax(scores / scale, dim=-1)
        else:
            probabilities = scores.clamp(min=0.0)

        if self.type_temperature != 1.0:
            probabilities = probabilities ** (1.0 / self.type_temperature)
            probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)

        total = probabilities.sum(dim=-1, keepdim=True)
        # an atom with no mass to draw from -- every entry non-positive, or the
        # estimate was not finite -- falls back to its argmax, so it is still
        # assigned an element. The threshold also keeps the division below away
        # from the denormals that would make it overflow.
        degenerate = ~(total > 1e-12)
        fallback = torch.zeros_like(probabilities)
        fallback.scatter_(-1, scores.argmax(dim=-1, keepdim=True), 1.0)
        safe_total = torch.where(degenerate, torch.ones_like(total), total)
        return torch.where(
            degenerate.expand_as(probabilities), fallback, probabilities / safe_total
        )

    def update_types(
        self,
        batch: Dict[str, torch.Tensor],
        h: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Set the noisy one-hot types and take each atom's element from them.
        """
        batch = self.update_batch(batch, updates={keys.h: h})
        h_curr = batch[keys.h]

        if self.type_readout == "argmax":
            indices = h_curr.argmax(dim=-1)
        else:
            probabilities = self._probabilities(h_curr)
            indices = torch.multinomial(
                probabilities, 1, generator=self._generator(probabilities.device)
            ).squeeze(-1)

        if self.type_readout == "commit":
            scale = self.type_scale
            if scale is None:
                # the estimate sums to type_scale for a categorical model; a
                # non-finite or non-positive sum means this atom's molecule has
                # already diverged, so fall back to 1 rather than writing NaN
                scale = torch.nan_to_num(
                    h_curr.sum(dim=-1, keepdim=True), nan=1.0, posinf=1.0, neginf=1.0
                )
                scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            committed = torch.zeros_like(h_curr)
            committed.scatter_(-1, indices.unsqueeze(-1), 1.0)
            batch = self.update_batch(batch, updates={keys.h: scale * committed})

        # atoms of converged molecules keep the element they were given
        converged_atoms = batch[keys.converged][batch[properties.idx_m]]
        elements = self.atom_types.to(indices.device)[indices]
        batch[properties.Z] = torch.where(
            converged_atoms, batch[properties.Z], elements
        )
        return batch


class ReadoutDirectDenoisingSampler(TypeReadout, DirectDenoisingSampler):
    """
    :class:`~gpff.sample.direct_denoising.DirectDenoisingSampler` with a
    configurable atom-type readout. See :class:`TypeReadout`.
    """
