"""
Atom types for models that generate them, and the processes that corrupt them.

A :class:`Vocabulary` orders the tokens a slot can hold: the dataset's elements, then
MASK, the absorbing state, which says that the type is not known. The clean classes a
model predicts are the vocabulary without MASK, which are its first ``n_classes`` tokens.

A type-generating model reads the types as one vector per slot over the vocabulary,
``keys.h``: a one-hot token for :class:`AbsorbingTypes`, a noisy scaled one-hot for
:class:`ContinuousTypes`. ``properties.Z`` holds the clean elements in training and the
types' atomic numbers in sampling (MASK_Z for MASK), for the output; such a model never
reads it.

A process defines the corruption of training (``corrupt``) and the state sampling starts
from (``prior``); :mod:`gpff.sample.atom_types` reverses it. The model's type head keeps
its process, so the samplers draw types the way the model was trained.
"""

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F


__all__ = ["MASK_Z", "Vocabulary", "AbsorbingTypes", "ContinuousTypes"]

# the atomic number properties.Z carries for a MASK slot; no dataset has einsteinium
MASK_Z = 99


class Vocabulary:
    """
    The tokens of a slot: the elements in their order, then MASK.

    Args:
        elements: atomic numbers of the dataset's elements.
    """

    def __init__(self, elements: Sequence[int]):
        elements = tuple(int(z) for z in elements)
        if not elements or len(set(elements)) != len(elements) or not all(0 < z < MASK_Z for z in elements):
            raise ValueError(f"elements must be distinct atomic numbers from 1 to {MASK_Z - 1}, got {elements}")
        self.elements = elements
        self.numbers = elements + (MASK_Z,)

    @property
    def size(self) -> int:
        return len(self.numbers)

    @property
    def n_classes(self) -> int:
        """The number of clean classes, the tokens before MASK."""
        return self.size - 1

    @property
    def mask_index(self) -> int:
        return self.size - 1

    def classes(self, numbers: torch.Tensor) -> torch.Tensor:
        """
        The clean class of each atomic number. Raises for MASK_Z and for atomic numbers
        outside the vocabulary.
        """
        table = torch.tensor(self.numbers[: self.n_classes], device=numbers.device)
        match = numbers.unsqueeze(-1) == table
        known = match.any(-1)
        if not bool(known.all()):
            unknown = sorted(set(numbers[~known].tolist()))
            raise ValueError(f"atomic numbers {unknown} are not classes of {self}")
        return (match.long() * torch.arange(len(table), device=numbers.device)).sum(-1)

    def atomic_numbers(self, tokens: torch.Tensor) -> torch.Tensor:
        """The atomic number of each token."""
        return torch.tensor(self.numbers, device=tokens.device)[tokens]

    def one_hot(self, tokens: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """(n, size) one-hot vectors of the tokens."""
        return F.one_hot(tokens, self.size).to(dtype)

    def masked(self, h: torch.Tensor) -> torch.Tensor:
        """True for the slots whose type vector is the MASK token."""
        return h[..., self.mask_index] > 0.5

    def __repr__(self) -> str:
        return f"Vocabulary({list(self.elements)})"


class AbsorbingTypes:
    r"""
    Absorbing-state (masked) discrete diffusion of the atom types: a continuous-time
    Markov chain in which a slot jumps to MASK and stays there (Austin et al. 2021,
    Campbell et al. 2022). At the masking rate :math:`r = 1 - \alpha_t` every slot is
    MASK independently with probability r, and a visible slot holds its clean class.

    The model predicts the clean class of every slot from the geometry and the visible
    types. For the linear schedule r = t the continuous-time NELBO is the cross-entropy
    on the masked slots weighted by 1/r (MDLM, Sahoo et al. 2024; Shi et al. 2024): with
    weight 1/r on the masked slots and 0 on the visible ones, the mean over the slots is
    unbiased whatever r was drawn.

    r is drawn per molecule uniformly from [min_rate, 1], independently of the noise level
    of the positions, so one model covers every pair of the two. Given the visible slots,
    the clean-type posterior of an absorbing process does not depend on r (Ou et al.
    2025), which is what a time-free model needs, and the sampler may unmask on any
    schedule.

    Args:
        vocabulary: the tokens.
        min_rate: the smallest rate drawn; bounds the weight by 1/min_rate.
    """

    def __init__(self, vocabulary: Vocabulary, min_rate: float = 0.01):
        if not 0 < min_rate <= 1:
            raise ValueError(f"min_rate must be in (0, 1], got {min_rate}")
        self.vocabulary = vocabulary
        self.min_rate = float(min_rate)

    def corrupt(self, classes: torch.Tensor, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The types of one molecule at a random masking rate.

        Args:
            classes: (n_atoms,) clean classes.
            sigma: (n_atoms, 1) noise level of the positions; only its dtype is used.

        Returns:
            h: (n_atoms, vocabulary.size) one-hot tokens, MASK or the clean class.
            weight: (n_atoms, 1) weight of each slot's cross-entropy.
        """
        dtype = sigma.dtype
        rate = self.min_rate + (1 - self.min_rate) * torch.rand((), dtype=dtype)
        masked = torch.rand(len(classes), dtype=dtype) < rate
        tokens = torch.where(masked, self.vocabulary.mask_index, classes)
        return self.vocabulary.one_hot(tokens, dtype), masked.unsqueeze(-1).to(dtype) / rate

    def prior(self, positions: torch.Tensor) -> torch.Tensor:
        """Every slot MASK: (n_atoms, vocabulary.size)."""
        tokens = torch.full((len(positions),), self.vocabulary.mask_index, device=positions.device)
        return self.vocabulary.one_hot(tokens, positions.dtype)


class ContinuousTypes:
    r"""
    Continuous diffusion of the atom types along with the positions. The clean type of a
    slot is the scaled one-hot :math:`h_0 = s\,\mathrm{onehot}(c)`, noised like the
    positions and with their noise level, :math:`h_t = h_0 + \sigma\epsilon` (VE). The
    pseudo-energy gains :math:`|h_t - h_0|^2`, whose force on the types is
    :math:`2(\mathbb{E}[h_0 \mid x_t, h_t] - h_t)` with
    :math:`\mathbb{E}[h_0 \mid x_t, h_t] = s\,p` and p the clean-class probabilities. The
    head predicts p with a cross-entropy and the samplers move h towards s p. The scale
    sets when the types resolve relative to the positions, since
    :math:`h_t / s = \mathrm{onehot}(c) + (\sigma / s)\,\epsilon`.

    Args:
        vocabulary: the tokens; the MASK channel stays zero.
        scale: s, the value of the one-hot entries.
        sigma_max: noise level of the prior, the positions' sigma_max.
    """

    def __init__(self, vocabulary: Vocabulary, scale: float = 0.3, sigma_max: float = 30.0):
        if not scale > 0 or not sigma_max > 0:
            raise ValueError(f"scale and sigma_max must be positive, got {scale} and {sigma_max}")
        self.vocabulary = vocabulary
        self.scale = float(scale)
        self.sigma_max = float(sigma_max)

    def corrupt(self, classes: torch.Tensor, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The types of one molecule at the noise level of its positions.

        Args:
            classes: (n_atoms,) clean classes.
            sigma: (n_atoms, 1) noise level of the positions.

        Returns:
            h: (n_atoms, vocabulary.size) the noisy types.
            weight: (n_atoms, 1) ones: every slot's cross-entropy counts alike.
        """
        n_classes = self.vocabulary.n_classes
        h = self.scale * self.vocabulary.one_hot(classes, sigma.dtype)
        h[:, :n_classes] += sigma * torch.randn(len(h), n_classes, dtype=sigma.dtype)
        return h, torch.ones_like(sigma)

    def prior(self, positions: torch.Tensor) -> torch.Tensor:
        """Types at sigma_max: (n_atoms, vocabulary.size)."""
        h = positions.new_zeros(len(positions), self.vocabulary.size)
        h[:, : self.vocabulary.n_classes] = self.sigma_max * torch.randn_like(h[:, : self.vocabulary.n_classes])
        return h

    def clean_estimate(self, logits: torch.Tensor) -> torch.Tensor:
        """s p over the vocabulary, (n_atoms, vocabulary.size), from the logits of the clean classes."""
        return F.pad(self.scale * torch.softmax(logits, dim=-1), (0, 1))
