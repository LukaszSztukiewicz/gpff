"""Validation and test batches with pinned pseudo-PES noise.

Each (molecule, draw) gets its noise from a generator seeded with
(seed, database index, draw) and its noise level from a stratified sample of
the training distribution: the n levels of n_molecules * n_draws draws are the
quantiles (stratum + u) / n of the geometry sampler's sigma distribution, dealt
out by a seeded permutation. The noise follows the training distribution and
depends on nothing but the seed and the split: not on workers, batch size,
Lightning or the training seed. Stratifying the noise level removes most of
the spread a single draw adds to the mean loss.

The transforms (including the permutation alignment) run once and the collated
batches are kept, so evaluating the set again is only the model's forward pass.
"""
import copy
from typing import Dict, List

import numpy as np
import torch
from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn
from torch.utils.data import DataLoader, Dataset

from gpff.data.transform import PseudoPESTransform
from gpff.pseudo_pes import VarianceExplodingGeometries

__all__ = ["pin_noise", "stratified_sigmas"]


def stratified_sigmas(
    sampler: VarianceExplodingGeometries, n: int, rng: np.random.Generator
) -> np.ndarray:
    """n noise levels of the sampler's training distribution, one per quantile stratum, shuffled."""
    return sampler.sigma_quantile((rng.permutation(n) + rng.random(n)) / n)


class _PinnedDraws(Dataset):
    """Draw i is molecule i // n_draws of ``dataset`` with its pinned noise."""

    def __init__(self, dataset, seed: int, n_draws: int):
        # the dataset's own transforms are applied here, with the noise passed in
        self.transforms = list(dataset.transforms)
        self.dataset = copy.copy(dataset)
        self.dataset.transforms = None
        pes = [t for t in self.transforms if isinstance(t, PseudoPESTransform)]
        if len(pes) != 1 or not isinstance(pes[0].geometry_sampler, VarianceExplodingGeometries):
            raise NotImplementedError(
                "pinned noise needs exactly one PseudoPESTransform with VarianceExplodingGeometries"
            )
        self.sigma_max = pes[0].geometry_sampler.sigma_max
        self.seed = seed
        self.n_draws = n_draws
        self.sigmas = stratified_sigmas(
            pes[0].geometry_sampler, len(dataset) * n_draws, np.random.default_rng(seed)
        )

    def __len__(self) -> int:
        return len(self.sigmas)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        inputs = self.dataset[i // self.n_draws]
        positions = inputs[properties.R]
        rng = np.random.default_rng((self.seed, int(inputs[properties.idx]), i % self.n_draws))
        noise = torch.from_numpy(rng.standard_normal(tuple(positions.shape))).to(positions.dtype)
        for transform in self.transforms:
            if isinstance(transform, PseudoPESTransform):
                inputs = transform(inputs, t=float(self.sigmas[i] / self.sigma_max), noise=noise)
            else:
                inputs = transform(inputs)
        return inputs


def pin_noise(
    dataset, seed: int, batch_size: int, n_draws: int = 1, num_workers: int = 0
) -> List[Dict[str, torch.Tensor]]:
    """Collated batches of ``dataset`` (a split with its transforms) with pinned noise.

    The n_draws draws of a molecule are consecutive, in the dataset's order.
    """
    loader = DataLoader(
        _PinnedDraws(dataset, seed, n_draws),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_atoms_collate_fn,
    )
    # clone out of the workers' shared memory, which holds a file descriptor per tensor
    return [{k: v.clone() for k, v in batch.items()} for batch in loader]
