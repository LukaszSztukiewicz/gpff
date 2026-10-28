"""
The forward process of absorbing-state discrete diffusion over the atom types.

Runs as its own transform, after the pseudo-PES transform, so the positions and
their labels are untouched and ``align_permutation`` has already matched the
noisy geometry to ``x_0`` **using the clean elements**. Masking after that
alignment is not optional: the Hungarian matching in
``gpff.utils.alignment.align_permutation`` puts a cost of 1e12 on every pair of
unequal elements, so masking first would make every pair mismatch and the
assignment would be meaningless.

Each atom is replaced by MASK independently with probability ``rate``, one rate
per molecule, so the state is

    q(a_t | a_0) = Cat( (1 - rate) * onehot(a_0) + rate * onehot(MASK) )

The masked types go back into ``properties.Z`` as the MASK atomic number, which
means PaiNN embeds them through its existing nn.Embedding and the model needs no
new input path -- unlike a continuous one-hot channel, which has to be expanded
in a basis first because it grows with sigma.

The transform writes the targets of the masked cross-entropy:

    keys.type_target  (n_atoms, n_tokens) one-hot of the clean type
    keys.type_weight  (n_atoms, 1)        1/rate on masked atoms, 0 elsewhere
    keys.type_mask    (n_atoms, 1)        whether the atom was masked
    keys.type_rate    (n_atoms, 1)        the rate it was masked with

The weight is the continuous-time NELBO weight of masked diffusion: for the
linear schedule alpha_t = 1 - t the bound is an expectation of (1/t) times the
cross-entropy over the masked positions (Sahoo et al. 2024, Shi et al. 2024).
Because a fraction ``rate`` of the atoms is masked, weighting them by 1/rate and
averaging over all atoms makes the objective O(1) and unbiased whatever the rate
was -- the loss over all atoms is already a mean, so nothing else normalizes it.

The rate is drawn **independently of the position noise level**, not tied to it.
One model then covers the whole (sigma, rate) square, including the two corners
a tied schedule never visits: clean geometry with unknown types (type infilling)
and noisy geometry with known types (which is plain GPFF, and so gives a free
like-for-like control). It is also the honest reading of GPFF being time
agnostic -- the model has to infer both corruption levels from its input.

Draws come from the torch CPU RNG inside the transform, so under
``callbacks.deterministic_data_seed.per_sample_train_noise`` they are a function
of (seed, epoch, database index) like the position noise.
"""

from typing import Dict, Optional

import torch

import schnetpack.transform as trn
from schnetpack import properties

from gpff import keys, vocab


__all__ = ["AbsorbingTypesTransform"]


class AbsorbingTypesTransform(trn.Transform):
    """
    Mask the atom types of a sample for absorbing-state discrete diffusion.

    Args:
        min_rate: smallest masking rate drawn, which also clamps the 1/rate
            weight to 1/min_rate. Purely a variance control: the weight of the
            exact bound diverges as the rate goes to zero.
        max_rate: largest masking rate drawn. 1.0 means every atom can be
            masked, which is the state the sampler starts from.
        rate_from_sigma: tie the rate to the position noise level instead of
            drawing it, using rate = sigma_t / sigma_max. The ablation of the
            independent draw described above; needs sigma_max.
        sigma_max: the run's sigma_max, only for rate_from_sigma.
    """

    def __init__(
        self,
        min_rate: float = 0.01,
        max_rate: float = 1.0,
        rate_from_sigma: bool = False,
        sigma_max: Optional[float] = None,
    ):
        super().__init__()
        if not 0.0 < min_rate <= max_rate <= 1.0:
            raise ValueError(
                f"Need 0 < min_rate <= max_rate <= 1, got {min_rate} and {max_rate}."
            )
        if rate_from_sigma and not sigma_max:
            raise ValueError("rate_from_sigma needs sigma_max.")
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.rate_from_sigma = rate_from_sigma
        self.sigma_max = sigma_max

    def _rate(self, inputs: Dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        if self.rate_from_sigma:
            sigma = inputs[keys.sigma_curr].reshape(-1)[0].to(dtype)
            rate = (sigma / self.sigma_max).clamp(self.min_rate, self.max_rate)
            return rate.reshape(1)
        return (
            torch.rand(1, dtype=dtype) * (self.max_rate - self.min_rate) + self.min_rate
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z = inputs[properties.Z]
        n_atoms = z.shape[0]
        dtype = inputs[properties.R].dtype

        # the clean type is the label; z is still the aligned, unmasked element
        target_index = vocab.z_to_index(z)

        rate = self._rate(inputs, dtype)
        masked = torch.rand(n_atoms, dtype=dtype) < rate

        masked_z = torch.where(
            masked, torch.full_like(z, vocab.MASK_Z), z
        )

        one_hot = torch.zeros(n_atoms, vocab.n_tokens(), dtype=dtype)
        one_hot.scatter_(-1, target_index.unsqueeze(-1), 1.0)

        masked_column = masked.unsqueeze(-1)
        inputs.update(
            {
                properties.Z: masked_z,
                keys.type_target: one_hot,
                keys.type_weight: masked_column.to(dtype) / rate,
                keys.type_mask: masked_column,
                keys.type_rate: rate.expand(n_atoms).unsqueeze(-1).clone(),
            }
        )
        return inputs
