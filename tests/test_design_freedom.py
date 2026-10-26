import torch
from schnetpack import properties

from gpff import keys
from gpff.data.transform.pseudo_pes import PseudoPESTransform
from gpff.pseudo_pes.geometries import VarianceExplodingGeometries
from gpff.pseudo_pes.labels import VEPseudoCalculator
from gpff.pseudo_pes.weights import SigmaWeights
from gpff.sample.conditions import DesignFreedomConditionGenerator
from gpff.utils.alignment import align_permutation


def test_align_permutation_returns_the_indices_it_applied():
    torch.manual_seed(0)
    numbers = torch.tensor([6, 1, 1, 8, 1, 6, 1])
    target_positions = torch.randn(7, 3, dtype=torch.float64)
    positions = target_positions + 0.8 * torch.randn_like(target_positions)

    aligned, aligned_numbers, indices = align_permutation(
        positions, target_positions, numbers, numbers, return_permutation=True
    )

    torch.testing.assert_close(aligned, positions[indices])
    assert torch.equal(aligned_numbers, numbers[indices])
    assert torch.equal(aligned_numbers, numbers)


def test_per_atom_sigma_follows_the_permuted_noisy_positions():
    torch.manual_seed(0)
    n_atoms = 12
    inputs = {
        properties.R: torch.randn(n_atoms, 3, dtype=torch.float64),
        # one element, so every atom can trade places with every other
        properties.Z: torch.ones(n_atoms, dtype=torch.long),
    }
    noise = torch.randn(n_atoms, 3, dtype=torch.float64)
    sampler = VarianceExplodingGeometries(sigma_max=5.0, sigma_spread=1.0)
    transform = PseudoPESTransform(
        geometry_sampler=sampler,
        energy_function=VEPseudoCalculator(),
        weight_generator=SigmaWeights(min_clipping=0.0, max_clipping=1.0),
        align_permutation=True,
    )

    torch.manual_seed(1)
    out = transform(dict(inputs), t=torch.tensor(0.5), noise=noise)
    # replay the sampler's draws to recover each atom's noisy position and level
    torch.manual_seed(1)
    x_0 = inputs[properties.R] - inputs[properties.R].mean(0)
    x_t, sigma_t, _, _ = sampler(x_0=x_0, t=torch.tensor(0.5), noise=noise)

    for i in range(n_atoms):
        source = int((x_t - out[properties.R][i]).norm(dim=-1).argmin())
        assert out[keys.sigma_curr][i] == sigma_t[source]
        assert out[keys.forces_weight][i] == (1 / sigma_t[source] ** 2).clamp(max=1.0)


def test_design_freedom_skips_ids_past_a_small_molecule():
    generator = DesignFreedomConditionGenerator(
        groups=[{"sigma": 0.0, "ids": [0, 1, 2, 3, 4, 5]}], default_sigma=3.0
    )
    batch = {
        properties.idx_m: torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
        properties.R: torch.zeros(11, 3),
    }

    field = generator.from_batch(batch)[keys.sigma_field].squeeze(-1)

    assert field[:3].eq(0.0).all()
    assert field[3:9].eq(0.0).all() and field[9:].eq(3.0).all()


def test_design_freedom_without_groups_is_flat():
    generator = DesignFreedomConditionGenerator(groups=[], default_sigma=0.3)
    batch = {properties.idx_m: torch.tensor([0, 0, 1]), properties.R: torch.zeros(3, 3)}

    field = generator.from_batch(batch)[keys.sigma_field]

    assert field.eq(0.3).all()
