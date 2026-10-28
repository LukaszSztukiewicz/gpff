import pytest
import torch

from schnetpack import properties

from gpff import keys, vocab
from gpff.data.transform.absorbing_types import AbsorbingTypesTransform
from gpff.sample.absorbing import AbsorbingDirectDenoisingSampler, AbsorbingPrior


QM9_Z = (1, 6, 7, 8, 9)


# ---------------------------------------------------------------- vocabulary


def test_mask_and_absent_are_outside_the_elements():
    assert vocab.TOKEN_Z[vocab.MASK_INDEX] == vocab.MASK_Z == 99
    assert vocab.TOKEN_Z[vocab.ABSENT_INDEX] == vocab.ABSENT_Z == 0
    assert 99 not in QM9_Z and 0 not in QM9_Z


def test_z_index_roundtrip():
    z = torch.tensor([1, 6, 7, 8, 9, 0, 99, 17, 35])
    torch.testing.assert_close(vocab.index_to_z(vocab.z_to_index(z)), z)


def test_z_outside_the_vocabulary_is_an_error_not_absent():
    # silently mapping an unknown element to ABSENT would delete an atom
    with pytest.raises(ValueError, match="outside the vocabulary"):
        vocab.z_to_index(torch.tensor([6, 42]))


def test_disallowed_mask_blocks_mask_absent_and_foreign_elements():
    mask = vocab.disallowed_mask(["H", "C", "N", "O", "F"])

    assert mask[vocab.MASK_INDEX]
    assert mask[vocab.ABSENT_INDEX]
    for z in QM9_Z:
        assert not mask[vocab.TOKEN_Z.index(z)]
    assert mask[vocab.index_of_token("S")]
    assert int((~mask).sum()) == 5


def test_disallowed_mask_can_allow_absent():
    mask = vocab.disallowed_mask(["C"], allow_absent=True)

    assert not mask[vocab.ABSENT_INDEX]
    assert mask[vocab.MASK_INDEX]


def test_allowed_may_be_given_as_atomic_numbers():
    by_name = vocab.disallowed_mask(["H", "C"])
    by_number = vocab.disallowed_mask(["1", "6"])

    torch.testing.assert_close(by_name, by_number)


def test_mask_cannot_be_asked_for_as_an_element():
    with pytest.raises(ValueError, match="not an element"):
        vocab.disallowed_mask(["C", "MASK"])


# ----------------------------------------------------------------- transform


def make_inputs(n_atoms=20, sigma=3.0):
    z = torch.tensor([QM9_Z[i % len(QM9_Z)] for i in range(n_atoms)])
    return {
        properties.Z: z,
        properties.R: torch.randn(n_atoms, 3, dtype=torch.float64),
        keys.sigma_curr: torch.full((n_atoms, 1), sigma, dtype=torch.float64),
    }


def test_masked_atoms_carry_the_mask_atomic_number():
    torch.manual_seed(0)
    out = AbsorbingTypesTransform(min_rate=0.5, max_rate=0.5)(make_inputs())

    masked = out[keys.type_mask].squeeze(-1)
    assert torch.all(out[properties.Z][masked] == vocab.MASK_Z)
    assert torch.all(out[properties.Z][~masked] != vocab.MASK_Z)


def test_the_target_is_the_clean_type_not_the_masked_one():
    torch.manual_seed(0)
    inputs = make_inputs()
    clean = inputs[properties.Z].clone()

    out = AbsorbingTypesTransform(min_rate=0.9, max_rate=0.9)(inputs)

    recovered = vocab.index_to_z(out[keys.type_target].argmax(dim=-1))
    torch.testing.assert_close(recovered, clean)
    # and it is a one-hot
    torch.testing.assert_close(
        out[keys.type_target].sum(dim=-1), torch.ones(len(clean), dtype=torch.float64)
    )


def test_weight_is_one_over_the_rate_on_masked_atoms_and_zero_elsewhere():
    torch.manual_seed(0)
    rate = 0.25
    out = AbsorbingTypesTransform(min_rate=rate, max_rate=rate)(make_inputs())

    masked = out[keys.type_mask].squeeze(-1)
    weight = out[keys.type_weight].squeeze(-1)
    torch.testing.assert_close(weight[masked], torch.full_like(weight[masked], 1 / rate))
    assert torch.all(weight[~masked] == 0)


@pytest.mark.parametrize("rate", [0.05, 0.5, 1.0])
def test_the_mean_weight_is_the_masked_share_over_the_rate(rate):
    # exact identity per draw: weighting masked atoms by 1/rate and averaging
    # over every atom is what makes the objective O(1) whatever the rate was
    torch.manual_seed(0)
    transform = AbsorbingTypesTransform(min_rate=rate, max_rate=rate)

    for _ in range(20):
        out = transform(make_inputs(n_atoms=20))
        assert float(out[keys.type_weight].mean()) == pytest.approx(
            float(out[keys.type_mask].to(torch.float64).mean()) / rate
        )


@pytest.mark.parametrize("rate", [0.05, 0.5, 1.0])
def test_the_mean_weight_is_one_in_expectation(rate):
    # ... and so has expectation 1, which is what keeps the two loss terms
    # comparable across masking rates
    torch.manual_seed(0)
    transform = AbsorbingTypesTransform(min_rate=rate, max_rate=rate)
    weights = [
        float(transform(make_inputs(n_atoms=50))[keys.type_weight].mean())
        for _ in range(400)
    ]

    assert sum(weights) / len(weights) == pytest.approx(1.0, abs=0.05)


def test_masking_rate_sets_the_share_of_masked_atoms():
    torch.manual_seed(0)
    transform = AbsorbingTypesTransform(min_rate=0.3, max_rate=0.3)
    masked = [
        float(transform(make_inputs(n_atoms=50))[keys.type_mask].float().mean())
        for _ in range(100)
    ]

    assert sum(masked) / len(masked) == pytest.approx(0.3, abs=0.03)


def test_rate_from_sigma_ties_the_rate_to_the_noise_level():
    torch.manual_seed(0)
    transform = AbsorbingTypesTransform(rate_from_sigma=True, sigma_max=30.0)

    out = transform(make_inputs(sigma=15.0))

    assert float(out[keys.type_rate][0]) == pytest.approx(0.5)


def test_rate_from_sigma_needs_sigma_max():
    with pytest.raises(ValueError, match="sigma_max"):
        AbsorbingTypesTransform(rate_from_sigma=True)


@pytest.mark.parametrize("bounds", [(0.0, 1.0), (0.5, 0.2), (0.1, 1.5)])
def test_bad_rate_bounds_are_refused(bounds):
    with pytest.raises(ValueError, match="min_rate"):
        AbsorbingTypesTransform(min_rate=bounds[0], max_rate=bounds[1])


# ------------------------------------------------------------------- sampler


def make_batch(n_molecules=3, n_atoms_each=6, device="cpu"):
    n_atoms = n_molecules * n_atoms_each
    return {
        properties.Z: torch.full((n_atoms,), vocab.MASK_Z, dtype=torch.long),
        properties.R: torch.randn(n_atoms, 3),
        properties.idx_m: torch.arange(n_molecules).repeat_interleave(n_atoms_each),
        properties.n_atoms: torch.full((n_molecules,), n_atoms_each),
        keys.converged: torch.zeros(n_molecules, dtype=torch.bool),
    }


def uniform_probabilities(n_atoms, allowed=QM9_Z):
    probabilities = torch.zeros(n_atoms, vocab.n_tokens())
    for z in allowed:
        probabilities[:, vocab.TOKEN_Z.index(z)] = 1.0 / len(allowed)
    return probabilities


@pytest.mark.parametrize("schedule", ["cosine", "linear"])
def test_the_masking_rate_falls_to_exactly_zero(schedule):
    sampler = AbsorbingDirectDenoisingSampler(n_steps=20, type_schedule=schedule)

    rates = [sampler.masking_rate(step) for step in range(20, 0, -1)]

    assert rates[0] > 0.9
    assert rates[-1] == 0.0
    assert all(a >= b for a, b in zip(rates, rates[1:]))
    # the last step always unmasks whatever is left
    assert sampler.unmask_probability(1) == 1.0


def test_a_full_loop_leaves_no_masked_atom():
    torch.manual_seed(0)
    sampler = AbsorbingDirectDenoisingSampler(n_steps=16)
    batch = make_batch(n_molecules=20, n_atoms_each=9)
    probabilities = uniform_probabilities(batch[properties.Z].shape[0])

    for step in range(16, 0, -1):
        batch = sampler.unmask_types(batch, probabilities, step)

    assert not bool((batch[properties.Z] == vocab.MASK_Z).any())
    assert set(batch[properties.Z].tolist()) <= set(QM9_Z)


def test_an_unmasked_type_is_never_rewritten():
    torch.manual_seed(0)
    sampler = AbsorbingDirectDenoisingSampler(n_steps=8)
    batch = make_batch()
    n_atoms = batch[properties.Z].shape[0]
    # carbon everywhere except the first two atoms, which stay masked
    batch[properties.Z] = torch.full((n_atoms,), 6, dtype=torch.long)
    batch[properties.Z][:2] = vocab.MASK_Z
    probabilities = uniform_probabilities(n_atoms)

    for step in range(8, 0, -1):
        batch = sampler.unmask_types(batch, probabilities, step)

    assert torch.all(batch[properties.Z][2:] == 6)
    assert not bool((batch[properties.Z][:2] == vocab.MASK_Z).any())


def test_only_allowed_tokens_can_be_drawn():
    torch.manual_seed(0)
    sampler = AbsorbingDirectDenoisingSampler(n_steps=1)
    batch = make_batch(n_molecules=50, n_atoms_each=8)
    probabilities = uniform_probabilities(batch[properties.Z].shape[0], allowed=(7, 8))

    batch = sampler.unmask_types(batch, probabilities, 1)

    assert set(batch[properties.Z].tolist()) == {7, 8}


def test_converged_molecules_are_not_unmasked():
    sampler = AbsorbingDirectDenoisingSampler(n_steps=1)
    batch = make_batch(n_molecules=2, n_atoms_each=4)
    batch[keys.converged][0] = True
    probabilities = uniform_probabilities(batch[properties.Z].shape[0])

    batch = sampler.unmask_types(batch, probabilities, 1)

    assert torch.all(batch[properties.Z][:4] == vocab.MASK_Z)
    assert not bool((batch[properties.Z][4:] == vocab.MASK_Z).any())


def test_non_finite_probabilities_still_draw():
    sampler = AbsorbingDirectDenoisingSampler(n_steps=1)
    batch = make_batch(n_molecules=1, n_atoms_each=4)
    probabilities = uniform_probabilities(4)
    probabilities[0] = float("nan")
    probabilities[1] = 0.0

    batch = sampler.unmask_types(batch, probabilities, 1)

    assert not bool((batch[properties.Z] == vocab.MASK_Z).any())


def test_a_molecule_with_an_undecided_type_is_not_converged():
    sampler = AbsorbingDirectDenoisingSampler(n_steps=4)
    batch = make_batch(n_molecules=2, n_atoms_each=3)
    batch[keys.nfe] = torch.zeros(2, dtype=torch.int32)
    # molecule 0 fully typed, molecule 1 still masked
    batch[properties.Z] = torch.tensor([6, 6, 6, vocab.MASK_Z, 6, 6])

    batch = sampler.check_convergence(batch, step=1)

    assert bool(batch[keys.converged][0])
    assert not bool(batch[keys.converged][1])


def test_nan_molecules_still_converge_even_with_masked_types():
    # otherwise a diverged molecule would keep burning function evaluations
    sampler = AbsorbingDirectDenoisingSampler(n_steps=4)
    batch = make_batch(n_molecules=2, n_atoms_each=3)
    batch[keys.nfe] = torch.zeros(2, dtype=torch.int32)
    batch[properties.Z] = torch.full((6,), vocab.MASK_Z, dtype=torch.long)
    batch[properties.R][4] = float("nan")

    batch = sampler.check_convergence(batch, step=2)

    assert not bool(batch[keys.converged][0])
    assert bool(batch[keys.converged][1])


def test_the_prior_masks_every_type_and_resamples_the_positions():
    batch = make_batch()
    batch[properties.Z] = torch.full_like(batch[properties.Z], 6)
    before = batch[properties.R].clone()

    batch = AbsorbingPrior(sigma=30.0).sample_from_batch(batch)

    assert torch.all(batch[properties.Z] == vocab.MASK_Z)
    assert not torch.allclose(batch[properties.R], before)


@pytest.mark.parametrize("bad", ["greedy", "quadratic"])
def test_unknown_schedule_is_refused(bad):
    with pytest.raises(ValueError, match="type_schedule"):
        AbsorbingDirectDenoisingSampler(type_schedule=bad)


# --------------------------------------------------------- remasking corrector


def decided_batch(n_molecules=40, n_atoms_each=9):
    batch = make_batch(n_molecules, n_atoms_each)
    batch[properties.Z] = torch.full_like(batch[properties.Z], 6)
    return batch


def test_remask_prob_zero_keeps_pure_carry_over():
    torch.manual_seed(0)
    batch = decided_batch()
    n = batch[properties.Z].shape[0]
    sampler = AbsorbingDirectDenoisingSampler(n_steps=16, atom_types=QM9_Z, remask_prob=0.0)

    out = sampler.remask_types(batch, uniform_probabilities(n), 8)

    assert not bool((out[properties.Z] == vocab.MASK_Z).any())


def test_remasking_returns_some_decided_atoms_to_mask():
    torch.manual_seed(0)
    batch = decided_batch()
    n = batch[properties.Z].shape[0]
    sampler = AbsorbingDirectDenoisingSampler(
        n_steps=16, atom_types=QM9_Z, remask_prob=1.0, remask_mode="random"
    )

    out = sampler.remask_types(batch, uniform_probabilities(n), 8)

    share = float((out[properties.Z] == vocab.MASK_Z).float().mean())
    # masking_rate(8) of 16 steps under the cosine schedule is about 0.63
    assert 0.3 < share < 0.95


def test_no_remasking_on_the_last_step_so_sampling_terminates():
    torch.manual_seed(0)
    sampler = AbsorbingDirectDenoisingSampler(
        n_steps=16, atom_types=QM9_Z, remask_prob=1.0, remask_mode="random"
    )
    assert sampler.masking_rate(1) == 0.0

    batch = make_batch(n_molecules=30, n_atoms_each=9)
    probabilities = uniform_probabilities(batch[properties.Z].shape[0])
    for step in range(16, 0, -1):
        batch = sampler.unmask_types(batch, probabilities, step)

    assert not bool((batch[properties.Z] == vocab.MASK_Z).any())
    assert set(batch[properties.Z].tolist()) <= set(QM9_Z)


def test_confidence_mode_remasks_the_doubted_atoms_far_more():
    torch.manual_seed(0)
    n_each = 2
    batch = decided_batch(n_molecules=1500, n_atoms_each=n_each)
    n = batch[properties.Z].shape[0]
    # every atom holds carbon; give the even ones p(C)=0.99 and the odd ones 0.01
    probabilities = torch.full((n, vocab.n_tokens()), 0.0)
    c, h = vocab.TOKEN_Z.index(6), vocab.TOKEN_Z.index(1)
    probabilities[0::2, c], probabilities[0::2, h] = 0.99, 0.01
    probabilities[1::2, c], probabilities[1::2, h] = 0.01, 0.99
    sampler = AbsorbingDirectDenoisingSampler(
        n_steps=16, atom_types=QM9_Z, remask_prob=1.0, remask_mode="confidence"
    )

    out = sampler.remask_types(batch, probabilities, 8)
    masked = out[properties.Z] == vocab.MASK_Z

    confident = float(masked[0::2].float().mean())
    doubted = float(masked[1::2].float().mean())
    assert doubted > 10 * max(confident, 1e-6)


def test_converged_molecules_are_not_remasked():
    torch.manual_seed(0)
    batch = decided_batch(n_molecules=2, n_atoms_each=6)
    batch[keys.converged][0] = True
    n = batch[properties.Z].shape[0]
    sampler = AbsorbingDirectDenoisingSampler(
        n_steps=16, atom_types=QM9_Z, remask_prob=1.0, remask_mode="random"
    )

    out = sampler.remask_types(batch, uniform_probabilities(n), 8)

    assert not bool((out[properties.Z][:6] == vocab.MASK_Z).any())


@pytest.mark.parametrize("kwargs", [{"remask_prob": -0.1}, {"remask_mode": "greedy"}])
def test_bad_remask_arguments_are_refused(kwargs):
    with pytest.raises(ValueError):
        AbsorbingDirectDenoisingSampler(atom_types=QM9_Z, **kwargs)
