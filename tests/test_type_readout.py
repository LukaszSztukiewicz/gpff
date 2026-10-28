import numpy as np
import pytest
import torch

from schnetpack import properties

from gpff import keys
from gpff.sample.direct_denoising import DirectDenoisingSampler
from gpff.sample.type_readout import ReadoutDirectDenoisingSampler


ATOM_TYPES = (1, 6, 7, 8, 9)


def make_batch(h):
    """A batch of one molecule holding the noisy one-hot types ``h``."""
    n_atoms = h.shape[0]
    return {
        keys.h: h.clone(),
        properties.Z: torch.zeros(n_atoms, dtype=torch.long),
        properties.idx_m: torch.zeros(n_atoms, dtype=torch.long),
        properties.n_atoms: torch.tensor([n_atoms]),
        keys.converged: torch.zeros(1, dtype=torch.bool),
    }


def categorical_prediction(probabilities, type_scale=0.3):
    """What a gpff_types_categorical head emits: type_scale * softmax."""
    return type_scale * probabilities


def test_argmax_readout_matches_the_plain_sampler():
    h = torch.randn(12, len(ATOM_TYPES))
    plain = DirectDenoisingSampler(atom_types=ATOM_TYPES)
    readout = ReadoutDirectDenoisingSampler(atom_types=ATOM_TYPES, type_readout="argmax")

    expected = plain.update_types(make_batch(h), h)
    actual = readout.update_types(make_batch(h), h)

    torch.testing.assert_close(actual[keys.h], expected[keys.h])
    torch.testing.assert_close(actual[properties.Z], expected[properties.Z])


def test_normalize_recovers_a_categorical_heads_marginal_exactly():
    probabilities = torch.softmax(torch.randn(64, len(ATOM_TYPES)), dim=-1)
    h_0 = categorical_prediction(probabilities, type_scale=0.3)
    sampler = ReadoutDirectDenoisingSampler(atom_types=ATOM_TYPES, type_readout="sample")

    torch.testing.assert_close(sampler._probabilities(h_0), probabilities)


def test_probabilities_fall_back_to_the_argmax_when_nothing_is_positive():
    h_0 = -torch.ones(3, len(ATOM_TYPES))
    h_0[:, 2] = -0.5  # the largest entry, still negative
    sampler = ReadoutDirectDenoisingSampler(atom_types=ATOM_TYPES, type_readout="sample")

    probabilities = sampler._probabilities(h_0)

    assert torch.all(probabilities.argmax(dim=-1) == 2)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(3))


def test_low_temperature_sharpens_towards_the_argmax():
    h_0 = categorical_prediction(torch.tensor([[0.5, 0.3, 0.2]]))
    sharp = ReadoutDirectDenoisingSampler(
        atom_types=(1, 6, 7), type_readout="sample", type_temperature=0.1
    )._probabilities(h_0)
    flat = ReadoutDirectDenoisingSampler(
        atom_types=(1, 6, 7), type_readout="sample", type_temperature=10.0
    )._probabilities(h_0)

    assert sharp[0, 0] > 0.9
    assert flat[0].max() - flat[0].min() < 0.1


def test_sampling_recovers_the_marginal_and_argmax_does_not():
    # one atom whose marginal puts 25% on a minority element
    probabilities = torch.tensor([[0.75, 0.25]])
    h_0 = categorical_prediction(probabilities).repeat(4000, 1)
    batch = make_batch(h_0)
    batch[properties.idx_m] = torch.arange(4000)
    batch[properties.n_atoms] = torch.ones(4000, dtype=torch.long)
    batch[keys.converged] = torch.zeros(4000, dtype=torch.bool)

    sampled = ReadoutDirectDenoisingSampler(
        atom_types=(1, 6), type_readout="sample", type_seed=0
    ).update_types(batch, h_0)
    minority = (sampled[properties.Z] == 6).float().mean().item()

    assert minority == pytest.approx(0.25, abs=0.03)

    argmaxed = ReadoutDirectDenoisingSampler(
        atom_types=(1, 6), type_readout="argmax"
    ).update_types(make_batch(h_0), h_0)
    assert (argmaxed[properties.Z] == 6).sum() == 0


def test_commit_replaces_h_by_the_scaled_one_hot_of_the_draw():
    type_scale = 0.3
    h_0 = categorical_prediction(
        torch.softmax(torch.randn(16, len(ATOM_TYPES)), dim=-1), type_scale
    )
    sampler = ReadoutDirectDenoisingSampler(
        atom_types=ATOM_TYPES, type_readout="commit", type_seed=1
    )

    batch = sampler.update_types(make_batch(h_0), h_0)

    h = batch[keys.h]
    assert torch.all((h == 0) | torch.isclose(h, torch.tensor(type_scale)))
    torch.testing.assert_close(h.sum(dim=-1), torch.full((16,), type_scale))
    drawn = torch.tensor(ATOM_TYPES)[h.argmax(dim=-1)]
    torch.testing.assert_close(batch[properties.Z], drawn)


def test_converged_molecules_keep_their_elements():
    h_0 = categorical_prediction(torch.softmax(torch.randn(6, 2), dim=-1))
    batch = make_batch(h_0)
    batch[properties.Z] = torch.full((6,), 99, dtype=torch.long)
    batch[keys.converged] = torch.ones(1, dtype=torch.bool)

    out = ReadoutDirectDenoisingSampler(
        atom_types=(1, 6), type_readout="sample"
    ).update_types(batch, h_0)

    assert torch.all(out[properties.Z] == 99)


def test_type_draws_do_not_consume_the_global_rng():
    h_0 = categorical_prediction(torch.softmax(torch.randn(32, len(ATOM_TYPES)), dim=-1))
    sampler = ReadoutDirectDenoisingSampler(atom_types=ATOM_TYPES, type_readout="sample")

    torch.manual_seed(7)
    sampler.update_types(make_batch(h_0), h_0)
    after_sampling = torch.randn(5)

    torch.manual_seed(7)
    reference = torch.randn(5)

    torch.testing.assert_close(after_sampling, reference)


@pytest.mark.parametrize("probabilities_from", ["normalize", "softmax"])
def test_non_finite_estimates_still_give_a_valid_distribution(probabilities_from):
    # a molecule whose positions have gone to NaN reaches update_types before
    # check_convergence freezes it; multinomial asserts on a non-finite input
    h_0 = categorical_prediction(torch.softmax(torch.randn(5, len(ATOM_TYPES)), dim=-1))
    h_0[0] = float("nan")
    h_0[1] = float("inf")
    h_0[2] = float("-inf")
    h_0[3] = 0.0
    sampler = ReadoutDirectDenoisingSampler(
        atom_types=ATOM_TYPES,
        type_readout="sample",
        probabilities_from=probabilities_from,
    )

    probabilities = sampler._probabilities(h_0)

    assert torch.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(5))
    # and the draw itself goes through
    sampler.update_types(make_batch(h_0), h_0)


def test_commit_does_not_write_nan_into_h():
    h_0 = categorical_prediction(torch.softmax(torch.randn(4, len(ATOM_TYPES)), dim=-1))
    h_0[0] = float("nan")
    sampler = ReadoutDirectDenoisingSampler(atom_types=ATOM_TYPES, type_readout="commit")

    batch = sampler.update_types(make_batch(h_0), h_0)

    assert torch.isfinite(batch[keys.h]).all()


@pytest.mark.parametrize("bad", ["greedy", ""])
def test_unknown_readout_is_refused(bad):
    with pytest.raises(ValueError, match="type_readout"):
        ReadoutDirectDenoisingSampler(atom_types=ATOM_TYPES, type_readout=bad)
