import pytest
import torch
import torch.nn.functional as F
from schnetpack import properties
from schnetpack.atomistic import PairwiseDistances
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.model import NeuralNetworkPotential
from schnetpack.nn.cutoff import CosineCutoff
from schnetpack.nn.radial import GaussianRBF

from gpff import keys
from gpff.atom_types import MASK_Z, AbsorbingTypes, ContinuousTypes, Vocabulary
from gpff.data.transform import AtomTypeTransform, PseudoPESTransform
from gpff.model.atom_types import TypeEmbedding, TypeLogits
from gpff.model.heads import GatedEquivariant
from gpff.model.painn import PaiNN
from gpff.pseudo_pes import VarianceExplodingGeometries, VEPseudoCalculator
from gpff.sample.atom_types import AbsorbingTypeSampler, ContinuousTypeSampler
from gpff.sample.base import Calculator
from gpff.task import CrossEntropyOutput
from gpff.utils.neighbors import compute_neighbors

QM9 = (1, 6, 7, 8, 9)
F64 = torch.float64


def test_vocabulary_orders_the_elements_then_mask():
    vocabulary = Vocabulary(QM9)
    assert vocabulary.numbers == (*QM9, MASK_Z)
    assert (vocabulary.n_classes, vocabulary.mask_index) == (5, 5)
    z = torch.tensor([6, 1, 9, 9, 7])
    assert torch.equal(vocabulary.atomic_numbers(vocabulary.classes(z)), z)


@pytest.mark.parametrize("z", [MASK_Z, 16, 0])
def test_vocabulary_refuses_what_is_not_a_clean_class(z):
    with pytest.raises(ValueError):
        Vocabulary(QM9).classes(torch.tensor([6, z]))


def test_absorbing_types_show_the_clean_class_or_mask_and_weight_the_masked_by_one_over_the_rate():
    torch.manual_seed(0)
    process = AbsorbingTypes(Vocabulary(QM9), min_rate=0.2)
    classes = torch.randint(5, (50,))
    sigma = torch.ones(50, 1, dtype=F64)
    weights = []
    for _ in range(200):
        h, weight = process.corrupt(classes, sigma)
        masked = process.vocabulary.masked(h)
        assert torch.equal(h.sum(-1), torch.ones(50, dtype=F64))
        assert torch.equal(h[~masked].argmax(-1), classes[~masked])
        assert (weight[~masked] == 0).all() and weight[masked].unique().numel() <= 1
        assert (weight <= 1 / 0.2 + 1e-12).all()
        weights.append(weight.mean())
    # the NELBO weight makes the mean over the slots unbiased whatever rate is drawn
    assert abs(torch.stack(weights).mean() - 1) < 0.1


def test_continuous_types_noise_a_scaled_one_hot_with_the_noise_level_of_the_positions():
    torch.manual_seed(0)
    process = ContinuousTypes(Vocabulary(QM9), scale=0.5, sigma_max=30.0)
    classes = torch.randint(5, (20000,))
    h, weight = process.corrupt(classes, torch.full((20000, 1), 1.5, dtype=F64))
    noise = h - 0.5 * F.one_hot(classes, 6).to(F64)
    assert (noise[:, -1] == 0).all()
    torch.testing.assert_close(noise.std(0)[:-1], torch.full((5,), 1.5, dtype=F64), rtol=0.03, atol=0)
    assert (weight == 1).all()
    prior = process.prior(torch.zeros(20000, 3, dtype=F64))
    assert (prior[:, -1] == 0).all()
    torch.testing.assert_close(prior.std(0)[:-1], torch.full((5,), 30.0, dtype=F64), rtol=0.03, atol=0)


def type_model(process, n_atom_basis=16):
    torch.manual_seed(0)
    return NeuralNetworkPotential(
        representation=PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=2,
            radial_basis=GaussianRBF(n_rbf=10, cutoff=5.0),
            cutoff_fn=CosineCutoff(5.0),
            return_vector_representation=True,
            embedding_key=keys.h,
            nuclear_embedding=TypeEmbedding(process.vocabulary, n_atom_basis),
        ),
        input_modules=[PairwiseDistances()],
        output_modules=[GatedEquivariant(n_in=n_atom_basis), TypeLogits(n_atom_basis, process)],
    ).to(F64)


def training_batch(transforms, elements=((6, 8, 1, 1), (7, 1, 1, 1, 9, 6))):
    torch.manual_seed(0)
    samples = []
    for k, z in enumerate(elements):
        sample = {
            properties.idx: torch.tensor([k]),
            properties.n_atoms: torch.tensor([len(z)]),
            properties.Z: torch.tensor(z),
            properties.R: torch.randn(len(z), 3, dtype=F64),
            properties.cell: torch.zeros(1, 3, 3, dtype=F64),
            properties.pbc: torch.zeros(3, dtype=torch.bool),
        }
        for transform in transforms:
            sample = transform(sample)
        samples.append(sample)
    batch = _atoms_collate_fn(samples)
    batch.update(compute_neighbors(batch, fully_connected=True, device="cpu"))
    return batch


PROCESSES = [
    AbsorbingTypes(Vocabulary(QM9)),
    ContinuousTypes(Vocabulary(QM9), scale=0.3, sigma_max=30.0),
]


@pytest.mark.parametrize("process", PROCESSES, ids=["absorbing", "continuous"])
def test_a_type_model_learns_the_clean_classes_without_reading_the_elements(process):
    pseudo_pes = PseudoPESTransform(
        VarianceExplodingGeometries(sigma_max=30.0, mean_free_noise=True),
        VEPseudoCalculator(),
        align_permutation=True,
        align_by_element=False,
    )
    batch = training_batch([pseudo_pes, AtomTypeTransform(process)])
    model = type_model(process)

    logits = model(dict(batch))[keys.type_logits]
    assert logits.shape == (10, 5)
    # the clean elements stay in properties.Z, but the model reads keys.h
    shuffled = dict(batch, **{properties.Z: batch[properties.Z].flip(0)})
    torch.testing.assert_close(model(shuffled)[keys.type_logits], logits)

    output = CrossEntropyOutput(name=keys.type_logits, target_property=keys.type_target, weights_key=keys.type_weight)
    nll = F.cross_entropy(logits, batch[keys.type_target], reduction="none")
    expected = (nll * batch[keys.type_weight].squeeze(-1)).mean()
    torch.testing.assert_close(output.calculate_loss({keys.type_logits: logits}, batch), expected)


class TypeToy(torch.nn.Module):
    """
    Forces 2 (x_0 - x) towards x_0 = 0, and the same class logits for every slot; records
    the type vectors of every call.
    """

    def __init__(self, process, logits):
        super().__init__()
        self.head = TypeLogits(1, process)
        self.register_buffer("logits", torch.as_tensor(logits, dtype=torch.float32))
        self.calls = []

    def forward(self, inputs):
        self.calls.append(inputs[keys.h].clone())
        inputs[keys.forces] = -2 * inputs[properties.R]
        inputs[keys.type_logits] = self.logits.expand(len(inputs[properties.R]), -1)
        return inputs


def prior_batch(sizes):
    torch.manual_seed(0)
    n = sum(sizes)
    return {
        properties.idx: torch.arange(len(sizes)),
        properties.n_atoms: torch.tensor(sizes),
        properties.idx_m: torch.arange(len(sizes)).repeat_interleave(torch.tensor(sizes)),
        properties.Z: torch.full((n,), 6),
        properties.R: 30 * torch.randn(n, 3, dtype=F64),
        properties.cell: torch.zeros(len(sizes), 3, 3, dtype=F64),
        properties.pbc: torch.zeros(len(sizes), 3, dtype=torch.bool),
    }


def sample(sampler_cls, process, logits, sizes=(4, 6), **kwargs):
    model = TypeToy(process, logits)
    sampler = sampler_cls(calculator=Calculator(model=model, device="cpu", dtype=F64), **kwargs)
    return sampler, sampler.denoise(prior_batch(list(sizes))), model.calls


def test_ctmc_unmasks_on_the_schedule_and_never_changes_a_decided_slot():
    process = AbsorbingTypes(Vocabulary(QM9))
    # every slot carbon or nitrogen
    sampler, out, calls = sample(AbsorbingTypeSampler, process, [-30.0, 0.0, 0.0, -30.0, -30.0], sizes=[20] * 100, n_steps=10)

    masked = [process.vocabulary.masked(h) for h in calls]
    assert masked[0].all() and not process.vocabulary.masked(out[keys.h]).any()
    for step, (before, after) in enumerate(zip(calls, calls[1:])):
        # a slot is MASK at the start of a step with the masking rate after the step before
        rate = sampler.masking_rate(sampler.n_steps - step)
        assert abs(masked[step + 1].double().mean() - rate) < 0.05
        decided = ~process.vocabulary.masked(before)
        assert torch.equal(after[decided], before[decided])
    assert set(out[properties.Z].tolist()) == {6, 7}


def test_ctmc_holds_the_convergence_of_molecules_with_mask_slots():
    process = AbsorbingTypes(Vocabulary(QM9))
    # every force is below f_max at once, but the types are not decided yet
    _, out, _ = sample(AbsorbingTypeSampler, process, [0.0, 30.0, 0.0, 0.0, 0.0], n_steps=10, f_max=1e9)
    assert (out[properties.Z] == 6).all()


def test_continuous_types_jump_to_the_clean_estimate():
    process = ContinuousTypes(Vocabulary(QM9), scale=0.3, sigma_max=30.0)
    _, out, calls = sample(ContinuousTypeSampler, process, [0.0, 0.0, 30.0, 0.0, 0.0], sizes=[50] * 4, n_steps=10)
    # the first call reads the prior, types at sigma_max
    assert abs(calls[0][:, :5].std() - 30) < 3 and (calls[0][:, 5] == 0).all()
    clean = process.clean_estimate(torch.tensor([[0.0, 0, 30, 0, 0]], dtype=F64))
    torch.testing.assert_close(out[keys.h], clean.expand(200, -1))
    assert (out[properties.Z] == 7).all()


def test_samplers_refuse_a_model_of_the_other_process():
    model = TypeToy(AbsorbingTypes(Vocabulary(QM9)), [0.0] * 5)
    sampler = ContinuousTypeSampler(calculator=Calculator(model=model, device="cpu", dtype=F64), n_steps=2)
    with pytest.raises(ValueError):
        sampler.denoise(prior_batch([3]))
