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
from gpff.atom_types import ABSENT_Z, MASK_Z, AbsorbingTypes, ContinuousTypes, Vocabulary
from gpff.data.transform import AtomTypeTransform, Canvas, PseudoPESTransform
from gpff.model.atom_types import TypeEmbedding, TypeLogits
from gpff.model.heads import GatedEquivariant
from gpff.model.painn import PaiNN
from gpff.pseudo_pes import VarianceExplodingGeometries, VEPseudoCalculator
from gpff.sample.atom_types import AbsorbingTypeSampler, CanvasPrior, ContinuousTypeSampler
from gpff.sample.base import Calculator
from gpff.task import CrossEntropyOutput
from gpff.utils.batching import batch_to_atoms_list
from gpff.utils.neighbors import compute_neighbors

QM9 = (1, 6, 7, 8, 9)
F64 = torch.float64


def test_vocabulary_orders_the_elements_then_mask():
    vocabulary = Vocabulary(QM9)
    assert vocabulary.numbers == (*QM9, MASK_Z)
    assert (vocabulary.n_classes, vocabulary.mask_index, vocabulary.absent_index) == (5, 5, None)
    z = torch.tensor([6, 1, 9, 9, 7])
    assert torch.equal(vocabulary.atomic_numbers(vocabulary.classes(z)), z)


def test_absent_is_the_last_clean_class():
    vocabulary = Vocabulary(QM9, absent=True)
    assert vocabulary.numbers == (*QM9, ABSENT_Z, MASK_Z)
    assert (vocabulary.n_classes, vocabulary.absent_index, vocabulary.mask_index) == (6, 5, 6)
    assert vocabulary.classes(torch.tensor([ABSENT_Z, 6])).tolist() == [5, 1]


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


def test_continuous_types_with_mask_hide_noisy_slots_at_a_random_rate():
    torch.manual_seed(0)
    process = ContinuousTypes(Vocabulary(QM9), scale=0.5, sigma_max=30.0, mask=True, min_rate=0.2)
    classes = torch.randint(5, (50,))
    shares = []
    for _ in range(20):
        h, weight = process.corrupt(classes, torch.full((50, 1), 0.01, dtype=F64))
        masked = process.vocabulary.masked(h)
        shares.append(masked.double().mean())
        assert torch.equal(h[masked], process.vocabulary.mask_tokens(int(masked.sum()), F64))
        # the visible slots are the noisy one-hot, with the weight of continuous types
        assert torch.equal(h[~masked].argmax(-1), classes[~masked]) and (h[~masked][:, -1] == 0).all()
        assert (weight[~masked] == 1).all() and weight[masked].unique().numel() <= 1 and (weight[masked] >= 1).all()
    assert 0.2 < torch.stack(shares).mean() < 0.9
    assert process.vocabulary.masked(process.prior(torch.zeros(10, 3, dtype=F64))).all()


def molecule(n_atoms=7, shift=5.0):
    generator = torch.Generator().manual_seed(0)
    return {
        properties.R: torch.randn(n_atoms, 3, dtype=F64, generator=generator) + shift,
        properties.Z: torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1][:n_atoms]),
        properties.n_atoms: torch.tensor([n_atoms]),
    }


def test_a_canvas_pads_a_molecule_with_absent_slots_at_its_centroid():
    positions = molecule()[properties.R]
    outputs = Canvas(size=12)(molecule())
    assert outputs[properties.n_atoms].tolist() == [12]
    assert outputs[properties.Z].tolist() == [6, 6, 8, 1, 1, 1, 1] + [ABSENT_Z] * 5
    torch.testing.assert_close(outputs[properties.R][7:], positions.mean(0).expand(5, 3))
    # the slots keep the centroid PseudoPESTransform centres at the origin
    torch.testing.assert_close(outputs[properties.R].mean(0), positions.mean(0))
    with pytest.raises(ValueError):
        Canvas(size=6)(molecule())


def test_a_canvas_with_extra_slots_has_up_to_max_extra_more():
    torch.manual_seed(0)
    sizes = {int(Canvas(max_extra=3)(molecule())[properties.n_atoms]) for _ in range(100)}
    assert sizes == {7, 8, 9, 10}
    assert Canvas()(molecule())[properties.Z].tolist() == molecule()[properties.Z].tolist()


def test_absent_slots_get_the_force_to_the_centroid():
    pseudo_pes = PseudoPESTransform(VarianceExplodingGeometries(sigma_max=30.0, mean_free_noise=True), VEPseudoCalculator())
    outputs = pseudo_pes(Canvas(size=12)(molecule()))
    absent = outputs[properties.Z] == ABSENT_Z
    torch.testing.assert_close(outputs[keys.x_0][absent], torch.zeros(5, 3, dtype=F64))
    torch.testing.assert_close(outputs[keys.x_0][~absent].mean(0), torch.zeros(3, dtype=F64))
    torch.testing.assert_close(outputs[keys.forces], 2 * (outputs[keys.x_0] - outputs[properties.R]))


def test_absent_slots_pair_only_among_themselves():
    x0 = molecule(4, shift=0.0)[properties.R]
    x0 = x0 - x0.mean(0)
    inputs = Canvas(size=5)({**molecule(4), properties.R: x0.clone()})
    # sigma = 1: the ABSENT slot lands on atom 0 and atom 0 on the centroid, which an
    # anonymous pairing of all the slots would exchange back
    noise = torch.zeros(5, 3, dtype=F64)
    noise[4], noise[0] = x0[0], -x0[0]
    pseudo_pes = PseudoPESTransform(
        VarianceExplodingGeometries(sigma_max=1.0),
        VEPseudoCalculator(),
        align_permutation=True,
        align_by_element=False,
    )
    outputs = pseudo_pes(inputs, t=1.0, noise=noise)
    torch.testing.assert_close(outputs[properties.R][4], x0[0])
    torch.testing.assert_close(outputs[keys.forces][4], -2 * x0[0])


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


PROCESSES = {
    "absorbing": lambda vocabulary: AbsorbingTypes(vocabulary),
    "continuous": lambda vocabulary: ContinuousTypes(vocabulary, scale=0.3, sigma_max=30.0),
    "continuous_mask": lambda vocabulary: ContinuousTypes(vocabulary, scale=0.3, sigma_max=30.0, mask=True),
}


@pytest.mark.parametrize("canvas", [None, 9])
@pytest.mark.parametrize("name", PROCESSES)
def test_a_type_model_learns_the_clean_classes_without_reading_the_elements(name, canvas):
    process = PROCESSES[name](Vocabulary(QM9, absent=canvas is not None))
    pseudo_pes = PseudoPESTransform(
        VarianceExplodingGeometries(sigma_max=30.0, mean_free_noise=True),
        VEPseudoCalculator(),
        align_permutation=True,
        align_by_element=False,
    )
    batch = training_batch([Canvas(size=canvas), pseudo_pes, AtomTypeTransform(process)])
    model = type_model(process)

    logits = model(dict(batch))[keys.type_logits]
    assert logits.shape == (10 if canvas is None else 18, process.vocabulary.n_classes)
    # the clean elements stay in properties.Z, but the model reads keys.h
    shuffled = dict(batch, **{properties.Z: batch[properties.Z].flip(0)})
    torch.testing.assert_close(model(shuffled)[keys.type_logits], logits)

    output = CrossEntropyOutput(name=keys.type_logits, target_property=keys.type_target, weights_key=keys.type_weight)
    nll = F.cross_entropy(logits, batch[keys.type_target], reduction="none")
    expected = (nll * batch[keys.type_weight].squeeze(-1)).mean()
    torch.testing.assert_close(output.calculate_loss({keys.type_logits: logits}, batch), expected)


class TypeToy(torch.nn.Module):
    """
    Forces 2 (x_0 - x) towards x_0 = 0, and class logits, the same for every slot or one
    row per slot; records the type vectors of every call.
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


def test_continuous_mask_slots_unmask_on_the_schedule_and_then_follow_the_channel():
    process = ContinuousTypes(Vocabulary(QM9), scale=0.3, sigma_max=30.0, mask=True)
    sampler, out, calls = sample(ContinuousTypeSampler, process, [0.0, 0.0, 30.0, 0.0, 0.0], sizes=[20] * 100, n_steps=10)

    masked = [process.vocabulary.masked(h) for h in calls]
    assert masked[0].all()
    for step, (before, after) in enumerate(zip(calls, calls[1:])):
        assert abs(masked[step + 1].double().mean() - sampler.masking_rate(sampler.n_steps - step)) < 0.05
        # a decided slot is perturbed and jumps, but is never MASK again
        assert not process.vocabulary.masked(after[~masked[step]]).any()
    clean = process.clean_estimate(torch.tensor([[0.0, 0, 30, 0, 0]], dtype=F64))
    torch.testing.assert_close(out[keys.h], clean.expand(2000, -1))
    assert (out[properties.Z] == 7).all()


def test_canvas_prior_replaces_the_molecules_by_canvases():
    data = prior_batch([3, 5])
    out = CanvasPrior(size=9).sample_from_batch(dict(data))
    assert out[properties.n_atoms].tolist() == [9, 9]
    assert torch.equal(out[properties.idx_m], torch.arange(2).repeat_interleave(9))
    assert out[properties.R].shape == (18, 3) and (out[properties.Z] == MASK_Z).all()
    extra = [CanvasPrior(max_extra=2).sample_from_batch(dict(data))[properties.n_atoms] - data[properties.n_atoms] for _ in range(50)]
    assert set(torch.cat(extra).tolist()) == {0, 1, 2}


@pytest.mark.parametrize(
    "sampler_cls, process",
    [
        (AbsorbingTypeSampler, AbsorbingTypes(Vocabulary(QM9, absent=True))),
        (ContinuousTypeSampler, ContinuousTypes(Vocabulary(QM9, absent=True), scale=0.3, sigma_max=30.0, mask=True)),
    ],
    ids=["absorbing", "continuous_mask"],
)
def test_slots_that_end_absent_leave_the_molecules(sampler_cls, process):
    # the slots are carbon and ABSENT in turn
    carbon, absent = [-30.0, 30, -30, -30, -30, -30], [-30.0, -30, -30, -30, -30, 30]
    model = TypeToy(process, torch.tensor([carbon, absent]).repeat(18, 1))
    sampler = sampler_cls(calculator=Calculator(model=model, device="cpu", dtype=F64), n_steps=10)
    out = sampler.denoise(CanvasPrior(size=12).sample_from_batch(prior_batch([3, 5, 4])))

    assert out[properties.Z].tolist() == [6, ABSENT_Z] * 18
    assert [m.get_chemical_symbols() for m in batch_to_atoms_list(out)] == [["C"] * 6] * 3


def test_samplers_refuse_a_model_of_the_other_process():
    model = TypeToy(AbsorbingTypes(Vocabulary(QM9)), [0.0] * 5)
    sampler = ContinuousTypeSampler(calculator=Calculator(model=model, device="cpu", dtype=F64), n_steps=2)
    with pytest.raises(ValueError):
        sampler.denoise(prior_batch([3]))
