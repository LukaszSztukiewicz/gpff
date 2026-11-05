from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.utils import str2class

import gpff.cli
from gpff import keys
from gpff.sample.base import Calculator
from gpff.sample.direct_denoising import DirectDenoisingSampler

CONFIG_DIR = Path(gpff.cli.__file__).parents[1] / "configs"
GEOM_ELEMENTS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 83]


def config_of(*overrides):
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        return compose("train", overrides=[
            "experiment=gpff_types_geom_drugs", "run.path=/runs", "run.data_dir=/data",
            "globals.n_atom_basis=16", "globals.n_interactions=2", "globals.n_rbf=12",
            "globals.n_type_rbf=9", "globals.knn_k=3", "globals.prenorm=true",
            "globals.degree_norm=true", "globals.bounded_forces=true", "globals.global_channel=true",
            *overrides,
        ])


def molecule(numbers, index):
    return {
        properties.idx: torch.tensor([index]),
        properties.n_atoms: torch.tensor([len(numbers)]),
        properties.Z: torch.tensor(numbers),
        properties.R: torch.randn(len(numbers), 3, dtype=torch.float64),
        properties.cell: torch.zeros(1, 3, 3, dtype=torch.float64),
        properties.pbc: torch.zeros(3, dtype=torch.bool),
    }


@pytest.mark.parametrize("view", ["top5", "all"])
@pytest.mark.parametrize("graph", ["knn", "size_radius"])
@pytest.mark.parametrize("basis", ["uniform", "hybrid", "log"])
def test_geom_types_train_with_both_views_graphs_and_radial_bases(view, graph, basis):
    torch.manual_seed(0)
    config = config_of(f"data=geom_drugs_revisited_{view}",
                       f"model/representation/graph={graph}", f"model/representation/radial_basis={basis}")
    assert list(config.globals.atom_types) == GEOM_ELEMENTS
    assert config.globals.n_types == len(GEOM_ELEMENTS)
    transform = hydra.utils.instantiate(config.generative_transform)
    inputs = _atoms_collate_fn([
        transform(molecule(GEOM_ELEMENTS, 0)),
        transform(molecule([1, 6, 7, 8, 15, 16, 17, 35], 1)),
    ])
    torch.testing.assert_close(inputs[keys.h_0].sum(-1), torch.ones(21, dtype=torch.float64))
    assert not torch.equal(inputs[keys.h], inputs[keys.h_0])
    model = hydra.utils.instantiate(config.model).double()
    task = hydra.utils.instantiate(config.task, model=model,
                                  optimizer_cls=str2class(config.task.optimizer_cls),
                                  scheduler_cls=str2class(config.task.scheduler_cls))
    targets = task._filter_targets(inputs)
    predicted = task.predict_without_postprocessing(dict(inputs))
    assert predicted[keys.type_logits].shape == (21, 13)
    loss = task.loss_fn(predicted, targets)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    for module in (model.representation.embedding, *model.output_modules):
        grads = [p.grad for p in module.parameters()]
        assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
        assert any(grad.abs().sum() > 0 for grad in grads)
    assert properties.idx_i not in inputs  # graph is built inside PaiNN
    # Clean atomic numbers must not leak into the denoiser of the noisy types.
    changed = dict(inputs, **{properties.Z: torch.ones_like(inputs[properties.Z])})
    other = model(changed)
    torch.testing.assert_close(predicted[keys.forces], other[keys.forces])
    torch.testing.assert_close(predicted[keys.type_logits], other[keys.type_logits])


def test_type_diffusion_rejects_elements_missing_from_the_vocabulary():
    transform = hydra.utils.instantiate(config_of().generative_transform)
    with pytest.raises(ValueError, match="Atomic numbers missing from atom_types.*10"):
        transform(molecule([1, 6, 10], 0))


@pytest.mark.parametrize("converged", [[False, False], [True, False], [True, True]])
def test_calculator_keeps_type_predictions_when_skipping_converged_molecules(converged):
    config = config_of()
    calculator = Calculator(model=hydra.utils.instantiate(config.model), device="cpu", dtype=torch.float64,
                            skip_converged=True)
    inputs = hydra.utils.instantiate(config.prior).sample_from_batch(
        _atoms_collate_fn([molecule(GEOM_ELEMENTS, 0), molecule([1, 6, 35], 1)])
    )
    inputs[keys.converged] = torch.zeros(2, dtype=torch.bool)
    inputs[keys.nfe] = torch.zeros(2, dtype=torch.long)
    full = calculator.calculate(dict(inputs), forces=True, x0=True)
    inputs[keys.converged] = torch.tensor(converged)
    result = calculator.calculate(dict(inputs), forces=True, x0=True)
    active = ~inputs[keys.converged][inputs[properties.idx_m]]
    torch.testing.assert_close(result[keys.h_0][active], full[keys.h_0][active])
    torch.testing.assert_close(result[keys.h_0][~active], inputs[keys.h][~active])
    assert properties.idx_i not in inputs


def test_direct_denoising_freezes_both_positions_and_types_of_converged_molecules():
    config = config_of()
    sampler = DirectDenoisingSampler(
        calculator=Calculator(model=hydra.utils.instantiate(config.model), device="cpu", dtype=torch.float64),
        atom_types=GEOM_ELEMENTS, n_steps=2,
    )
    inputs = hydra.utils.instantiate(config.prior).sample_from_batch(
        _atoms_collate_fn([molecule(GEOM_ELEMENTS, 0), molecule([1, 6, 35], 1)])
    )
    inputs[keys.converged] = torch.tensor([True, False])
    inputs[keys.nfe] = torch.zeros(2, dtype=torch.long)
    positions, types = inputs[properties.R].clone(), inputs[keys.h].clone()
    result = sampler.denoise_step(inputs, step=2)
    torch.testing.assert_close(result[properties.R][:13], positions[:13])
    torch.testing.assert_close(result[keys.h][:13], types[:13])
    assert set(result[properties.Z].tolist()) <= set(GEOM_ELEMENTS)
