import math
from pathlib import Path
from types import SimpleNamespace

import hydra
import torch
from hydra import compose, initialize_config_dir
from schnetpack.utils import str2class

import gpff.cli  # registers the resolvers the configs use
from gpff import keys
from gpff.optim import block_param_groups
from gpff.task import AtomisticTask, LearnedSigmaWeights, ModelOutput

CONFIG_DIR = Path(gpff.cli.__file__).parents[1] / "configs"


def test_learned_sigma_weights_start_from_the_plain_loss():
    torch.manual_seed(0)
    weighting = LearnedSigmaWeights()
    loss = torch.rand(6, 3)
    sigma = torch.tensor([0.0, 0.01, 0.1, 1.0, 10.0, 30.0]).unsqueeze(-1)

    torch.testing.assert_close(weighting(loss, sigma), loss)


def test_learned_sigma_weights_leave_the_global_rng_alone():
    # the task is built before the data split is drawn from the global RNG
    torch.manual_seed(999)
    expected = torch.rand(5)
    torch.manual_seed(999)
    weighting = LearnedSigmaWeights()
    torch.testing.assert_close(torch.rand(5), expected)

    torch.testing.assert_close(weighting.frequencies, LearnedSigmaWeights().frequencies)
    assert not torch.equal(weighting.frequencies, LearnedSigmaWeights(seed=1).frequencies)


def test_learned_sigma_weights_learn_the_mean_loss_per_noise_level():
    torch.manual_seed(0)
    weighting = LearnedSigmaWeights()
    optimizer = torch.optim.Adam(weighting.parameters(), lr=0.05)
    # two noise levels whose losses differ by a factor of 1000
    sigma = torch.tensor([0.05, 5.0]).repeat_interleave(256).unsqueeze(-1)
    mean_loss = torch.tensor([0.002, 2.0]).repeat_interleave(256).unsqueeze(-1)

    for _ in range(2000):
        loss = mean_loss * torch.distributions.Exponential(1.0).sample((512, 3))
        optimizer.zero_grad()
        weighting(loss, sigma).mean().backward()
        optimizer.step()

    learned = weighting.log_variance(torch.tensor([0.05, 5.0])).exp()
    torch.testing.assert_close(learned, torch.tensor([0.002, 2.0]), rtol=0.1, atol=0)


def test_learned_sigma_weights_tolerance_bounds_the_weight():
    weighting = LearnedSigmaWeights(tolerance=0.01)
    with torch.no_grad():
        weighting.bias.fill_(-20.0)

    log_variance = weighting.log_variance(torch.tensor([0.0, 0.1, 1.0]))
    assert torch.all(log_variance.exp() >= 0.01)


def test_learned_sigma_weights_with_a_tolerance_start_at_one_over_one_plus_tolerance():
    loss = torch.rand(4, 3)
    sigma = torch.full((4, 1), 0.3)

    weighted = LearnedSigmaWeights(tolerance=0.5)(loss, sigma)

    torch.testing.assert_close(weighted, loss / 1.5 + math.log(1.5))


def test_model_output_applies_learned_sigma_weights_to_the_weighted_loss():
    torch.manual_seed(0)
    output = ModelOutput(
        name=keys.forces,
        metrics={},
        weights_key=keys.forces_weight,
        sigma_weighting=LearnedSigmaWeights(),
    )
    sigma = torch.tensor([0.1, 0.1, 2.0, 2.0]).unsqueeze(-1)
    pred = {keys.forces: torch.randn(4, 3)}
    target = {
        keys.forces: torch.randn(4, 3),
        keys.forces_weight: 1 / sigma**2,
        keys.sigma_curr: sigma,
    }

    loss = output.calculate_loss(pred, target)
    expected = torch.mean(target[keys.forces_weight] * (pred[keys.forces] - target[keys.forces]) ** 2)
    torch.testing.assert_close(loss, expected)

    loss.backward()
    assert output.sigma_weighting.weight.grad.abs().sum() > 0
    assert output.sigma_weighting.bias.grad.abs() > 0


def test_task_passes_sigma_to_learned_sigma_weights_and_logs_them():
    output = ModelOutput(
        name=keys.forces, metrics={}, sigma_weighting=LearnedSigmaWeights(log_sigmas=(0.1, 1.0))
    )
    batch = {keys.forces: torch.zeros(2, 3), keys.sigma_curr: torch.ones(2, 1)}
    logged = {}
    task = SimpleNamespace(
        outputs=[output],
        log=lambda name, value, **kwargs: logged.__setitem__(name, value),
    )

    targets = AtomisticTask._filter_targets(task, batch)
    assert keys.sigma_curr in targets

    AtomisticTask.on_validation_epoch_end(task)
    assert set(logged) == {"forces_logvar/sigma_0.1", "forces_logvar/sigma_1"}


def test_task_without_learned_sigma_weights_neither_asks_for_sigma_nor_logs():
    output = ModelOutput(name=keys.forces, metrics={})
    logged = {}
    task = SimpleNamespace(outputs=[output], log=lambda name, value, **kwargs: logged.update({name: value}))

    targets = AtomisticTask._filter_targets(task, {keys.forces: torch.zeros(2, 3)})
    AtomisticTask.on_validation_epoch_end(task)

    assert keys.sigma_curr not in targets and not logged


def test_gpff_experiments_train_the_learned_weights_with_the_block_optimizer():
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        config = compose(
            "train",
            overrides=[
                "experiment=gpff",
                "run.path=/runs",
                "run.data_dir=/data",
                "task/optimizer=block_adam",
                "globals.sigma_weighting={_target_:gpff.task.LearnedSigmaWeights}",
            ],
        )
    model = hydra.utils.instantiate(config.model)
    task = hydra.utils.instantiate(
        config.task,
        model=model,
        optimizer_cls=str2class(config.task.optimizer_cls),
        scheduler_cls=str2class(config.task.scheduler_cls),
    )

    weighting = task.outputs[0].sigma_weighting
    assert isinstance(weighting, LearnedSigmaWeights)
    learned = {id(p) for p in weighting.parameters()}
    assert learned <= {id(p) for p in task.parameters()}

    # the weighting is not part of the model, so it is no block and takes no weight decay
    blocks, rest = block_param_groups(task.parameters(), task.model, config.task.block_scope)
    assert learned.isdisjoint(id(p) for p in blocks["params"])
    assert learned <= {id(p) for p in rest["params"]}
