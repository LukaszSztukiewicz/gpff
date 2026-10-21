from types import SimpleNamespace

import torch

from gpff import keys
from gpff.task import AtomisticTask, LearnedSigmaWeights, ModelOutput


def test_learned_sigma_weights_start_from_the_plain_loss():
    torch.manual_seed(0)
    weighting = LearnedSigmaWeights()
    loss = torch.rand(6, 3)
    sigma = torch.tensor([0.0, 0.01, 0.1, 1.0, 10.0, 30.0]).unsqueeze(-1)

    torch.testing.assert_close(weighting(loss, sigma), loss)


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
        weighting.linear.bias.fill_(-20.0)

    log_variance = weighting.log_variance(torch.tensor([0.0, 0.1, 1.0]))
    assert torch.all(log_variance.exp() >= 0.01)


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
    assert output.sigma_weighting.linear.weight.grad.abs().sum() > 0


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
