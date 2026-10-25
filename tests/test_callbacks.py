import math
from types import SimpleNamespace

import pytest
import torch
from schnetpack import properties

from gpff import keys
from gpff.callbacks import DenoisingErrorCurve, DeterministicDataSeed


def epoch_seed(base_seed, epoch):
    trainer = SimpleNamespace(current_epoch=epoch, fit_loop=SimpleNamespace(_data_fetcher=None))
    DeterministicDataSeed(base_seed=base_seed).on_train_epoch_start(trainer, None)
    return torch.initial_seed()


def test_epoch_seed_depends_only_on_seed_and_epoch():
    assert epoch_seed(999, 3) == epoch_seed(999, 3)


def test_nearby_seeds_do_not_share_shifted_epochs():
    assert epoch_seed(1000, 0) != epoch_seed(999, 1)


def test_denoising_error_curve_logs_without_touching_global_rng():
    sizes = [3, 5]
    x_0 = torch.randn(sum(sizes), 3, generator=torch.Generator().manual_seed(0))
    batch = {
        keys.x_0: x_0 - x_0.mean(0),
        properties.Z: torch.ones(sum(sizes), dtype=torch.long),
        properties.idx_m: torch.repeat_interleave(torch.arange(2), torch.tensor(sizes)),
        properties.n_atoms: torch.tensor(sizes),
    }
    logged = {}
    pl_module = SimpleNamespace(
        # the identity removes no noise
        predict_without_postprocessing=lambda inputs: {keys.forces: torch.zeros_like(inputs[properties.R])},
        log=lambda name, value, **kwargs: logged.setdefault(name, (float(value), kwargs)),
    )
    callback = DenoisingErrorCurve(sigmas=[1.0, 0.1], aligned=False)

    rng_state = torch.get_rng_state()
    callback.on_validation_batch_end(SimpleNamespace(current_epoch=0), pl_module, None, batch, 0)
    assert torch.equal(torch.get_rng_state(), rng_state)

    assert set(logged) == {"val/e_sigma_0.1", "val/e_sigma_1", "val/e_log_sigma_integral"}
    assert logged["val/e_sigma_0.1"][1]["batch_size"] == 2
    # the identity's error does not depend on sigma, so the integral is e * log(10)
    error = logged["val/e_sigma_1"][0]
    assert logged["val/e_sigma_0.1"][0] == pytest.approx(error)
    assert logged["val/e_log_sigma_integral"][0] == pytest.approx(error * math.log(10))
