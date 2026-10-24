from types import SimpleNamespace

import torch
from schnetpack.train.callbacks import ExponentialMovingAverage as SPKExponentialMovingAverage
from torch_ema import ExponentialMovingAverage as EMA

from gpff.callbacks import DeterministicDataSeed, ExponentialMovingAverage, _ForeachEMA


def epoch_seed(base_seed, epoch):
    trainer = SimpleNamespace(current_epoch=epoch, fit_loop=SimpleNamespace(_data_fetcher=None))
    DeterministicDataSeed(base_seed=base_seed).on_train_epoch_start(trainer, None)
    return torch.initial_seed()


def test_epoch_seed_depends_only_on_seed_and_epoch():
    assert epoch_seed(999, 3) == epoch_seed(999, 3)


def test_nearby_seeds_do_not_share_shifted_epochs():
    assert epoch_seed(1000, 0) != epoch_seed(999, 1)


def test_foreach_ema_update_is_bit_identical_to_torch_ema():
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(shape)) for shape in [(7, 3), (5,), (2, 2, 2)]]
    params.append(torch.nn.Parameter(torch.randn(4, dtype=torch.float64)))
    reference, ema = EMA(params, decay=0.99), _ForeachEMA(params, decay=0.99)

    for _ in range(50):
        with torch.no_grad():
            for param in params:
                param.add_(torch.randn_like(param))
        reference.update()
        ema.update()

    assert ema.num_updates == reference.num_updates
    for shadow, shadow_reference in zip(ema.shadow_params, reference.shadow_params):
        assert torch.equal(shadow, shadow_reference)


def test_ema_callback_state_loads_from_schnetpack_checkpoints():
    # Lightning stores callback state under the state key
    assert ExponentialMovingAverage(decay=0.9).state_key == SPKExponentialMovingAverage(decay=0.9).state_key
