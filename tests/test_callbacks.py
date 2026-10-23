from types import SimpleNamespace

import torch

from gpff.callbacks import DeterministicDataSeed


def epoch_seed(base_seed, epoch):
    trainer = SimpleNamespace(current_epoch=epoch, fit_loop=SimpleNamespace(_data_fetcher=None))
    DeterministicDataSeed(base_seed=base_seed).on_train_epoch_start(trainer, None)
    return torch.initial_seed()


def test_epoch_seed_depends_only_on_seed_and_epoch():
    assert epoch_seed(999, 3) == epoch_seed(999, 3)


def test_nearby_seeds_do_not_share_shifted_epochs():
    assert epoch_seed(1000, 0) != epoch_seed(999, 1)
