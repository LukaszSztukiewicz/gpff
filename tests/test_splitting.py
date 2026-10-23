import torch
from schnetpack.data.splitting import RandomSplit

from gpff.data import SeededRandomSplit


def test_seeded_split_ignores_the_global_rng():
    torch.manual_seed(0)
    first = SeededRandomSplit(seed=999).split(range(1000), 800, 100, None)
    torch.rand(100)  # e.g. a model initialization in between
    second = SeededRandomSplit(seed=999).split(range(1000), 800, 100, None)
    assert first == second


def test_seeded_split_matches_random_split_after_manual_seed():
    # how the split of the GPFF paper's baseline was drawn, with seed 999
    torch.manual_seed(999)
    expected = RandomSplit().split(range(1000), 800, 100, None)
    assert SeededRandomSplit(seed=999).split(range(1000), 800, 100, None) == expected
