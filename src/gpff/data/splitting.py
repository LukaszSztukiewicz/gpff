import torch
from schnetpack.data.splitting import SplittingStrategy, absolute_split_sizes

__all__ = ["SeededRandomSplit"]


class SeededRandomSplit(SplittingStrategy):
    """
    Random split drawn from its own generator, so the partitions depend only on
    the seed and the dataset size, and not on what used the global RNG before
    (the model initialization, for one). It gives the same partitions as
    schnetpack's RandomSplit right after ``torch.manual_seed(seed)``; seed 999
    is the split of the GPFF paper's baseline.
    """

    def __init__(self, seed: int = 999):
        """
        Args:
            seed: seed of the generator the permutation is drawn from.
        """
        super().__init__()
        self.seed = seed

    def split(self, dataset, *split_sizes):
        dsize = len(dataset)
        split_sizes = absolute_split_sizes(dsize, split_sizes)
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(dsize, generator=generator).tolist()
        offsets = torch.cumsum(torch.tensor(split_sizes), dim=0)
        return [
            indices[offset - length : offset]
            for offset, length in zip(offsets, split_sizes)
        ]
