from typing import Dict, Optional, Union

import torch
import schnetpack.transform as trn
from schnetpack import properties

from gpff import keys
from gpff.atom_types import ABSENT_Z, AbsorbingTypes, ContinuousTypes


__all__ = ["Canvas", "AtomTypeTransform"]


class Canvas(trn.Transform):
    """
    Pad a molecule with ABSENT slots, for a model that generates the number of atoms
    (Vocabulary(absent=True)). Runs before PseudoPESTransform, which then noises the slots
    like atoms: an ABSENT slot is an atom of the type ABSENT (Z = 0) whose clean position
    is the molecule's centroid. So the slots get exact force labels, leave the centroid
    PseudoPESTransform centres at the origin, and the sampler parks them there. The
    permutation alignment pairs them among themselves only.

    The canvas has ``size`` slots, or the molecule's atoms and U{0, ..., max_extra} more;
    with neither it is the molecule. Only the positions and the atomic numbers are padded.

    Args:
        size: number of slots of every canvas; must hold the largest molecule.
        max_extra: largest number of ABSENT slots added to a molecule, instead of a size.
    """

    def __init__(self, size: Optional[int] = None, max_extra: int = 0):
        super().__init__()
        if size is not None and max_extra:
            raise ValueError("a canvas has a size or extra slots, not both")
        if (size is not None and size < 1) or max_extra < 0:
            raise ValueError(f"invalid canvas: size {size}, max_extra {max_extra}")
        self.size = size
        self.max_extra = max_extra

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        positions, numbers = inputs[properties.R], inputs[properties.Z]
        n_atoms = len(numbers)
        if self.size is not None:
            n_slots = self.size
        else:
            # no draw without extra slots, which keeps the random stream
            n_slots = n_atoms + (int(torch.randint(self.max_extra + 1, ())) if self.max_extra else 0)
        if n_slots < n_atoms:
            raise ValueError(f"a canvas of {n_slots} slots cannot hold a molecule of {n_atoms} atoms")
        n_absent = n_slots - n_atoms
        inputs[properties.R] = torch.cat([positions, positions.mean(0).expand(n_absent, 3)])
        inputs[properties.Z] = torch.cat([numbers, numbers.new_full((n_absent,), ABSENT_Z)])
        inputs[properties.n_atoms] = inputs[properties.n_atoms].new_tensor([n_slots])
        return inputs


class AtomTypeTransform(trn.Transform):
    """
    Corrupt the atom types of a sample with a type process, for a model that generates
    them. Runs after PseudoPESTransform, which has already aligned the noisy positions
    using the clean elements, and reads its noise level.

    Writes the type vectors the model reads (keys.h), the clean classes (keys.type_target)
    and the weights of their cross-entropy (keys.type_weight). properties.Z keeps the clean
    elements; the model does not read it.

    Args:
        process: AbsorbingTypes or ContinuousTypes.
    """

    def __init__(self, process: Union[AbsorbingTypes, ContinuousTypes]):
        super().__init__()
        self.process = process

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        classes = self.process.vocabulary.classes(inputs[properties.Z])
        h, weight = self.process.corrupt(classes, inputs[keys.sigma_curr])
        inputs.update({keys.h: h, keys.type_target: classes, keys.type_weight: weight})
        return inputs
