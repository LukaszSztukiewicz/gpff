from typing import Dict, Union

import torch
import schnetpack.transform as trn
from schnetpack import properties

from gpff import keys
from gpff.atom_types import AbsorbingTypes, ContinuousTypes


__all__ = ["AtomTypeTransform"]


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
