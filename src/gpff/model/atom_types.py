"""
The type input and the type head of models that generate the atom types (gpff.atom_types).
"""

from typing import Callable, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import schnetpack.nn as snn

from gpff import keys
from gpff.atom_types import AbsorbingTypes, ContinuousTypes, Vocabulary


__all__ = ["TypeEmbedding", "TypeLogits"]


class TypeEmbedding(nn.Module):
    r"""
    Atom embedding of the type vectors h (keys.h), for PaiNN(embedding_key=keys.h): a linear
    map of :math:`h / \sqrt{1 + |h|^2}` and of :math:`\ln(1 + |h|^2)`. These are EDM's c_in
    and c_noise with the noise level read off h itself, as a time-free model has to: noisy
    continuous types, :math:`|h|^2 \approx n \sigma^2` at large sigma, stay O(1) at every
    noise level, and their size enters through the logarithm. A one-hot token is scaled by a
    constant, so for absorbing types this is an embedding lookup.

    The weights of h start as nn.Embedding's, N(0, 1), the weight of the logarithm and the
    bias at zero.

    Args:
        vocabulary: the tokens of h.
        n_atom_basis: size of the embedding.
    """

    def __init__(self, vocabulary: Vocabulary, n_atom_basis: int):
        super().__init__()
        self.linear = nn.Linear(vocabulary.size + 1, n_atom_basis)
        with torch.no_grad():
            nn.init.normal_(self.linear.weight)
            self.linear.weight[:, -1] = 0
            nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        norm2 = h.square().sum(dim=-1, keepdim=True)
        return self.linear(torch.cat([h * torch.rsqrt(1 + norm2), torch.log1p(norm2)], dim=-1))


class TypeLogits(nn.Module):
    """
    Invariant head: the logits of every slot's clean class, (n_atoms, n_classes), from
    PaiNN's scalar representation. It keeps the type process the model is trained with,
    from which the samplers of gpff.sample.atom_types take the vocabulary and the process's
    constants.

    Args:
        n_in: size of the scalar representation.
        process: the type process, AbsorbingTypes or ContinuousTypes.
        n_hidden: size of the hidden layers (see schnetpack.nn.build_mlp).
        n_layers: number of layers.
        activation: activation function.
        output_key: key of the logits.
    """

    def __init__(
        self,
        n_in: int,
        process: Union[AbsorbingTypes, ContinuousTypes],
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        output_key: str = keys.type_logits,
    ):
        super().__init__()
        self.outnet = snn.build_mlp(
            n_in=n_in,
            n_out=process.vocabulary.n_classes,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )
        self.process = process
        self.output_key = output_key
        self.model_outputs = [output_key]

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs[self.output_key] = self.outnet(inputs["scalar_representation"])
        return inputs
