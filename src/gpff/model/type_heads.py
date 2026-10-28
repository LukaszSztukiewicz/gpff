"""
Invariant head predicting the clean atom type of every atom.

The element is a scalar property, so the head is an MLP on PaiNN's scalar
representation, beside the equivariant force head. It predicts the **clean**
type distribution p(a_0 | x_t, a_t), which for an absorbing forward process does
not depend on the noise level at all (Ou et al., ICLR 2025: the concrete score of
an absorbing process factorizes into a time-independent clean-data conditional
times a scalar function of t). So the head needs no time input, which is what
lets it sit inside a time-agnostic model without changing anything else.

Tokens the model must not emit -- MASK always, and the elements its dataset does
not contain -- get a large negative logit rather than -inf: the masked
cross-entropy multiplies the log-softmax by a one-hot target, and 0 * -inf is
NaN, while 0 * -1e4 is 0. The value is far enough below the others that the
softmax probability is zero in float32.
"""

from typing import Callable, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.nn as snn

from gpff import keys, vocab


__all__ = ["TypeLogits"]


class TypeLogits(nn.Module):
    """
    Per-atom logits over the type vocabulary.

    Args:
        n_in: width of the scalar representation.
        allowed: the element tokens the model may emit, by name ("C") or atomic
            number ("6"). MASK is always excluded.
        allow_absent: whether ABSENT is a legal prediction, i.e. whether the
            model decides how many atoms there are.
        n_hidden: hidden sizes, as in schnetpack.nn.build_mlp.
        n_layers: number of layers.
        activation: activation function.
        logits_key: key the logits are stored under.
        probabilities_key: key the softmax is stored under, for the sampler.
        disallowed_logit: logit given to the tokens that must not be emitted.
    """

    def __init__(
        self,
        n_in: int,
        allowed: Sequence[str] = vocab.QM9_ELEMENTS,
        allow_absent: bool = False,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 3,
        activation: Callable = F.silu,
        logits_key: str = keys.type_logits,
        probabilities_key: str = "type_probabilities",
        disallowed_logit: float = -1e4,
    ):
        super().__init__()
        self.outnet = snn.build_mlp(
            n_in=n_in,
            n_out=vocab.n_tokens(),
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )
        self.logits_key = logits_key
        self.probabilities_key = probabilities_key
        self.disallowed_logit = disallowed_logit
        self.model_outputs = [logits_key, probabilities_key]
        # a buffer, so it moves with the model and is restored with it
        self.register_buffer(
            "disallowed",
            vocab.disallowed_mask(allowed, allow_absent=allow_absent),
            persistent=True,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits = self.outnet(inputs["scalar_representation"])
        logits = logits.masked_fill(self.disallowed, self.disallowed_logit)
        inputs[self.logits_key] = logits
        inputs[self.probabilities_key] = torch.softmax(logits, dim=-1)
        return inputs
