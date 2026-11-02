"""AdamW with a controlled update size for the weight matrices of equivariant blocks.

An equivariant layer mixes channels with one matrix that is shared over the 2l + 1
components of a feature (in PaiNN: the 3 components of a vector). Adam rescales every
entry on its own, so one learning rate still gives these matrices very different step
sizes. ``BlockAdamW`` rescales the update of each such matrix ("block") so that the
learning rate is its size, or replaces it by Muon's update.

Manolache & Niepert, "Equivariance Breaks the Learning Rate" (NeurIPS 2026 workshop).
"""
import math
from typing import Iterable, List

import torch
from schnetpack.nn import GatedEquivariantBlock
from schnetpack.representation.painn import PaiNNInteraction, PaiNNMixing
from torch import Tensor, nn
from torch.optim import Optimizer

__all__ = ["BlockAdamW", "block_param_groups"]

# the equivariant blocks, with the name of their layer acting on vector features (None: no such layer)
_BLOCKS = {PaiNNInteraction: None, PaiNNMixing: "mu_channel_mix", GatedEquivariantBlock: "mix_vectors"}
# the RMS of an AdamW update entry, which Muon's update is scaled to
_ADAMW_RMS = 0.2


def rms_norm(update: Tensor) -> Tensor:
    return update.pow(2).mean().sqrt()


def spectral_norm(update: Tensor) -> Tensor:
    """Schatten-16 norm: an upper bound of the largest singular value, at most rank**(1/16) times it
    (exact for rank one). Two squarings of the Gram matrix replace an SVD, which is too slow for wide matrices."""
    frobenius = update.norm()
    # singular values of at most one cannot overflow in the 16th power
    unit = update / frobenius.clamp_min(torch.finfo(update.dtype).tiny)
    gram = unit.T @ unit if unit.shape[0] >= unit.shape[1] else unit @ unit.T
    gram = gram @ gram
    gram = gram @ gram
    return frobenius * gram.norm() ** (1 / 8)


def newton_schulz(matrix: Tensor, steps: int = 5) -> Tensor:
    """Muon's orthogonalization: the matrix with the singular vectors of ``matrix`` and singular values near one."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix / matrix.norm().clamp_min(1e-7)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if tall else x


_NORMS = {"rms": rms_norm, "spectral": spectral_norm}


class BlockAdamW(Optimizer):
    """AdamW; the matrices of parameter groups with ``block=True`` (see ``block_param_groups``) take
    ``block_update`` steps instead:

    * ``"adam"``: AdamW's.
    * ``"rms"``, ``"spectral"``: AdamW's update divided by its RMS or spectral norm, so that the step has
      exactly this norm times ``lr`` and the direction of AdamW's. Single learning rate for all blocks;
      zero updates stay zero.
    * ``"muon"``: Muon's update, Nesterov ``momentum`` and Newton-Schulz orthogonalization (no Adam moments),
      scaled to the RMS of an AdamW update, so ``lr`` is comparable to AdamW's.

    ``weight_decay`` is decoupled, as in AdamW: a step shrinks the weights by ``lr * weight_decay`` before the
    update, whichever update it is. It is off by default; the groups of ``block_param_groups`` decay the block
    matrices only.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        block_update: str = "rms",
        momentum: float = 0.95,
    ):
        if block_update not in ("adam", "muon", *_NORMS):
            raise ValueError(f"unknown block_update {block_update!r}")
        defaults = dict(
            lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay,
            block=False, block_update=block_update, momentum=momentum,
        )
        super().__init__(params, defaults)

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        if group["block"] and any(p.ndim != 2 for p in group["params"]):
            raise ValueError("the parameters of a block group have to be weight matrices")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mode = group["block_update"] if group["block"] else "adam"
            for p in group["params"]:
                if p.grad is None:
                    continue
                if group["weight_decay"]:
                    p.mul_(1 - lr * group["weight_decay"])
                if mode == "muon":
                    update = self._muon_update(p, group)
                else:
                    update = self._adam_update(p, group)
                    if mode in _NORMS:
                        update = update / _NORMS[mode](update).clamp_min(torch.finfo(update.dtype).tiny)
                p.add_(update, alpha=-lr)
        return loss

    def _adam_update(self, p: Tensor, group: dict) -> Tensor:
        state = self.state[p]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
        beta1, beta2 = group["betas"]
        state["step"] += 1
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        exp_avg.lerp_(p.grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
        denominator = (exp_avg_sq / (1 - beta2 ** state["step"])).sqrt_().add_(group["eps"])
        return exp_avg / (1 - beta1 ** state["step"]) / denominator

    def _muon_update(self, p: Tensor, group: dict) -> Tensor:
        state = self.state[p]
        if not state:
            state["momentum_buffer"] = torch.zeros_like(p)
        buffer = state["momentum_buffer"]
        buffer.lerp_(p.grad, 1 - group["momentum"])
        update = newton_schulz(p.grad.lerp(buffer, group["momentum"]))
        return update * (_ADAMW_RMS * math.sqrt(max(p.shape)))


def block_param_groups(params: Iterable[Tensor], model: nn.Module, scope: str = "all") -> List[dict]:
    """Split ``params`` into the weight matrices of the equivariant blocks of ``model`` (``block=True``) and the rest.

    ``scope="all"``: every linear layer of the PaiNN interaction and mixing blocks and of the gated equivariant
    blocks of the head. ``scope="vector"``: only the ones acting on vector features (``mu_channel_mix``,
    ``mix_vectors``), which mix the channels the same way for all 3 components: the l = 1 irrep blocks.
    Embeddings, the filter network, biases and normalization gains are never blocks, and the rest never takes
    weight decay: ``weight_decay`` of the optimizer decays the block matrices only.
    """
    if scope not in ("all", "vector"):
        raise ValueError(f"unknown block scope {scope!r}")
    blocks = {}
    for module in model.modules():
        for block_cls, vector_layer in _BLOCKS.items():
            if isinstance(module, block_cls):
                for name, layer in module.named_modules():
                    if isinstance(layer, nn.Linear) and (scope == "all" or name == vector_layer):
                        blocks[id(layer.weight)] = layer.weight
    params = list(params)
    matrices = [p for p in params if id(p) in blocks]
    if not matrices:
        raise ValueError(f"no weight matrices of equivariant blocks in the parameters for block scope {scope!r}")
    rest = [p for p in params if id(p) not in blocks]
    return [{"params": matrices, "block": True}] + ([{"params": rest, "weight_decay": 0.0}] if rest else [])
