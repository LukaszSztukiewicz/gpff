"""EDM2-inspired linear weights, without mutating parameters during forward.

Use ordinary modules rather than torch parametrizations: GPFF exports whole
pickled models, which parametrized modules do not support.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def _row_norm(w):
    # Accumulate low-precision reductions in float32, but retain float64 for
    # derivative/equivariance checks.
    v = w.float() if w.dtype in (torch.float16, torch.bfloat16) else w
    return torch.linalg.vector_norm(v, dim=1, keepdim=True).clamp_min(1e-12)


def soft_unit_vectors(v):
    """Per-channel equivariant direction coordinates, smooth at the zero vector.

    Unlike v/|v|, this map has no singular Jacobian at symmetry-induced zeros.
    Used as tensor-product operands, not as a force/output clipping operation.
    """
    return v * torch.rsqrt(1 + v.square().sum(dim=-2, keepdim=True))


class ScalarRMSNorm(nn.Module):
    """Per-atom soft RMS normalization, retaining scalar means; no learned gain."""

    def forward(self, q):
        return q * torch.rsqrt(1 + q.square().mean(dim=-1, keepdim=True))


class NormalizedDense(nn.Module):
    """Unit-L2 effective rows; unit-RMS stored rows equalize relative update scales.

    Normalization is differentiated, so gradients are tangent to each weight
    sphere. ``project_`` must also run after optimizer updates (AtomisticTask
    does this), keeping the effective learning rate from drifting with norms.
    The bias is a weight on a constant input and shares the row norm, so it
    cannot bypass normalization. SiLU's expected RMS attenuation is compensated.
    """

    def __init__(self, layer: nn.Linear):
        super().__init__()
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.weight = layer.weight
        self.bias = layer.bias
        self.activation = getattr(layer, "activation", nn.Identity())
        if torch.any(self._augmented_weight().detach().square().sum(dim=1) == 0):
            raise ValueError("weight_norm requires nonzero affine rows; use output_gain=0 for a zero output")
        self.project_()

    def _augmented_weight(self):
        return (self.weight if self.bias is None else
                torch.cat([self.weight, self.bias[:, None]], dim=1))

    def normalized_parameters(self):
        norm = _row_norm(self._augmented_weight())
        weight = (self.weight / norm).to(self.weight.dtype)
        bias = None if self.bias is None else (self.bias / norm[:, 0]).to(self.bias.dtype)
        return weight, bias

    @torch.no_grad()
    def project_(self):
        fan_in = self.in_features + (self.bias is not None)
        scale = (math.sqrt(fan_in) / _row_norm(self._augmented_weight())).to(self.weight.dtype)
        self.weight.mul_(scale)
        if self.bias is not None:
            self.bias.mul_(scale[:, 0])

    def forward(self, x):
        weight, bias = self.normalized_parameters()
        y = F.linear(x, weight, bias)
        if self.activation is F.silu:
            return F.silu(y) / 0.596
        return self.activation(y) if self.activation is not None else y


def normalize_linears(module: nn.Module):
    """Replace linear/Dense children, retaining shared modules and parameter names."""
    replaced = {}
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                if id(child) not in replaced:
                    replaced[id(child)] = NormalizedDense(child)
                setattr(parent, name, replaced[id(child)])


@torch.no_grad()
def project_weights(module: nn.Module):
    for child in module.modules():
        if isinstance(child, NormalizedDense):
            child.project_()
