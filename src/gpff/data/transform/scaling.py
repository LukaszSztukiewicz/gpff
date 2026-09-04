from typing import Dict
import torch
from schnetpack import properties
from gpff import keys
import torch.nn as nn

__all__ = [
    "RemoveVariance",
    "AddVariance",
]


class RemoveVariance(nn.Module):
    def __init__(self, property_key: str = properties.R, variance_key: str = keys.variance):
        super().__init__()
        self.property_key = property_key
        self.variance_key = variance_key

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        property_tensor = inputs[self.property_key]
        idx_m = inputs["_idx_m"]
        n_molecules = inputs[properties.n_atoms].shape[0]
        
        # NOTE: we do not use Bessel Correction like torch does by default

        # flatten property and idx_m
        property_flat = property_tensor.flatten()
        idx_m_flat = idx_m.unsqueeze(1).repeat(1, 3).flatten()

        # calculate mean
        mean = torch.zeros(n_molecules, dtype=property_tensor.dtype, device=property_tensor.device)
        mean = mean.scatter_reduce(dim=0, index=idx_m_flat, src=property_flat, reduce="mean", include_self=False)[idx_m_flat]

        # calculate variance and std
        variance = torch.zeros(n_molecules, dtype=property_tensor.dtype, device=property_tensor.device)
        sq_diff = (property_flat - mean)**2
        variance = variance.scatter_reduce(dim=0, index=idx_m_flat, src=sq_diff, reduce="mean", include_self=False)
        std = torch.sqrt(variance + 1e-8)

        # scale property by std
        inputs[self.property_key] = property_tensor / std[idx_m].unsqueeze(-1)
        inputs[self.variance_key] = variance

        return inputs


class AddVariance(nn.Module):
    def __init__(self, property_key: str, variance_key: str = keys.variance):
        super().__init__()
        self.property_key = property_key
        self.variance_key = variance_key
        self.force_apply = True

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        idx_m = inputs["_idx_m"]
        unscaled_property = inputs[self.property_key]

        # get variance for each atom
        std = torch.sqrt(inputs[self.variance_key])
        std = std[idx_m]

        # scale property by variance
        inputs[self.property_key] = unscaled_property * std.unsqueeze(-1)
        return inputs
