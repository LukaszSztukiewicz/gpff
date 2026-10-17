import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Callable, Dict, Union
from schnetpack.nn import build_gated_equivariant_mlp
import schnetpack.nn as snn
import schnetpack.properties as properties
from schnetpack.atomistic import Atomwise as SPKAtomwise

from gpff import keys


__all__ = ["GatedEquivariant", "Atomwise", "TimeAwareEquivariant", "DiffusionTime"]


class GatedEquivariant(nn.Module):
    """
    Equivariant head for pseudo force predictions without energy.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        include_time: bool = False,
        n_hidden: Optional[int | Sequence[int]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        output_key: str = keys.forces,
        time_head: Optional[nn.Module] = None,
        time_key: Optional[str] = None,
        detach_time_head: bool = False,
    ):
        """
        Args:
            n_in: input dimension without time.
            n_out: output dimension for the target property.
            include_time: whether to append time as input feature.
            n_hidden: size of hidden layers.
                    If an integer, same number of node is used for all hidden
                        layers resulting in a rectangular network.
                    If None, the number of neurons is divided
                        by two after each layer starting
                        n_in resulting in a pyramidal network.
            n_layers: number of hidden layers.
            activation: activation function.
            output_key: the key under which the result will be stored.
            time_head: time head to predict time.
            time_key: time key for input to the equivariant module.
            detach_time_head: detach time head from computational graph.
        """

        super().__init__()

        self.outnet = build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )

        self.output_key = output_key
        self.model_outputs = [self.output_key]

        # todo: add time as input scalar feature
        # self.include_time = include_time
        # if self.include_time:
        #    self.outnet[0] = spk.nn.GatedEquivariantBlock(
        #        n_sin=self.outnet[0].n_sin + 1,
        #        n_vin=self.outnet[0].n_vin,
        #        n_sout=self.outnet[0].n_sout,
        #        n_vout=self.outnet[0].n_vout,
        #        n_hidden=self.outnet[0].n_hidden,
        #        activation=activation,
        #        sactivation=activation,
        #    )
        # time prediction
        # self.time_outnet = time_head
        # self.detach_time_head = detach_time_head
        # self.time_key = time_key
        # set default time key
        # if self.include_time and self.time_key is None:
        #    raise ValueError(
        #        "Argument 'time_key' must be set when 'include_time' is True."
        #    )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        l0 = inputs["scalar_representation"]
        l1 = inputs["vector_representation"]

        # todo: implement time later
        # append time to representation
        # if self.include_time:
        #    # predict time
        #    if self.time_outnet is not None:
        #        t = self.time_outnet(inputs)[self.time_key]
        #        if self.time_outnet.aggregation_mode is not None:
        #            t = t[inputs[properties.idx_m]]
        #        if self.detach_time_head:
        #            t = t.detach()
        #    # use true time from input
        #    else:
        #        t = inputs[self.time_key]  # type: ignore
        #    t = t.unsqueeze(-1)
        #
        #    # append time to scalar representation features
        #    l0 = torch.cat((l0, t), dim=-1)

        # predict equivariant output
        _, x = self.outnet((l0, l1))
        x = torch.squeeze(x, -1)
        inputs[self.output_key] = x

        return inputs


class TimeConditionedGatedEquivariant(nn.Module):
    """
    Time-conditioned Equivariant head for pseudo force predictions without energy.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_hidden: Optional[int | Sequence[int]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        output_key: str = keys.forces,
        time_key: Optional[str] = "timestep",
    ):
        """
        Args:
            n_in: input dimension without time.
            n_out: output dimension for the target property.
            n_hidden: size of hidden layers.
                    If an integer, same number of node is used for all hidden
                        layers resulting in a rectangular network.
                    If None, the number of neurons is divided
                        by two after each layer starting
                        n_in resulting in a pyramidal network.
            n_layers: number of hidden layers.
            activation: activation function.
            output_key: the key under which the result will be stored.
            time_key: time key for input to the equivariant module.
        """

        super().__init__()

        self.outnet = build_gated_equivariant_mlp(
            n_in=n_in+1,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )

        self.time_key = time_key
        self.output_key = output_key
        self.model_outputs = [self.output_key]

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        l0 = inputs["scalar_representation"]
        l1 = inputs["vector_representation"]

        # append time to scalar and vector representations
        t = inputs[self.time_key]
        l0 = torch.cat((l0, t), dim=-1)
        t = t.unsqueeze(-1).expand(t.shape[0], 3, t.shape[-1])
        l1 = torch.cat((l1, t), dim=-1)

        # predict equivariant output
        _, x = self.outnet((l0, l1))
        x = torch.squeeze(x, -1)
        inputs[self.output_key] = x

        return inputs


class Atomwise(SPKAtomwise):
    """
    Time-aware Atomwise head for energy prediction.
    Overwrites the ``spk.atomistic.Atomwise`` module.
    Can be used to predict diffusion noise as the forces, i.e. derivative of energy.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        time_conditioning: bool = False,
        time_head: Optional[nn.Module] = None,
        detach_time_head: bool = False,
        apply_scaling: bool = False,
        noise_schedule: Optional[Callable] = None,
        **kwargs
    ):
        """
        Args:
            n_in: input dimension without time.
            n_out: output dimension for the target property.
            time_conditioning: whether to append time as input feature.
            time_head: time head to predict time.
            detach_time_head: detachs time head from computational graph.
            apply_scaling: whether to apply scaling to the output.
            noise_schedule: noise schedule to apply to the output.
        """

        # append time as input feature
        self.time_conditioning = time_conditioning
        if self.time_conditioning:
            n_in += 1

        super().__init__(n_in, n_out=n_out, **kwargs)

        # time_prediction
        self.time_outnet = time_head
        self.detach_time_head = detach_time_head
        self.time_key = (
            self.time_outnet.output_key if self.time_outnet is not None else None
        )

        # scaling
        self.apply_scaling = apply_scaling
        self.noise_schedule = noise_schedule

        #  checks
        if self.time_conditioning and self.time_outnet is None:
            raise ValueError(
                "Argument 'time_outnet' must be set when 'time_conditioning' is True."
            )
        if self.apply_scaling and self.noise_schedule is None:
            raise ValueError(
                "Argument 'noise_schedule' must be set when 'apply_scaling' is True."
            )
        if self.apply_scaling and self.time_outnet is None:
            raise ValueError(
                "Argument 'time_outnet' must be set when 'apply_scaling' is True."
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # prediction timestep
        if self.time_outnet is not None:
            t = self.time_outnet(inputs)[self.time_key]
            # broadcast molecule-level time to all atoms
            if self.time_outnet.aggregation_mode is not None:
                t = t[inputs[properties.idx_m]]
            # detach to stop gradient flow
            if self.detach_time_head:
                t = t.detach()

            t = t.unsqueeze(-1)

            # append timestep to scalar representation
            if self.time_conditioning:
                inputs["scalar_representation"] = torch.cat(
                    (inputs["scalar_representation"], t), dim=-1
                )

        # predict atomwise contributions
        y = self.outnet(inputs["scalar_representation"])

        # scaling
        if self.apply_scaling:
            if self.noise_schedule is not None:
                sigma = self.noise_schedule(t)
                y = y * sigma**2
            # todo: add else

        # accumulate the per-atom output if necessary
        if self.per_atom_output_key is not None:
            inputs[self.per_atom_output_key] = y

        # aggregate
        if self.aggregation_mode is not None:
            idx_m = inputs[properties.idx_m]
            maxm = int(idx_m[-1]) + 1
            y = snn.scatter_add(y, idx_m, dim_size=maxm)
            y = torch.squeeze(y, -1)

            if self.aggregation_mode == "avg":
                y = y / inputs[properties.n_atoms]

        inputs[self.output_key] = y

        return inputs


class TimeAwareEquivariant(nn.Module):
    """
    Time-aware Equivariant head for diffusion noise.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        include_time: bool = False,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        output_key: str = keys.eps_pred,
        time_head: Optional[nn.Module] = None,
        time_key: Optional[str] = None,
        detach_time_head: bool = False,
    ):
        """
        Args:
            n_in: input dimension without time.
            n_out: output dimension for the target property.
            include_time: whether to append time as input feature.
            n_hidden: size of hidden layers.
                    If an integer, same number of node is used for all hidden
                        layers resulting in a rectangular network.
                    If None, the number of neurons is divided
                        by two after each layer starting
                        n_in resulting in a pyramidal network.
            n_layers: number of hidden layers.
            activation: activation function.
            output_key: the key under which the result will be stored.
            time_head: time head to predict time.
            time_key: time key for input to the equivariant module.
            detach_time_head: detach time head from computational graph.
        """

        super().__init__()

        self.outnet = snn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )

        self.output_key = output_key
        self.model_outputs = [output_key]

        self.include_time = include_time
        # add time as input scalar feature
        if self.include_time:
            self.outnet[0] = snn.GatedEquivariantBlock(
                n_sin=self.outnet[0].n_sin + 1,
                n_vin=self.outnet[0].n_vin,
                n_sout=self.outnet[0].n_sout,
                n_vout=self.outnet[0].n_vout,
                n_hidden=self.outnet[0].n_hidden,
                activation=activation,
                sactivation=activation,
            )

        # time prediction
        self.time_outnet = time_head
        self.detach_time_head = detach_time_head

        self.time_key = time_key
        # set default time key
        if self.include_time and self.time_key is None:
            raise ValueError(
                "Argument 'time_key' must be set when 'include_time' is True."
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        l0 = inputs["scalar_representation"]
        l1 = inputs["vector_representation"]

        # append time to representation
        if self.include_time:
            # predict time
            if self.time_outnet is not None:
                t = self.time_outnet(inputs)[self.time_key]
                if self.time_outnet.aggregation_mode is not None:
                    t = t[inputs[properties.idx_m]]
                if self.detach_time_head:
                    t = t.detach()
            # use true time from input
            else:
                t = inputs[self.time_key]  # type: ignore

            # append time to scalar representation features
            l0 = torch.cat((l0, t), dim=-1)

        # predict equivariant output
        _, x = self.outnet((l0, l1))
        x = torch.squeeze(x, -1)
        inputs[self.output_key] = x

        return inputs


class DiffusionTime(nn.Module):
    """
    Invariant time prediction head using the spk.atomistic.Atomwise module.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        output_key: str = "t_pred",
        aggregation_mode: Optional[str] = None,
        detach_representation: bool = False,
    ):
        """
        Args:
            n_in: input dimension without time.
            n_out: time output dimension.
            n_hidden: size of hidden layers.
            n_layers: number of hidden layers.
            activation: activation function.
            output_key: the key under which the result will be stored.
            aggregation_mode: aggregation mode for the atomwise module.
            detach_representation: detachs representation module
                                from computational graph.
        """
        super().__init__()

        self.n_out = n_out

        self.aggregation_mode = aggregation_mode
        self.output_key = output_key
        self.model_outputs = [output_key]

        # time predictor MLP
        self.time_outnet = SPKAtomwise(
            n_in=n_in,
            n_out=self.n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            aggregation_mode=aggregation_mode,
            output_key=self.output_key,
            per_atom_output_key=self.output_key,
        )

        self.detach_representation = detach_representation

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # stop gradient flow through backbone representation.
        if self.detach_representation:
            det_inputs = {key: val.detach() for key, val in inputs.items()}
            det_inputs = self.time_outnet(det_inputs)
            inputs[self.output_key] = det_inputs[self.output_key]

        else:
            inputs = self.time_outnet(inputs)

        if self.aggregation_mode is None:
            inputs[self.output_key] = inputs[self.output_key]

        return inputs