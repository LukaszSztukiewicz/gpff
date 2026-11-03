from typing import Dict, Optional

import torch

import schnetpack.transform as trn
from schnetpack import properties

from gpff.pseudo_pes import LossWeights, Geometries, PseudoCalculator
from gpff.utils.alignment import align_rotation, align_permutation
from gpff import keys


__all__ = ["PseudoPESTransform"]


class PseudoPESTransform(trn.Transform):

    def __init__(
        self,
        geometry_sampler: Geometries,
        energy_function: PseudoCalculator,
        weight_generator: Optional[LossWeights] = None,
        align_permutation: bool = False,
        align_rotation: bool = False,
    ):
        """
        Args:
            geometry_sampler: sampler for the geometry
            energy_function: generator for the energy function
            weight_generator: generator for the weights
            x0_key: key to store the x0 positions
            sigmas_key: key to store the sigmas
            energy_weight_key: key to store the energy weight
            force_weight_key: key to store the force weight
            use_best_permutation: whether to use the best permutation to align 
                the noisy geometry to the reference geometry
            align_rotation: whether to align the rotation of 
                the noisy geometry to the reference geometry
        """
        super().__init__()
        self.geometry_sampler = geometry_sampler
        self.energy_function = energy_function
        self.weight_generator = weight_generator
        self.align_permutation = align_permutation
        self.align_rotation = align_rotation

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Define the forward diffusion transformation.

        Args:
            inputs: dictionary of input tensors as in SchNetPack.
        """
        x_0 = inputs[properties.R].clone()
        atom_types = inputs[properties.Z].clone()

        # center x_0
        x_0 = x_0 - x_0.mean(0)

        # sample noisy geometry
        x_t, sigma_t, t, noise = self.geometry_sampler(x_0=x_0, atom_types=atom_types, t=t, noise=noise)
            
        # align best permutation of x_t to x_0
        if self.align_permutation:
            x_t, atom_types = align_permutation(
                positions=x_t,
                target_positions=x_0,
                numbers=atom_types,
                target_numbers=atom_types,
            )

        # align rotation of x_t to x_0
        if self.align_rotation:
            x_t = align_rotation(positions=x_t, target_positions=x_0)

        inputs.update(
            {
                properties.position: x_t,
                properties.Z: atom_types,
                keys.x_0: x_0,
                keys.sigma_curr: sigma_t,
                keys.t: t,
            }
        )
        # alignment moves x_t away from x_0 + sigma_t * noise, so the drawn noise
        # is a target only without it; the force labels below are exact either way
        if not (self.align_permutation or self.align_rotation):
            inputs[keys.noise] = noise

        # generate labels
        energy = self.energy_function.get_energy(x_0=x_0, x_t=x_t)
        forces = self.energy_function.get_forces_analytic(x_0=x_0, x_t=x_t)

        inputs.update(
            {
                keys.energy: energy,
                keys.forces: forces,
            }
        )

        # generate weights
        if self.weight_generator is not None:
            energy_weight, force_weight = self.weight_generator(
                sigma=sigma_t,
                x_0=x_0,
                x_t=x_t,
                energy=energy,
                forces=forces,
            )
            inputs.update(
                {
                    keys.energy_weight: energy_weight,
                    keys.forces_weight: force_weight,
                }
            )

        return inputs


if __name__ == "__main__":
    from schnetpack.datasets import QM9
    from ase.db import connect
    from ase.visualize import view
    from ase import Atoms
    from gpff.utils import dict_to_atoms
    import matplotlib.pyplot as plt
    import numpy as np
    from gpff.utils.alignment import sample_covariance_noise, align_to_principal_components
    from gpff.utils import compute_covariance_matrix

    from gpff.pseudo_pes import (
        VariancePreservingGeometries,
        VarianceExplodingGeometries,
        VEPseudoCalculator,
    )

    # set seed
    seed = 420694
    torch.manual_seed(seed)
    np.random.seed(seed)

    with connect("qm9.db") as db:
        print(len(db))

    # load batch
    qm9 = QM9(
        datapath="datasets/qm9.db",
        batch_size=10,
        remove_uncharacterized=True,
        num_train=100,
        num_val=100,
        num_test=100,
        split_file=None,
    )
    qm9.setup()
    dataset = qm9.train_dataset

    # apply transforms
    for inputs in dataset:
        break
        
    #noise = torch.randn_like(inputs["_positions"])
    noise = sample_covariance_noise(inputs["_positions"])
    atoms_list = []
    forces_list = []
    noise_levels = []
    for t_i in range(1, 1000, 1):
        print("t_i", t_i)
        #geometry_sampler = VariancePreservingGeometries(
        #    beta_min=0.0001,
        #    beta_max=0.01,
        #    timesteps=1000,
        #)
        t_i = t_i / 1000
        geometry_sampler = VarianceExplodingGeometries(
            sigma_max=30,
            sigma_mean=-1.2,
            sigma_std=1.2,
        )

        energy_function = VEPseudoCalculator()

        transforms = PseudoPESTransform(
            energy_function=energy_function,
            geometry_sampler=geometry_sampler,
            weight_generator=None,
            align_permutation=True,
            align_rotation=False,
        )
        noisy_inputs = transforms({k: v.clone() for k, v in inputs.items()}, t_i, noise=noise)
        noise_levels.append(noisy_inputs[keys.sigma_curr].mean())
        aligned_positions = align_to_principal_components(noisy_inputs["_positions"])
        covariance = compute_covariance_matrix(aligned_positions)
        print(covariance)
        atoms_list.append(dict_to_atoms(noisy_inputs))

    sort_ids = np.argsort(noise_levels)
    atoms_list = [atoms_list[i] for i in sort_ids]
    print([noise_levels[i] for i in sort_ids])
    view(atoms_list)
