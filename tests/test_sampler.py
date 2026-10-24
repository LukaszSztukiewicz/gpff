import pytest
import torch
from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.model import NeuralNetworkPotential
from schnetpack.atomistic import PairwiseDistances
from schnetpack.nn.cutoff import CosineCutoff
from schnetpack.nn.radial import GaussianRBF

from gpff import keys
from gpff.model.heads import GatedEquivariant
from gpff.model.painn import PaiNN
from gpff.sample.ancestral_sampling import AdaptiveAncestralSampler
from gpff.sample.base import Calculator
from gpff.utils.neighbors import compute_neighbors


# Lightning runs validation, and the SamplerCallback with it, in inference mode
@pytest.mark.parametrize("inference_mode", [False, True])
def test_std_per_molecule_reuses_the_layout_of_the_same_idx_m(inference_mode):
    torch.manual_seed(0)
    calculator = Calculator(device="cpu")
    forces = torch.randn(7, 3)

    with torch.inference_mode(inference_mode):
        for idx_m in (torch.tensor([0, 0, 1, 1, 1, 2, 2]), torch.tensor([0, 1, 1, 1, 1, 2, 2])):
            for _ in range(2):
                sigma = calculator._compute_std_per_molecule(idx_m, forces)
                expected = torch.stack([forces[idx_m == m].std() for m in range(3)])[idx_m]
                torch.testing.assert_close(sigma, expected.unsqueeze(-1))
            assert calculator._layout_cache[0] is idx_m


def batch(sizes, seed=0):
    g = torch.Generator().manual_seed(seed)
    molecules = [
        {
            properties.idx: torch.tensor([k]),
            properties.n_atoms: torch.tensor([n]),
            properties.Z: torch.randint(1, 10, (n,), generator=g),
            properties.R: torch.randn(n, 3, generator=g, dtype=torch.float64) * 2,
            properties.cell: torch.zeros(1, 3, 3, dtype=torch.float64),
            properties.pbc: torch.zeros(3, dtype=torch.bool),
        }
        for k, n in enumerate(sizes)
    ]
    return _atoms_collate_fn(molecules)


def test_converged_molecules_skip_the_model():
    torch.manual_seed(0)
    model = NeuralNetworkPotential(
        representation=PaiNN(
            n_atom_basis=16,
            n_interactions=2,
            radial_basis=GaussianRBF(n_rbf=10, cutoff=5.0),
            cutoff_fn=CosineCutoff(5.0),
            return_vector_representation=True,
        ),
        input_modules=[PairwiseDistances()],
        output_modules=[GatedEquivariant(n_in=16)],
    )
    calculator = Calculator(model=model, device="cpu", dtype=torch.float64)
    inputs = batch([5, 1, 7, 3, 9])
    inputs.update(compute_neighbors(inputs, fully_connected=True, device="cpu"))
    inputs[keys.converged] = torch.zeros(5, dtype=torch.bool)
    full = calculator._predict(dict(inputs))

    inputs[keys.converged] = torch.tensor([False, True, True, False, False])
    inputs[keys.forces] = full
    prediction = calculator._predict(inputs)

    active = ~inputs[keys.converged][inputs[properties.idx_m]]
    torch.testing.assert_close(prediction[active], full[active])
    assert (prediction[~active] == 0).all()
    # the batch keeps the last forces predicted for the converged molecules
    assert torch.equal(inputs[keys.forces], full)


class Relax(torch.nn.Module):
    """Forces pulling each atom to its target, with a stiffness per molecule."""

    def __init__(self):
        super().__init__()
        self.atoms_seen = 0

    def forward(self, inputs):
        self.atoms_seen += len(inputs[properties.R])
        inputs[keys.forces] = 2 * inputs["stiffness"] * (inputs["target"] - inputs[properties.R])
        return inputs


@pytest.mark.parametrize("skip_converged", [False, True])
def test_skipping_converged_molecules_keeps_the_samples(skip_converged):
    sizes = [5, 9, 3, 12, 7, 4, 8, 6]
    inputs = batch(sizes)
    g = torch.Generator().manual_seed(1)
    stiffness = torch.rand(len(sizes), generator=g, dtype=torch.float64) * 0.9 + 0.05
    inputs["stiffness"] = stiffness[inputs[properties.idx_m]].unsqueeze(-1)
    inputs["target"] = inputs[properties.R].clone()
    inputs[properties.R] = inputs[properties.R] + 20 * torch.randn(
        inputs[properties.R].shape, generator=g, dtype=torch.float64
    )

    results = {}
    for skip in (False, True):
        model = Relax()
        sampler = AdaptiveAncestralSampler(
            calculator=Calculator(model=model, device="cpu", dtype=torch.float64, skip_converged=skip),
            n_steps=40,
        )
        torch.manual_seed(0)
        results[skip] = sampler.denoise({k: v.clone() for k, v in inputs.items()}), model.atoms_seen

    (reference, seen_all), (out, seen) = results[False], results[skip_converged]
    assert torch.equal(out[properties.R], reference[properties.R])
    assert torch.equal(out[keys.nfe], reference[keys.nfe])
    assert (seen < seen_all) == skip_converged
