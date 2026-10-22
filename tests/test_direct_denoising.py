import pytest
import torch
from schnetpack import properties

from gpff import keys
from gpff.sample.direct_denoising import DirectDenoisingSampler, remove_rigid_motion


def _two_molecules():
    torch.manual_seed(0)
    idx_m = torch.tensor([0] * 5 + [1] * 7)
    positions = torch.randn(12, 3, dtype=torch.float64)
    return positions, idx_m


def _per_molecule_sums(vectors, positions, idx_m):
    net = torch.zeros(2, 3, dtype=vectors.dtype).index_add(0, idx_m, vectors)
    centroids = torch.zeros(2, 3, dtype=vectors.dtype).index_add(0, idx_m, positions) / torch.bincount(idx_m)[:, None]
    torque = torch.zeros(2, 3, dtype=vectors.dtype).index_add(
        0, idx_m, torch.cross(positions - centroids[idx_m], vectors, dim=1))
    return net, torque


def test_remove_rigid_motion_removes_a_pure_rigid_motion():
    positions, idx_m = _two_molecules()
    translation = torch.tensor([[0.3, -0.2, 0.1], [-0.5, 0.4, 0.2]], dtype=torch.float64)
    omega = torch.tensor([[0.1, 0.7, -0.2], [-0.4, 0.0, 0.3]], dtype=torch.float64)
    rigid = translation[idx_m] + torch.cross(omega[idx_m], positions, dim=1)

    torch.testing.assert_close(remove_rigid_motion(rigid, positions, idx_m), torch.zeros_like(rigid))


def test_remove_rigid_motion_leaves_no_net_force_or_torque():
    positions, idx_m = _two_molecules()
    forces = torch.randn_like(positions)
    net, torque = _per_molecule_sums(remove_rigid_motion(forces, positions, idx_m), positions, idx_m)

    torch.testing.assert_close(net, torch.zeros_like(net))
    torch.testing.assert_close(torque, torch.zeros_like(torque))


def test_transport_grid_runs_from_sigma_max_to_transport_sigma():
    sampler = DirectDenoisingSampler(transport_sigma=0.3, transport_steps=10, transport_sigma_max=30.0)
    levels = sampler.transport_levels

    assert len(levels) == 11
    torch.testing.assert_close(levels[[0, -1]], torch.tensor([30.0, 0.3], dtype=levels.dtype))
    assert (levels[1:] < levels[:-1]).all()


def test_transport_ratios_step_to_the_first_level_below_sigma_and_at_least_one_level():
    sampler = DirectDenoisingSampler(transport_sigma=0.3, transport_steps=10, transport_sigma_max=30.0)
    levels = sampler.transport_levels
    idx_m = torch.tensor([0, 0, 1, 1, 1])
    batch = {
        properties.idx_m: idx_m,
        properties.n_atoms: torch.tensor([2, 3]),
        keys.transport_level: torch.tensor([0, 4]),
    }
    # molecule 0 reads as far below level 1, molecule 1 as above its current level
    sigma_mol = torch.tensor([float(levels[5]) * 1.01, float(levels[3])], dtype=torch.float64)
    sigma = sigma_mol[idx_m].unsqueeze(-1)
    in_transport = torch.ones_like(sigma, dtype=torch.bool)

    ratio = sampler.transport_ratios(batch, sigma, in_transport)

    assert batch[keys.transport_level].tolist() == [5, 5]
    torch.testing.assert_close(ratio[[0, 2], 0], levels[[5, 5]] / sigma_mol)


def test_transport_steps_needs_transport_sigma():
    with pytest.raises(ValueError):
        DirectDenoisingSampler(transport_steps=10)
