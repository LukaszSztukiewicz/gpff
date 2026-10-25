import math

import pytest
import torch
from schnetpack import properties

from gpff import keys
from gpff.analysis.landscape import (
    center_per_molecule,
    denoise,
    denoiser_jacobians,
    denoising_errors,
    jacobian_statistics,
    mean_free_noise,
)


def make_batch(sizes, seed=0):
    generator = torch.Generator().manual_seed(seed)
    n = sum(sizes)
    return {
        properties.R: torch.randn(n, 3, generator=generator, dtype=torch.float64),
        properties.Z: torch.randint(0, 2, (n,), generator=generator) * 5 + 1,  # H and C
        properties.idx_m: torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes)),
        properties.n_atoms: torch.tensor(sizes),
    }


def scaling_model(factor):
    """D(x) = c + factor * (x - c) with c each molecule's centroid."""
    def model(inputs):
        centered = center_per_molecule(inputs[properties.R], inputs[properties.idx_m])
        return {keys.forces: 2 * (factor - 1) * centered}
    return model


def cubic_model(inputs):
    """A nonlinear model that couples the atoms of a molecule (via the centroid)."""
    centered = center_per_molecule(inputs[properties.R], inputs[properties.idx_m])
    return {keys.forces: 0.2 * centered**3 + 0.1 * centered.roll(1, dims=-1)}


def test_denoising_errors_known_values():
    batch = make_batch([3, 5, 8])
    x_0 = center_per_molecule(batch[properties.R], batch[properties.idx_m])
    noise = mean_free_noise(x_0, batch[properties.idx_m], torch.Generator().manual_seed(1))
    sigmas = [0.1, 1.0, 10.0]

    # the identity removes nothing: e = |noise|^2 / (3N - 3) at every sigma
    identity = denoising_errors(scaling_model(1.0), batch, x_0, sigmas, noise)
    sizes = batch[properties.n_atoms].tolist()
    expected = torch.stack([(n**2).sum() / (3 * len(n) - 3) for n in noise.split(sizes)])
    torch.testing.assert_close(identity, expected[:, None].expand(-1, len(sigmas)))

    # an oracle that knows x_0 removes everything
    oracle = lambda inputs: {keys.forces: 2 * (x_0 - inputs[properties.R])}
    torch.testing.assert_close(
        denoising_errors(oracle, batch, x_0, sigmas, noise), torch.zeros(3, len(sigmas), dtype=x_0.dtype)
    )

    # aligning the pairing can only bring the noisy atoms closer to x_0
    aligned = denoising_errors(scaling_model(1.0), batch, x_0, sigmas, noise, aligned=True)
    assert (aligned <= identity + 1e-12).all()
    assert (aligned[:, -1] < identity[:, -1]).all()


def test_mean_free_noise_is_centered():
    batch = make_batch([4, 6])
    noise = mean_free_noise(batch[properties.R], batch[properties.idx_m])
    for part in noise.split(batch[properties.n_atoms].tolist()):
        torch.testing.assert_close(part.mean(0), torch.zeros(3, dtype=noise.dtype))


def test_batched_jacobians_match_per_molecule_autograd():
    batch = make_batch([3, 6, 4], seed=2)
    jacobians = denoiser_jacobians(cubic_model, batch, batch[properties.R])
    sizes = batch[properties.n_atoms].tolist()
    for m, (positions, jacobian) in enumerate(zip(batch[properties.R].split(sizes), jacobians)):
        single = {
            properties.idx_m: torch.zeros(sizes[m], dtype=torch.long),
            properties.n_atoms: torch.tensor([sizes[m]]),
        }
        f = lambda x: denoise(cubic_model, single, x.reshape(-1, 3)).reshape(-1)
        expected = torch.autograd.functional.jacobian(f, positions.reshape(-1))
        torch.testing.assert_close(jacobian, expected)


@pytest.mark.parametrize("factor", [1.0, 0.5, 0.0])
def test_jacobian_statistics_of_scaling(factor):
    # J = factor * I + (1 - factor) * (translations): translations are kept,
    # rotations and internal modes keep a fraction `factor` of the noise
    batch = make_batch([7], seed=3)
    (jacobian,) = denoiser_jacobians(scaling_model(factor), batch, batch[properties.R])
    stats = jacobian_statistics(jacobian, batch[properties.R])
    assert stats["asymmetry"] == pytest.approx(0, abs=1e-12)
    assert stats["rigid_residual"] == pytest.approx((1 - factor) / math.sqrt(2))
    assert stats["internal_mean"] == pytest.approx(factor)
    assert stats["internal_max"] == pytest.approx(factor)


def test_jacobian_statistics_of_linear_molecule():
    # 2 rotations instead of 3: 3N - 5 internal directions
    positions = torch.tensor([[0.0, 0, 0], [1.1, 0, 0], [2.3, 0, 0]], dtype=torch.float64)
    stats = jacobian_statistics(torch.eye(9, dtype=torch.float64), positions)
    assert stats["rigid_residual"] == pytest.approx(0, abs=1e-12)
    assert stats["internal_mean"] == pytest.approx(1)


def test_jacobian_statistics_asymmetry():
    jacobian = torch.eye(9, dtype=torch.float64)
    jacobian[0, 1] = 0.5
    positions = torch.randn(3, 3, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    assert jacobian_statistics(jacobian, positions)["asymmetry"] > 0.1
