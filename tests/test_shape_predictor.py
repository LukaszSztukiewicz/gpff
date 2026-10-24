import numpy as np
import torch
from schnetpack import properties

from gpff.sample.shape_predictor import ShapeGMM
from gpff.utils import batch_compute_principal_variances


def training_set():
    g = torch.Generator().manual_seed(0)
    scale = torch.tensor([3.0, 1.0, 0.3], dtype=torch.float64)
    return [
        {properties.R: torch.randn(n, 3, generator=g, dtype=torch.float64) * scale}
        for n in [5, 6, 7] * 20
    ]


def shape_gmm():
    np.random.seed(0)
    return ShapeGMM(training_set(), n_components=2)


def test_fit_uses_the_per_structure_variances(monkeypatch):
    fitted = []
    monkeypatch.setattr("sklearn.mixture.GaussianMixture.fit", lambda self, x: fitted.append(x) or self)
    shape_gmm()

    for n_atoms, log_variances in zip([5, 6, 7], fitted):
        expected = [
            batch_compute_principal_variances(s[properties.R], torch.zeros(n_atoms, dtype=torch.long))[0]
            for s in training_set()
            if len(s[properties.R]) == n_atoms
        ]
        np.testing.assert_allclose(log_variances, torch.stack(expected).log().numpy(), rtol=1e-12)


def test_sample_from_batch_draws_once_per_molecule_size():
    gmm = shape_gmm()
    calls = []
    sample = gmm.sample
    gmm.sample = lambda n_atoms, num_samples, target_ratio=None: (
        calls.append((n_atoms, num_samples)) or sample(n_atoms, num_samples, target_ratio)
    )
    idx_m = torch.repeat_interleave(torch.arange(6), torch.tensor([5, 6, 5, 7, 5, 6]))

    variances = gmm.sample_from_batch({properties.R: torch.zeros(len(idx_m), 3), properties.idx_m: idx_m})

    assert sorted(calls) == [(5, 3), (6, 2), (7, 1)]
    assert variances.shape == (6, 3)
    assert (variances[:, :-1] >= variances[:, 1:]).all()


def test_sample_rows_are_not_grouped_by_component():
    gmm = shape_gmm()
    # two far apart components; GaussianMixture.sample returns its draws grouped by component
    mixture = gmm.models[5]
    mixture.weights_ = np.array([0.5, 0.5])
    mixture.means_ = np.array([[2.0, 0.0, -2.0], [-2.0, -4.0, -6.0]])
    mixture.covariances_ = np.stack([np.eye(3) * 1e-2] * 2)
    np.random.seed(0)

    first_component = (gmm.sample(5, 200)[:, 0] > 1.0).float()

    assert 0.3 < first_component[:100].mean() < 0.7
    assert 0.3 < first_component[100:].mean() < 0.7
