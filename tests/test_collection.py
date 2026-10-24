import numpy as np
import pytest

from gpff.analysis import calculate_js_divergence


def test_js_divergence_penalizes_far_outliers():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=100_000)
    samples = rng.normal(0.1, 1.05, size=10_000)
    with_outlier = np.append(samples[:-1], 100.0)

    # an outlier is mass where the reference has none: it must not make the
    # samples look closer to the reference by stretching the bins
    assert calculate_js_divergence(reference, with_outlier) > calculate_js_divergence(reference, samples)


def test_js_divergence_rejects_non_finite_values():
    with pytest.raises(ValueError):
        calculate_js_divergence(np.random.rand(100), np.array([0.5, np.nan]))
