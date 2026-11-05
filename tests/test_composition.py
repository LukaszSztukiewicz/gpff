import numpy as np
import pytest

from gpff.analysis.composition import composition_metrics


def mols(*molecules):
    numbers = [np.array(m) for m in molecules]
    sizes = np.array([len(m) for m in numbers], dtype=int)
    return numbers, sizes


def test_element_and_molecule_fractions():
    # 2 molecules, 6 atoms: H H C / H C N
    numbers, sizes = mols([1, 1, 6], [1, 6, 7])

    metrics = composition_metrics(numbers, sizes)

    assert metrics["n_molecules"] == 2
    assert metrics["element_fraction_H"] == pytest.approx(3 / 6)
    assert metrics["element_fraction_C"] == pytest.approx(2 / 6)
    assert metrics["element_fraction_N"] == pytest.approx(1 / 6)
    assert metrics["element_fraction_O"] == 0.0
    # N is in one of the two molecules, H and C in both
    assert metrics["molecule_fraction_N"] == pytest.approx(0.5)
    assert metrics["molecule_fraction_H"] == pytest.approx(1.0)
    assert metrics["heteroatom_fraction"] == pytest.approx(1 / 6)
    assert metrics["mean_n_atoms"] == pytest.approx(3.0)


def test_identical_sets_have_zero_distance_to_the_reference():
    numbers, sizes = mols([1, 6, 7], [1, 1, 8, 8])

    metrics = composition_metrics(numbers, sizes, numbers, sizes)

    assert metrics["element_tv"] == pytest.approx(0.0)
    assert metrics["size_w1"] == pytest.approx(0.0)


def test_element_tv_measures_the_swapped_share():
    reference, reference_sizes = mols([6, 6, 7, 7])  # 50% C, 50% N
    numbers, sizes = mols([6, 6, 6, 7])  # 75% C, 25% N

    metrics = composition_metrics(numbers, sizes, reference, reference_sizes)

    # total variation = 0.5 * (|0.75-0.5| + |0.25-0.5|)
    assert metrics["element_tv"] == pytest.approx(0.25)


def test_element_tv_counts_an_element_outside_the_reported_set():
    reference, reference_sizes = mols([6, 6])
    numbers, sizes = mols([6, 16])  # half sulfur, not in QM9_ELEMENTS

    metrics = composition_metrics(numbers, sizes, reference, reference_sizes)

    assert metrics["element_tv"] == pytest.approx(0.5)


def test_size_w1_is_the_mean_shift_for_a_constant_offset():
    reference, reference_sizes = mols([1] * 3, [1] * 3)
    numbers, sizes = mols([1] * 5, [1] * 5)

    metrics = composition_metrics(numbers, sizes, reference, reference_sizes)

    assert metrics["size_w1"] == pytest.approx(2.0)


def test_empty_input_reports_no_molecules():
    metrics = composition_metrics([], np.array([], dtype=int))

    assert metrics == {"n_molecules": 0.0}
