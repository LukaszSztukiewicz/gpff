import numpy as np
import pytest

from gpff.analysis import collect_element_counts
from gpff.analysis import formula
from gpff.analysis.composition import composition_metrics


def counts(*molecules, symbols=None):
    """The count matrix of molecules given by their atomic numbers, and its symbols."""
    table = collect_element_counts(molecules)
    symbols = symbols or formula.symbols_of(table)
    return formula.count_matrix(table, symbols), symbols


def test_element_and_molecule_fractions():
    # 2 molecules, 6 atoms: H H C / H C N
    matrix, symbols = counts([1, 1, 6], [1, 6, 7], symbols=["H", "C", "N", "O"])

    metrics = composition_metrics(matrix, symbols)

    assert metrics["element_fraction_H"] == pytest.approx(3 / 6)
    assert metrics["element_fraction_C"] == pytest.approx(2 / 6)
    assert metrics["element_fraction_N"] == pytest.approx(1 / 6)
    assert metrics["element_fraction_O"] == 0.0
    # N is in one of the two molecules, H and C in both
    assert metrics["molecule_fraction_N"] == pytest.approx(0.5)
    assert metrics["molecule_fraction_H"] == pytest.approx(1.0)
    assert metrics["heteroatom_fraction"] == pytest.approx(1 / 6)
    assert metrics["mean_n_atoms"] == pytest.approx(3.0)
    assert "element_tv" not in metrics


def test_identical_sets_have_zero_distance_to_the_reference():
    matrix, symbols = counts([1, 6, 7], [1, 1, 8, 8])

    metrics = composition_metrics(matrix, symbols, matrix)

    assert metrics["element_tv"] == pytest.approx(0.0)
    assert metrics["size_w1"] == pytest.approx(0.0)


def test_element_tv_measures_the_swapped_share():
    symbols = ["C", "N"]
    reference, _ = counts([6, 6, 7, 7], symbols=symbols)  # 50% C, 50% N
    matrix, _ = counts([6, 6, 6, 7], symbols=symbols)  # 75% C, 25% N

    metrics = composition_metrics(matrix, symbols, reference)

    # total variation = 0.5 * (|0.75-0.5| + |0.25-0.5|)
    assert metrics["element_tv"] == pytest.approx(0.25)


def test_element_tv_counts_an_element_only_one_side_has():
    symbols = ["C", "S"]
    reference, _ = counts([6, 6], symbols=symbols)
    matrix, _ = counts([6, 16], symbols=symbols)  # half sulfur

    metrics = composition_metrics(matrix, symbols, reference)

    assert metrics["element_tv"] == pytest.approx(0.5)
    assert metrics["heteroatom_fraction"] == pytest.approx(0.5)


def test_size_w1_is_the_mean_shift_for_a_constant_offset():
    reference, symbols = counts([1] * 3, [1] * 3)
    matrix, _ = counts([1] * 5, [1] * 5)

    metrics = composition_metrics(matrix, symbols, reference)

    assert metrics["size_w1"] == pytest.approx(2.0)


def test_empty_input_reports_nothing():
    assert composition_metrics(np.zeros((0, 2), dtype=int), ["H", "C"]) == {}
