import numpy as np
import pytest

from gpff.analysis import formula

SYMBOLS = ["H", "C", "N", "O", "F"]


def counts(*rows):
    return np.array(rows, dtype=np.int64)


def test_symbols_follow_the_atomic_numbers():
    symbols = formula.symbols_of({"O": 1, "H": 1}, {"C": 1, "Es": 1, "X": 1})

    assert symbols == ["X", "H", "C", "O", "Es"]


def test_a_missing_symbol_is_a_column_of_zeros():
    table = {"C": np.array([2, 3]), "H": np.array([6, 8])}

    assert formula.count_matrix(table, ["H", "C", "N"]).tolist() == [[6, 2, 0], [8, 3, 0]]
    assert formula.count_matrix({}, ["H"], n_rows=2).shape == (2, 1)


def test_hill_formula_puts_carbon_and_hydrogen_first():
    assert formula.hill_formula([11, 7, 1, 1, 0], SYMBOLS) == "C7H11NO"
    assert formula.hill_formula([3, 1, 0, 0, 1], SYMBOLS) == "CH3F"
    # without carbon every element is in alphabetical order, hydrogen included
    assert formula.hill_formula([2, 0, 0, 1, 0], SYMBOLS) == "H2O"


def test_the_same_formula_has_the_same_id_in_every_set():
    a = counts([6, 2, 0, 0, 0], [8, 3, 0, 0, 0])
    b = counts([8, 3, 0, 0, 0], [6, 2, 0, 0, 0], [6, 2, 0, 0, 0])

    (ids_a, ids_b), rows = formula.formula_ids(a, b)

    assert len(rows) == 2
    assert ids_a[0] == ids_b[1] == ids_b[2]
    assert ids_a[1] == ids_b[0]
    assert ids_a[0] != ids_a[1]


def test_distances_of_equal_distributions_are_zero():
    p = np.array([0.5, 0.5, 0.0])

    for value in formula.distances(p, p).values():
        assert value == pytest.approx(0.0)


def test_distances_of_distributions_without_a_common_formula_are_one():
    p, q = np.array([1.0, 0.0]), np.array([0.0, 1.0])

    for value in formula.distances(p, q).values():
        assert value == pytest.approx(1.0)


def test_total_variation_is_the_share_that_has_to_move():
    assert formula.total_variation(np.array([0.75, 0.25]), np.array([0.5, 0.5])) == pytest.approx(0.25)


def test_jensen_shannon_is_finite_where_a_formula_is_missing_on_one_side():
    p, q = np.array([0.5, 0.5, 0.0]), np.array([0.5, 0.0, 0.5])

    assert formula.jensen_shannon(p, q) == pytest.approx(0.5)


def test_formula_distances_see_what_the_pooled_shares_miss():
    # two N in one molecule and none in the other, against one N in each: the same
    # pooled element shares, no formula in common
    generated = counts([3, 1, 2, 0, 0], [5, 1, 0, 0, 0])
    reference = counts([4, 1, 1, 0, 0], [4, 1, 1, 0, 0])

    measured, p, q, rows = formula.formula_distances(generated, reference)

    shares = formula.element_shares(generated), formula.element_shares(reference)
    assert formula.total_variation(*shares) == pytest.approx(0.0)
    assert measured["tv"] == pytest.approx(1.0)
    assert len(rows) == 3 and p.sum() == pytest.approx(1.0) and q.sum() == pytest.approx(1.0)


def test_resampling_keeps_every_size_and_takes_the_nearest_for_a_missing_one():
    pool = counts([4, 1, 0, 0, 0], [6, 2, 0, 0, 0], [8, 3, 0, 0, 0])  # 5, 8 and 11 atoms

    drawn = formula.resample_by_size(pool, np.array([5, 8, 8, 11, 9]), np.random.default_rng(0))

    assert drawn.sum(axis=1).tolist() == [5, 8, 8, 11, 8]


def test_independent_atoms_have_the_sizes_and_the_shares():
    shares = np.array([0.5, 0.5, 0.0, 0.0, 0.0])

    drawn = formula.independent_atoms(shares, np.full(2000, 6), np.random.default_rng(0))

    assert (drawn.sum(axis=1) == 6).all()
    assert drawn[:, 2:].sum() == 0
    assert drawn[:, 0].mean() == pytest.approx(3.0, abs=0.1)


def test_the_floor_is_zero_for_one_formula_and_positive_for_two_of_one_size():
    one = counts(*[[4, 1, 0, 0, 0]] * 50)
    assert formula.oracle_floor(one, one[:20], n_draws=3)["tv"] == pytest.approx((0.0, 0.0))

    # CH4-like and CH3N-like molecules, both of 5 atoms, at random
    pair = counts([4, 1, 0, 0, 0], [3, 1, 1, 0, 0])[np.random.default_rng(1).integers(0, 2, 500)]
    floor = formula.oracle_floor(pair, pair[:200], n_draws=10)

    # two samples of 200 from a fair coin differ by about 0.04 in the share of each
    assert 0.005 < floor["tv"][0] < 0.15
    assert floor["tv"][1] >= 0.0


def test_feasibility_follows_the_valence_parity_and_the_unsaturation():
    rows = counts(
        [4, 1, 0, 0, 0],  # CH4
        [3, 1, 0, 0, 0],  # CH3: the valences add up to an odd number
        [3, 1, 1, 0, 0],  # CH3N, methanimine
        [8, 1, 0, 0, 0],  # CH8: more H than a carbon can hold
        [2, 0, 0, 1, 0],  # H2O
        [1, 0, 0, 0, 0],  # a lone H
    )

    feasible = formula.formula_feasible(rows, SYMBOLS)

    assert feasible.tolist() == [True, False, True, False, True, False]


def test_an_element_outside_qm9_makes_a_formula_infeasible():
    rows = counts([4, 1, 0], [3, 1, 1])

    assert formula.formula_feasible(rows, ["H", "C", "Es"]).tolist() == [True, False]


def test_standardized_validity_weights_the_strata_by_the_reference():
    symbols = ["H", "C", "N", "O"]
    # three molecules without N, all valid, and one with N, invalid: validity 0.75
    generated = counts([4, 1, 0, 0], [4, 1, 0, 0], [4, 1, 0, 0], [3, 1, 1, 0])
    valid = np.array([True, True, True, False])
    # the reference has N in half of its molecules
    reference = counts([4, 1, 0, 0], [3, 1, 1, 0])

    value = formula.standardized_validity(
        valid,
        formula.composition_strata(generated, symbols),
        formula.composition_strata(reference, symbols),
    )

    assert value == pytest.approx(0.5)


def test_a_stratum_the_set_never_generates_counts_as_invalid():
    symbols = ["H", "C", "N", "O"]
    generated = counts([4, 1, 0, 0], [4, 1, 0, 0])
    reference = counts([4, 1, 0, 0], [3, 1, 1, 0])

    value = formula.standardized_validity(
        np.array([True, True]),
        formula.composition_strata(generated, symbols),
        formula.composition_strata(reference, symbols),
    )

    assert value == pytest.approx(0.5)


def test_the_count_distribution_caps_the_count():
    rows = counts([1, 0], [1, 1], [1, 2], [1, 5])

    distribution = formula.count_distribution(rows, ["H", "N"], "N", cap=3)

    assert distribution.tolist() == [0.25, 0.25, 0.25, 0.25]
    assert formula.count_distribution(rows, ["H", "N"], "O", cap=3).tolist() == [1.0, 0.0, 0.0, 0.0]


def test_count_w1_is_the_shift_of_the_counts():
    assert formula.count_w1(np.array([0, 0]), np.array([2, 2])) == pytest.approx(2.0)
    assert formula.count_w1(np.array([1, 3]), np.array([1, 3])) == pytest.approx(0.0)
