import json
import sys

import numpy as np
import pytest

from gpff.analysis import FAILURES, collect_element_counts
from gpff.cli import calculate_metrics
from gpff.cli.calculate_metrics import (
    compute_metrics,
    load_test_compositions,
    missing_reference_inputs,
)


def per_sample_arrays():
    """Four samples: two valid ethanols, one fragmented, one without bond orders."""
    return {
        "rdkit_valid": np.array([True, False, False, True]),
        "failure": np.array(["valid", "fragments", "bond_orders", "valid"]),
        "smiles": ["CCO", "", "", "CCO"],
        "elongation": np.array([3.0, np.nan, np.nan, 3.5]),
        "element_counts": collect_element_counts([[6, 6, 8], [6, 6, 8], [6, 7], [6, 6, 8]]),
    }


def test_failure_shares_and_validity_sum_to_one():
    results = compute_metrics(per_sample_arrays(), ["rdkit_validity", "failures"], {}, None)

    shares = [results[f"failure_{cause}"] for cause in FAILURES[1:]]
    assert results["failure_fragments"] == pytest.approx(0.25)
    assert results["failure_bond_orders"] == pytest.approx(0.25)
    assert results["rdkit_validity"] + sum(shares) == pytest.approx(1.0)


def test_vun_counts_each_novel_molecule_once_over_all_samples():
    results = compute_metrics(per_sample_arrays(), ["vun"], {}, {"C"})

    # two valid samples share one SMILES: one of four is valid, unique and novel
    assert results["vun"] == pytest.approx(0.25)
    assert compute_metrics(per_sample_arrays(), ["vun"], {}, {"CCO"})["vun"] == 0.0


def test_elongation_w1_of_the_valid_molecules():
    reference = {"elongation": np.array([3.0, 3.5])}

    results = compute_metrics(per_sample_arrays(), ["elongation_w1"], reference, None)
    assert results["elongation_w1"] == pytest.approx(0.0)

    shifted = {"elongation": np.array([3.5, 4.0])}
    results = compute_metrics(per_sample_arrays(), ["elongation_w1"], shifted, None)
    assert results["elongation_w1"] == pytest.approx(0.5)


def test_composition_against_the_test_molecules_the_samples_came_from():
    arrays = per_sample_arrays()
    same = [np.array(m) for m in ([6, 6, 8], [6, 6, 8], [6, 7], [6, 6, 8])]
    # the third sample has an N where its test molecule has an O
    other = [np.array(m) for m in ([6, 6, 8], [6, 6, 8], [6, 8], [6, 6, 8])]

    results = compute_metrics(arrays, ["composition"], {}, None, test_compositions=same)
    assert results["n_molecules"] == 4
    assert results["element_tv"] == pytest.approx(0.0)
    assert results["size_w1"] == pytest.approx(0.0)
    assert results["molecule_fraction_N"] == pytest.approx(0.25)

    results = compute_metrics(arrays, ["composition"], {}, None, test_compositions=other)
    # one of eleven atoms moved from O to N
    assert results["element_tv"] == pytest.approx(1 / 11)


def test_composition_needs_the_element_counts_and_the_split():
    missing = missing_reference_inputs(["composition"], {}, None, None, "train_val")

    assert any("--reference_element_counts_path" in line for line in missing)
    assert any("--split_path" in line for line in missing)


def test_test_compositions_follow_the_split_order(tmp_path):
    counts = collect_element_counts([[1], [6], [7], [8]])
    np.savez(tmp_path / "counts.npz", **counts)
    np.savez(tmp_path / "split.npz", train_idx=[0], val_idx=[1], test_idx=[3, 2])

    compositions = load_test_compositions(tmp_path / "counts.npz", tmp_path / "split.npz")

    assert [m.tolist() for m in compositions] == [[8], [7]]


def test_the_command_line_computes_the_new_metrics(tmp_path, monkeypatch):
    evaluation = tmp_path / "evaluation"
    metadata = evaluation / "metadata"
    metadata.mkdir(parents=True)
    arrays = per_sample_arrays()
    for key in ("rdkit_valid", "failure", "elongation"):
        np.save(metadata / f"{key}.npy", arrays[key])
    (metadata / "smiles.txt").write_text("\n".join(arrays["smiles"]) + "\n")
    np.savez(metadata / "element_counts.npz", **arrays["element_counts"])
    # the reference database: rows 4-7 are the test molecules, in order
    reference = [[1, 1], [6], [7], [8], [6, 6, 8], [6, 6, 8], [6, 7], [6, 6, 8]]
    np.savez(tmp_path / "counts.npz", **collect_element_counts(reference))
    (tmp_path / "smiles.txt").write_text("\n".join(["C", "N", "O", "CC", "", "", "", ""]) + "\n")
    np.save(tmp_path / "elongation.npy", np.array([3.0, 3.5]))
    np.savez(tmp_path / "split.npz", train_idx=[0, 1], val_idx=[2, 3], test_idx=[4, 5, 6, 7])

    monkeypatch.setattr(sys, "argv", [
        "gpff_calculate_metrics", "--evaluation_dir", str(evaluation), "--quiet",
        "--metrics", "rdkit_validity", "failures", "vun", "elongation_w1", "composition",
        "--reference_smiles_path", str(tmp_path / "smiles.txt"),
        "--reference_elongation_path", str(tmp_path / "elongation.npy"),
        "--reference_element_counts_path", str(tmp_path / "counts.npz"),
        "--split_path", str(tmp_path / "split.npz"),
    ])
    assert calculate_metrics.main() == 0

    results = json.loads((evaluation / "results.json").read_text())
    assert results["failure_fragments"] == pytest.approx(0.25)
    assert results["vun"] == pytest.approx(0.25)
    assert results["elongation_w1"] == pytest.approx(0.0)
    assert results["element_tv"] == pytest.approx(0.0)
