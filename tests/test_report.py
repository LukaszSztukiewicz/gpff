import json
import sys

import numpy as np
import pytest

from gpff.analysis import calculate_bond_length_w1, collect_element_counts
from gpff.analysis import report
from gpff.cli import report as report_cli

N_SAMPLES = 200
# 3 to 6 heavy atoms, 50 samples each
NUMBERS = [[6] * (3 + i % 4) + [1] * 4 for i in range(N_SAMPLES)]


def bond_table(shift: float) -> dict:
    rng = np.random.default_rng(0)
    environment = np.array(["C(sp3)-C(sp3)"] * 60 + ["C(sp3)-H"] * 40)
    length = np.r_[1.53 + 0.01 * rng.standard_normal(60), 1.09 + 0.005 * rng.standard_normal(40)]
    return {"molecule": np.zeros(100, dtype=int), "bond_type": environment,
            "environment": environment, "length": length + shift}


def write_evaluation(model_dir, config: str, validity: float, seed: int) -> None:
    evaluation = model_dir / config / "seed42069" / "evaluation"
    (evaluation / "metadata").mkdir(parents=True)
    rng = np.random.default_rng(seed)
    valid = rng.random(N_SAMPLES) < validity
    failure = np.where(valid, "valid", rng.choice(["fragments", "bond_orders"], N_SAMPLES))
    np.save(evaluation / "metadata" / "rdkit_valid.npy", valid)
    np.save(evaluation / "metadata" / "failure.npy", failure)
    np.savez(evaluation / "metadata" / "element_counts.npz", **collect_element_counts(NUMBERS))
    breakdown = calculate_bond_length_w1(bond_table(0.0), bond_table(0.01 * (1 - validity)))
    (evaluation / "bond_length_w1.json").write_text(json.dumps(breakdown))
    (evaluation / "results.json").write_text(json.dumps(
        {"rdkit_validity": float(valid.mean()), "vun": 0.5, "bond_length_w1": breakdown["w1"]}
    ))


def test_wilson_interval_contains_the_proportion():
    low, high = report.wilson(30, 100)

    assert low < 0.3 < high
    assert 0 < low and high < 1


def test_paired_difference_uses_the_discordant_samples():
    sizes = np.array([3, 4, 5, 6])
    a = {"seed1": {"sizes": sizes, "valid": np.array([True, True, False, False])}}
    b = {"seed1": {"sizes": sizes, "valid": np.array([True, False, True, True])}}

    delta, sd, paired = report.difference(a, b, lambda e: e["valid"])

    assert paired
    assert delta == pytest.approx(25.0)
    assert sd == pytest.approx(100 * np.sqrt(3) / 4)

    b["seed1"]["sizes"] = sizes[::-1]
    assert not report.difference(a, b, lambda e: e["valid"])[2]


def test_default_names_are_run_and_epoch(tmp_path):
    repro = tmp_path / "key-moray" / "frozen" / "repro"

    assert report_cli.default_name(repro / "epoch_5000_model") == "key-moray:5000"
    assert report_cli.default_name(repro / "best_epoch_1961_model") == "key-moray:best"
    assert report_cli.default_name(tmp_path / "somewhere") == "somewhere"


def test_the_report_writes_every_figure_and_the_page(tmp_path, monkeypatch):
    model_dirs = []
    for run, epoch, validity in (("run-a", 100, 0.6), ("run-a", 200, 0.8), ("run-b", 100, 0.7)):
        model_dir = tmp_path / run / "frozen" / "repro" / f"epoch_{epoch}_model"
        for i, config in enumerate(("dd16", "sdd256")):
            write_evaluation(model_dir, config, validity, seed=epoch + i)
        model_dirs.append(str(model_dir))
    output = tmp_path / "report"

    monkeypatch.setattr(sys, "argv", ["gpff_report", *model_dirs, "--control", "run-a:100",
                                      "--output_dir", str(output)])
    assert report_cli.main() == 0

    page = (output / "index.html").read_text()
    assert "run-b:100" in page and "Valid % [95% CI]" in page
    for name in ("deltas", "failures", "validity_by_size", "epoch_rdkit_validity", "epoch_vun",
                 "bonds_0", "bonds_2"):
        assert (output / "figures" / f"{name}.png").stat().st_size > 0
        assert f"figures/{name}.png" in page
