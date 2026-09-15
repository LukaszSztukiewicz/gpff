"""C2 regression tests: reference asset derivation and export.

Covers the full chain, with no precomputed arrays anywhere::

    tiny_raw_db -> build_filtered_qm9 -> gpff_reference_assets -> five files

plus the atomrefs/SMILES metadata C2a adds to freshly filtered databases, and
the precomputed-array, dependency-resolution and error-reporting behavior C2b
adds to the exporter. Most branch/error tests call `main()` in-process
(monkeypatching `sys.argv`) for speed; the full export and the
missing-atomrefs error keep a real subprocess, per the ground rules.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from ase.db import connect
from ase.units import Hartree

from gpff.analysis import analyze_with_rdkit, compute_elongation
from gpff.cli import reference_assets as ra
from tests.conftest import (
    ATOMREF_ENERGY_U0_HARTREE,
    N_FIXTURE_MOLECULES,
    atomrefs_energy_U0_array,
    copy_db,
    run_cli,
    venv_console_script,
)

# Appendix A: the marker new filtered databases declare (kept independent of
# gpff.cli.reference_assets.SMILES_FORMAT_MARKER so this is a real regression
# anchor, not a tautology against the implementation's own constant).
SMILES_FORMAT_MARKER = "gpff_explicit_h_isomeric_v1"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _variant_db(tiny_filtered_db, tmp_path, name) -> str:
    """A private copy of tiny_filtered_db this test can mutate freely."""
    return copy_db(tiny_filtered_db, str(tmp_path / name))


def _get_metadata(db_path) -> dict:
    with connect(db_path) as db:
        return dict(db.metadata)


def _set_metadata(db_path, metadata: dict) -> None:
    with connect(db_path) as db:
        db.metadata = metadata


def _replace_row_data(db_path, row_id: int, new_data: dict) -> None:
    """Replace one row's `data` dict entirely (not merged, unlike db.update)."""
    with connect(db_path) as db:
        row = db.get(row_id)
        db.write(row.toatoms(), data=new_data, key_value_pairs=row.key_value_pairs, id=row_id)


def _run_main(monkeypatch, reference_db_path, assets_dir, targets=None, prefix: str = "qm9"):
    """Call reference_assets.main() in-process with a given argv."""
    argv = [
        "--reference_db_path", str(reference_db_path),
        "--assets_dir", str(assets_dir),
        "--prefix", prefix,
        "--quiet",
    ]
    if targets:
        argv += ["--targets", *targets]
    monkeypatch.setattr(sys, "argv", ["gpff_reference_assets", *argv])
    return ra.main()


def _load_array(assets_dir, key: str, prefix: str = "qm9") -> np.ndarray:
    return np.load(Path(assets_dir) / f"{prefix}_{key}.npy")


def _load_smiles_lines(assets_dir, prefix: str = "qm9") -> list:
    text = (Path(assets_dir) / f"{prefix}_smiles_values.txt").read_text()
    return text.splitlines()


def _assert_no_assets_written(assets_dir) -> None:
    assert not assets_dir.exists() or not any(assets_dir.iterdir())


# ---------------------------------------------------------------------------
# the full chain: tiny_raw_db -> build_filtered_qm9 -> gpff_reference_assets
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exported_assets(tmp_path_factory, tiny_filtered_db):
    """Full export of tiny_filtered_db via the installed CLI, shared across assertions."""
    assets_dir = tmp_path_factory.mktemp("assets_full")
    run_dir = tmp_path_factory.mktemp("run_full")
    run_cli(
        [
            venv_console_script("gpff_reference_assets"),
            "--reference_db_path", tiny_filtered_db,
            "--assets_dir", str(assets_dir),
            "--quiet",
        ],
        cwd=run_dir,
        timeout=90,
    )
    arrays = {key: _load_array(assets_dir, key) for key in ra.METADATA_ARRAY_TARGETS}
    arrays["smiles_values"] = _load_smiles_lines(assets_dir)
    arrays["dir"] = assets_dir
    return arrays


def test_atomrefs_preserved(tiny_raw_db, tiny_filtered_db):
    raw_atomrefs = _get_metadata(tiny_raw_db)["atomrefs"]["energy_U0"]
    filtered_atomrefs = _get_metadata(tiny_filtered_db)["atomrefs"]["energy_U0"]
    assert len(filtered_atomrefs) == 100
    assert filtered_atomrefs == raw_atomrefs


def test_five_assets_written(exported_assets):
    assets_dir = Path(exported_assets["dir"])
    for name in (
        "qm9_energy_U0_values.npy",
        "qm9_gap_values.npy",
        "qm9_elongation_values.npy",
        "qm9_energy_U0_atomrefs_eV.npy",
        "qm9_smiles_values.txt",
    ):
        assert (assets_dir / name).exists(), name


def test_lengths_match_rows(exported_assets):
    for key in ("energy_U0_values", "gap_values", "elongation_values"):
        assert exported_assets[key].shape == (N_FIXTURE_MOLECULES,)
    assert exported_assets["energy_U0_atomrefs_eV"].shape == (100,)
    assert len(exported_assets["smiles_values"]) == N_FIXTURE_MOLECULES


def test_atomrefs_units(exported_assets):
    expected = ATOMREF_ENERGY_U0_HARTREE[6] * Hartree
    np.testing.assert_allclose(exported_assets["energy_U0_atomrefs_eV"][6], expected, rtol=1e-9)


def test_energy_is_atomization(exported_assets, tiny_filtered_db):
    atomrefs_hartree = atomrefs_energy_U0_array()
    with connect(tiny_filtered_db) as db:
        rows = list(db.select())
    for i, row in enumerate(rows):
        raw = float(row.data["energy_U0"][0])
        expected = raw * Hartree - sum(atomrefs_hartree[z] * Hartree for z in row.numbers)
        np.testing.assert_allclose(exported_assets["energy_U0_values"][i], expected, rtol=1e-9)


def test_gap_units_and_order(exported_assets, tiny_filtered_db):
    with connect(tiny_filtered_db) as db:
        rows = list(db.select())
    for i, row in enumerate(rows):
        expected = float(row.data["gap"][0]) * Hartree
        np.testing.assert_allclose(exported_assets["gap_values"][i], expected, rtol=1e-9, atol=0)
    # the fixture assigns strictly increasing gap values by (filtered) row order
    assert np.all(np.diff(exported_assets["gap_values"]) > 0)


def test_filtered_row_alignment(exported_assets, tiny_raw_db, tiny_filtered_db, tiny_split):
    with connect(tiny_raw_db) as raw_db, connect(tiny_filtered_db) as filtered_db:
        for row in filtered_db.select():
            orig = int(row.data["original_index"][0])
            raw_row = raw_db.get(orig + 1)
            assert tuple(int(z) for z in row.numbers) == tuple(int(z) for z in raw_row.numbers)
            np.testing.assert_allclose(row.positions, raw_row.positions, rtol=0, atol=0)
            assert float(row.data["energy_U0"][0]) == float(raw_row.data["energy_U0"][0])
            assert float(row.data["gap"][0]) == float(raw_row.data["gap"][0])

    split = np.load(tiny_split)
    with connect(tiny_filtered_db) as db:
        rows = list(db.select())
    for partition_name in ("train_idx", "val_idx", "test_idx"):
        idx = split[partition_name]
        expected = np.array([float(rows[i].data["gap"][0]) * Hartree for i in idx])
        np.testing.assert_allclose(exported_assets["gap_values"][idx], expected, rtol=1e-9)


def test_smiles_matches_sample_path(exported_assets, tiny_filtered_db):
    with connect(tiny_filtered_db) as db:
        row = db.get(1)
    expected = analyze_with_rdkit(row.toatoms())["smiles"]
    assert exported_assets["smiles_values"][0] == expected


def test_elongation_matches(exported_assets, tiny_filtered_db):
    with connect(tiny_filtered_db) as db:
        rows = list(db.select())
    for i, row in enumerate(rows):
        expected = compute_elongation(row.toatoms())
        np.testing.assert_allclose(exported_assets["elongation_values"][i], expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# precomputed arrays and selective requests
# ---------------------------------------------------------------------------


def test_precomputed_only(tiny_filtered_db, tmp_path, monkeypatch):
    db_path = _variant_db(tiny_filtered_db, tmp_path, "precomputed_only.db")
    with connect(db_path) as db:
        n_rows = db.count()

    energy = (np.arange(n_rows, dtype=float) + 1) * 111.0
    gap = (np.arange(n_rows, dtype=float) + 1) * 222.0
    elongation = (np.arange(n_rows, dtype=float) + 1) * 3.3
    atomrefs_eV = (np.arange(100, dtype=float) + 1) * 7.0
    _set_metadata(
        db_path,
        {
            "energy_U0_values": energy.tolist(),
            "gap_values": gap.tolist(),
            "elongation_values": elongation.tolist(),
            "energy_U0_atomrefs_eV": atomrefs_eV.tolist(),
        },
    )

    assets_dir = tmp_path / "assets"
    rc = _run_main(monkeypatch, db_path, assets_dir, targets=list(ra.METADATA_ARRAY_TARGETS))
    assert rc == 0

    np.testing.assert_array_equal(_load_array(assets_dir, "energy_U0_values"), energy)
    np.testing.assert_array_equal(_load_array(assets_dir, "gap_values"), gap)
    np.testing.assert_array_equal(_load_array(assets_dir, "elongation_values"), elongation)
    np.testing.assert_array_equal(_load_array(assets_dir, "energy_U0_atomrefs_eV"), atomrefs_eV)


def test_precomputed_takes_precedence(tiny_filtered_db, tmp_path, monkeypatch):
    db_path = _variant_db(tiny_filtered_db, tmp_path, "precedence.db")
    with connect(db_path) as db:
        n_rows = db.count()

    # metadata keeps the real raw energy_U0/gap/atomrefs/_property_unit_dict;
    # conflicting precomputed arrays are added on top and must win.
    metadata = _get_metadata(db_path)
    conflicting_gap = (np.arange(n_rows, dtype=float) + 1) * 999.0
    conflicting_atomrefs = (np.arange(100, dtype=float) + 1) * -5.0
    metadata["gap_values"] = conflicting_gap.tolist()
    metadata["energy_U0_atomrefs_eV"] = conflicting_atomrefs.tolist()
    _set_metadata(db_path, metadata)

    assets_dir = tmp_path / "assets"
    rc = _run_main(monkeypatch, db_path, assets_dir, targets=["gap_values", "energy_U0_atomrefs_eV"])
    assert rc == 0

    np.testing.assert_array_equal(_load_array(assets_dir, "gap_values"), conflicting_gap)
    np.testing.assert_array_equal(_load_array(assets_dir, "energy_U0_atomrefs_eV"), conflicting_atomrefs)


def test_mixed_sources(tiny_filtered_db, tmp_path, monkeypatch):
    db_path = _variant_db(tiny_filtered_db, tmp_path, "mixed.db")
    metadata = _get_metadata(db_path)
    precomputed_atomrefs_eV = np.asarray(atomrefs_energy_U0_array()) * Hartree
    del metadata["atomrefs"]  # raw atomrefs absent: only the precomputed array is available
    metadata["energy_U0_atomrefs_eV"] = precomputed_atomrefs_eV.tolist()
    _set_metadata(db_path, metadata)

    assets_dir = tmp_path / "assets"
    rc = _run_main(monkeypatch, db_path, assets_dir, targets=["energy_U0_values", "gap_values"])
    assert rc == 0

    with connect(db_path) as db:
        rows = list(db.select())
    expected_energy = np.array(
        [
            float(row.data["energy_U0"][0]) * Hartree
            - sum(precomputed_atomrefs_eV[z] for z in row.numbers)
            for row in rows
        ]
    )
    expected_gap = np.array([float(row.data["gap"][0]) * Hartree for row in rows])
    np.testing.assert_allclose(_load_array(assets_dir, "energy_U0_values"), expected_energy, rtol=1e-9)
    np.testing.assert_allclose(_load_array(assets_dir, "gap_values"), expected_gap, rtol=1e-9)


def test_selected_target_dependencies(tiny_filtered_db, tmp_path, monkeypatch):
    db_path = _variant_db(tiny_filtered_db, tmp_path, "selected.db")

    # energy-only: resolves atomrefs internally (raw atomrefs present) but
    # writes only the energy file.
    assets_dir = tmp_path / "assets_energy_only"
    rc = _run_main(monkeypatch, db_path, assets_dir, targets=["energy_U0_values"])
    assert rc == 0
    assert (assets_dir / "qm9_energy_U0_values.npy").exists()
    assert not (assets_dir / "qm9_energy_U0_atomrefs_eV.npy").exists()

    # gap-only: must succeed even with energy properties and atomrefs gone.
    metadata = _get_metadata(db_path)
    del metadata["atomrefs"]
    unit_dict = dict(metadata["_property_unit_dict"])
    del unit_dict["energy_U0"]
    metadata["_property_unit_dict"] = unit_dict
    _set_metadata(db_path, metadata)

    assets_dir2 = tmp_path / "assets_gap_only"
    rc2 = _run_main(monkeypatch, db_path, assets_dir2, targets=["gap_values"])
    assert rc2 == 0
    assert (assets_dir2 / "qm9_gap_values.npy").exists()


# ---------------------------------------------------------------------------
# SMILES compatibility and fast path
# ---------------------------------------------------------------------------


def test_smiles_marker_and_fields(tiny_filtered_db):
    metadata = _get_metadata(tiny_filtered_db)
    assert metadata["smiles_format"] == SMILES_FORMAT_MARKER
    with connect(tiny_filtered_db) as db:
        for row in db.select():
            analysis = analyze_with_rdkit(row.toatoms())
            assert row.get("smiles") == analysis["smiles"]
            assert row.get("canonical_smiles") == analysis["canonical_smiles"]


def test_stored_smiles_fast_path(tiny_filtered_db, tmp_path, monkeypatch):
    db_path = _variant_db(tiny_filtered_db, tmp_path, "fast_path.db")
    with connect(db_path) as db:
        db.update(1, smiles="")  # a stored empty string must retain its line

    def _raise(*args, **kwargs):
        raise AssertionError("analyze_with_rdkit should not be called on the fast path")

    monkeypatch.setattr(ra, "analyze_with_rdkit", _raise)

    assets_dir = tmp_path / "assets"
    rc = _run_main(monkeypatch, db_path, assets_dir, targets=["smiles_values"])
    assert rc == 0

    lines = _load_smiles_lines(assets_dir)
    assert len(lines) == N_FIXTURE_MOLECULES
    assert lines[0] == ""


def test_legacy_smiles_recomputed(tiny_filtered_db, tmp_path, monkeypatch):
    db_path = _variant_db(tiny_filtered_db, tmp_path, "legacy.db")
    metadata = _get_metadata(db_path)
    del metadata["smiles_format"]

    with connect(db_path) as db:
        legacy_by_id = {row.id: row.get("canonical_smiles") for row in db.select()}
    with connect(db_path) as db:
        for row_id, canonical in legacy_by_id.items():
            db.update(row_id, smiles=canonical)
    _set_metadata(db_path, metadata)

    assets_dir = tmp_path / "assets"
    rc = _run_main(monkeypatch, db_path, assets_dir, targets=["smiles_values"])
    assert rc == 0
    lines = _load_smiles_lines(assets_dir)

    with connect(db_path) as db:
        rows = list(db.select())
    mismatch_found = False
    for i, row in enumerate(rows):
        fresh = analyze_with_rdkit(row.toatoms())["smiles"]
        assert lines[i] == fresh
        if fresh != legacy_by_id[row.id]:
            mismatch_found = True
    assert mismatch_found, "expected at least one fixture molecule with H representations that differ"


@pytest.mark.parametrize(
    "make_variant",
    ["unknown_marker", "missing_column", "one_missing_value"],
)
def test_smiles_fallback(tiny_filtered_db, tmp_path, monkeypatch, make_variant):
    db_path = _variant_db(tiny_filtered_db, tmp_path, f"fallback_{make_variant}.db")

    if make_variant == "unknown_marker":
        metadata = _get_metadata(db_path)
        metadata["smiles_format"] = "some_unrecognised_marker_v0"
        _set_metadata(db_path, metadata)
    elif make_variant == "missing_column":
        with connect(db_path) as db:
            ids = [row.id for row in db.select()]
        with connect(db_path) as db:
            for row_id in ids:
                db.update(row_id, delete_keys=["smiles"])
    elif make_variant == "one_missing_value":
        with connect(db_path) as db:
            db.update(1, delete_keys=["smiles"])

    assets_dir = tmp_path / "assets"
    rc = _run_main(monkeypatch, db_path, assets_dir, targets=["smiles_values"])
    assert rc == 0
    lines = _load_smiles_lines(assets_dir)
    assert len(lines) == N_FIXTURE_MOLECULES

    with connect(db_path) as db:
        rows = list(db.select())
    for i, row in enumerate(rows):
        assert lines[i] == analyze_with_rdkit(row.toatoms())["smiles"]


def test_empty_smiles_export(tmp_path, monkeypatch):
    db_path = str(tmp_path / "empty.db")
    with connect(db_path) as db:
        db.metadata = {
            "_property_unit_dict": {"energy_U0": "Ha", "gap": "Ha"},
            "_distance_unit": "Ang",
            "atomrefs": {"energy_U0": atomrefs_energy_U0_array()},
            "smiles_format": SMILES_FORMAT_MARKER,
        }

    assets_dir = tmp_path / "assets"
    rc = _run_main(monkeypatch, db_path, assets_dir, targets=["smiles_values"])
    assert rc == 0

    out_path = assets_dir / "qm9_smiles_values.txt"
    assert out_path.exists()
    assert out_path.stat().st_size == 0


# ---------------------------------------------------------------------------
# actionable errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mutate", ["absent", "empty_dict", "missing_energy_key"])
def test_missing_atomrefs_errors(tiny_filtered_db, tmp_path, mutate):
    db_path = _variant_db(tiny_filtered_db, tmp_path, f"atomrefs_{mutate}.db")
    metadata = _get_metadata(db_path)
    if mutate == "absent":
        del metadata["atomrefs"]
    elif mutate == "empty_dict":
        metadata["atomrefs"] = {}
    elif mutate == "missing_energy_key":
        metadata["atomrefs"] = {"gap": [0.0] * 100}
    _set_metadata(db_path, metadata)

    assets_dir = tmp_path / f"assets_{mutate}"
    result = run_cli(
        [
            venv_console_script("gpff_reference_assets"),
            "--reference_db_path", db_path,
            "--assets_dir", str(assets_dir),
            "--targets", "energy_U0_atomrefs_eV",
            "--quiet",
        ],
        cwd=tmp_path,
        timeout=60,
        check=False,
    )
    assert result.returncode != 0
    assert "atomrefs" in result.stderr
    assert db_path in result.stderr
    _assert_no_assets_written(assets_dir)


@pytest.mark.parametrize("target,property_name", [("gap_values", "gap"), ("energy_U0_values", "energy_U0")])
@pytest.mark.parametrize("bad_unit", [None, "eV"])
def test_property_units_errors(tiny_filtered_db, tmp_path, monkeypatch, capsys, target, property_name, bad_unit):
    db_path = _variant_db(tiny_filtered_db, tmp_path, f"unit_{property_name}_{bad_unit}.db")
    metadata = _get_metadata(db_path)
    unit_dict = dict(metadata["_property_unit_dict"])
    if bad_unit is None:
        del unit_dict[property_name]
    else:
        unit_dict[property_name] = bad_unit
    metadata["_property_unit_dict"] = unit_dict
    _set_metadata(db_path, metadata)

    assets_dir = tmp_path / "assets"
    rc = _run_main(monkeypatch, db_path, assets_dir, targets=[target])
    captured = capsys.readouterr()

    assert rc != 0
    assert property_name in captured.err
    assert db_path in captured.err
    if bad_unit is not None:
        assert bad_unit in captured.err
    _assert_no_assets_written(assets_dir)


@pytest.mark.parametrize("bad_unit", [None, "Bohr"])
def test_distance_units_errors(tiny_filtered_db, tmp_path, monkeypatch, capsys, bad_unit):
    db_path = _variant_db(tiny_filtered_db, tmp_path, f"distance_{bad_unit}.db")
    metadata = _get_metadata(db_path)
    if bad_unit is None:
        del metadata["_distance_unit"]
    else:
        metadata["_distance_unit"] = bad_unit
    _set_metadata(db_path, metadata)

    assets_dir = tmp_path / "assets"
    rc = _run_main(monkeypatch, db_path, assets_dir, targets=["elongation_values"])
    captured = capsys.readouterr()

    assert rc != 0
    assert db_path in captured.err
    if bad_unit is not None:
        assert bad_unit in captured.err
    _assert_no_assets_written(assets_dir)


@pytest.mark.parametrize("target,property_name", [("gap_values", "gap"), ("energy_U0_values", "energy_U0")])
def test_missing_row_property_errors(tiny_filtered_db, tmp_path, monkeypatch, capsys, target, property_name):
    db_path = _variant_db(tiny_filtered_db, tmp_path, f"missing_row_{property_name}.db")
    with connect(db_path) as db:
        target_row = db.get(3)
    new_data = dict(target_row.data)
    del new_data[property_name]
    _replace_row_data(db_path, target_row.id, new_data)

    assets_dir = tmp_path / "assets"
    rc = _run_main(monkeypatch, db_path, assets_dir, targets=[target])
    captured = capsys.readouterr()

    assert rc != 0
    assert property_name in captured.err
    assert str(target_row.id) in captured.err
    assert db_path in captured.err
    _assert_no_assets_written(assets_dir)

    # rows are never silently skipped: the row itself is still there
    with connect(db_path) as db:
        assert db.count() == N_FIXTURE_MOLECULES


def _malformed_nonscalar_row(db_path):
    with connect(db_path) as db:
        row = db.get(1)
    new_data = dict(row.data)
    new_data["gap"] = np.array([0.1, 0.2])
    _replace_row_data(db_path, row.id, new_data)
    return ["gap_values"]


def _malformed_nonfinite_row(db_path):
    with connect(db_path) as db:
        row = db.get(1)
    new_data = dict(row.data)
    new_data["energy_U0"] = np.array([np.nan])
    _replace_row_data(db_path, row.id, new_data)
    return ["energy_U0_values"]


def _malformed_atomrefs_length(db_path):
    metadata = _get_metadata(db_path)
    metadata["atomrefs"] = {"energy_U0": [0.0] * 50}
    _set_metadata(db_path, metadata)
    return ["energy_U0_atomrefs_eV"]


def _malformed_atomrefs_nonfinite(db_path):
    metadata = _get_metadata(db_path)
    bad = atomrefs_energy_U0_array()
    bad[6] = float("nan")
    metadata["atomrefs"] = {"energy_U0": bad}
    _set_metadata(db_path, metadata)
    return ["energy_U0_atomrefs_eV"]


def _malformed_precomputed_length(db_path):
    metadata = _get_metadata(db_path)
    metadata["gap_values"] = [1.0] * (N_FIXTURE_MOLECULES - 1)
    _set_metadata(db_path, metadata)
    return ["gap_values"]


def _malformed_precomputed_nonfinite(db_path):
    metadata = _get_metadata(db_path)
    values = [1.0] * N_FIXTURE_MOLECULES
    values[0] = float("nan")
    metadata["gap_values"] = values
    _set_metadata(db_path, metadata)
    return ["gap_values"]


@pytest.mark.parametrize(
    "make_malformed",
    [
        _malformed_nonscalar_row,
        _malformed_nonfinite_row,
        _malformed_atomrefs_length,
        _malformed_atomrefs_nonfinite,
        _malformed_precomputed_length,
        _malformed_precomputed_nonfinite,
    ],
    ids=[
        "nonscalar_row_property",
        "nonfinite_row_property",
        "malformed_atomrefs_length",
        "malformed_atomrefs_nonfinite",
        "precomputed_wrong_length",
        "precomputed_nonfinite",
    ],
)
def test_malformed_values_errors(tiny_filtered_db, tmp_path, monkeypatch, make_malformed):
    db_path = _variant_db(tiny_filtered_db, tmp_path, f"{make_malformed.__name__}.db")
    targets = make_malformed(db_path)

    assets_dir = tmp_path / "assets"
    rc = _run_main(monkeypatch, db_path, assets_dir, targets=targets)
    assert rc != 0
    _assert_no_assets_written(assets_dir)


def test_failed_export_writes_no_assets(tiny_filtered_db, tmp_path, monkeypatch):
    db_path = _variant_db(tiny_filtered_db, tmp_path, "failed_export.db")
    with connect(db_path) as db:
        row = db.get(2)
    new_data = dict(row.data)
    del new_data["gap"]
    _replace_row_data(db_path, row.id, new_data)

    assets_dir = tmp_path / "assets"
    rc = _run_main(monkeypatch, db_path, assets_dir, targets=["all"])
    assert rc != 0
    _assert_no_assets_written(assets_dir)
