import os
import time

from ase import Atoms
from ase.build import molecule

from gpff.analysis import analyze_with_rdkit, get_posebusters_validity
from gpff.analysis import structure as structure_module


def test_a_timed_out_analysis_looks_like_an_invalid_one(monkeypatch):
    # the alarm can fire after the inner function's try block, e.g. while RDKit
    # is handling an error of its own, and then escapes it
    def raise_timeout(*args, **kwargs):
        raise structure_module._RDKitTimeout()

    monkeypatch.setattr(structure_module, "_analyze_with_rdkit", raise_timeout)
    analysis = analyze_with_rdkit(molecule("CH3CH2OH"))

    assert analysis["rdkit_valid"] is False
    assert analysis["bonds"] == {"bond_type": [], "environment": [], "length": []}


def test_every_analysis_has_the_same_keys():
    # the metadata pass reads the same keys off every molecule, so a valid, an
    # invalid and a timed-out analysis have to agree on them
    valid = analyze_with_rdkit(molecule("CH3CH2OH"))
    two_fragments = analyze_with_rdkit(Atoms(numbers=[6, 6], positions=[[0, 0, 0], [5, 0, 0]]))

    assert valid["rdkit_valid"] and not two_fragments["rdkit_valid"]
    assert set(valid) == set(two_fragments) == set(structure_module._invalid_analysis())


def test_posebusters_timeout_counts_a_slow_check_as_invalid(monkeypatch):
    monkeypatch.setattr(structure_module, "_posebusters_validity", lambda atoms: time.sleep(30))
    started = time.monotonic()

    assert get_posebusters_validity(molecule("CH3CH2OH"), timeout=0.5) is False
    assert time.monotonic() - started < 10


def test_posebusters_timeout_passes_the_answer_through(monkeypatch):
    for answer in (True, False):
        monkeypatch.setattr(structure_module, "_posebusters_validity", lambda atoms: answer)
        assert get_posebusters_validity(molecule("CH3CH2OH"), timeout=30) is answer


def test_posebusters_without_a_timeout_runs_in_this_process(monkeypatch):
    def not_forked(*args, **kwargs):
        raise AssertionError("no child process for an unlimited check")

    monkeypatch.setattr(structure_module._FORK, "Process", not_forked)

    assert get_posebusters_validity(molecule("CH3CH2OH")) is True


def test_posebusters_timeout_survives_a_dead_child(monkeypatch):
    # the child answers nothing, so the pipe closes without a value
    monkeypatch.setattr(structure_module, "_posebusters_validity", lambda atoms: os._exit(0))

    assert get_posebusters_validity(molecule("CH3CH2OH"), timeout=30) is False
