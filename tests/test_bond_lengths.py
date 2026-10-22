from collections import Counter

import numpy as np
import pytest
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

from gpff.analysis import analyze_with_rdkit, calculate_bond_length_w1, collect_bonds
from gpff.cli.calculate_metrics import compute_metrics


def embedded_atoms(smiles: str) -> Atoms:
    """An ASE Atoms object of the molecule, embedded with a fixed seed."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    AllChem.MMFFOptimizeMolecule(mol)
    return Atoms(
        numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
        positions=mol.GetConformer().GetPositions(),
    )


def bond_table(groups: dict[tuple[str, str], np.ndarray]) -> dict[str, np.ndarray]:
    """A bond table with the given lengths per (bond type, environment)."""
    bond_type, environment, length = [], [], []
    for (type_label, environment_label), lengths in groups.items():
        bond_type += [type_label] * len(lengths)
        environment += [environment_label] * len(lengths)
        length += list(lengths)
    return {
        "molecule": np.zeros(len(length), dtype=int),
        "bond_type": np.array(bond_type),
        "environment": np.array(environment),
        "length": np.array(length, dtype=float),
    }


def test_bond_labels_of_acetaldehyde():
    bonds = analyze_with_rdkit(embedded_atoms("CC=O"))["bonds"]

    assert Counter(bonds["bond_type"]) == {"C-H": 4, "C-C": 1, "C=O": 1}
    assert Counter(bonds["environment"]) == {
        "C(sp3)-H": 3,
        "C(sp2)-H": 1,
        "C(sp2)-C(sp3)": 1,
        "C(sp2)=O(sp2)": 1,
    }
    lengths = dict(zip(bonds["environment"], bonds["length"]))
    assert lengths["C(sp2)=O(sp2)"] == pytest.approx(1.22, abs=0.03)
    assert lengths["C(sp2)-C(sp3)"] == pytest.approx(1.50, abs=0.03)


def test_bond_labels_mark_small_rings():
    bonds = analyze_with_rdkit(embedded_atoms("C1CC1C#N"))["bonds"]

    environments = Counter(bonds["environment"])
    assert environments["C(sp3)-C(sp3) r3"] == 3
    assert environments["C(sp)-C(sp3)"] == 1
    assert environments["C(sp)#N(sp)"] == 1


def test_invalid_molecule_has_no_bonds():
    # two far-apart carbon atoms: two fragments, so invalid
    atoms = Atoms(numbers=[6, 6], positions=[[0, 0, 0], [5, 0, 0]])
    analysis = analyze_with_rdkit(atoms)

    assert not analysis["rdkit_valid"]
    assert analysis["bonds"] == {"bond_type": [], "environment": [], "length": []}


def test_collect_bonds_keeps_molecule_index():
    no_bonds = {"bond_type": [], "environment": [], "length": []}
    one_bond = {"bond_type": ["C-H"], "environment": ["C(sp3)-H"], "length": [1.09]}
    table = collect_bonds([one_bond, no_bonds, one_bond])

    np.testing.assert_array_equal(table["molecule"], [0, 2])
    np.testing.assert_array_equal(table["environment"], ["C(sp3)-H", "C(sp3)-H"])


def test_w1_of_a_shift_is_the_shift():
    rng = np.random.default_rng(0)
    reference = bond_table({
        ("C-C", "C(sp3)-C(sp3)"): rng.normal(1.53, 0.01, 1000),
        ("C=O", "C(sp2)=O(sp2)"): rng.normal(1.21, 0.01, 500),
    })
    observed = dict(reference, length=reference["length"] + 0.02)

    summary = calculate_bond_length_w1(reference, observed)

    assert summary["w1"] == pytest.approx(0.02)
    for row in summary["labels"].values():
        assert row["median_shift"] == pytest.approx(0.02)
        assert row["spread_ratio"] == pytest.approx(1.0)


def test_swapped_peaks_are_invisible_pooled_but_not_per_environment():
    # half the C-C bonds conjugated at 1.46, half sp3 at 1.53; the observed set
    # swaps them, so its pooled C-C histogram is the reference's
    conjugated = np.linspace(1.455, 1.465, 200)
    saturated = np.linspace(1.525, 1.535, 200)
    reference = bond_table({
        ("C-C", "C(sp2)-C(sp2)"): conjugated,
        ("C-C", "C(sp3)-C(sp3)"): saturated,
    })
    observed = bond_table({
        ("C-C", "C(sp2)-C(sp2)"): saturated,
        ("C-C", "C(sp3)-C(sp3)"): conjugated,
    })

    assert calculate_bond_length_w1(reference, observed, by="bond_type")["w1"] == pytest.approx(0.0)
    assert calculate_bond_length_w1(reference, observed)["w1"] == pytest.approx(0.07)


def test_a_different_mix_of_environments_alone_does_not_count():
    conjugated = np.linspace(1.455, 1.465, 200)
    saturated = np.linspace(1.525, 1.535, 200)
    reference = bond_table({
        ("C-C", "C(sp2)-C(sp2)"): conjugated,
        ("C-C", "C(sp3)-C(sp3)"): saturated,
    })
    # every bond where it belongs, but three times as many conjugated ones
    observed = bond_table({
        ("C-C", "C(sp2)-C(sp2)"): np.tile(conjugated, 3),
        ("C-C", "C(sp3)-C(sp3)"): saturated,
    })

    assert calculate_bond_length_w1(reference, observed)["w1"] == pytest.approx(0.0)
    assert calculate_bond_length_w1(reference, observed, by="bond_type")["w1"] > 0.01


def test_rare_environments_are_reported_but_not_compared():
    reference = bond_table({
        ("C-C", "C(sp3)-C(sp3)"): np.full(100, 1.53),
        ("N-N", "N(sp3)-N(sp3)"): np.full(100, 1.45),
    })
    observed = bond_table({
        ("C-C", "C(sp3)-C(sp3)"): np.full(100, 1.54),
        ("N-N", "N(sp3)-N(sp3)"): np.full(5, 1.80),
        ("O-O", "O(sp3)-O(sp3)"): np.full(30, 1.47),
    })

    summary = calculate_bond_length_w1(reference, observed, min_count=20)

    assert summary["w1"] == pytest.approx(0.01)
    assert summary["reference_weight_compared"] == pytest.approx(0.5)
    assert summary["observed_fraction_compared"] == pytest.approx(100 / 135)
    assert summary["labels"]["N(sp3)-N(sp3)"]["n_observed"] == 5
    assert np.isnan(summary["labels"]["N(sp3)-N(sp3)"]["w1"])
    assert summary["labels"]["O(sp3)-O(sp3)"]["n_reference"] == 0


def test_compute_metrics_reads_the_bond_table_next_to_per_sample_arrays():
    reference = bond_table({("C-C", "C(sp3)-C(sp3)"): np.full(100, 1.53)})
    observed = bond_table({("C-C", "C(sp3)-C(sp3)"): np.full(100, 1.55)})
    arrays = {"rdkit_valid": np.array([True, True, False]), "bond_lengths": observed}

    results = compute_metrics(
        arrays,
        ["rdkit_validity", "bond_length_w1"],
        reference_distributions={},
        reference_smiles=None,
        reference_bond_lengths=reference,
    )

    assert results["n_molecules"] == 3
    assert results["bond_length_w1"] == pytest.approx(0.02)
