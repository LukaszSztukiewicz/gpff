import hashlib
import json
import os
import pickle
from pathlib import Path

import hydra
import numpy as np
import pytest
from ase.db import connect
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from schnetpack.data import AtomsDataModuleError

import gpff.cli  # registers the resolvers the configs use
from gpff.cli import prepare_geom_revisited
from gpff.data.datasets import PreparedAtomsData
from gpff.data.datasets import geom_revisited
from gpff.data.datasets.geom_revisited import (
    UPSTREAM,
    RawMol,
    build_geom_drugs_revisited,
    check_upstream,
    clean_molecule,
    file_digests,
    load_records,
    split_file_name,
)

CONFIG_DIR = Path(gpff.cli.__file__).parents[1] / "configs"


def conformers(smiles, n, seed=7):
    """A molecule the way the MiDi pickles hold it: one RDKit molecule per conformer."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert len(AllChem.EmbedMultipleConfs(mol, n, randomSeed=seed)) == n
    mols = []
    for conformer in mol.GetConformers():
        single = Chem.Mol(mol)
        single.RemoveAllConformers()
        single.AddConformer(Chem.Conformer(conformer), assignId=True)
        mols.append(single)
    return mols


def disconnected(mol):
    """The same atoms and positions without the first bond: two fragments."""
    editable = Chem.RWMol(mol)
    editable.RemoveBond(0, 1)
    return editable.GetMol()


def stretched(mol, factor=2.0):
    """The same molecule with every bond too long."""
    mol = Chem.Mol(mol)
    conformer = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conformer.GetAtomPosition(i)
        conformer.SetAtomPosition(i, Point3D(factor * p.x, factor * p.y, factor * p.z))
    return mol


@pytest.fixture(scope="module")
def propanol():
    return conformers("CCCO", 4)


@pytest.fixture
def source(tmp_path, propanol):
    """Three small splits with every case the cleaning has to handle."""
    mols_of_propanol = list(propanol)
    mols_of_propanol[1] = disconnected(propanol[1])  # this conformer is dropped
    splits = {
        "train": [
            ("CCO", conformers("CCO", 7)),  # more than 5 conformers
            ("c1ccccc1O", conformers("c1ccccc1O", 3)),  # fewer than 5
            ("not a smiles", conformers("CCO", 2)),  # removed: the SMILES does not parse
            ("CCN", [stretched(mol) for mol in conformers("CCN", 2)]),  # removed: bonds too long
            ("CC(=O)O", conformers("CC(=O)O", 6)),
            ("CCCO", mols_of_propanol),  # 3 of 4 conformers stay
        ],
        "val": [
            ("CCCl", conformers("CCCl", 2)),
            ("CC", [disconnected(mol) for mol in conformers("CC", 2)]),  # removed: none stays
        ],
        "test": [
            ("CCBr", conformers("CCBr", 1)),
            ("CCO", conformers("CCO", 2)),  # a molecule of train, too
        ],
    }
    directory = tmp_path / "source"
    directory.mkdir()
    for split, records in splits.items():
        with open(directory / f"{split}_data.pickle", "wb") as file:
            pickle.dump(records, file)
    return directory


def build(source, tmp_path, name="target", **kwargs):
    target = tmp_path / name
    target.mkdir()
    counts = build_geom_drugs_revisited(source, target, progress=False, **kwargs)
    return target, counts


def read_rows(target):
    with connect(target / "geom_drugs.db") as db:
        return [
            (row.smiles, row.mol_id, row.conf_rank, row.numbers.tolist(), row.positions)
            for row in db.select()
        ]


def test_the_cleaning_script_is_the_pinned_copy():
    check_upstream()
    assert UPSTREAM["sha256"] == hashlib.sha256(
        Path(geom_revisited.upstream.__file__).read_bytes()
    ).hexdigest()


def test_molecules_are_loaded_as_bytes_and_cleaned_like_the_molecules_themselves(source):
    path = source / "train_data.pickle"
    with open(path, "rb") as file:
        eager = pickle.load(file)  # the molecules the pickle holds, built by RDKit
    lazy = load_records(path)

    assert len(eager) == len(lazy) == 6
    assert all(isinstance(mol, RawMol) for _, mols in lazy for mol in mols)
    for (smiles, mols), (lazy_smiles, raw) in zip(eager, lazy):
        assert smiles == lazy_smiles
        (conformers, reason), (lazy_conformers, lazy_reason) = (
            clean_molecule(smiles, mols), clean_molecule(lazy_smiles, raw)
        )
        assert reason == lazy_reason
        assert (conformers is None) == (lazy_conformers is None)
        for (numbers, positions), (lazy_numbers, lazy_positions) in zip(conformers or [], lazy_conformers or []):
            assert np.array_equal(numbers, lazy_numbers) and np.array_equal(positions, lazy_positions)


def test_molecules_and_conformers_are_cleaned_as_the_script_does(source, tmp_path):
    target, counts = build(source, tmp_path)

    train = counts["cleaning"]["train"]
    assert train["molecules_read"] == 6
    assert train["molecules_kept"] == 4
    assert train["molecules_removed_smiles"] == 1
    assert train["molecules_removed_topology"] == 1
    assert train["molecules_removed_no_conformer"] == 0
    assert train["conformers_kept"] == 7 + 3 + 6 + 3
    val = counts["cleaning"]["val"]
    assert val["molecules_kept"] == 1 and val["molecules_removed_no_conformer"] == 1
    assert counts["cleaning"]["test"]["molecules_kept"] == 2

    assert counts["n_rows"] == 19 + 2 + 3
    assert (counts["n_train"], counts["n_val"], counts["n_test"]) == (19, 2, 3)
    assert (counts["n_train_molecules"], counts["n_val_molecules"], counts["n_test_molecules"]) == (4, 1, 2)
    assert counts["molecules_shared_train_test"] == 1
    assert counts["molecules_shared_train_val"] == counts["molecules_shared_val_test"] == 0
    # H C O Cl Br; the only molecule with N is removed
    assert set(counts["elements"]) == {1, 6, 8, 17, 35}


def test_a_row_is_a_conformer_with_its_molecule_and_rank(source, tmp_path, propanol):
    target, _ = build(source, tmp_path)
    rows = read_rows(target)

    assert [row[2] for row in rows[:7]] == [1, 2, 3, 4, 5, 6, 7]  # CCO
    assert {row[0] for row in rows[:7]} == {"CCO"} and {row[1] for row in rows[:7]} == {0}
    assert [row[1] for row in rows] == [0] * 7 + [1] * 3 + [2] * 6 + [3] * 3 + [4] * 2 + [5] * 1 + [6] * 2

    # the dropped conformer leaves no gap: ranks count the conformers that stayed
    propanol_rows = rows[16:19]
    assert [row[2] for row in propanol_rows] == [1, 2, 3]
    for row, original in zip(propanol_rows, (propanol[0], propanol[2], propanol[3])):
        assert row[3] == [atom.GetAtomicNum() for atom in original.GetAtoms()]
        assert np.allclose(row[4], original.GetConformer().GetPositions())

    with connect(target / "geom_drugs.db") as db:
        assert db.metadata["protocol"] == "geom_drugs_revisited"
        assert db.metadata["_distance_unit"] == "Ang"


def test_the_two_views_list_rows_of_one_database(source, tmp_path):
    target, counts = build(source, tmp_path)
    rows = read_rows(target)
    all_view = np.load(target / split_file_name(None))
    top5_view = np.load(target / "split_top5.npz")

    assert [all_view[f"{s}_idx"].tolist() for s in ("train", "val", "test")] == [
        list(range(0, 19)), [19, 20], [21, 22, 23]
    ]
    for split in ("train", "val", "test"):
        view = top5_view[f"{split}_idx"]
        assert set(view) <= set(all_view[f"{split}_idx"].tolist())
        assert all(rows[i][2] <= 5 for i in view)
        per_molecule = {}
        for i in all_view[f"{split}_idx"]:
            per_molecule.setdefault(rows[i][1], []).append(i)
        assert view.tolist() == [i for ids in per_molecule.values() for i in ids[:5]]
    assert len(top5_view["train_idx"]) == 5 + 3 + 5 + 3
    assert counts["views"]["top5"] == {
        "split_file": "split_top5.npz", "top_k": 5, "n_train": 16, "n_val": 2, "n_test": 3
    }
    assert counts["views"]["all"]["n_train"] == 19
    # there is no split.npz that a config for another GEOM-Drugs folder could pick up
    assert sorted(path.name for path in target.iterdir()) == [
        "geom_drugs.db", "split_all.npz", "split_top5.npz"
    ]


def test_top_k_names_the_second_split_file(source, tmp_path):
    target, counts = build(source, tmp_path, top_k=2)
    assert (target / "split_top2.npz").exists() and not (target / "split_top5.npz").exists()
    assert counts["views"]["top2"]["n_train"] == 2 + 2 + 2 + 2


def test_the_build_does_not_depend_on_the_number_of_workers(source, tmp_path):
    one, _ = build(source, tmp_path, name="one", n_workers=1)
    two, _ = build(source, tmp_path, name="two", n_workers=2)
    rows_one, rows_two = read_rows(one), read_rows(two)
    assert len(rows_one) == len(rows_two)
    for a, b in zip(rows_one, rows_two):
        assert a[:4] == b[:4] and np.array_equal(a[4], b[4])
    for name in ("split_all.npz", "split_top5.npz"):
        for key in ("train_idx", "val_idx", "test_idx"):
            assert np.array_equal(np.load(one / name)[key], np.load(two / name)[key])


def test_max_molecules_reads_the_first_molecules_of_each_split(source, tmp_path):
    _, counts = build(source, tmp_path, max_molecules=1)
    assert counts["cleaning"]["train"]["molecules_read"] == 1
    assert counts["n_train"] == 7 and counts["n_val"] == 2 and counts["n_test"] == 1


def test_files_of_another_md5_are_not_unpickled(source, tmp_path):
    known = {
        name: file_digests(source / name)["md5"]
        for name in ("train_data.pickle", "val_data.pickle", "test_data.pickle")
    }
    build(source, tmp_path, known_md5=known)  # the right files are accepted

    known["val_data.pickle"] = "0" * 32
    with pytest.raises(ValueError, match="refusing to unpickle"):
        build(source, tmp_path, name="other", known_md5=known)


def test_a_pickle_that_runs_code_is_refused(tmp_path):
    class Evil:
        def __reduce__(self):
            return (os.system, ("echo should never run",))

    path = tmp_path / "evil.pickle"
    with open(path, "wb") as file:
        pickle.dump([("CCO", [Evil()])], file)
    with pytest.raises(pickle.UnpicklingError, match="refusing to unpickle"):
        load_records(path)


def test_an_existing_database_is_not_overwritten(source, tmp_path):
    target, _ = build(source, tmp_path)
    with pytest.raises(FileExistsError):
        build_geom_drugs_revisited(source, target, progress=False)


@pytest.mark.parametrize("view, sizes", [("split_all.npz", (19, 2, 3)), ("split_top5.npz", (16, 2, 3))])
def test_the_data_module_reads_both_views(source, tmp_path, view, sizes):
    target, _ = build(source, tmp_path)
    module = PreparedAtomsData(
        datapath=str(target / "geom_drugs.db"),
        batch_size=2,
        split_file=str(target / view),
        load_properties=[],
        num_workers=0,
    )
    module.prepare_data()
    module.setup()
    assert (len(module.train_dataset), len(module.val_dataset), len(module.test_dataset)) == sizes


def test_the_split_file_of_geom_drugs_does_not_exist_in_the_new_folder(source, tmp_path):
    """The config of the other GEOM-Drugs folders reads split.npz: here it must fail,
    not make a random split."""
    target, _ = build(source, tmp_path)
    module = PreparedAtomsData(
        datapath=str(target / "geom_drugs.db"),
        batch_size=2,
        split_file=str(target / "split.npz"),  # what data=geom_drugs asks for
        num_train=None,
        num_val=None,
        load_properties=[],
        num_workers=0,
    )
    module.prepare_data()
    with pytest.raises(AtomsDataModuleError):
        module.setup()
    assert not (target / "split.npz").exists()


def test_the_command_writes_a_manifest_and_names_the_folder_by_its_files(source, tmp_path):
    root = tmp_path / "geom_drugs"
    assert prepare_geom_revisited.main(
        ["--source_dir", str(source), "--data_root", str(root), "--allow_unknown_source", "--quiet"]
    ) == 0
    (folder,) = [path for path in root.iterdir()]
    manifest = json.loads((folder / "manifest.json").read_text())

    assert len(folder.name) == 16 and manifest["dataset_id"] == folder.name
    assert manifest["dataset"] == "geom_drugs" and manifest["protocol"] == "geom_drugs_revisited"
    assert manifest["settings"]["heavy_atoms"] is False and manifest["settings"]["top_k"] == 5
    assert set(manifest["files"]) == {"geom_drugs.db", "split_all.npz", "split_top5.npz"}
    for name, entry in manifest["files"].items():
        assert entry["sha256"] == file_digests(folder / name)["sha256"]
    assert manifest["cleaning"]["script"]["commit"] == UPSTREAM["commit"]
    assert manifest["cleaning"]["statistics"]["train"]["molecules_kept"] == 4
    assert set(manifest["source"]["files"]) == {
        "train_data.pickle", "val_data.pickle", "test_data.pickle"
    }
    assert manifest["counts"]["n_rows"] == 24
    expected = hashlib.sha256(
        "".join(f"{e['sha256']}  {n}\n" for n, e in manifest["files"].items()).encode()
    ).hexdigest()[:16]
    assert expected == folder.name


def test_the_command_refuses_files_that_are_not_the_zenodo_ones_and_leaves_nothing(source, tmp_path):
    root = tmp_path / "geom_drugs"
    with pytest.raises(ValueError, match="refusing to unpickle"):
        prepare_geom_revisited.main(["--source_dir", str(source), "--data_root", str(root), "--quiet"])
    assert list(root.iterdir()) == []


@pytest.mark.parametrize(
    "name, split_file",
    [("geom_drugs_revisited_all", "split_all.npz"), ("geom_drugs_revisited_top5", "split_top5.npz")],
)
def test_the_revisited_configs_differ_from_geom_drugs_only_in_the_split_file(name, split_file):
    def load(config):
        return OmegaConf.to_container(
            OmegaConf.load(CONFIG_DIR / "data" / f"{config}.yaml"), resolve=False
        )

    config, base = load(name), load("geom_drugs")
    assert config.pop("split_file") == f"${{run.data_dir}}/{split_file}"
    assert base.pop("split_file") == "${run.data_dir}/split.npz"
    assert config == base


@pytest.mark.parametrize(
    "name, split_file",
    [("geom_drugs_revisited_all", "split_all.npz"), ("geom_drugs_revisited_top5", "split_top5.npz")],
)
def test_the_revisited_configs_compose_and_build_the_data_module(name, split_file):
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        config = compose(
            "train",
            overrides=["experiment=gpff", f"data={name}", "run.path=/runs", "run.data_dir=/data"],
        )
    assert config.data.split_file == f"/data/{split_file}"
    assert config.data.datapath == "/data/geom_drugs.db"
    assert config.data.num_train is None and config.data.num_val is None
    hydra.utils.instantiate(config.data, train_sampler_cls=None)
