"""Formula support, provenance and the real generation entry-point contract."""
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.db import connect
from omegaconf import OmegaConf
from schnetpack import properties as P

from gpff import keys
from gpff.sample.formula import FormulaPrior, build_formula_prior


@pytest.fixture
def formula_files(tmp_path):
    db, split, table = [tmp_path / p for p in ("molecules.db", "split.npz", "formula.json")]
    with connect(str(db)) as source:
        for z in ([1, 6], [1, 6], [1, 7, 8], [9] * 4, [8] * 5):
            source.write(Atoms(numbers=z, positions=np.zeros((len(z), 3))))
    np.savez(split, train_idx=[0, 1, 2], val_idx=[3], test_idx=[4])
    build_formula_prior(db, split, table)
    return db, split, table


def test_joint_frequencies_and_no_held_out_atoms(formula_files):
    db, split, table = formula_files
    prior = FormulaPrior(table, mode="joint", seed=7)
    prior.validate_source(db, split)
    assert prior.payload["frequencies"] == [2, 1]
    out = prior.sample_from_batch({P.n_atoms: torch.full((1000,), 100), "secret": 42})
    assert "secret" not in out and not (out[P.Z] == 9).any()
    assert set(out[P.n_atoms].tolist()) == {2, 3}
    assert abs(float((out[P.n_atoms] == 2).float().mean()) - 2 / 3) < .06
    prior.verify(out)
    out[P.Z][0] = 9
    with pytest.raises(ValueError, match="preserve"):
        prior.verify(out)


def test_source_mismatch_and_overlap_rejected(formula_files, tmp_path):
    db, split, table = formula_files
    bad = tmp_path / "bad.npz"
    np.savez(bad, train_idx=[0, 1, 2], val_idx=[2], test_idx=[4])
    with pytest.raises(ValueError, match="another dataset or training split"):
        FormulaPrior(table).validate_source(db, bad)
    with pytest.raises(ValueError, match="disjoint"):
        build_formula_prior(db, bad, tmp_path / "bad.json")


def test_conditional_size_and_batch_independent_streams(formula_files):
    _, _, table = formula_files
    prior = FormulaPrior(table, mode="conditional_size")
    out = prior.sample_from_batch({P.n_atoms: torch.tensor([2, 3])})
    assert out[P.n_atoms].tolist() == [2, 3]
    with pytest.raises(ValueError, match="No training formula"):
        prior.sample_from_batch({P.n_atoms: torch.tensor([4])})
    one, chunks = [FormulaPrior(table, mode="joint", seed=42) for _ in range(2)]
    whole = one.sample_from_batch({P.n_atoms: torch.zeros(7, dtype=torch.long)})
    parts = [chunks.sample_from_batch({P.n_atoms: torch.zeros(n, dtype=torch.long)}) for n in (3, 4)]
    for key in (P.Z, P.R, P.n_atoms, P.idx, "requested_formula"):
        torch.testing.assert_close(whole[key], torch.cat([p[key] for p in parts]))


class ZeroForce(torch.nn.Module):
    def forward(self, batch):
        return {keys.forces: torch.zeros_like(batch[P.R])}


@pytest.mark.parametrize("include_condition_generator", [True, False])
def test_joint_generation_never_loads_test_molecules(formula_files, tmp_path, monkeypatch, include_condition_generator):
    from gpff.cli import generate
    db, split, table = formula_files
    model = tmp_path / "model"
    model.touch()
    monkeypatch.setattr(generate, "load_model", lambda *a, **k: ZeroForce())
    monkeypatch.setattr(generate, "print_config", lambda *a, **k: None)
    monkeypatch.setattr(generate, "get_rdkit_validity", lambda *a, **k: False)
    def forbidden(*args, **kwargs):
        raise AssertionError("Joint formula sampling must not load held-out molecules")
    monkeypatch.setattr(generate, "QM9Valid", forbidden)
    config = OmegaConf.create(dict(
        model_path=str(model), split_path=str(split), datapath=str(db), seed=13,
        prior=dict(_target_="gpff.sample.formula.FormulaPrior", formula_path=str(table), mode="joint", seed=13),
        sampler=dict(_target_="gpff.sample.direct_denoising.DirectDenoisingSampler", n_steps=2,
                     calculator=dict(_target_="gpff.sample.base.Calculator", device="cpu")),
        condition_generator=None, n_samples=7, batch_size=3, num_workers=0,
        data_workdir=None, output_db=str(tmp_path / "output.db"), store_trajectories=False,
    ))
    if not include_condition_generator:
        del config.condition_generator  # Hydra omits a group whose default is null.
    generate.sample_molecules.__wrapped__(config)
    with connect(config.output_db) as output:
        assert output.count() == 7  # more than the test split, exact final batch
        for row in output.select():
            assert row.data["requested_formula"] == [(row.numbers == z).sum() for z in (1, 6, 7, 8, 9)]
            assert row.data[keys.nfe] == 2
    with pytest.raises(FileExistsError):
        generate.sample_molecules.__wrapped__(config)
