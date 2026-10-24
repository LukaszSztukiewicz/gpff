import subprocess
import sys

import numpy as np
import schnetpack as spk
import torch
from ase import Atoms
from ase.db import connect

from gpff.model import GatedEquivariant, PaiNN


def test_generate_writes_samples(tmp_path):
    # a stand-in for the filtered QM9 database, its split and a trained model
    rng = np.random.default_rng(0)
    with connect(tmp_path / "data.db") as db:
        for _ in range(6):
            db.write(Atoms("CH4", positions=rng.normal(size=(5, 3))))
        db.metadata = {"_property_unit_dict": {}, "_distance_unit": "Ang", "filter_flags": ["valid"]}
    np.savez(tmp_path / "split.npz", train_idx=[0, 1], val_idx=[2], test_idx=[3, 4, 5])
    representation = PaiNN(
        n_atom_basis=8,
        n_interactions=1,
        radial_basis=spk.nn.GaussianRBF(5, 5.0),
        cutoff_fn=spk.nn.CosineCutoff(5.0),
        return_vector_representation=True,
    )
    model = spk.model.NeuralNetworkPotential(
        representation=representation,
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[GatedEquivariant(n_in=8)],
    )
    torch.save(model, tmp_path / "best_model")

    overrides = [
        f"model_path={tmp_path / 'best_model'}",
        f"split_path={tmp_path / 'split.npz'}",
        f"datapath={tmp_path / 'data.db'}",
        "n_samples=3",
        "batch_size=2",
        "num_workers=0",
        "output_db=samples.db",  # no directory part
        "sampler.n_steps=3",
        "sampler.calculator.device=cpu",
    ]
    run = subprocess.run(
        [sys.executable, "-c", "from gpff.cli.generate import sample_molecules; sample_molecules()", *overrides],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert run.returncode == 0, run.stderr[-2000:]
    with connect(tmp_path / "samples.db") as db:
        assert db.count() == 3
