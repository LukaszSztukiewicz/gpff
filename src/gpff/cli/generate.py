"""Sampling entry point (``gpff_generate``)."""
import logging
import os
import socket
from typing import Optional

import hydra
import numpy as np
import torch
from ase.db import connect
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from schnetpack.utils import load_model
from schnetpack import properties as P
from schnetpack.utils.script import print_config

from gpff import keys
from gpff.analysis import get_rdkit_validity
from gpff.cli import header
from gpff.data.datasets.qm9 import QM9Valid
from gpff.sample import Calculator, ConditionGenerator, Prior, Sampler
from gpff.sample.observers import TrajectoryRecorder
from gpff.sample.results import batches_to_relaxed, batches_to_trajectories
from gpff.sample.formula import FormulaPrior

log = logging.getLogger(__name__)


def _get_trajectory_db_path(output_db: str) -> str:
    """
    Derive the trajectory database path from the sample database path.

    Uses ``samples -> trajectories`` replacement when possible, otherwise adds
    a ``_trajectories`` suffix before the extension.
    """
    directory, filename = os.path.split(output_db)
    if "samples" in filename:
        return os.path.join(directory, filename.replace("samples", "trajectories", 1))
    base, extension = os.path.splitext(output_db)
    if extension == "":
        return f"{output_db}_trajectories.db"
    return f"{base}_trajectories{extension}"


@hydra.main(config_path="../configs", config_name="generate")
def sample_molecules(config: DictConfig):
    """
    Sample molecules from the model.
    """
    print(header)
    log.info("Running on host: " + str(socket.gethostname()))

    # print config
    print_config(
        config,
        fields=[
            "prior",
            "condition_generator",
            "sampler",
            "model_path",
            "split_path",
        ],
        resolve=True,
    )

    # Set matmul precision if specified
    if "matmul_precision" in config and config.matmul_precision is not None:
        log.info(f"Setting float32 matmul precision to <{config.matmul_precision}>")
        torch.set_float32_matmul_precision(config.matmul_precision)

    # seed torch, numpy and python.random: priors, stochastic samplers and the
    # shape GMM fit all draw from the global RNGs
    log.info(f"Seed with <{config.seed}>")
    seed_everything(seed=config.seed, workers=True)

    # load model; the checkpoint always comes from a local file, there is no
    # remote lookup that could substitute a different model
    model_path = OmegaConf.select(config, "model_path")
    if not model_path:
        raise ValueError("model_path is required: pass the path of a local checkpoint.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model checkpoint at {model_path}.")
    log.info(f"Loading model from: {model_path}")
    model = load_model(model_path, device="cpu")

    # load split; it has to match the model, so it is required as well
    split_path = OmegaConf.select(config, "split_path")
    if not split_path:
        raise ValueError(
            "split_path is required: pass the split file of the run that trained the model."
        )
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"No split file at {split_path}.")
    log.info(f"Using split file: {split_path}")

    prior: Prior = hydra.utils.instantiate(config.prior)
    formula_prior = isinstance(prior, FormulaPrior)
    if formula_prior:
        prior.validate_source(config.datapath, split_path)
        if OmegaConf.select(config, "condition_generator") is not None:
            raise ValueError("Formula generation requires condition_generator=null")
    joint_formula = formula_prior and prior.mode == "joint"

    qm9_valid = None if joint_formula else QM9Valid(
        datapath=config.datapath,
        batch_size=config.batch_size,
        split_file=split_path,
        load_properties=[],
        num_workers=config.num_workers,
        data_workdir=config.data_workdir,
    )
    if qm9_valid is not None:
        qm9_valid.prepare_data()
        qm9_valid.setup()

    # initialize sampler; the calculator is built first so it can take the
    # model from the loaded checkpoint
    calculator: Calculator = hydra.utils.instantiate(
        config.sampler.calculator,
        model=model,
    )
    sampler: Sampler = hydra.utils.instantiate(
        config.sampler,
        calculator=calculator,
    )

    # optional condition generator, building the conditions dict shared by
    # prior and sampler
    condition_generator_config = OmegaConf.select(config, "condition_generator")
    condition_generator: Optional[ConditionGenerator] = None
    if condition_generator_config is not None:
        overrides = {}
        if OmegaConf.select(condition_generator_config, "shape_predictor") is not None:
            overrides["shape_predictor"] = {"train_dataset": qm9_valid._train_dataset}
        condition_generator = hydra.utils.instantiate(condition_generator_config, **overrides)

    if qm9_valid is not None and config.n_samples > len(qm9_valid._test_dataset):
        raise ValueError(
            f"Requested {config.n_samples} samples, but test split only contains "
            f"{len(qm9_valid._test_dataset)} structures."
        )

    # create samples directory and initialize database
    run_metadata = {"config": OmegaConf.to_container(config, resolve=True)}
    if formula_prior:
        run_metadata["formula_prior"] = prior.payload
        if os.path.exists(config.output_db):
            raise FileExistsError(config.output_db)
    if os.path.exists(config.output_db):
        os.remove(config.output_db)
    os.makedirs(os.path.dirname(config.output_db), exist_ok=True)
    with connect(config.output_db) as db:
        db.metadata = run_metadata

    trajectory_db_path = _get_trajectory_db_path(config.output_db)
    if config.store_trajectories:
        if os.path.exists(trajectory_db_path):
            os.remove(trajectory_db_path)
        with connect(trajectory_db_path) as db:
            db.metadata = run_metadata

    # run sampling loop
    n_samples_written = 0
    trajectory_idx = 0
    # In joint mode only the requested number of slots is supplied; no held-out
    # molecule, size, geometry or identity participates in generation.
    test_loader = (
        ({P.n_atoms: torch.zeros(min(config.batch_size, config.n_samples - start), dtype=torch.long)}
         for start in range(0, config.n_samples, config.batch_size))
        if joint_formula else qm9_valid.test_dataloader()
    )
    for batch in test_loader:
        if n_samples_written >= config.n_samples:
            break

        # prior and sampler share the same conditions, e.g. one target variance
        # vector per molecule for shape-prescribing runs; conditions are built
        # from the original data positions, before the prior overwrites them
        conditions = (
            condition_generator.from_batch(batch) if condition_generator is not None else {}
        )
        batch = prior.sample_from_batch(batch, conditions=conditions)
        if formula_prior:
            batch = calculator.update_neighbors(calculator.cast_batch(batch))

        # denoise and process
        recorder = TrajectoryRecorder() if config.store_trajectories else None
        final_batch = sampler.denoise(
            batch,
            conditions=conditions,
            observers=[recorder] if recorder is not None else None,
        )
        if formula_prior:
            prior.verify(final_batch)
        relaxed_atoms = batches_to_relaxed([final_batch])
        if config.store_trajectories:
            atoms_trajectory = batches_to_trajectories(recorder.snapshots)

        n_to_store = min(len(relaxed_atoms), config.n_samples - n_samples_written)
        relaxed_atoms_to_store = [
            (atoms, {keys.nfe: int(data[keys.nfe])})
            for atoms, data in relaxed_atoms[:n_to_store]
        ]
        if formula_prior:
            for i, (_, data) in enumerate(relaxed_atoms_to_store):
                data["requested_formula"] = final_batch["requested_formula"][i].tolist()

        # log intermediate sampling progress; validity is a progress signal only,
        # gpff_prepare_metadata recomputes it for the metrics
        nfe = float(np.mean([data[keys.nfe] for _, data in relaxed_atoms_to_store]))
        validity = float(
            np.mean([get_rdkit_validity(atoms) for atoms, _ in relaxed_atoms_to_store])
        )
        log_str = (
            f"| n_samples: {n_samples_written + n_to_store} --- avg nfe: {nfe} --- "
            f"batch validity: {validity} |"
        )
        print("-"*len(log_str))
        print(log_str)
        print("-"*len(log_str))

        # save relaxed atoms to database
        with connect(config.output_db) as db:
            for atoms, data in relaxed_atoms_to_store:
                db.write(atoms, data=data)

        # save trajectories to database
        if config.store_trajectories:
            trajectories_to_store = atoms_trajectory[:n_to_store]
            with connect(trajectory_db_path) as db:
                # Write full trajectory i before full trajectory i+1.
                for trajectory in trajectories_to_store:
                    # Unique id for this trajectory across the entire db.
                    for atoms, data in trajectory:
                        filtered_data = {
                            k: v
                            for k, v in data.items()
                            if k not in ("scalar_representation", "vector_representation")
                        }
                        db.write(
                            atoms,
                            data={**filtered_data, "trajectory_idx": trajectory_idx},
                        )
                    trajectory_idx += 1

        n_samples_written += n_to_store

    # log final sample count
    print(f"+ final samples written: {n_samples_written} +")
