"""Sampling entry point (``gpff_generate``)."""
import logging
import os
import socket
from typing import Any, Callable, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
from ase.db import connect
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from schnetpack.utils import load_model
from schnetpack.utils.script import print_config

from gpff import keys
from gpff.analysis import get_rdkit_validity
from gpff.cli import header
from gpff.data.datasets.qm9 import QM9Valid
from gpff.sample import Calculator, ConditionGenerator, Prior, Sampler
from gpff.sample.observers import DeferredWork, TrajectoryRecorder
from gpff.sample.results import batches_to_relaxed, batches_to_trajectories

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


def _store_samples(
    output_db: str,
    samples: List[Tuple[Any, Dict[str, Any]]],
    n_samples_written: int,
) -> List[Callable[[], None]]:
    """
    Appending a batch's samples to the database and logging its progress, as
    small steps for DeferredWork: the writes first, then one RDKit validity
    check per molecule, then the log line.
    """
    validity = []

    def write():
        with connect(output_db) as db:
            for atoms, data in samples:
                db.write(atoms, data=data)

    def check(atoms):
        return lambda: validity.append(get_rdkit_validity(atoms))

    def log_progress():
        # validity is a progress signal only, gpff_prepare_metadata
        # recomputes it for the metrics
        nfe = float(np.mean([data[keys.nfe] for _, data in samples]))
        log_str = (
            f"| n_samples: {n_samples_written} --- avg nfe: {nfe} --- "
            f"batch validity: {float(np.mean(validity))} |"
        )
        print("-"*len(log_str))
        print(log_str)
        print("-"*len(log_str))

    return [write] + [check(atoms) for atoms, _ in samples] + [log_progress]


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
    if config.get("compile") is not None:
        log.info(f"Compiling the model with <{config.compile}>")
        model.compile(**OmegaConf.to_container(config.compile))

    # load split; it has to match the model, so it is required as well
    split_path = OmegaConf.select(config, "split_path")
    if not split_path:
        raise ValueError(
            "split_path is required: pass the split file of the run that trained the model."
        )
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"No split file at {split_path}.")
    log.info(f"Using split file: {split_path}")

    qm9_valid = QM9Valid(
        datapath=config.datapath,
        batch_size=config.batch_size,
        split_file=split_path,
        load_properties=[],
        num_workers=config.num_workers,
        data_workdir=config.data_workdir,
    )
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

    # initialize prior
    prior: Prior = hydra.utils.instantiate(config.prior)

    # optional condition generator, building the conditions dict shared by
    # prior and sampler
    condition_generator_config = OmegaConf.select(config, "condition_generator")
    condition_generator: Optional[ConditionGenerator] = None
    if condition_generator_config is not None:
        overrides = {}
        if OmegaConf.select(condition_generator_config, "shape_predictor") is not None:
            overrides["shape_predictor"] = {"train_dataset": qm9_valid._train_dataset}
        condition_generator = hydra.utils.instantiate(condition_generator_config, **overrides)

    if config.n_samples > len(qm9_valid._test_dataset):
        raise ValueError(
            f"Requested {config.n_samples} samples, but test split only contains "
            f"{len(qm9_valid._test_dataset)} structures."
        )

    # create samples directory and initialize database
    run_metadata = {"config": OmegaConf.to_container(config, resolve=True)}
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

    # run sampling loop. Each batch's samples are written and its progress
    # logged while the next batch denoises, in the time the GPU takes for its
    # steps; the queue is flushed when the loop ends, also on an error.
    n_samples_written = 0
    trajectory_idx = 0
    test_loader = qm9_valid.test_dataloader()
    deferred = DeferredWork()
    try:
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

            # denoise and process; the deferred work goes first, so that it runs
            # before the recorder copies the batch (and waits for the GPU)
            recorder = TrajectoryRecorder() if config.store_trajectories else None
            final_batch = sampler.denoise(
                batch,
                conditions=conditions,
                observers=[deferred] + ([recorder] if recorder is not None else []),
            )
            deferred.finish()
            relaxed_atoms = batches_to_relaxed([final_batch])
            if config.store_trajectories:
                atoms_trajectory = batches_to_trajectories(recorder.snapshots)

            n_to_store = min(len(relaxed_atoms), config.n_samples - n_samples_written)
            relaxed_atoms_to_store = [
                (atoms, {keys.nfe: int(data[keys.nfe])})
                for atoms, data in relaxed_atoms[:n_to_store]
            ]

            # save relaxed atoms to database and log progress
            deferred.add(
                *_store_samples(config.output_db, relaxed_atoms_to_store, n_samples_written + n_to_store)
            )

            # save trajectories to database
            if config.store_trajectories:
                trajectories_to_store = atoms_trajectory[:n_to_store]
                with connect(trajectory_db_path) as db:
                    # Write full trajectory i before full trajectory i+1.
                    for trajectory in trajectories_to_store:
                        # Unique id for this trajectory across the entire db.
                        for atoms, data in trajectory:
                            db.write(
                                atoms,
                                data={**data, "trajectory_idx": trajectory_idx},
                            )
                        trajectory_idx += 1

            n_samples_written += n_to_store
    finally:
        deferred.finish()

    # log final sample count
    print(f"+ final samples written: {n_samples_written} +")
