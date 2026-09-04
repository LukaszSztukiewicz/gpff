import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from typing import Optional
import os
from pathlib import Path
from schnetpack.utils import load_model
import numpy as np
import torch


__all__ = [
    "upload_artifact",
    "load_latest_model",
    "load_split",
    "load_last_state_dict",
]


def upload_artifact(run_id: str, artifact_type: str, artifact_description: str, artifact_file: str, trainer: Optional[pl.Trainer] = None):
        """Upload an artifact to W&B. No-op when no WandbLogger is attached to the trainer."""
        wandb_logger = None
        if trainer is not None:
            logger = trainer.logger
            if isinstance(logger, WandbLogger):
                wandb_logger = logger
            else:
                try:
                    for lg in logger:
                        if isinstance(lg, WandbLogger):
                            wandb_logger = lg
                            break
                except TypeError:
                    pass  # logger is not iterable

        if wandb_logger is None:
            return

        artifact = wandb.Artifact(
            name=f"{run_id}.{artifact_type}",
            type=artifact_type,
            description=artifact_description,
        )
        artifact.add_file(artifact_file)
        wandb_logger.experiment.log_artifact(artifact)

def load_latest_model(entity: str, project: str, run_id: str, device: str = "cpu"):
    """Load a model from W&B."""
    # download artifact
    api = wandb.Api()
    artifact = api.artifact(f"{entity}/{project}/{run_id}.model:latest", type="model")
    artifact_dir = artifact.download()

    # load model
    model_path = os.path.join(artifact_dir, "best_model")
    model = load_model(model_path, device=device)

    return model

def load_split(entity: str, project: str, run_id: str, return_path: bool = False):
    """Load a model from W&B."""
    # download artifact
    api = wandb.Api()
    artifact = api.artifact(f"{entity}/{project}/{run_id}.split:latest", type="split")
    artifact_dir = artifact.download()

    # load split
    split_path = os.path.join(artifact_dir, "split.npz")

    if return_path:
        return split_path
    else:
        return np.load(split_path)


def load_last_state_dict(
    wandb_project: str,
    run_id: str,
    base_dir: str = "experiments/runs",
    map_location: str = "cpu",
) -> Optional[dict]:
    """Load the state dict of a run's *last* checkpoint from the local run tree.

    Unlike :func:`load_latest_model`, which fetches the best-model artifact from
    W&B, this searches the local Hydra output tree
    ``<base_dir>/<wandb_project>/<date>/<run_id>/checkpoints/last.ckpt``. Returns
    ``None`` when no such checkpoint exists.

    TODO: Find clean solution for saving or getting last checkpoint.
    """
    base_path = Path(base_dir) / wandb_project

    if not base_path.exists():
        return None

    # Search through all date folders
    for date_folder in base_path.iterdir():
        if not date_folder.is_dir():
            continue

        # Check if the experiment exists in this date folder
        experiment_path = date_folder / run_id
        if experiment_path.exists() and experiment_path.is_dir():
            ckpt_path = experiment_path / "checkpoints" / "last.ckpt"
            if ckpt_path.exists():
                state_dict = torch.load(
                    ckpt_path, map_location=map_location, weights_only=False
                )["state_dict"]
                # Remove "model." prefix from all keys
                new_state_dict = {
                    k.replace("model.", "", 1): v
                    for k, v in state_dict.items()
                    if k.startswith("model.")
                }
                # Include non-prefixed keys as-is
                for k, v in state_dict.items():
                    if not k.startswith("model."):
                        new_state_dict[k] = v
                return new_state_dict

    return None