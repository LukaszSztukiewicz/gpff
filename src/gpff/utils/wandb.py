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


def load_last_state_dict(run_dir: str, map_location: str = "cpu") -> Optional[dict]:
    """State dict (EMA weights, ``model.`` prefix stripped) of a run's newest last*.ckpt.

    Picks the newest ``last*.ckpt`` like Lightning's ``ckpt_path="last"``, so a
    ``last-v1.ckpt`` written after a moved run is resumed is found too. Returns
    ``None`` when the run has no last checkpoint.
    """
    ckpts = list(Path(run_dir, "checkpoints").glob("last*.ckpt"))
    if not ckpts:
        return None
    ckpt = torch.load(
        max(ckpts, key=lambda p: p.stat().st_mtime),
        map_location=map_location,
        weights_only=False,
    )
    return {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
