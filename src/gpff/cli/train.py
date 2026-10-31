"""Training entry point (``gpff_train``)."""
import logging
import os
import socket
from typing import List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.loggers.logger import Logger

import schnetpack as spk
from schnetpack.utils import str2class
from schnetpack.utils.script import log_hyperparameters, print_config

from gpff.cli import header

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="train")
def train(config: DictConfig):
    """
    General training routine for all models defined by the provided hydra configs.

    """
    print(header)
    log.info("Running on host: " + str(socket.gethostname()))

    if OmegaConf.is_missing(config, "run.data_dir"):
        log.error(
            f"Config incomplete! You need to specify the data directory `data_dir`."
        )
        return

    if not ("model" in config and "data" in config):
        log.error(
            f"""
        Config incomplete! You have to specify at least `data` and `model`!
        For an example, try one of our pre-defined experiments:
        > gpff_train experiment=gpff
        """
        )
        return

    # Set matmul precision if specified
    if "matmul_precision" in config and config.matmul_precision is not None:
        log.info(f"Setting float32 matmul precision to <{config.matmul_precision}>")
        torch.set_float32_matmul_precision(config.matmul_precision)

    # Set seed for random number generators in pytorch, numpy and python.random
    log.info(f"Seed with <{config.seed}>")
    seed_everything(seed=config.seed, workers=True)

    if config.get("print_config"):
        print_config(config, resolve=True)

    if not os.path.exists(config.run.data_dir):
        os.makedirs(config.run.data_dir)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.data,
        train_sampler_cls=(
            str2class(config.data.train_sampler_cls)
            if config.data.train_sampler_cls
            else None
        ),
    )

    # Init model
    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)
    if config.get("init_weights"):
        if config.run.ckpt_path not in (None, ""):
            raise ValueError("init_weights requires run.ckpt_path=null (fresh matched optimizer restart)")
        from gpff.model.initialization import initialize_weights
        log.info("Weights initialization: %s", initialize_weights(
            model, config.init_weights, allow_sigma=bool(config.get("init_allow_sigma", False))))
    if config.get("compile") is not None:
        log.info(f"Compiling the model with <{config.compile}>")
        # in place, so state_dict keys and the pickled model keep their names
        model.compile(**OmegaConf.to_container(config.compile))

    # Init LightningModule
    log.info(f"Instantiating task <{config.task._target_}>")
    scheduler_cls = (
        str2class(config.task.scheduler_cls) if config.task.scheduler_cls else None
    )

    task: spk.AtomisticTask = hydra.utils.instantiate(
        config.task,
        model=model,
        optimizer_cls=str2class(config.task.optimizer_cls),
        scheduler_cls=scheduler_cls,
    )

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[Logger] = []

    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                l = hydra.utils.instantiate(lg_conf)
                logger.append(l)

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=os.path.join(config.run.id),
        _convert_="partial",
    )

    # log.info("Logging hyperparameters.")
    log_hyperparameters(config=config, model=task, trainer=trainer)

    # Our checkpoints pickle schnetpack model objects (task hyperparameters), so
    # every Lightning load below needs weights_only=False; Lightning >= 2.6
    # otherwise falls back to torch's weights_only=True default.

    # Train the model
    log.info("Starting training.")
    if config.get("init_weights"):
        # New module initialization must not change the matched data/RNG draws.
        seed_everything(seed=config.seed, workers=True)
    trainer.fit(model=task, datamodule=datamodule, ckpt_path=config.run.ckpt_path, weights_only=False)

    # Evaluate model on test set after training
    log.info("Starting testing.")
    trainer.test(model=task, datamodule=datamodule, ckpt_path="best", weights_only=False)

    # Store best model
    best_path = trainer.checkpoint_callback.best_model_path
    log.info(f"Best checkpoint path:\n{best_path}")

    log.info(f"Store best model")
    best_task = type(task).load_from_checkpoint(best_path, weights_only=False)

    best_task.save_model(config.globals.model_path, do_postprocessing=True)
    log.info(f"Best model stored at {os.path.abspath(config.globals.model_path)}")
