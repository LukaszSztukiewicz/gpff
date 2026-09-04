from token import OP
from typing import Optional, Dict, List, Type, Any
import logging
import os
import datetime
import torch
import numpy as np
from torch import nn as nn
from torchmetrics import Metric
from schnetpack.task import ModelOutput as SPKModelOutput
from schnetpack.task import AtomisticTask as SPKAtomisticTask
from schnetpack.task import UnsupervisedModelOutput
from schnetpack.model.base import AtomisticModel


log = logging.getLogger(__name__)


__all__ = ["ModelOutput", "AtomisticTask"]


def loss_fn(pred, target, weights=None, scale_loss: Optional[str] = None, epsilon: float = 1e-8):
    # todo: do we need to normalize with sum(weights)?
    if weights is None:
        weights = torch.ones_like(target)

    weighted_diff = weights * (pred - target)**2

    if scale_loss is None:
        pass
    elif scale_loss == "sqrt":
        weighted_diff = torch.sqrt(weighted_diff)
    elif scale_loss == "log":
        weighted_diff = torch.log(weighted_diff + 1)
    else:
        raise ValueError(f"Invalid scale_loss: {scale_loss}")

    return torch.mean(weighted_diff)


class ModelOutput(SPKModelOutput):
    r"""
    Subclass of SPKModelOutput that adds magnitude weighting of elements in the loss.
    This is for example used in score matching models to account for different magnitudes:

    .. math::

       $L(\theta;\{\sigma_{i}\}_{i=1}^L) \coloneqq &\frac{1}{L} \sum_{i=1}^{L} \sigma^2\ell(\theta,\sigma_{i}) \\$

    PLEASE NOTE:
    Please only use squared losses! The weights are multiplied with the targets and predictions and to account for the
    squared loss, we sqrt(weights). I.e. L = ||sqrt(weights)*preds - sqrt(weights)*targets||**2.

    """

    def __init__(
        self,
        name: str,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        target_property: Optional[str] = None,
        weights_key: Optional[str] = None,
        scale_loss: Optional[str] = None,
        warmup_steps: int = 0,
    ):
        r"""
        Args:
            name: name of output in results dict
            target_property: Name of target in training batch. Only required for supervised training.
                If not given, the output name is assumed to also be the target name.
            loss_fn: function to compute the loss
            loss_weight: loss weight in the composite loss: $l = w_1 l_1 + \dots + w_n l_n$
            metrics: dictionary of metrics with names as keys
            weights_key: Name of weights key in training batch.
            max_weight: Maximum weight for the loss.
        """
        super().__init__(
            name=name,
            loss_weight=loss_weight,
            metrics=metrics,
            target_property=target_property,
        )
        self.weights_key = weights_key
        self.scale_loss = scale_loss
        self.warmup_steps = warmup_steps
        self.global_step = 0

    def calculate_loss(self, pred, target):
        # calculate weighted loss
        weights = None
        if self.weights_key is not None:
            weights = target[self.weights_key]

            if self.global_step < self.warmup_steps:
                dampening = self.global_step / self.warmup_steps
                weights = (1-dampening) * torch.ones_like(weights) + dampening * weights

        loss = self.loss_weight * loss_fn(
            pred[self.name], target[self.target_property], weights, self.scale_loss
        )

        return loss


class AtomisticTask(SPKAtomisticTask):
    """
    The modified SchNetPack learning task, which ties model, loss and optimizer together.
    This adds the possibility to pass loss weights of the model_outputs to the loss function.
    """

    def __init__(
        self,
        model: AtomisticModel,
        outputs: List[SPKModelOutput],
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
        warmup_steps: int = 0,
        skip_exploding_batches: bool = True,
        debug_folder: Optional[str] = None,
        exit_after_n_skipped_train_batches: Optional[int] = None,
        loss_threshold: float = 1e3,
    ):
        """
        Args:
            model: the neural network model
            outputs: list of outputs an optional loss functions
            optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
            optimizer_args: dict of optimizer keyword arguments
            scheduler_cls: type of torch learning rate scheduler
            scheduler_args: dict of scheduler keyword arguments
            scheduler_monitor: name of metric to be observed for ReduceLROnPlateau
            warmup_steps: number of steps used to increase the learning rate from zero
              linearly to the target learning rate at the beginning of training
            skip_exploding_batches: skip exploding batches in backward pass
            debug_folder: folder to save skipped batches and model weights (if not None)
            exit_after_n_skipped_train_batches: number of skipped batches after which the training should exit
            loss_threshold: threshold for skipping exploding batches
        """
        super().__init__(
            model=model,
            outputs=outputs,
            optimizer_cls=optimizer_cls,
            optimizer_args=optimizer_args,
            scheduler_cls=scheduler_cls,
            scheduler_args=scheduler_args,
            scheduler_monitor=scheduler_monitor,
            warmup_steps=warmup_steps,
        )
        self.skip_exploding_batches = skip_exploding_batches
        self.debug_folder = debug_folder
        self.n_skipped_train_batches = 0
        self.n_skipped_val_batches = 0
        self.exit_after_n_skipped_train_batches = exit_after_n_skipped_train_batches
        self.loss_threshold = loss_threshold

    def setup(self, stage=None):
        """
        Overwrite the pytorch lightning task setup function.
        Collects postprocessors that have force_apply=True.
        """
        # call the parent atomistic task setup
        super().setup(stage=stage)

        # force some post-processing transforms during training
        forced_postprocessors = []
        for pp in self.model.postprocessors:
            if hasattr(pp, "force_apply"):
                if pp.force_apply:
                    forced_postprocessors.append(pp)
        self.model.forced_postprocessors = nn.ModuleList(forced_postprocessors)

    def predict_without_postprocessing(self, batch):
        """
        Predict without post-processing transforms.
        Note: forced post-processing transforms will still be applied.

        Args:
            batch: input batch.
        """
        tmp_postprocessors = self.model.postprocessors
        forced_postprocessors = getattr(self.model, "forced_postprocessors", nn.ModuleList())
        self.model.postprocessors = forced_postprocessors
        pred = self(batch)
        self.model.postprocessors = tmp_postprocessors

        return pred

    def _filter_targets(self, batch):
        targets = dict()
        for output in self.outputs:
            # add target for supervised loss
            if not isinstance(output, UnsupervisedModelOutput):
                targets[output.target_property] = batch[output.target_property]
            # add loss weights if available
            if hasattr(output, "weights_key") and output.weights_key is not None:
                targets[output.weights_key] = batch[output.weights_key]

        # todo: what is this? Why do we need try except here?
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        return targets

    def log_metrics(self, pred, targets, subset):
        for output in self.outputs:
            output.update_metrics(pred, targets, subset)
            for metric_name, metric in output.metrics[subset].items():
                self.log(
                    f"{subset}/{output.name}_{metric_name}",
                    metric,
                    on_step=(subset == "train"),
                    on_epoch=(subset != "train"),
                    prog_bar=False,
                )

    def training_step(self, batch, batch_idx):

        targets = self._filter_targets(batch)

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        # skip exploding batches in backward pass
        if self.skip_exploding_batches and (
            torch.isnan(loss) or torch.isinf(loss) or loss > self.loss_threshold
        ):
            self.n_skipped_train_batches += 1
            log.warning(
                f"Loss is {loss} for train batch_idx {batch_idx} and training step "
                f"{self.global_step}, training step will be skipped!"
            )

            # Save batch and model weights if debug_folder is set
            if self.debug_folder is not None:
                os.makedirs(self.debug_folder, exist_ok=True)
                # Use global_step and batch_idx for unique filenames
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_file = os.path.join(
                    self.debug_folder,
                    f"skipped_batch_step{self.global_step}_batch{batch_idx}_{timestamp}.pt",
                )
                model_file = os.path.join(
                    self.debug_folder,
                    f"model_weights_step{self.global_step}_batch{batch_idx}_{timestamp}.pt",
                )
                try:
                    torch.save(batch, batch_file)
                except Exception as e:
                    log.warning(f"Failed to save skipped batch: {e}")
                try:
                    torch.save(self.model.state_dict(), model_file)
                except Exception as e:
                    log.warning(f"Failed to save model weights: {e}")

            if (
                self.exit_after_n_skipped_train_batches is not None
                and self.n_skipped_train_batches
                >= self.exit_after_n_skipped_train_batches
            ):
                raise RuntimeError(
                    f"Too many skipped batches, stopping training after {self.exit_after_n_skipped_train_batches} skipped batches!"
                )

            self.log(
                "train/skipped_nan",
                self.n_skipped_train_batches,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

            return None

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log_metrics(pred, targets, "train")
        # log cumulative skipped batches
        self.log(
            "train/skipped_nan",
            self.n_skipped_train_batches,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        targets = self._filter_targets(batch)

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        # skip and log nan loss
        # if self.skip_exploding_batches and (
        #     torch.isnan(loss) or torch.isinf(loss) or loss > 1e10
        # ):
        #     self.n_skipped_val_batches += 1

        #     return None

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pred, targets, "val")

        # log skipped nan
        self.log(
            "val/skipped_nan",
            self.n_skipped_val_batches,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"val/loss": loss}

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        targets = self._filter_targets(batch)

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pred, targets, "test")
        return {"test/loss": loss}
