from types import SimpleNamespace

import pytorch_lightning.callbacks as pl_callbacks
import torch

from gpff.callbacks import ModelCheckpoint


def test_best_model_is_saved_once_per_improvement(monkeypatch, tmp_path):
    # stand in for Lightning's checkpoint bookkeeping: every call is a new best
    def update_best(self, current, trainer, monitor_candidates):
        self.best_model_score = current

    monkeypatch.setattr(pl_callbacks.ModelCheckpoint, "_update_best_and_save", update_best)

    saved = []
    checkpoint = ModelCheckpoint(model_path="best_model", monitor="val/loss", dirpath=str(tmp_path))
    checkpoint.trainer = SimpleNamespace(strategy=SimpleNamespace(local_rank=0))
    checkpoint.task = SimpleNamespace(save_model=lambda path, do_postprocessing: saved.append(path))

    checkpoint._update_best_and_save(torch.tensor(0.5), checkpoint.trainer, {})

    assert saved == ["best_model"]
