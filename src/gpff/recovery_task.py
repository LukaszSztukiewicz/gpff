"""Opt-in recovery task. Frozen teacher is excluded from optimizer/checkpoints."""
import json
from schnetpack.utils import load_model
from gpff.task import AtomisticTask


class RecoveryTask(AtomisticTask):
    def __init__(self, *args, recovery, teacher_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.recovery, self.teacher_path = recovery, teacher_path
        self.save_hyperparameters("recovery", "teacher_path")
        object.__setattr__(self, "_teacher", None)
        self.teacher_batch_calls = 0

    def on_train_start(self):
        super().on_train_start()
        teacher = load_model(self.teacher_path, device=self.device).eval()
        teacher.requires_grad_(False)
        object.__setattr__(self, "_teacher", teacher)

    def training_step(self, batch, batch_idx):
        batch, metrics = self.recovery(batch, self._teacher, self.current_epoch, batch_idx)
        if self.recovery.mode != "ordinary":
            self.teacher_batch_calls += 1
        if self.trainer.is_global_zero and metrics:
            with open("recovery_errors.jsonl", "a") as stream:
                stream.write(json.dumps({"epoch": self.current_epoch, "batch": batch_idx,
                    "mode": self.recovery.mode, "records": self.recovery.last_records}) + "\n")
        for name, value in metrics.items():
            self.log("recovery/" + name, value, on_step=False, on_epoch=True,
                     batch_size=len(batch["_n_atoms"]))
        return super().training_step(batch, batch_idx)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["recovery_teacher_batch_calls"] = self.teacher_batch_calls

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        self.teacher_batch_calls = checkpoint.get("recovery_teacher_batch_calls",0)
