"""Fixed-update study checkpoints and independent displacement logs."""
import json
import hashlib
from pathlib import Path
from pytorch_lightning import Callback


class UpdateCheckpoint(Callback):
    """Save raw student weights at exact optimizer-update counts (not EMA)."""
    def __init__(self, updates=(100,500,1000,2000), dirpath="updates"):
        self.updates, self.dirpath = set(updates), Path(dirpath)
        self.saved = set()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step not in self.updates or step in self.saved:
            return
        self.dirpath.mkdir(parents=True,exist_ok=True)
        trainer.save_checkpoint(str(self.dirpath / f"step_{step}.ckpt"))
        if trainer.is_global_zero:
            pl_module.save_model(str(self.dirpath / f"step_{step}_model"), do_postprocessing=True)
            with (self.dirpath / "manifest.jsonl").open("a") as stream:
                stream.write(json.dumps({"updates":step,"epoch":trainer.current_epoch,
                    "weights":"raw_student", "skipped_batches":pl_module.n_skipped_train_batches,
                    "teacher_batch_calls":getattr(pl_module,"teacher_batch_calls",0)})+"\n")
        self.saved.add(step)


class InputAudit(Callback):
    """Hash ordinary corruption before augmentation to verify matched data draws."""
    def __init__(self, steps=(0,1,2,99,499,999,1999), path="input_audit.jsonl"):
        self.steps,self.path=set(steps),Path(path)

    def on_train_batch_start(self,trainer,pl_module,batch,batch_idx):
        if trainer.global_step not in self.steps or not trainer.is_global_zero:
            return
        from schnetpack import properties as P
        from gpff import keys
        digest=hashlib.sha256()
        for key in (P.idx,P.n_atoms,P.R,P.Z,keys.sigma_curr,keys.type_mask):
            value=batch[key].detach().contiguous().cpu()
            digest.update(key.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes())
        with self.path.open("a") as stream:
            stream.write(json.dumps({"step":trainer.global_step,"epoch":trainer.current_epoch,
                "batch":batch_idx,"sha256":digest.hexdigest(),"indices":batch[P.idx].reshape(-1).tolist()})+"\n")
