"""Actual Lightning training with the GPFF transform, loss, metrics and EMA.
The 512 read-only QM9 structures are cached in memory and repeated to avoid
database and split writes, while preserving molecule-size and worker costs.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from schnetpack import properties as P
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.transform import CastTo32
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from gpff import keys as K
from gpff.data.transform import PseudoPESTransform
from gpff.pseudo_pes import VarianceExplodingGeometries, VEPseudoCalculator, SigmaWeights
from gpff.utils.neighbors import AllToAllNeighborList
from gpff.model.painn import PaiNN
from gpff.task import AtomisticTask, ModelOutput
import audit


class Molecules(Dataset):
    def __init__(self,molecules,length):
        self.molecules=molecules
        self.length=length
        self.transform=PseudoPESTransform(
            geometry_sampler=VarianceExplodingGeometries(sigma_max=30.,mean_free_noise=True),
            energy_function=VEPseudoCalculator(),weight_generator=SigmaWeights(min_clipping=0.,max_clipping=1.),
            align_permutation=True,align_rotation=False)
        self.neighbors=AllToAllNeighborList()
        self.cast=CastTo32()
    def __len__(self): return self.length
    def __getitem__(self,i):
        torch.manual_seed(999+i)
        return self.cast(self.neighbors(self.transform(audit.clone(self.molecules[i%len(self.molecules)]))))


class Timing(pl.Callback):
    def __init__(self):
        self.values=[]
        self.start=None
    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        if batch_idx==19 or (batch_idx>=29 and (batch_idx-19)%10==0):
            audit.sync()
            now=time.perf_counter()
            if self.start is not None:
                self.values.append((now-self.start)*1000/10)
            self.start=now


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--width',type=int,default=256)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--compile',action='store_true')
    p.add_argument('--deterministic',action='store_true')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    audit.setup(a)
    pl.seed_everything(999,workers=True)
    molecules,_=audit.data(512)
    dataset=Molecules(molecules,a.batch*80)
    loader=DataLoader(dataset,batch_size=a.batch,num_workers=a.workers,
        collate_fn=_atoms_collate_fn,pin_memory=True,persistent_workers=a.workers>0)
    net=audit.model(PaiNN,a.width)
    if a.compile:
        net.compile(dynamic=True)
    task=AtomisticTask(model=net,outputs=[ModelOutput(K.forces,
        metrics={'mae':MeanAbsoluteError(),'mse':MeanSquaredError()},weights_key=K.forces_weight)],
        optimizer_cls=torch.optim.AdamW,optimizer_args={'lr':1e-4,'weight_decay':0.})
    try:
        from gpff.callbacks import ExponentialMovingAverage
    except ImportError:
        from schnetpack.train.callbacks import ExponentialMovingAverage
    timer=Timing()
    trainer=pl.Trainer(accelerator='gpu',devices=1,max_epochs=1,max_steps=80,
        logger=False,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,
        num_sanity_val_steps=0,limit_val_batches=0,callbacks=[ExponentialMovingAverage(decay=.995),timer],
        deterministic=a.deterministic,
        default_root_dir=str(audit.ROOT/'lightning-audit'))
    trainer.fit(task,train_dataloaders=loader)
    audit.emit({'test':'lightning/train_pipeline','samples_ms':timer.values,
        'median_ms':float(torch.tensor(timer.values).median()),'batch':a.batch,'width':a.width,
        'workers':a.workers,'compiled':a.compile,'skipped':task.n_skipped_train_batches})
    Path(a.out).write_text(json.dumps(audit.ROWS,indent=2)+'\n')


if __name__=='__main__': main()
