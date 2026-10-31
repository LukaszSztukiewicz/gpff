#!/bin/bash
# Source on Hydra. Shared immutable inputs and training choices.
G=/home/lsztuk/gpff
SIF=$G/containers/gpff-py313-torch214-gcc.sif
DATASET_PATH=$G/data/qm9/b36d9cd218f5998d/qm9_filtered.db
REFERENCE=$G/runs/default/26-10-28/proven-ox-999
SPLIT=$REFERENCE/frozen/split.npz
export SIF DATASET_PATH SPLIT REFERENCE
TRAIN_ARGS=(experiment=gpff seed=999 trainer.max_epochs=100000 globals.lr=1e-4
  data.batch_size=512 data.load_properties=[] data.cache_in_memory=true
  data.persistent_workers=true callbacks.deterministic_data_seed.per_sample_train_noise=true
  'compile={dynamic:true}' trainer.deterministic=true +callbacks=milestones
  'callbacks.milestone_checkpoint.epochs=[100,200,500,1000,2000,5000,10000,12000,12560,15000,20000,30000,40000,50000]'
  "+data.split_file=$SPLIT")
