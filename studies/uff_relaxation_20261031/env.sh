# Paths shared by the UFF evaluation jobs (hydra).
GPFF=/home/lsztuk/gpff
SIF=$GPFF/containers/gpff-py313-torch214.sif
# gpff source of this study, rsynced from the feat/uff-relaxation worktree
CODE=${CODE:-$GPFF/scratch/uff_eval/code/src}
FROZEN=$GPFF/runs/default/26-10-20/fancy-monkey/frozen
MODEL=${MODEL:-$FROZEN/epoch_5000_model}
SAMPLES=$FROZEN/repro/epoch_5000_model
HERE=$GPFF/scratch/uff_eval
py() { apptainer exec --nv --env PYTHONPATH=$CODE "$SIF" python "$@"; }
# fail early if the container's editable install shadows CODE
check_code() { py -c "import gpff, sys; sys.exit(0 if gpff.__file__.startswith('$CODE') else f'gpff from {gpff.__file__}, not $CODE')"; }
# GFN2-xTB 6.7.1, official release binary (sha256-checked), one thread per run
XTB=$GPFF/tools/xtb-6.7.1/xtb-dist
py_xtb() { apptainer exec --env PYTHONPATH=$CODE,XTBPATH=$XTB/share/xtb,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1 "$SIF" python "$@" --xtb_command $XTB/bin/xtb; }
# MACE-OFF24 medium (git blob 413df80 = GitHub's), mace-torch 0.3.16 and its missing deps in their own folder
MACE_SITE=$GPFF/tools/mace/site
MACE_MODEL=$GPFF/tools/mace/models/MACE-OFF24_medium.model
py_mace() { apptainer exec --nv --env PYTHONPATH=$CODE:$MACE_SITE,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$SIF" python "$@"; }
