"""Short CPU Slurm integration checks, separate from scientific pilot budgets."""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(os.environ["SNAPSHOT"])
study=Path(os.environ["STUDY"])
source=os.environ["SOURCE"]
dataset=os.environ["DATASET"]
label=os.environ.get("SMOKE_LABEL", "smoke4")
results=[]
def run(name,command):
    log=study/(label+"-"+name+".log")
    print(json.dumps({"name":name,"command":command,"log":str(log)}),flush=True)
    with log.open("x") as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    results.append({"name":name,"returncode":result.returncode,"log":str(log)})
base=["--dataset",dataset,"--split",source+"/frozen/split.npz","--model",source+"/frozen/epoch_2000_model",
      "--training-config",source+"/.hydra/config.yaml","--n-samples","4","--batch-size","4","--device","cpu"]
for name, command, extra in (("d2","conditionals",["--sigmas","0",".5","--mask-rates",".75","--decode","--decode-sigmas","0",".5"]),
                             ("d1","pilot",["--steps","4","--diagnostics"]),
                             ("t1","pilot",["--steps","4","--kernel","random","--recompute"]),
                             ("scheduled","pilot",["--steps","4","--scheduled"]),
                             ("given","pilot",["--steps","4","--types-given"])):
    command=[sys.executable,"-m","gpff.cli.absorbing_study",command,*base,"--output",str(study/label/name),*extra]
    run(name,command)
tasks=[json.loads(line) for line in (root/"studies/absorbing_coupling/train_pilots.jsonl").read_text().splitlines()]
for task in tasks:
    args=[]
    for value in task["args"]:
        if value.startswith("run.id="):
            value=value.replace("-pilot","-"+label)
        replacements={"data.batch_size":"4","data.num_workers":"0","data.cache_in_memory":"false",
            "data.persistent_workers":"false","+trainer.max_steps":"2","trainer.val_check_interval":"1",
            "trainer.limit_val_batches":"1","trainer.limit_test_batches":"1"}
        for key,replacement in replacements.items():
            if value.startswith(key+"="):
                value=key+"="+replacement
        args.append(os.path.expandvars(value))
    args.extend(["trainer.accelerator=cpu","+trainer.limit_train_batches=2"])
    # limit_train_batches already exists in base config, override without +.
    args[-1]="trainer.limit_train_batches=2"
    args.append("callbacks.study_updates.updates=[1,2]")
    args.append("run.git_hash="+str(root))
    run("-".join(task["name"].split()[:2]), ["gpff_train",*args])
with (study/(label+"-summary.json")).open("x") as stream:
    json.dump(results,stream,indent=2)
sys.exit(int(any(r["returncode"] for r in results)))
