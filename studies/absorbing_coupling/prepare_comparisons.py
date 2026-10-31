"""Prepare predeclared main training and confirmation matrices; never submits jobs."""
import argparse
import json
import os
from pathlib import Path


def write_tasks(path,tasks):
    with Path(path).open("x") as stream:
        for task in tasks:
            stream.write(json.dumps(task)+"\n")


def training(root,study,output):
    tasks=[]
    for line in (root/"studies/absorbing_coupling/train_pilots.jsonl").read_text().splitlines():
        task=json.loads(line)
        args=[]
        for arg in task["args"]:
            if arg.startswith("run.id="):
                arg=arg.replace("-pilot","-main")
            for key,value in (("+trainer.max_steps","2000"),("trainer.val_check_interval","100"),
                              ("trainer.limit_val_batches","1.0"),("trainer.limit_test_batches","1.0")):
                if arg.startswith(key+"="):
                    arg=key+"="+value
            args.append(arg)
        args.extend(["compile=null","+task.exit_after_n_skipped_train_batches=1"])
        task["args"]=args
        task["name"]+=" 2000-update fresh matched restart"
        tasks.append(task)
    write_tasks(output,tasks)


def confirmation(root,study,selection,output,matrix=None):
    # Only validation-selected settings may enter this file. It is recorded
    # before test generation and includes mandatory cosine/linear controls.
    selected=json.loads(Path(selection).read_text())
    if not {"abs256","abs256lin"}<=set(selected):
        raise ValueError("Confirmation requires both cosine and linear controls")
    tasks=[]
    source_matrix=Path(matrix) if matrix else root/"studies/absorbing_coupling/pilots.jsonl"
    for line in source_matrix.read_text().splitlines():
        task=json.loads(line)
        args=task["args"]
        old_output=args[args.index("--output")+1]
        name=Path(old_output).name
        if name not in selected:
            continue
        for seed,type_seed in ((42069,0),(42070,1),(42071,2)):
            argv=[a.replace("epoch_2000_model","epoch_5000_model") for a in args]
            argv[0]="generate"
            argv[argv.index("--output")+1]=f"$STUDY/confirmation5000/{name}/seed{seed}"
            argv[argv.index("--n-samples")+1]="10000"
            if "--diagnostics" in argv:
                argv.remove("--diagnostics")
            argv.extend(["--seed",str(seed),"--type-seed",str(type_seed)])
            tasks.append({"name":f"e5000 {name} seed{seed}","module":task["module"],"args":argv})
    if {Path(t["args"][t["args"].index("--output")+1]).parent.name for t in tasks} != set(selected):
        raise ValueError("A selected setting was not found in the pilot matrix")
    write_tasks(output,tasks)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("training","confirmation"))
    parser.add_argument("output")
    parser.add_argument("--selection")
    parser.add_argument("--pilot-matrix")
    args=parser.parse_args()
    root,study=Path(os.environ["SNAPSHOT"]),Path(os.environ["STUDY"])
    if args.mode=="training":
        training(root,study,args.output)
    else:
        if not args.selection:
            parser.error("Confirmation requires a recorded validation selection JSON")
        confirmation(root,study,args.selection,args.output,args.pilot_matrix)
