"""Freeze explicit task matrices inside a Slurm container before submission."""
import json
import os
from pathlib import Path

root=Path(os.environ["SNAPSHOT"])
study=Path(os.environ["STUDY"])
version=os.environ.get("STUDY_VERSION","4")
origin=root/"studies/absorbing_coupling"
tasks=[]
for line in (origin/"pilots.jsonl").read_text().splitlines():
    task=json.loads(line)
    task["args"]=[a.replace("$STUDY/pilots/",f"$STUDY/pilots{version}/") for a in task["args"]]
    tasks.append(task)
base=tasks[2]["args"]
for name, extra in (("types-given",["--types-given"]),
                    ("e1-conditional",["--formula","$STUDY/formula_prior.json","--formula-mode","conditional_size"]),
                    ("e1-joint",["--formula","$STUDY/formula_prior.json","--formula-mode","joint"])):
    args=base.copy()
    args[args.index("--output")+1]=f"$STUDY/pilots{version}/"+name
    # Formula and composition-given direct denoising do not have a type schedule.
    i=args.index("--type-schedule")
    del args[i:i+2]
    tasks.append({"name":name,"module":"gpff.cli.absorbing_study","args":args+extra})
for filename, values in ((f"pilots{version}.jsonl",tasks),):
    with (study/filename).open("x") as stream:
        for value in values:
            stream.write(json.dumps(value)+"\n")
with (study/f"evaluation-list-pilot{version}.txt").open("x") as stream:
    for task in tasks:
        arguments=task["args"]
        directory=os.path.expandvars(arguments[arguments.index("--output")+1])
        stream.write(directory+"\n")
print(f"Prepared {len(tasks)} tasks at {study}/pilots{version}.jsonl",flush=True)
