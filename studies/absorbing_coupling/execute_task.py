"""Run one explicit JSONL task inside the Slurm container; no shell evaluation."""
import json
import os
from pathlib import Path
import subprocess
import sys

task_file=Path(sys.argv[1])
index=int(os.environ.get("SLURM_ARRAY_TASK_ID","0"))
task=json.loads(task_file.read_text().splitlines()[index])
args=[os.path.expandvars(a) for a in task["args"]]
if task.get("entrypoint") == "gpff_train":
    args.append("run.git_hash="+os.environ["SNAPSHOT"])
command=([task["entrypoint"]] if "entrypoint" in task else [sys.executable,"-m",task["module"]])+args
print(json.dumps({"task":task["name"],"command":command,"job":os.environ["SLURM_JOB_ID"]}),flush=True)
subprocess.run(command,check=True)
