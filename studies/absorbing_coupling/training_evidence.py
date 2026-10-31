"""Audit matched inputs and summarize data-anchored rollout errors from a run set."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from gpff.sample.formula import file_sha256


def summarize(study,suffix,output):
    study=Path(study)
    result={"runs":[],"matched_input_audits":True}
    baseline=None
    for name in ("t2-ordinary","t2-rollout","t2-isotropic","t3-unconditioned","t3-sigma"):
        run=study/"training"/(name+"-"+suffix)
        updates=run/"updates"
        checkpoint=updates/"step_100.ckpt"
        if not checkpoint.exists():
            raise ValueError(f"Qualification has not finished: {checkpoint}")
        stored=torch.load(checkpoint,map_location="cpu",weights_only=False)
        audit=[json.loads(s) for s in (run/"input_audit.jsonl").read_text().splitlines()]
        hashes={r["step"]:r["sha256"] for r in audit}
        if baseline is None:
            baseline=hashes
        if hashes != baseline:
            result["matched_input_audits"]=False
        entry={"run":str(run),"checkpoint_sha256":file_sha256(checkpoint),"global_step":stored["global_step"],
            "initialization":json.loads((run/"initialization.json").read_text()),"audit":audit,
            "teacher_batch_calls":stored.get("recovery_teacher_batch_calls",0)}
        errors=run/"recovery_errors.jsonl"
        if errors.exists():
            rows=[json.loads(s) for s in errors.read_text().splitlines()]
            values=[r for row in rows for r in row["records"]]
            entry["augmented_molecules"]=len(values)
            for field in ("initial_sigma","displacement_rms","rollout_error_rms"):
                array=np.array([r[field] for r in values])
                entry[field]={"mean":float(array.mean()),"quantiles_0_50_90_99_100":np.quantile(array,[0,.5,.9,.99,1]).tolist(),
                    "all_finite":bool(np.isfinite(array).all())}
        result["runs"].append(entry)
    # The two ordinary arms use the same objective, initialization and draws.
    control_states=[]
    for name in ("t2-ordinary","t3-unconditioned"):
        path=study/"training"/(name+"-"+suffix)/"updates/step_100.ckpt"
        state=torch.load(path,map_location="cpu",weights_only=False)["state_dict"]
        control_states.append({k:v for k,v in state.items() if k.startswith("model.")})
    differences=[float((control_states[0][k]-control_states[1][k]).abs().max())
                 for k in control_states[0] if control_states[0][k].is_floating_point()]
    result["ordinary_controls_max_parameter_difference"]=max(differences,default=0)
    with Path(output).open("x") as stream:
        json.dump(result,stream,indent=2)
    if not result["matched_input_audits"]:
        raise ValueError("Input draws differ; do not treat these runs as a matched training comparison")
    print(json.dumps({"matched_input_audits":result["matched_input_audits"],
        "ordinary_controls_max_parameter_difference":result["ordinary_controls_max_parameter_difference"]}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study")
    parser.add_argument("output")
    parser.add_argument("--suffix",default="pilot")
    args=parser.parse_args()
    summarize(args.study,args.suffix,args.output)
