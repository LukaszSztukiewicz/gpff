"""D1 operation-aligned count-proxy changes and D2 joint-completion summaries."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from gpff import vocab


def trajectory_summary(directory):
    directory=Path(directory)
    logs=[json.loads(s) for s in (directory/"decisions.jsonl").read_text().splitlines()]
    states={}
    batch=-1
    index=0
    for row in logs:
        if row["phase"]=="transition":
            continue
        if row["phase"]=="decision_input" and (batch<0 or row["step"]>previous_step):
            batch+=1
        previous_step=row["step"]
        path=directory/f"state_{index:05d}.pt"
        index+=1
        if not path.exists():
            continue
        state=torch.load(path,map_location="cpu",weights_only=False)
        if (state["phase"],state["step"]) != (row["phase"],row["step"]):
            raise ValueError("Saved state and diagnostic row differ")
        states[(batch,row["step"],row["phase"])]=state
    pairs=[("geometry","decision_input","after_geometry_probe"),
           ("types","after_geometry_probe","after_types_probe"),
           ("types_first","decision_input","after_types_before_geometry_probe"),
           ("geometry_conditioned","type_conditioned_force","after_geometry_probe")]
    records=[]
    for (batch,step,phase),start in states.items():
        for operation,before,after in pairs:
            if phase!=before or (batch,step,after) not in states:
                continue
            # T1 geometry is compared after types, not across both operations.
            if operation=="geometry" and (batch,step,"type_conditioned_force") in states:
                continue
            end=states[(batch,step,after)]
            a,b=start["expected_count_proxy"],end["expected_count_proxy"]
            n=min(len(a),len(b))
            for m in range(n):
                good_a=bool(start["finite"][start["idx_m"]==m].all())
                good_b=bool(end["finite"][end["idx_m"]==m].all())
                if good_a and good_b and bool(torch.isfinite(a[m]).all()) and bool(torch.isfinite(b[m]).all()):
                    records.append({"batch":batch,"molecule":m,"step":step,"operation":operation,
                                    "before":a[m].tolist(),"after":b[m].tolist(),"delta":(b[m]-a[m]).tolist()})
    summary=[]
    grouped=defaultdict(list)
    for row in records:
        grouped[(row["step"],row["operation"])].append(row)
    rng=np.random.default_rng(725)
    for (step,operation),members in grouped.items():
        delta=np.array([r["delta"] for r in members])
        nitrogen=delta[:,vocab.TOKEN_Z.index(7)]
        means=[float(nitrogen[rng.integers(len(nitrogen),size=len(nitrogen))].mean()) for _ in range(1000)]
        summary.append({"step":step,"operation":operation,"n_molecules":len(members),
            "mean_delta":delta.mean(0).tolist(),"mean_delta_N":float(nitrogen.mean()),
            "molecule_bootstrap_N_95pct":np.quantile(means,[.025,.975]).tolist()})
    return {"records":records,"summary":summary,
        "interpretation":"Count proxy changes use identical finite molecules per pair. They are not a per-trajectory conservation law or a causal identification test."}


def decoding_summary(directory):
    path=Path(directory)/"decodings.jsonl"
    records=[json.loads(s) for s in path.read_text().splitlines()]
    grouped=defaultdict(list)
    for row in records:
        grouped[(row["sigma"],row["mask_rate"],row["corruption"],row["context"],row["parallel"])].append(row)
    result=[]
    for key,members in grouped.items():
        counts=np.array([r["formula"] for r in members])
        reference=np.array([r["reference_formula"] for r in members])
        freq=Counter(map(tuple,counts.tolist()))
        n=len(members)
        row=dict(zip(("sigma","mask_rate","corruption","context","parallel"),key))
        row.update(n_molecules=len({r["index"] for r in members}),n_completions=n,
                   mean_counts=counts.mean(0).tolist(),reference_mean_counts=reference.mean(0).tolist(),
                   element_shares=(counts.sum(0)/counts.sum()).tolist(),
                   n_o_cooccurrence=float(((counts[:,2]>0)&(counts[:,3]>0)).mean()),
                   formula_distribution={str(k):v/n for k,v in freq.items()},
                   reconstruction_formula_fraction=float((counts==reference).all(-1).mean()))
        # Repeated draws are clustered within molecules, never independent atoms.
        clusters=defaultdict(list)
        for item in members:
            clusters[item["index"]].append(item["formula"][2])
        averages=np.array([np.mean(c) for c in clusters.values()])
        rng=np.random.default_rng(725)
        bootstrap=[float(averages[rng.integers(len(averages),size=len(averages))].mean()) for _ in range(1000)]
        row["molecule_bootstrap_mean_N_95pct"]=np.quantile(bootstrap,[.025,.975]).tolist()
        result.append(row)
    return result
