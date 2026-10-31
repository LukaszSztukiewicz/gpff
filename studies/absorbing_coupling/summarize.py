"""Write compact tables, plots and an artifact manifest from completed outputs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import os
import numpy as np
from gpff import vocab
from gpff.sample.formula import file_sha256
from gpff.analysis.coupling_evidence import trajectory_summary,decoding_summary
from gpff.analysis.absorbing_report import failure_flags


def summarize(study,output):
    study,output=Path(study),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    rows,manifest=[],[]
    paths=sorted([*study.glob("pilots*/**/study_evaluation/metrics.json"),
                  *study.glob("confirmation*/**/study_evaluation/metrics.json")])
    for path in paths:
        run=path.parents[1]
        generation=json.loads((run/"generation.json").read_text())
        origin=json.loads((run/"manifest.json").read_text())
        if origin.get("status")!="completed":
            continue
        metric=json.loads(path.read_text())
        molecules=path.parent/"molecules.jsonl"
        flags=failure_flags([json.loads(s) for s in molecules.read_text().splitlines()]) if molecules.exists() else metric.get("failure_flags",{})
        mass=metric["all_output"]
        geometry=metric.get("structural_metrics") or {}
        row={"name":str(run.relative_to(study)),"partition":generation["partition"],
             "seed":origin["arguments"]["seed"],"n":metric["attempts"],"validity":metric["validity"],
             "vun":metric["vun"],"uniqueness":metric["uniqueness"],"element_tv":mass.get("element_tv"),
             "atom_N":mass["element_shares"][2],"atom_O":mass["element_shares"][3],
             "mol_N":mass["molecule_presence"][2],"mol_O":mass["molecule_presence"][3],
             "cooccurrence":mass["n_o_cooccurrence"],"mean_nfe":generation["mean_nfe"],
             "bond_w1":geometry.get("bond_length_w1"),"angle_w1":geometry.get("bond_angle_w1"),
             "wall_seconds":generation["wall_seconds"],"failures":metric["failures"],"failure_flags":flags,
             "bootstrap":metric["molecule_bootstrap_95pct"]}
        rows.append(row)
        manifest.append({"row":row["name"],"manifest":origin,"metrics":str(path),"metrics_sha256":file_sha256(path)})
    with (output/"summary.json").open("x") as stream:
        json.dump(rows,stream,indent=2)
    with (output/"manifest.json").open("x") as stream:
        json.dump({"reporter_snapshot":os.environ.get("PYTHONPATH"),"job":os.environ.get("SLURM_JOB_ID"),"rows":manifest},stream,indent=2)
    lines=["| Generation | Partition | Attempts | Valid % | VUN % | Unique / valid % | Element TV | N atom % | O atom % | NFE | Bond W1 | Angle W1 |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        def fmt(v,percent=False):
            return "—" if v is None else f"{v*(100 if percent else 1):.4f}"
        values=[r["name"],r["partition"],str(r["n"]),*[fmt(r[k],True) for k in ("validity","vun","uniqueness")],
                fmt(r["element_tv"]),fmt(r["atom_N"],True),fmt(r["atom_O"],True),fmt(r["mean_nfe"]),fmt(r["bond_w1"]),fmt(r["angle_w1"])]
        lines.append("| "+" | ".join(values)+" |")
    (output/"comparison.md").write_text("\n".join(lines)+"\n")
    failures=["| Generation | Attempts | Nonfinite molecules | MASK molecules | MASK atoms | ABSENT molecules | ABSENT atoms |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        flags=row["failure_flags"]
        counts=[str(flags.get(k,"—")) for k in ("nonfinite_geometry_molecules","MASK_molecules","MASK_atoms","ABSENT_molecules","ABSENT_atoms")]
        failures.append("| "+" | ".join([row["name"],str(row["n"]),*counts])+" |")
    (output/"failure_counts.md").write_text("\n".join(failures)+"\n")
    groups=defaultdict(list)
    for row in rows:
        if row["name"].startswith("confirmation"):
            groups[Path(row["name"]).parent.name].append(row)
    aggregate=[]
    for name,members in groups.items():
        item={"setting":name,"n_generation_seeds":len(members),"attempts_per_seed":[r["n"] for r in members],
              "seeds":[r["seed"] for r in members],"note":"Per-seed metrics; this does not measure training-seed variability or pooled uniqueness."}
        for metric in ("validity","vun","uniqueness","element_tv","atom_N","atom_O","mean_nfe","bond_w1","angle_w1"):
            values=[r[metric] for r in members if r[metric] is not None]
            item[metric]={"mean":float(np.mean(values)),"std_across_generation_seeds":float(np.std(values,ddof=1)) if len(values)>1 else None} if values else None
        aggregate.append(item)
    (output/"confirmation_by_seed.json").write_text(json.dumps(aggregate,indent=2)+"\n")
    conditionals=[]
    for path in sorted(study.glob("pilots*/**/conditionals.jsonl")):
        records=[json.loads(s) for s in path.read_text().splitlines()]
        grouped=defaultdict(list)
        for r in records:
            grouped[(r["corruption"],r["sigma"],r["requested_mask_rate"])].append(r)
        for (mode,sigma,rate),members in grouped.items():
            good=[r for r in members if r["n_atoms"] and not r.get("failed")]
            n=sum(r["n_atoms"] for r in good)
            if not n:
                continue
            p=np.sum([r["expected_counts"] for r in good],axis=0)/n
            q=np.sum([r["true_counts"] for r in good],axis=0)/n
            i=vocab.TOKEN_Z.index(7)
            row={"source":str(path.parent.relative_to(study)),"corruption":mode,"sigma":sigma,"rate":rate,
                 "molecules":len(members),"masked_atoms":n,"N_expected":float(p[i]),"N_reference":float(q[i])}
            for metric in ("nll","entropy","brier","sample_argmax_disagreement"):
                row[metric]=sum(r[metric]*r["n_atoms"] for r in good)/n
            conditionals.append(row)
    (output/"conditionals.json").write_text(json.dumps(conditionals,indent=2)+"\n")
    for path in sorted(study.glob("pilots*/**/diagnostics/decisions.jsonl")):
        label="-".join(path.parent.parent.relative_to(study).parts)
        (output/(label+"-trajectory.json")).write_text(json.dumps(trajectory_summary(path.parent),indent=2)+"\n")
    for path in sorted(study.glob("pilots*/**/decodings.jsonl")):
        if not path.stat().st_size:
            continue
        label="-".join(path.parent.relative_to(study).parts)
        (output/(label+"-decodings.json")).write_text(json.dumps(decoding_summary(path.parent),indent=2)+"\n")
    if conditionals:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout="constrained")
        sources=sorted({r["source"] for r in conditionals})
        # Full-grid pilot is separate from the smaller sequential population.
        source=next((s for s in sources if s.endswith("d2-conditionals")),sources[0])
        molecule_count=next(r["molecules"] for r in conditionals if r["source"]==source)
        reference=None
        for mode in ("training_transform","raw_gaussian"):
            subset=sorted([r for r in conditionals if r["source"]==source and r["rate"]==1 and r["corruption"]==mode],key=lambda r:r["sigma"])
            if subset:
                style={"marker":"o","linestyle":"-","label":"Training transform"} if mode=="training_transform" else {"marker":"x","linestyle":"--","label":"Raw Gaussian"}
                axes[0].plot([r["sigma"] for r in subset],[100*r["N_expected"] for r in subset],**style)
                axes[1].plot([r["sigma"] for r in subset],[r["nll"] for r in subset],**style)
                reference=100*subset[0]["N_reference"]
        if reference is not None:
            axes[0].axhline(reference,color="gray",linestyle=":",label="Subset N share")
        for ax in axes:
            ax.set_xscale("symlog",linthresh=.1)
            ax.set_xlim(0,35)
            ax.set_xlabel("Declared corruption sigma (angstrom)")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Expected N share among masked atoms (%)")
        axes[1].set_ylabel("Masked negative log likelihood")
        fig.suptitle(f"D2 validation pilot: {molecule_count} molecules, all types masked")
        fig.savefig(output/"conditionals.png",dpi=180)
        plt.close(fig)
    print((output/"comparison.md").read_text(),flush=True)
    for row in conditionals:
        if row["rate"]==1:
            print(json.dumps(row),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study")
    parser.add_argument("output")
    args=parser.parse_args()
    summarize(args.study,args.output)
