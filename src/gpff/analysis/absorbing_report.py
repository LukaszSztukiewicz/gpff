"""Failure-preserving, molecule-level reports and conditional diagnostic tables."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from ase.db import connect
from gpff.analysis import analyze_with_rdkit
from gpff.sample.formula import file_sha256

ELEMENTS=(1,6,7,8,9)
GROUPS={"amine":"[NX3;!$(N-C=O)]", "amide":"[NX3][CX3](=[OX1])", "carbonyl":"[CX3]=[OX1]",
        "nitrile":"[CX2]#[NX1]", "hydroxyl":"[OX2H]", "nitro":"[$([NX3](=O)=O),$([NX3+](=O)[O-])]"}


def failure_flags(records):
    """Count overlapping sampler failures independently of first-cause labels."""
    result={"nonfinite_geometry_molecules":0,"MASK_molecules":0,"MASK_atoms":0,
            "ABSENT_molecules":0,"ABSENT_atoms":0}
    for row in records:
        masked=int(row.get("unresolved_mask",row.get("numbers",[]).count(99)))
        absent=int(row.get("absent",row.get("numbers",[]).count(0)))
        result["nonfinite_geometry_molecules"]+=int(bool(row.get("nonfinite_geometry",False)))
        result["MASK_molecules"]+=int(masked>0)
        result["MASK_atoms"]+=masked
        result["ABSENT_molecules"]+=int(absent>0)
        result["ABSENT_atoms"]+=absent
    return result


def reduce_records(records, reference, novelty):
    n = len(records)
    valid = [r for r in records if r["valid"]]
    unique = {r["smiles"] for r in valid}
    unseen = unique - novelty
    metrics={"attempts":n,"valid_count":len(valid),"validity":len(valid)/n if n else None,
        "unique_valid_count":len(unique),"uniqueness":len(unique)/len(valid) if valid else None,
        "uniqueness_denominator":"valid_attempts", "vun_count":len(unseen),"vun":len(unseen)/n if n else None,
        "failures":dict(Counter(r["failure"] for r in records if not r["valid"])),
        "failure_flags":failure_flags(records)}
    ref_counts=np.array(reference,dtype=int)
    ref_share=ref_counts.sum(0)/max(int(ref_counts.sum()),1)
    ref_hist=Counter(map(tuple,reference))
    for label, subset in (("all_output",records),("valid_only",valid)):
        if not subset:
            metrics[label]={"molecules":0}
            continue
        counts=np.array([r["formula"] for r in subset])
        chemical=int(counts.sum())
        shares=counts.sum(0)/max(chemical,1)
        hist=Counter(map(tuple,counts.tolist()))
        keys=set(hist)|set(ref_hist)
        formula_tv=.5*sum(abs(hist[k]/len(subset)-ref_hist[k]/len(reference)) for k in keys)
        n_presence=counts[:,2]>0
        o_presence=counts[:,3]>0
        metrics[label]={"molecules":len(subset),"chemical_atoms":chemical,
            "excluded_nonphysical_atoms":sum(len(r["numbers"])-sum(r["formula"]) for r in subset),
            "element_shares":shares.tolist(),"element_tv":float(.5*np.abs(shares-ref_share).sum()),
            "molecule_presence":(counts>0).mean(0).tolist(),
            "n_o_cooccurrence":float((n_presence&o_presence).mean()),
            "n_o_count_distribution":{str(k):v/len(subset) for k,v in Counter(map(tuple,counts[:,2:4].tolist())).items()},
            "formula_tv":formula_tv,"formula_distribution":{str(k):v/len(subset) for k,v in hist.items()},
            "size_distribution":{str(k):v/len(subset) for k,v in Counter(r["size"] for r in subset).items()},
            "mean_size":float(np.mean([r["size"] for r in subset]))}
    metrics["functional_groups_valid"]={name:sum(r.get("groups",{}).get(name,0)>0 for r in valid)/len(valid) if valid else None for name in GROUPS}
    return metrics


def reference_counts(dataset,split,partition):
    with np.load(split) as data:
        indices=data[partition+"_idx"]
    rows=[]
    with connect(dataset) as db:
        for i in indices:
            z=db.get(int(i)+1).numbers
            rows.append([int((z==k).sum()) for k in ELEMENTS])
    return rows


def evaluate(directory, dataset, split, assets, reference_partition="test", bootstrap=200):
    from rdkit import Chem
    directory=Path(directory)
    destination=directory/"study_evaluation"
    destination.mkdir(exist_ok=False)
    start=time.perf_counter()
    patterns={k:Chem.MolFromSmarts(v) for k,v in GROUPS.items()}
    smiles_rows=Path(assets).read_text().splitlines()
    with connect(dataset) as db:
        if len(smiles_rows)!=len(db):
            raise ValueError("Novelty assets must have one line per row of the matching dataset")
    with np.load(split) as partitions:
        novelty_indices=np.concatenate([partitions["train_idx"],partitions["val_idx"]])
    novelty={smiles_rows[int(i)].strip() for i in novelty_indices if smiles_rows[int(i)].strip()}
    reference=reference_counts(dataset,split,reference_partition)
    records=[]
    metadata=directory/"evaluation/metadata"
    cached_valid=np.load(metadata/"rdkit_valid.npy").astype(bool) if (metadata/"rdkit_valid.npy").exists() else None
    cached_failures=np.load(metadata/"failure.npy").astype(str) if (metadata/"failure.npy").exists() else None
    cached_smiles=(metadata/"smiles.txt").read_text().splitlines() if (metadata/"smiles.txt").exists() else None
    with connect(str(directory/"samples.db")) as db:
        if cached_valid is not None and (len(cached_valid)!=len(db) or cached_smiles is None or len(cached_smiles)!=len(db)):
            raise ValueError("Pinned evaluation metadata does not cover every generation attempt")
        for index,row in enumerate(db.select()):
            r=dict(row.data)
            # ASE serializes arrays, normalize to plain JSON values.
            for key in ("numbers","formula","reference_numbers","requested_formula"):
                if isinstance(r.get(key),np.ndarray):
                    r[key]=r[key].tolist()
            if r["nonfinite_geometry"]:
                validity,smiles,failure=False,"","nonfinite_geometry"
            elif r["unresolved_mask"] or r["absent"]:
                validity,smiles,failure=False,"","unresolved_token"
            elif cached_valid is not None:
                validity,smiles=bool(cached_valid[index]),cached_smiles[index]
                failure=None if validity else str(cached_failures[index]) if cached_failures is not None else "rdkit_invalid"
            else:
                result=analyze_with_rdkit(row.toatoms())
                validity,smiles=bool(result["rdkit_valid"]),result["smiles"]
                failure=None if validity else "rdkit_invalid"
            r.update(valid=validity,smiles=smiles,failure=failure)
            if validity:
                mol=Chem.MolFromSmiles(smiles)
                r["groups"]={k:len(mol.GetSubstructMatches(p)) for k,p in patterns.items()}
            records.append(r)
    with (destination/"molecules.jsonl").open("x") as stream:
        for record in records:
            stream.write(json.dumps(record,allow_nan=False)+"\n")
    metrics=reduce_records(records,reference,novelty)
    structural=directory/"evaluation/results.json"
    structural_metrics=json.loads(structural.read_text()) if structural.exists() else None
    metrics.update(reference_partition=reference_partition,reference_molecules=len(reference),
        elements=list(ELEMENTS),novelty_assets_sha256=file_sha256(assets),dataset_sha256=file_sha256(dataset),
        split_sha256=file_sha256(split),evaluation_wall_seconds=time.perf_counter()-start,
        novelty_reference="train_val",structural_metrics=structural_metrics,
        structural_results_sha256=file_sha256(structural) if structural.exists() else None,
        structural_status="completed" if structural.exists() else "not yet evaluated with the pinned structural evaluator")
    rng=np.random.default_rng(90125)
    values={k:[] for k in ("validity","all_element_tv","valid_element_tv")}
    for _ in range(bootstrap):
        chosen=[records[i] for i in rng.integers(len(records),size=len(records))]
        reduced=reduce_records(chosen,reference,novelty)
        for k in ("validity",):
            if reduced[k] is not None:
                values[k].append(reduced[k])
        for name,prefix in (("all_output","all"),("valid_only","valid")):
            if "element_tv" in reduced[name]:
                values[prefix+"_element_tv"].append(reduced[name]["element_tv"])
    metrics["molecule_bootstrap_95pct"]={k:np.quantile(v,[.025,.975]).tolist() if v else None for k,v in values.items()}
    metrics["bootstrap_note"]="Whole attempts resampled, reference fixed. Occupancy metrics uniqueness and VUN have no naive bootstrap interval because resampling artificially creates duplicate molecules."
    (destination/"metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False)+"\n")
    return metrics


def conditional_table(directory):
    directory=Path(directory)
    rows=[json.loads(s) for s in (directory/"conditionals.jsonl").read_text().splitlines()]
    groups={}
    for r in rows:
        size_bin="<=10" if r["size"]<=10 else "11-20" if r["size"]<=20 else ">20"
        mask_bin="<=.5" if r["actual_mask_fraction"]<=.5 else ">.5"
        for subgroup in ("all",f"size:{size_bin}",f"mask:{mask_bin}",f"size:{size_bin}/mask:{mask_bin}"):
            key=(r["corruption"],r["sigma"],r["requested_mask_rate"],subgroup)
            groups.setdefault(key,[]).append(r)
    result=[]
    for key, members in groups.items():
        count=sum(r["n_atoms"] for r in members)
        good=[r for r in members if r["n_atoms"] and not r.get("failed")]
        denom=sum(r["n_atoms"] for r in good)
        row=dict(zip(("corruption","sigma","requested_mask_rate","subgroup"),key))
        row.update(n_molecules=len(members),n_masked_atoms=count,failed_molecules=sum(bool(r.get("failed")) for r in members))
        if denom:
            for name in ("nll","brier","entropy","sample_argmax_disagreement"):
                row[name]=sum(r[name]*r["n_atoms"] for r in good)/denom
            row["expected_share"]=(np.sum([r["expected_counts"] for r in good],axis=0)/denom).tolist()
            row["true_share"]=(np.sum([r["true_counts"] for r in good],axis=0)/denom).tolist()
            row["elements"]={}
            for element in ("1","6","7","8","9"):
                entries=[r["elements"][element] for r in good]
                n_true=sum(e["n_true"] for e in entries)
                element_row={"n_true":n_true}
                for metric in ("argmax_recall","sample_recall"):
                    element_row[metric]=sum(e["n_true"]*(e[metric] or 0) for e in entries)/n_true if n_true else None
                calibration=[]
                for b in range(10):
                    cells=[e["reliability"][b] for e in entries]
                    n=sum(c["n"] for c in cells)
                    calibration.append({"n":n,"mean_probability":sum(c["n"]*(c["mean_probability"] or 0) for c in cells)/n if n else None,
                        "frequency":sum(c["n"]*(c["frequency"] or 0) for c in cells)/n if n else None})
                element_row["reliability"]=calibration
                row["elements"][element]=element_row
        result.append(row)
    (directory/"conditional_table.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--conditionals",action="store_true")
    for name in ("dataset","split","assets"):
        parser.add_argument("--"+name)
    parser.add_argument("--reference-partition",choices=("val","test"),default="test")
    args=parser.parse_args()
    if args.conditionals:
        conditional_table(args.directory)
    else:
        if not all((args.dataset,args.split,args.assets)):
            parser.error("evaluation needs --dataset --split --assets")
        evaluate(args.directory,args.dataset,args.split,args.assets,args.reference_partition)


if __name__=="__main__":
    main()
