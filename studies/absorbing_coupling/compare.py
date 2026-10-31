"""Paired molecule bootstrap and exact McNemar for compatible attempts."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest


def compare(left,right,formula_pairing=False,draws=2000):
    def read(path):
        return [json.loads(s) for s in Path(path,"study_evaluation/molecules.jsonl").read_text().splitlines()]
    a,b=read(left),read(right)
    if len(a)!=len(b):
        raise ValueError("Pairing requires equal attempt counts")
    for ra,rb in zip(a,b):
        field="requested_formula" if formula_pairing else "size_source_index"
        if ra[field] != rb[field] or (formula_pairing and ra[field] is None):
            raise ValueError("Pairing keys differ")
    ma,mb=[json.loads(Path(p,"manifest.json").read_text()) for p in (left,right)]
    aa,bb=ma["arguments"],mb["arguments"]
    if aa["seed"]!=bb["seed"] or aa["n_samples"]!=bb["n_samples"]:
        raise ValueError("Pairing requires same size/formula/position seed and sample count")
    if aa["batch_size"]!=bb["batch_size"]:
        raise ValueError("Initial position pairing requires identical batching")
    if bool(aa.get("formula")) != bool(bb.get("formula")):
        raise ValueError("Formula and size-only generators have different position streams")
    if formula_pairing and aa["formula_mode"]!=bb["formula_mode"]:
        raise ValueError("Formula modes differ")
    valid_a=np.array([r["valid"] for r in a])
    valid_b=np.array([r["valid"] for r in b])
    a_only=int((valid_a&~valid_b).sum())
    b_only=int((~valid_a&valid_b).sum())
    difference=valid_b.astype(float)-valid_a.astype(float)
    rng=np.random.default_rng(81991)
    means=[float(difference[rng.integers(len(a),size=len(a))].mean()) for _ in range(draws)]
    return {"left":str(left),"right":str(right),"n_pairs":len(a),"left_only_valid":a_only,
        "right_only_valid":b_only,"validity_difference":float(difference.mean()),
        "paired_bootstrap_95pct":np.quantile(means,[.025,.975]).tolist(),
        "mcnemar_exact_p":float(binomtest(a_only,a_only+b_only,.5).pvalue) if a_only+b_only else 1.,
        "pairing":"sampled formula and position seed" if formula_pairing else "size source and initial position seed",
        "noise_path_note":"Step counts and remasking draws may differ. This does not assert identical stochastic paths."}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--formula-pairing",action="store_true")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    with open(args.output,"x") as stream:
        json.dump(compare(args.left,args.right,args.formula_pairing),stream,indent=2)
