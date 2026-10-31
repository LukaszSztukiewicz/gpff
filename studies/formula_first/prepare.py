"""Fit the existing empirical formula prior; run only in Hydra Slurm."""
import os
import sys
from pathlib import Path
from gpff.sample.formula import FormulaPrior, build_formula_prior

if __name__ == "__main__":
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run inside Slurm")
    dataset, split, output = sys.argv[1:]
    if Path(output).exists():
        prior = FormulaPrior(output)
        prior.validate_source(dataset, split)
        result = prior.payload
        print("Reusing the verified existing formula table; no refit or overwrite.")
    else:
        result = build_formula_prior(dataset, split, output)
    print({key: result[key] for key in ("training_samples", "dataset_sha256", "split_sha256")})
    print("distinct_formulas:", len(result["formulas"]))
