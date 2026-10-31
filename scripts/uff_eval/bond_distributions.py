"""Bond-length distributions per bond type against QM9: JS divergence and 1-D Wasserstein.

    python bond_distributions.py <report.md> [--json out.json]

For fancy-monkey (full/) and SOB999 (sob/), raw and after xTB and MACE-OFF24(M), every set's bond
lengths of one type are compared with the reference: the raw bonds of fancy-monkey's QM9 test set
(DFT geometries). SOB999's QM9 test set, other real molecules, gives the floor of each metric for
10k-molecule sets. Bonds come from the valid molecules, measured on the bonds RDKit read from the
raw structure, before and after relaxation (as in relax_eval.py).
- W1: scipy's 1-D Wasserstein distance (earth mover's), in mA; no binning.
- W1 / spread: W1 over the reference's standard deviation of that bond type, so bond types with
  narrow (O-H) and wide (C-C) natural spreads compare: 1 is an error the size of the real spread.
- JS: Jensen-Shannon divergence (base 2, in [0, 1]) of histograms with fixed 5 mA bins over the
  reference's range +- 0.1 A, plus one overflow bin at each end.
"""
import argparse
import json
import os
import pickle

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

MODELS = {"fancy-monkey": "full", "SOB999": "sob"}
SETS = ["qm9", "dd16", "sdd16", "sheun16", "sdd256", "sheun256", "sdd16_dd", "sheun16_dd"]
LABELS = {"qm9": "QM9 (DFT)", "dd16": "DD16", "sdd16": "SDD16", "sheun16": "SHeun16", "sdd256": "SDD256",
          "sheun256": "SHeun256", "sdd16_dd": "SDD16 → DD", "sheun16_dd": "SHeun16 → DD"}
BIN = 0.005
PAD = 0.1


RELAXERS = {"xTB": "eval_xtb", "MACE": "eval_mace"}
STATES = ["raw"] + list(RELAXERS)


def load(folder, evaluation, name):
    with open(os.path.join(folder, evaluation, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)


def lengths(records, key):
    """Bond lengths per type over the valid molecules, before or after relaxation."""
    out = {}
    for record in records:
        for label, values in (record[key] or {}).items():
            out.setdefault(label, []).append(values)
    return {label: np.concatenate(values) for label, values in out.items()}


def js_divergence(reference, observed):
    edges = np.arange(reference.min() - PAD, reference.max() + PAD + BIN, BIN)
    edges = np.concatenate([[-np.inf], edges, [np.inf]])
    p, _ = np.histogram(reference, bins=edges)
    q, _ = np.histogram(observed, bins=edges)
    return float(jensenshannon(p, q, base=2) ** 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    reference = lengths(load("full", "eval_xtb", "qm9"), "bonds_before")
    labels = [label for label, values in sorted(reference.items(), key=lambda item: -len(item[1]))
              if len(values) >= 500][:12]
    spread = {label: float(reference[label].std()) for label in labels}

    results = {}
    for model, folder in MODELS.items():
        for name in SETS:
            # the raw bonds are the same in every relaxer's records
            sources = {"raw": (RELAXERS["xTB"], "bonds_before")}
            sources.update({state: (evaluation, "bonds_after") for state, evaluation in RELAXERS.items()})
            if model == "fancy-monkey" and name == "qm9":
                del sources["raw"]  # the reference itself
            for state, (evaluation, key) in sources.items():
                observed = lengths(load(folder, evaluation, name), key)
                results[(model, name, state)] = {
                    label: {"n": int(len(observed[label])),
                            "w1": float(wasserstein_distance(reference[label], observed[label])),
                            "js": js_divergence(reference[label], observed[label])}
                    for label in labels if label in observed
                }
                for label, entry in results[(model, name, state)].items():
                    entry["w1_rel"] = entry["w1"] / spread[label]

    def over_types(entry, metric, reduce):
        return float(reduce([entry[label][metric] for label in labels if label in entry]))

    lines = ["# Bond-length distributions against QM9 (DFT)", "",
             "Reference: raw bonds of fancy-monkey's QM9 test set. Floor: SOB999's QM9 test set (other real "
             "molecules). W1 in mA, JS base 2 with 5 mA bins. Mean and median: unweighted over the "
             f"{len(labels)} most common bond types. W1 / spread: W1 over the reference's standard deviation of "
             "the bond type.", "",
             "Reference spread (std), mA: " + ", ".join(f"{label} {1000 * spread[label]:.1f}" for label in labels),
             ""]
    metrics = [("w1", 1000, ".1f", "W1, mA"), ("w1_rel", 1, ".2f", "W1 / spread"), ("js", 1, ".3f", "JS")]
    for reduce_name, reduce in [("Median", np.median), ("Mean", np.mean)]:
        lines += [f"## {reduce_name} over bond types"]
        for metric, scale, fmt, title in metrics:
            lines += ["", f"{title}", "",
                      "| set | " + " | ".join(f"{model} {state}" for model in MODELS for state in STATES) + " |",
                      "|---|" + "---|" * (len(MODELS) * len(STATES))]
            for name in SETS:
                cells = []
                for model in MODELS:
                    for state in STATES:
                        entry = results.get((model, name, state))
                        cells.append("-" if entry is None else format(scale * over_types(entry, metric, reduce), fmt))
                label = LABELS[name] + (" (floor)" if name == "qm9" else "")
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
            lines.append("")

    for metric, scale, fmt, title in metrics:
        for model in MODELS:
            lines += ["", f"## {model}: {title} per bond type", "",
                      "| set | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
            for name in SETS:
                for state in STATES:
                    entry = results.get((model, name, state))
                    if entry is None:
                        continue
                    row = " | ".join("-" if label not in entry else format(scale * entry[label][metric], fmt)
                                     for label in labels)
                    lines.append(f"| {LABELS[name]}, {state} | {row} |")
    with open(args.report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"spread": spread, "results": {"/".join(key): value for key, value in results.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
