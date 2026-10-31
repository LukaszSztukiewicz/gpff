"""Report: fancy-monkey ep5000 against SOB999, raw and relaxed with xTB and MACE-OFF24(M), per bond type.

    python compare_models.py <report.md> [--json report.json]

Reads the relax_eval.py records of full/ (fancy-monkey; eval_uff, eval_xtb, eval_mace) and sob/
(SOB999; eval_xtb, eval_mace). All bond errors use one yardstick: the mean length per bond type of
fancy-monkey's QM9 reference set (DFT geometries), so both models are measured the same way.
"""
import argparse
import json
import os
import pickle

from relax_eval import reference_lengths, summarize

MODELS = {"fancy-monkey": "full", "SOB999": "sob"}
METHODS = {"uff": "UFF", "xtb": "xTB", "mace": "MACE"}
SETS = ["qm9", "dd16", "sdd16", "sheun16", "sdd256", "sheun256", "sdd16_dd", "sheun16_dd"]
LABELS = {"qm9": "QM9 (DFT)", "dd16": "DD16", "sdd16": "SDD16", "sheun16": "SHeun16", "sdd256": "SDD256",
          "sheun256": "SHeun256", "sdd16_dd": "SDD16 → DD", "sheun16_dd": "SHeun16 → DD"}


def load(folder, method, name):
    path = os.path.join(folder, f"eval_{method}", f"{name}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    reference = reference_lengths(load("full", "xtb", "qm9"))
    results = {}
    for model, folder in MODELS.items():
        for method in METHODS:
            for name in SETS:
                records = load(folder, method, name)
                if records is not None:
                    results[(model, method, name)] = summarize(records, reference)

    lines = ["# GPFF samples relaxed with xTB and MACE-OFF24(M)", "",
             "10k molecules per set. Bond error: mean |length - QM9 DFT mean of its bond type| in mA "
             "(valid molecules); the QM9 row is the floor. Strain: median energy released, kcal/mol.", "",
             "| model | set | NFE | valid % | after xTB | after MACE | bond error | after xTB | after MACE "
             "| strain xTB | strain MACE |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        for name in SETS:
            x, m = results.get((model, "xtb", name)), results.get((model, "mace", name))
            if x is None and m is None:
                continue
            base = x or m
            cell = lambda r, key, scale, fmt: "-" if r is None else format(scale * r[key], fmt)
            nfe = "-" if base["nfe"] != base["nfe"] else f"{base['nfe']:.0f}"
            lines.append(
                f"| {model} | {LABELS[name]} | {nfe} | {100 * base['valid_before']:.1f} "
                f"| {cell(x, 'valid_after', 100, '.1f')} | {cell(m, 'valid_after', 100, '.1f')} "
                f"| {1000 * base['bond_error_before']:.1f} | {cell(x, 'bond_error_after', 1000, '.1f')} "
                f"| {cell(m, 'bond_error_after', 1000, '.1f')} | {cell(x, 'strain_median', 1, '.1f')} "
                f"| {cell(m, 'strain_median', 1, '.1f')} |")

    # per bond type: the most common types, mean absolute deviation (signed bias in brackets)
    labels = [label for label, entry in results[("fancy-monkey", "xtb", "qm9")]["bond_types"].items()
              if entry["n"] >= 500][:12]
    def type_cells(r, suffix):
        cells = []
        for label in labels:
            e = r["bond_types"].get(label)
            cells.append("-" if e is None else f"{1000 * e['mad_' + suffix]:.1f} ({1000 * e['bias_' + suffix]:+.1f})")
        return " | ".join(cells)

    for model in MODELS:
        lines += ["", f"## {model}: bond error per bond type", "",
                  "mA, mean absolute deviation from the QM9 DFT mean, signed bias in brackets. Each set: raw, "
                  "then after each relaxation (the same bonds).", "",
                  "| set | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
        for name in SETS:
            relaxed = [(METHODS[m], results[(model, m, name)]) for m in METHODS if (model, m, name) in results]
            if not relaxed:
                continue
            lines.append(f"| **{LABELS[name]}** | " + type_cells(relaxed[0][1], "before") + " |")
            for method_label, r in relaxed:
                lines.append(f"| → {method_label} | " + type_cells(r, "after") + " |")
    with open(args.report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"reference_lengths": reference,
                       "results": {"/".join(key): value for key, value in results.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
