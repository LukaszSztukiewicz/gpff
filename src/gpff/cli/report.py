"""Figures and a summary page of evaluated sample sets (``gpff_report``).

    gpff_report <model dir>... --output_dir DIR [--names A B ...] [--control A]

A model dir is <run>/frozen/repro/<model>, whose <config>/seed<N>/evaluation/
folders gpff_prepare_metadata and gpff_calculate_metrics filled. Nothing is
computed again; the report only reads them and writes

    DIR/index.html      a results table (validity with its 95% Wilson interval)
                        and the figures
    DIR/figures/*.png

The figures: the causes RDKit rejects samples for, validity by heavy-atom count,
the bond and angle environments of every set, and with --control every other
set's difference to it (paired when both come from the same test molecules and
seeds). Sets named <run>:<epoch>, the default name of a milestone, are also
drawn against the epoch. Only the configs every set has are shown, unless
--configs names them.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

import numpy as np

from gpff.analysis import report
import matplotlib.pyplot as plt  # after report, which picks the Agg backend

# results table: results.json key, header, scale, digits
COLUMNS = (
    ("vun", "VUN %", 100, 1),
    ("uniqueness", "Unique %", 100, 1),
    ("novelty", "Novel %", 100, 1),
    ("bond_length_w1", "Bond W1 mÅ", 1000, 2),
    ("bond_angle_w1", "Angle W1 °", 1, 2),
    ("elongation_w1", "Elong. W1 Å", 1, 3),
    ("element_tv", "Element TV", 1, 4),
)
# epoch figures: results.json key, scale, axis label
EPOCH_METRICS = (
    ("rdkit_validity", 100, "RDKit valid (%)"),
    ("vun", 100, "valid, unique and novel (%)"),
    ("bond_length_w1", 1000, "bond length W1 (mÅ)"),
)
PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b;
       background: #f9f9f7; margin: 24px; max-width: 1400px; }}
table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 14px; }}
th, td {{ padding: 3px 10px; border-bottom: 1px solid #e1e0d9; text-align: right; }}
th:nth-child(-n+2), td:nth-child(-n+2) {{ text-align: left; }}
img {{ max-width: 100%; }}
p, li {{ color: #52514e; max-width: 80ch; }}
</style></head><body>
<h1>{title}</h1>
{body}
</body></html>
"""


def default_name(model_dir: Path) -> str:
    """<run>:<epoch> for <run>/frozen/repro/epoch_<epoch>_model, <run>:best for a best model."""
    parts = model_dir.resolve().parts
    model = parts[-1]
    if len(parts) < 4 or parts[-3:-1] != ("frozen", "repro"):
        return model
    match = re.fullmatch(r"epoch_(\d+)_model", model)
    if match:
        short = match[1]
    elif re.fullmatch(r"best(_epoch_\d+)?_model", model):
        short = "best"
    else:
        short = model.removesuffix("_model")
    return f"{parts[-4]}:{short}"


def results_table(sets: dict, configs) -> str:
    """The headline metrics per config and set, as an HTML table, means over seeds."""
    columns = [column for column in COLUMNS
               if any(column[0] in e["results"]
                      for data in sets.values() for config in configs
                      for e in data[config].values())]
    header = "".join(f"<th>{html.escape(name)}</th>" for _, name, _, _ in columns)
    lines = [f"<table><tr><th>config</th><th>set</th><th>seeds</th><th>Valid % [95% CI]</th>"
             f"{header}</tr>"]
    for config in configs:
        for label, data in sets.items():
            evaluations = data[config]
            valid = report.pooled(evaluations, lambda e: e["rdkit_valid"])
            if valid is not None:
                low, high = report.wilson(valid.sum(), len(valid))
                validity = f"{100 * valid.mean():.1f} [{100 * low:.1f}, {100 * high:.1f}]"
            else:
                values = [e["results"]["rdkit_validity"] for e in evaluations.values()
                          if "rdkit_validity" in e["results"]]
                validity = f"{100 * np.mean(values):.1f}" if values else "-"
            cells = []
            for key, _, scale, digits in columns:
                values = [e["results"][key] for e in evaluations.values() if key in e["results"]]
                cells.append(f"{scale * np.mean(values):.{digits}f}" if values else "-")
            lines.append(
                f"<tr><td>{html.escape(config)}</td><td>{html.escape(label)}</td>"
                f"<td>{len(evaluations)}</td><td>{validity}</td>"
                + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"
            )
    lines.append("</table>")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model_dirs", nargs="+", type=Path,
                        help="<run>/frozen/repro/<model> folders, one set each")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--names", nargs="+", default=None,
                        help="One name per set. Default: <run>:<epoch> or <run>:best.")
    parser.add_argument("--control", default=None,
                        help="Name of the set the others are compared with.")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Configs to show. Default: those every set has.")
    parser.add_argument("--seed_band", type=float, default=1.5,
                        help="Points of validity separately trained runs differ by; "
                             "shaded in the comparison. Default: 1.5.")
    parser.add_argument("--title", default="GPFF evaluation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = args.names or [default_name(path) for path in args.model_dirs]
    if len(names) != len(args.model_dirs) or len(set(names)) != len(names):
        print("ERROR: give one distinct name per model dir (--names).", file=sys.stderr)
        return 1
    if args.control is not None and args.control not in names:
        print(f"ERROR: --control {args.control} is none of {', '.join(names)}.", file=sys.stderr)
        return 1
    sets = {name: report.load_set(path) for name, path in zip(names, args.model_dirs)}
    empty = [name for name, data in sets.items() if not data]
    if empty:
        print(f"ERROR: no evaluation/results.json under {', '.join(empty)}.", file=sys.stderr)
        return 1

    shared = set.intersection(*(set(data) for data in sets.values()))
    configs = sorted(args.configs or shared, key=report.config_key)
    lacking = [f"{name}: {config}" for name, data in sets.items()
               for config in configs if config not in data]
    if not configs or lacking:
        print("ERROR: no config every set has" if not configs
              else f"ERROR: not evaluated: {'; '.join(lacking)}", file=sys.stderr)
        return 1

    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sections = [
        "<p>Sets: " + "; ".join(f"<b>{html.escape(name)}</b> {html.escape(str(path))}"
                                for name, path in zip(names, args.model_dirs)) + "</p>",
        results_table(sets, configs),
    ]

    def add(figure, filename: str, heading: str, caption: str) -> None:
        if figure is None:
            return
        figure.savefig(figures_dir / filename, dpi=130, bbox_inches="tight")
        plt.close(figure)
        sections.append(f"<h2>{html.escape(heading)}</h2><p>{caption}</p>"
                        f'<img src="figures/{filename}" alt="{html.escape(heading)}">')

    if args.control is not None:
        add(report.delta_figure(sets, args.control, configs, args.seed_band), "deltas.png",
            f"Difference to {args.control}",
            "Points of the samples, ±1.96 sd. A filled dot is paired (same test molecules "
            "and seeds, McNemar sd from the discordant samples), a hollow one unpaired. "
            f"The grey band on the validity panel is ±{args.seed_band:g} points, how far "
            "separately trained runs lie apart; a difference inside it needs a second "
            "training seed.")
    add(report.failure_figure(sets, configs), "failures.png", "Why samples are invalid",
        "nonfinite: NaN or infinite coordinates. bond_orders: no bond orders fit the "
        "geometry. radical: an atom is left with an unpaired electron. fragments: the "
        "bond graph falls apart. The bar's length is 100 minus the validity.")
    add(report.size_figure(sets, configs), "validity_by_size.png", "Validity by size",
        f"RDKit validity per heavy-atom count with its 95% Wilson band; counts with "
        f"fewer than {report.MIN_SAMPLES} samples are left out.")
    for metric, scale, ylabel in EPOCH_METRICS:
        add(report.epoch_figure(sets, configs, metric, scale, ylabel), f"epoch_{metric}.png",
            f"{ylabel} by epoch", "Mean over seeds, one line per run.")
    for i, (name, data) in enumerate(sets.items()):
        color = report.SERIES[i % len(report.SERIES)]
        for metric, what, larger in (("bond_length_w1", "bond", "longer bonds"),
                                     ("bond_angle_w1", "angle", "wider angles")):
            add(report.environment_figure(data, configs, metric, color),
                f"{what}s_{i}.png", f"{what.capitalize()} environments of {name}",
                f"One dot per chemical environment, its area the environment's share of the "
                f"reference {what}s; a perfect model sits on the crossing at (0, 1). Right of "
                f"0: {larger} than the reference; above 1: more spread out. The three "
                f"environments that add most to the W1 are named.")

    index = args.output_dir / "index.html"
    index.write_text(PAGE.format(title=html.escape(args.title), body="\n".join(sections)))
    print(f"Report       : {index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
