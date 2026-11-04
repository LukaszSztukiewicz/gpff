"""Figures and a summary page of evaluated sample sets (``gpff_report``).

    gpff_report <model dir>... --output_dir DIR [--names A B ...] [--control A]

A model dir is <run>/frozen/repro/<model>, whose <config>/seed<N>/evaluation/
folders gpff_prepare_metadata and gpff_calculate_metrics filled. No sample is
evaluated again; the report reads them and writes

    DIR/index.html      a results table (validity and VUN with their 95% Wilson
                        intervals), a composition table, and the figures
    DIR/figures/*.png

The figures: the causes RDKit rejects samples for, validity by heavy-atom count,
the elongation and ring sizes of the valid molecules, the heteroatom shares
(when a set generates the atom types), the bond and angle environments of every
set, and with --control every other set's difference to it, overall and per
heavy-atom count (paired when both come from the same test molecules and
seeds). The composition table and the figures of atoms per molecule and of
formula distributions (gpff.analysis.formula) compare the samples' formulas with
those of the test molecules they were generated from, which they read from the
reference files the evaluations recorded (provenance.json); a set whose
evaluation did not record them is left out of those. Sets named <run>:<epoch>, the default name of a milestone, are also
drawn against the epoch, and --control <run> compares every such set with the
control run's same epoch, adding a config x epoch map of the differences. Only the configs every set has are shown, unless
--configs names them. A gallery at the end shows --gallery samples per failure
cause, config and set in a 3D viewer (3Dmol.js, loaded from jsDelivr when the
page opens; drawn when a section is opened).
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

# results table: results.json key, header, scale, digits (valid and vun come first, with their
# intervals)
COLUMNS = (
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
table.wide {{ font-size: 12px; }}
.scroll {{ overflow-x: auto; }}
th, td {{ padding: 3px 10px; border-bottom: 1px solid #e1e0d9; text-align: right; }}
th:nth-child(-n+2), td:nth-child(-n+2) {{ text-align: left; }}
img {{ max-width: 100%; }}
p, li {{ color: #52514e; max-width: 80ch; }}
summary {{ cursor: pointer; padding: 3px 0; }}
.mols {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 12px; }}
.mol {{ width: 220px; }}
.mol .view {{ width: 220px; height: 200px; position: relative; background: #fcfcfb;
              border: 1px solid #e1e0d9; }}
.mol p {{ font-size: 12px; margin: 2px 0; }}
</style></head><body>
<h1>{title}</h1>
{body}
</body></html>
"""
# draws the molecules of a gallery section the first time it is opened
GALLERY_SCRIPT = """<script src="https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"></script>
<script>
document.querySelectorAll("details.gallery").forEach(function (section) {
  section.addEventListener("toggle", function () {
    section.querySelectorAll(".mol[data-xyz]:not(.drawn)").forEach(function (cell) {
      cell.classList.add("drawn");
      var viewer = $3Dmol.createViewer(cell.querySelector(".view"), {backgroundColor: "#fcfcfb"});
      viewer.addModel(cell.dataset.xyz, "xyz");
      viewer.setStyle({}, {stick: {radius: 0.12}, sphere: {scale: 0.22}});
      viewer.zoomTo();
      viewer.render();
    });
  });
});
</script>"""


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


def _share_of_samples(evaluations: dict, key: str) -> str:
    """A results.json share of all samples with its 95% Wilson interval, pooled over seeds.

    "x [low, high]" in percent, "-" if an evaluation lacks the share or the number of
    its samples. Approximate for vun: it counts the first molecule with each SMILES, so
    the samples are not quite independent.
    """
    shares, sizes = [], []
    for evaluation in evaluations.values():
        n = evaluation["results"].get("n_molecules")
        if n is None and "rdkit_valid" in evaluation:
            n = len(evaluation["rdkit_valid"])
        shares.append(evaluation["results"].get(key))
        sizes.append(n)
    if any(value is None for value in shares + sizes):
        return "-"
    total = float(sum(sizes))
    hits = float(sum(share * n for share, n in zip(shares, sizes)))
    low, high = report.wilson(hits, total)
    return f"{100 * hits / total:.1f} [{100 * low:.1f}, {100 * high:.1f}]"


def _types(stats) -> str:
    """"given" if the samples have exactly their test molecules' formulas, else "generated"."""
    if stats is None:
        return "?"
    return "given" if stats["distances"]["tv"] == 0.0 else "generated"


def results_table(sets: dict, configs, formulas=None) -> str:
    """The headline metrics per config and set, as an HTML table, means over seeds.

    With the formula statistics (report.formula_table) a column says whether the
    atom types were given, the samples having exactly their test molecules'
    formulas, or generated: a set that was handed them has no composition to judge.
    """
    columns = [column for column in COLUMNS
               if any(column[0] in e["results"]
                      for data in sets.values() for config in configs
                      for e in data[config].values())]
    with_vun = any("vun" in e["results"] for data in sets.values() for config in configs
                   for e in data[config].values())
    with_types = formulas is not None and any(
        formulas[name][config] is not None for name in sets for config in configs)
    header = "".join(f"<th>{html.escape(name)}</th>" for _, name, _, _ in columns)
    lines = [f"<table><tr><th>config</th><th>set</th><th>seeds</th>"
             f"{'<th>Types</th>' if with_types else ''}<th>Valid % [95% CI]</th>"
             f"{'<th>VUN % [95% CI]</th>' if with_vun else ''}{header}</tr>"]
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
            vun = f"<td>{_share_of_samples(evaluations, 'vun')}</td>" if with_vun else ""
            types = f"<td>{_types(formulas[label][config])}</td>" if with_types else ""
            lines.append(
                f"<tr><td>{html.escape(config)}</td><td>{html.escape(label)}</td>"
                f"<td>{len(evaluations)}</td>{types}<td>{validity}</td>{vun}"
                + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"
            )
    lines.append("</table>")
    return "\n".join(lines)


RESULTS_NOTE = (
    "<p><b>Types</b>: <i>given</i> if the samples have exactly their test molecules' "
    "formulas, which means the model was handed the atom types and generated none, "
    "<i>generated</i> if it chose them; ? if the evaluation did not record its reference. "
    "Compare a set that was given the types with one that generated them only for the "
    "geometry. VUN is a share of all samples, with the Wilson interval of the pooled samples "
    "(approximate: only the first molecule with a SMILES counts, so the samples are not "
    "quite independent). Unique and Novel are shares of the <i>valid</i> molecules only, so "
    "a set with lower validity can show a higher Novel: compare VUN, and the intervals "
    "before reading a difference.</p>"
)

COMPOSITION_NOTE = (
    "<p>How the atoms of the samples compare with those of the test molecules they were "
    "generated from (sample i has test molecule i's number of atoms). <b>Element TV</b> and "
    "<b>JS</b> compare the pooled element shares, <b>N share ×</b> is the pooled share of N "
    "relative to the test molecules' (1 is right). Pooled shares say nothing about which "
    "molecules the atoms are in, so the next columns compare whole formulas. <b>Formula "
    "TV</b> is the smallest share of the molecules whose formula would have to change to make "
    "the two sets equal; <b>Formula JS</b> is the Jensen-Shannon divergence in bits, finite "
    "where one set has formulas the other lacks and weighing rare formulas more, on a scale "
    "of its own. Read both against the <b>floor</b>: what a perfect model scores, a random "
    "training molecule of each test molecule's size (mean ± sd of 20 draws). Two finite "
    "samples of one distribution are never equal, so 0 is out of reach; a set that copies "
    "the formulas scores 0 by construction. <b>Indep. atoms</b>: atoms drawn independently "
    "with the set's own element shares, which is all that matching the shares buys. "
    "<b>#N W1</b>: Wasserstein distance of the number of N atoms per molecule, in atoms. "
    "<b>New formulas</b>: share of the molecules whose formula no training or validation "
    "molecule has (the test molecules' share in brackets). <b>Feasible</b>: share of the "
    "formulas a neutral closed-shell molecule can have (even valence sum, not more H and F "
    "than the heavy atoms hold; QM9's elements only); any other is a radical whatever the "
    "geometry. <b>Valid | feasible</b>: validity of the samples whose formula is feasible, "
    "the geometry's share. <b>Valid, std.</b>: validity if the set had the test molecules' "
    "mix of N and O counts (0 for a mix it cannot generate), so that leaving out the hard "
    "molecules does not pay.</p>"
)


def feasibility_check(formulas: dict) -> str:
    """A sentence for the feasibility caption if samples are valid despite an impossible formula."""
    worst = max((stats["infeasible_valid"] for per_config in formulas.values()
                 for stats in per_config.values()
                 if stats is not None and "infeasible_valid" in stats), default=0.0)
    if worst <= 0:
        return ""
    return (f" Up to {100 * worst:.2f}% of a set's samples are valid in spite of an "
            "impossible formula (outside the bars): the rule does not cover them.")


def _pm(pair) -> str:
    mean, sd = pair
    return f"{mean:.3f} ± {sd:.3f}"


# composition table: header, what to show of formula_stats (a string, or None for "-")
COMPOSITION_COLUMNS = (
    ("Element TV", lambda s: f"{s['element']['tv']:.4f}"),
    ("Element JS 10⁻³ bit", lambda s: f"{1e3 * s['element']['js']:.2f}"),
    ("N share ×", lambda s: f"{s['ratio']['N']:.2f}"),
    ("O share ×", lambda s: f"{s['ratio']['O']:.2f}"),
    ("Formula TV", lambda s: f"{s['distances']['tv']:.3f}"),
    ("TV floor", lambda s: _pm(s["floor"]["tv"])),
    ("TV indep. atoms", lambda s: f"{s['independent']['tv']:.3f}"),
    ("Formula JS bit", lambda s: f"{s['distances']['js']:.3f}"),
    ("JS floor", lambda s: _pm(s["floor"]["js"])),
    ("JS indep. atoms", lambda s: f"{s['independent']['js']:.3f}"),
    ("#N W1", lambda s: f"{s['count_w1']['N']:.3f}"),
    ("#O W1", lambda s: f"{s['count_w1']['O']:.3f}"),
    ("New formulas % (test)",
     lambda s: f"{100 * s['unseen']:.1f} ({100 * s['unseen_reference']:.1f})"),
    ("Feasible %", lambda s: f"{100 * s['feasible']:.1f}"),
    ("Valid | feasible %", lambda s: f"{100 * s['valid_given_feasible']:.1f}"),
    ("Valid, std. %", lambda s: f"{100 * s['standardized_validity']:.1f}"),
)


def composition_table(sets: dict, configs, formulas: dict) -> str:
    """The composition of the samples against their test molecules, per config and set.

    Empty when no set has the statistics (see report.formula_stats).
    """
    if not any(formulas[name][config] is not None for name in sets for config in configs):
        return ""
    header = "".join(f"<th>{html.escape(name)}</th>" for name, _ in COMPOSITION_COLUMNS)
    lines = [f'<div class="scroll"><table class="wide"><tr><th>config</th><th>set</th>{header}</tr>']
    for config in configs:
        for label in sets:
            stats = formulas[label][config]
            cells = []
            for _, show in COMPOSITION_COLUMNS:
                try:
                    value = "-" if stats is None else show(stats)
                except (KeyError, TypeError):
                    value = "-"
                cells.append("-" if "nan" in value else value)
            lines.append(f"<tr><td>{html.escape(config)}</td><td>{html.escape(label)}</td>"
                         + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    lines.append("</table></div>")
    return "\n".join(lines)


def gallery_html(sets: dict, configs, n: int) -> str:
    """The gallery: one collapsible section per set, config and failure cause."""
    parts = [
        "<h2>Gallery</h2>",
        f"<p>Up to {n} samples per cause of every set and config, drawn at random (seed 0) "
        "from its first seed; a section draws them when it is opened. # is the row in the "
        "samples database. The viewer guesses its bonds from the distances, which need "
        "not be RDKit's.</p>",
    ]
    for name, data in sets.items():
        groups: dict = {}
        for entry in report.gallery(data, configs, n):
            groups.setdefault((entry["config"], entry["cause"]), []).append(entry)
        for (config, cause), entries in groups.items():
            cells = []
            for entry in entries:
                label = f"#{entry['index']} {html.escape(entry['formula'])}"
                if entry["xyz"] is None:
                    cells.append(f'<div class="mol"><p>{label}: non-finite coordinates</p></div>')
                else:
                    cells.append(f'<div class="mol" data-xyz="{html.escape(entry["xyz"])}">'
                                 f'<div class="view"></div><p>{label}</p></div>')
            parts.append(
                f'<details class="gallery"><summary>{html.escape(name)} · {html.escape(config)} · '
                f'{cause}: {entries[0]["n_cause"]} of {entries[0]["n"]} samples</summary>'
                f'<div class="mols">{"".join(cells)}</div></details>'
            )
    parts.append(GALLERY_SCRIPT)
    return "\n".join(parts)


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
                        help="The set the others are compared with, or a run: then every "
                             "<run>:<epoch> is compared with the control run's same epoch.")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Configs to show. Default: those every set has.")
    parser.add_argument("--seed_band", type=float, default=1.5,
                        help="Points of validity separately trained runs differ by; "
                             "shaded in the comparison. Default: 1.5.")
    parser.add_argument("--gallery", type=int, default=4,
                        help="Samples per failure cause, config and set in the gallery; "
                             "0 leaves it out. Default: 4.")
    parser.add_argument("--title", default="GPFF evaluation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = args.names or [default_name(path) for path in args.model_dirs]
    if len(names) != len(args.model_dirs) or len(set(names)) != len(names):
        print("ERROR: give one distinct name per model dir (--names).", file=sys.stderr)
        return 1
    try:
        styles = report.set_styles(names)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    sets = {name: report.load_set(path) for name, path in zip(names, args.model_dirs)}
    empty = [name for name, data in sets.items() if not data]
    if empty:
        print(f"ERROR: no evaluation/results.json under {', '.join(empty)}.", file=sys.stderr)
        return 1
    pairs = report.control_pairs(sets, args.control) if args.control is not None else {}
    if args.control is not None and not pairs:
        print(f"ERROR: --control {args.control} is no set, nor a run with an epoch another "
              f"set has; the sets are {', '.join(names)}.", file=sys.stderr)
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
    formulas = report.formula_table(sets, configs)
    sections = [
        "<p>Sets: " + "; ".join(f"<b>{html.escape(name)}</b> {html.escape(str(path))}"
                                for name, path in zip(names, args.model_dirs)) + "</p>",
        results_table(sets, configs, formulas),
        RESULTS_NOTE,
    ]
    table = composition_table(sets, configs, formulas)
    if table:
        sections += ["<h2>Composition</h2>", COMPOSITION_NOTE, table]

    def add(figure, filename: str, heading: str, caption: str) -> None:
        if figure is None:
            return
        figure.savefig(figures_dir / filename, dpi=130, bbox_inches="tight")
        plt.close(figure)
        sections.append(f"<h2>{html.escape(heading)}</h2><p>{caption}</p>"
                        f'<img src="figures/{filename}" alt="{html.escape(heading)}">')

    if pairs:
        against = (f"Each set against {args.control} at its own epoch. "
                   if args.control not in sets else "")
        add(report.epoch_delta_figure(sets, pairs, configs, args.seed_band, args.control),
            "deltas_by_epoch.png", f"Difference to {args.control} by epoch",
            "RDKit validity of each run minus the control run at the same epoch, in points; "
            "blue is ahead, red behind. * marks a difference outside both its 95% interval "
            f"and ±{args.seed_band:g} points, the spread of separately trained runs.")
        add(report.delta_figure(sets, pairs, configs, args.seed_band, args.control),
            "deltas.png", f"Difference to {args.control}",
            f"{against}Points of the samples, ±1.96 sd. A filled dot is paired "
            "(same test molecules and seeds, McNemar sd from the discordant samples), a "
            "hollow one unpaired. The grey band on the validity panel is "
            f"±{args.seed_band:g} points, how far separately trained runs lie apart; a "
            "difference inside it needs a second training seed.")
        add(report.size_delta_figure(sets, pairs, configs, args.control), "deltas_by_size.png",
            f"Difference to {args.control} by size",
            "The validity difference per heavy-atom count (or size bin, as in validity by "
            "size), ±1.96 sd from the samples of that size valid in only one of the two; "
            "paired comparisons only. Shows which molecules a change helps.")
    add(report.failure_figure(sets, configs), "failures.png", "Why samples are invalid",
        "nonfinite: NaN or infinite coordinates. bond_orders: no bond orders fit the "
        "geometry. radical: an atom is left with an unpaired electron. fragments: the "
        "bond graph falls apart. The bar's length is 100 minus the validity.")
    add(report.size_figure(sets, configs), "validity_by_size.png", "Validity by size",
        f"RDKit validity per heavy-atom count with its 95% Wilson band; counts with "
        f"fewer than {report.MIN_SAMPLES} samples are left out. Counts spanning more than "
        f"{report.MAX_SIZES} values (GEOM-Drugs) are grouped into up to {report.MAX_SIZES} "
        "bins of equal sample count, each drawn at its mean size.")
    add(report.composition_figure(sets, configs), "composition.png", "Heteroatoms",
        "The share of the atoms and of the molecules each heteroatom has, for models that "
        "generate the atom types; an untyped set has its test molecules' composition. "
        "Fewer N or O than that makes validity easier without the model getting better.")
    add(report.count_figure(sets, configs, formulas), "formula_counts.png",
        "Atoms per molecule",
        "The share of the molecules with 0, 1, 2 and 3 or more N atoms (top) and O atoms "
        "(bottom); the black dashes are the test molecules'. The pooled shares above cannot "
        "tell a few molecules with many N from many molecules with one.")
    add(report.formula_distance_figure(sets, configs, formulas), "formula_distances.png",
        "Formula distributions",
        "Distance between the formulas (C7H11NO) of the samples and those of the test "
        "molecules they were generated from: total variation on top, Jensen-Shannon below. "
        "The black dash is the oracle floor, what a model that draws from the data scores "
        "(mean ± sd of draws); it is not 0. The open circle is atoms drawn independently with "
        "the set's own element shares: a bar there has learned no more than the shares. A set "
        "that copies the formulas has no bar.")
    add(report.formula_scatter_figure(sets, configs, formulas), "formula_scatter.png",
        "Formula frequencies",
        "The share of every formula in the samples against its share in the test molecules, "
        "log-log; the line is equality. Grey: one oracle draw, how far a perfect model "
        "scatters by chance, so a set is worse only where its cloud is wider than the grey "
        "one. A formula missing on one side sits on the half-sample line. The three formulas "
        "that differ most are named.")
    add(report.feasibility_figure(sets, configs, formulas), "formula_feasibility.png",
        "Invalid samples: formula or geometry",
        "Height: the share of all samples that are invalid. Dark: the formula is one no "
        "neutral closed-shell molecule can have (odd valence sum, or more H and F than the "
        "heavy atoms hold), so no geometry could be valid: the type generation's doing. "
        "Light: the formula is possible, the geometry still is not."
        + feasibility_check(formulas))
    add(report.elongation_figure(sets, configs), "elongation.png", "Elongation",
        "Cumulative distribution of the valid molecules' largest atom distance; dashed, "
        "the reference elongation_w1 was computed against, when an evaluation recorded it.")
    add(report.ring_figure(sets, configs), "rings.png", "Rings",
        "Rings per valid molecule by ring size. For scale: the QM9 database itself has "
        "0.50, 0.45, 0.48 and 0.16 rings of 3, 4, 5 and 6 atoms per molecule (43%, 35%, "
        "39% and 14% of its molecules have one).")
    for metric, scale, ylabel in EPOCH_METRICS:
        add(report.epoch_figure(sets, configs, metric, scale, ylabel), f"epoch_{metric}.png",
            f"{ylabel} by epoch", "Mean over seeds, one line per run.")
    for i, (name, data) in enumerate(sets.items()):
        for metric, what, larger in (("bond_length_w1", "bond", "longer bonds"),
                                     ("bond_angle_w1", "angle", "wider angles")):
            add(report.environment_figure(data, configs, metric, styles[name]["color"]),
                f"{what}s_{i}.png", f"{what.capitalize()} environments of {name}",
                f"One dot per chemical environment, its area the environment's share of the "
                f"reference {what}s; a perfect model sits on the crossing at (0, 1). Right of "
                f"0: {larger} than the reference; above 1: more spread out. The three "
                f"environments that add most to the W1 are named.")
    if args.gallery > 0:
        sections.append(gallery_html(sets, configs, args.gallery))

    index = args.output_dir / "index.html"
    index.write_text(PAGE.format(title=html.escape(args.title), body="\n".join(sections)))
    print(f"Report       : {index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
