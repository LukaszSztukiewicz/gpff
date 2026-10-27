"""Figures of evaluated sample sets, for ``gpff_report``.

A set is one model's repro folder, <run>/frozen/repro/<model>, holding
<config>/seed<N>/evaluation/ per sampler config. The figures read the
per-sample arrays of gpff_prepare_metadata (rdkit_valid, failure,
element_counts, elongation, ring_counts) and the results and breakdowns of
gpff_calculate_metrics; what an evaluation lacks is left out of the figures,
never filled in. The gallery also reads a few rows of the sample database.
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.data import chemical_symbols  # noqa: E402
from ase.db import connect  # noqa: E402

from gpff.analysis.structure import FAILURES  # noqa: E402

__all__ = [
    "CONFIG_ORDER",
    "composition_figure",
    "control_pairs",
    "delta_figure",
    "difference",
    "elongation_figure",
    "environment_figure",
    "epoch_delta_figure",
    "epoch_figure",
    "failure_figure",
    "gallery",
    "load_set",
    "pooled",
    "reference_path",
    "ring_figure",
    "set_styles",
    "size_delta_figure",
    "size_figure",
    "wilson",
]

# categorical colors, used in this order and never repeated: a run (see
# set_styles), or the i-th failure cause, gets SERIES[i]
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
# what tells the sets of one run apart: lines, markers and bar hatches, in order
LINESTYLES = ("-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)))
MARKERS = ("o", "s", "^", "D", "v")
HATCHES = ("", "//", "..", "xx", "\\\\")
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
)
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
    "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
    "font.size": 9, "lines.linewidth": 2, "legend.frameon": False,
}
# the repro configs in the paper's order; any other config follows by name
CONFIG_ORDER = ("dd16", "sdd16", "sddshape16", "heun16", "sheun16", "anc16",
                "dd256", "sdd256", "sddshape256", "heun256", "sheun256", "anc256")
# fewest samples a point of the by-size figures needs
MIN_SAMPLES = 20
# heavy-atom counts spanning more than this many values (GEOM-Drugs) are binned
MAX_SIZES = 15


def config_key(config: str) -> tuple:
    if config in CONFIG_ORDER:
        return (0, CONFIG_ORDER.index(config), "")
    return (1, 0, config)


def set_styles(labels) -> dict:
    """{label: color, linestyle, marker, hatch}, so that a set looks the same in every figure.

    A set <run>:<model> takes its run's color, the runs in order of first
    appearance, and the sets of one run differ by line style, marker and bar
    hatch. Raises ValueError for more runs than SERIES has colors: a repeated
    color would make two runs look alike.
    """
    runs = list(dict.fromkeys(label.split(":")[0] for label in labels))
    if len(runs) > len(SERIES):
        raise ValueError(f"{len(runs)} runs, but only {len(SERIES)} colors; report fewer at once.")
    styles, seen = {}, {}
    for label in labels:
        run = label.split(":")[0]
        k = seen.get(run, 0)
        seen[run] = k + 1
        styles[label] = {"color": SERIES[runs.index(run)],
                         "linestyle": LINESTYLES[k % len(LINESTYLES)],
                         "marker": MARKERS[k % len(MARKERS)],
                         "hatch": HATCHES[k % len(HATCHES)]}
    return styles


def load_evaluation(evaluation_dir: Path) -> dict:
    """The results and the per-sample arrays the figures use of one evaluation folder."""
    metadata = evaluation_dir / "metadata"
    evaluation = {"results": json.loads((evaluation_dir / "results.json").read_text())}
    # the database the arrays were made from: Open Babel's H for heavy-atom data
    for name in ("samples_h.db", "samples.db"):
        if (evaluation_dir.parent / name).exists():
            evaluation["db"] = evaluation_dir.parent / name
            break
    for key in ("rdkit_valid", "failure", "elongation", "ring_counts"):
        if (metadata / f"{key}.npy").exists():
            evaluation[key] = np.load(metadata / f"{key}.npy")
    if (metadata / "element_counts.npz").exists():
        with np.load(metadata / "element_counts.npz") as table:
            evaluation["sizes"] = sum(table[symbol] for symbol in table.files)
            evaluation["heavy_atoms"] = evaluation["sizes"] - (
                table["H"] if "H" in table.files else 0
            )
    for name in ("bond_length_w1", "bond_angle_w1", "provenance"):
        if (evaluation_dir / f"{name}.json").exists():
            evaluation[name] = json.loads((evaluation_dir / f"{name}.json").read_text())
    return evaluation


def reference_path(sets: dict, metric: str, argument: str):
    """The reference file `metric` was computed against, as provenance.json records it.

    The first evaluation that has one decides; None if no set recorded it.
    """
    for data in sets.values():
        for evaluations in data.values():
            for evaluation in evaluations.values():
                entry = evaluation.get("provenance", {}).get("results", {}).get(metric, {})
                path = entry.get("inputs", {}).get(argument, {}).get("path")
                if path and Path(path).exists():
                    return Path(path)
    return None


def load_set(model_dir) -> dict:
    """{config: {seed: evaluation}} of every evaluated config of a model folder."""
    data: dict = {}
    for results_path in sorted(Path(model_dir).glob("*/seed*/evaluation/results.json")):
        seed_dir = results_path.parent.parent
        data.setdefault(seed_dir.parent.name, {})[seed_dir.name] = load_evaluation(
            results_path.parent
        )
    return data


def pooled(evaluations: dict, get, seeds=None):
    """get(evaluation) of every seed (or of `seeds`) one after the other; None if any lacks it."""
    try:
        return np.concatenate([get(evaluations[seed]) for seed in sorted(seeds or evaluations)])
    except KeyError:
        return None


def wilson(k, n, z: float = 1.96):
    """The Wilson score interval of k successes in n trials (arrays work too)."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    p = k / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return center - half, center + half


def _same_rows(a: dict, b: dict) -> bool:
    return "sizes" in a and "sizes" in b and np.array_equal(a["sizes"], b["sizes"])


def _size_bins(heavy: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """(x, mask) of every point of a by-size figure.

    One point per heavy-atom count with MIN_SAMPLES samples or more; counts that
    span more than MAX_SIZES values are grouped instead into up to MAX_SIZES bins
    of equal sample count, each drawn at its mean size, so that a sparse tail of
    large molecules does not draw lines across sizes nobody sampled.
    """
    sizes, n = np.unique(heavy, return_counts=True)
    if sizes.max() - sizes.min() <= MAX_SIZES:
        return [(float(size), heavy == size) for size, count in zip(sizes, n)
                if count >= MIN_SAMPLES]
    n_bins = int(min(MAX_SIZES, len(heavy) // MIN_SAMPLES))
    edges = np.unique(np.quantile(heavy, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.searchsorted(edges, heavy, side="right") - 1, 0, len(edges) - 2)
    return [(float(heavy[bins == b].mean()), bins == b) for b in range(len(edges) - 1)
            if (bins == b).sum() >= MIN_SAMPLES]


def _paired_seeds(a: dict, b: dict) -> list:
    """The seeds a and b share, if all of them hold the same test molecules row by row; else []."""
    common = sorted(set(a) & set(b))
    return common if common and all(_same_rows(a[seed], b[seed]) for seed in common) else []


def _mcnemar(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """y minus x in points of two paired booleans, and its sd from the discordant pairs."""
    n10, n01 = int((x & ~y).sum()), int((~x & y).sum())
    return 100 * (n01 - n10) / len(x), 100 * np.sqrt(n10 + n01) / len(x)


def difference(a: dict, b: dict, get) -> tuple[float, float, bool]:
    """b minus a in points of a per-sample boolean, its sd, and whether it is paired.

    a and b are {seed: evaluation} of one config. They are paired when they share
    seeds and every shared seed's samples have the same atom counts row by row,
    i.e. were generated from the same test molecules with the same prior draws;
    the sd then comes from the discordant samples (McNemar), otherwise from two
    independent proportions. nan if either lacks the array.
    """
    common = _paired_seeds(a, b)
    if common:
        x, y = pooled(a, get, common), pooled(b, get, common)
        if x is None or y is None:
            return np.nan, np.nan, True
        return (*_mcnemar(x, y), True)
    x, y = pooled(a, get), pooled(b, get)
    if x is None or y is None:
        return np.nan, np.nan, False
    px, py = x.mean(), y.mean()
    sd = np.sqrt(px * (1 - px) / len(x) + py * (1 - py) / len(y))
    return 100 * (py - px), 100 * sd, False


def _causes(sets: dict, configs) -> list[str]:
    """The failure causes that occur in any of the sets' configs, in FAILURES order."""
    seen = set()
    for data in sets.values():
        for config in configs:
            failure = pooled(data[config], lambda e: e["failure"])
            if failure is not None:
                seen.update(np.unique(failure).tolist())
    return [cause for cause in FAILURES[1:] if cause in seen]


def failure_figure(sets: dict, configs):
    """The share of the samples RDKit rejects, split by cause: one bar per config and set."""
    causes = _causes(sets, configs)
    if not causes:
        return None
    rows = [(config, label) for config in configs for label in sets]
    with plt.rc_context(STYLE):
        figure, ax = plt.subplots(figsize=(7.5, 0.28 * len(rows) + 1.4))
        y = np.arange(len(rows)) + np.repeat(np.arange(len(configs)), len(sets)) * 0.6
        left = np.zeros(len(rows))
        for i, cause in enumerate(causes):
            shares = []
            for config, label in rows:
                failure = pooled(sets[label][config], lambda e: e["failure"])
                shares.append(np.nan if failure is None else 100 * np.mean(failure == cause))
            shares = np.array(shares)
            ax.barh(y, shares, left=left, height=0.8, color=SERIES[i], label=cause,
                    edgecolor=SURFACE, linewidth=1)
            left += np.nan_to_num(shares)
        ax.set_yticks(y, [f"{config}  {label}" for config, label in rows])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("samples RDKit rejects (%), by cause")
        ax.legend(ncols=len(causes), loc="lower left", bbox_to_anchor=(0, 1.0))
        figure.tight_layout()
    return figure


def size_figure(sets: dict, configs):
    """RDKit validity against the number of heavy atoms, with 95% Wilson bands."""
    styles = set_styles(sets)
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(
            1, len(configs), sharey=True, squeeze=False, figsize=(2.5 * len(configs) + 1, 3.2)
        )
        drawn = False
        for ax, config in zip(axes[0], configs):
            for label, data in sets.items():
                heavy = pooled(data[config], lambda e: e["heavy_atoms"])
                valid = pooled(data[config], lambda e: e["rdkit_valid"])
                if heavy is None or valid is None:
                    continue
                points = _size_bins(heavy)
                if not points:
                    continue
                sizes = np.array([x for x, _ in points])
                k = np.array([valid[mask].sum() for _, mask in points])
                n = np.array([mask.sum() for _, mask in points])
                low, high = wilson(k, n)
                style = styles[label]
                ax.fill_between(sizes, 100 * low, 100 * high, color=style["color"], alpha=0.12,
                                linewidth=0)
                ax.plot(sizes, 100 * k / n, color=style["color"], linestyle=style["linestyle"],
                        label=label)
                drawn = True
            ax.set_title(config)
            ax.set_xlabel("heavy atoms")
        if not drawn:
            plt.close(figure)
            return None
        axes[0][0].set_ylabel("RDKit valid (%)")
        if len(sets) > 1:
            figure.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center",
                          ncols=min(len(sets), 4), bbox_to_anchor=(0.5, 1.02))
        figure.tight_layout(rect=(0, 0, 1, 0.9 if len(sets) > 1 else 1))
    return figure


def environment_figure(data: dict, configs, metric: str, color: str, n_labels: int = 3):
    """Each chemical environment's median shift and spread ratio against the reference.

    One panel per config (first seed), one dot per compared environment, its area
    the environment's share of the reference. A perfect model sits at (0, 1);
    the n_labels environments that contribute most to the W1 are named.
    """
    scale, unit = (1000, "mÅ") if metric == "bond_length_w1" else (1, "°")
    panels = [(config, next(iter(data[config].values())).get(metric)) for config in configs]
    panels = [(config, breakdown) for config, breakdown in panels if breakdown]
    if not panels:
        return None
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(
            1, len(panels), sharey=True, squeeze=False, figsize=(2.6 * len(panels) + 1, 3.2)
        )
        for ax, (config, breakdown) in zip(axes[0], panels):
            rows = [(name, row) for name, row in breakdown["labels"].items()
                    if row["w1"] is not None and np.isfinite(row["w1"])]
            shift = np.array([row["median_shift"] for _, row in rows]) * scale
            spread = np.array([row["spread_ratio"] for _, row in rows], dtype=float)
            weight = np.array([row["weight"] for _, row in rows])
            ax.axvline(0, color=AXIS, linewidth=1)
            ax.axhline(1, color=AXIS, linewidth=1)
            ax.scatter(shift, spread, s=600 * weight + 8, color=color, alpha=0.6,
                       edgecolors=SURFACE, linewidths=1)
            for j in np.argsort(-weight * np.array([row["w1"] for _, row in rows]))[:n_labels]:
                ax.annotate(rows[j][0], (shift[j], spread[j]), fontsize=7, color=INK_SECONDARY,
                            xytext=(4, 4), textcoords="offset points")
            ax.set_yscale("log")
            ax.set_title(f"{config}   W1 {breakdown['w1'] * scale:.2f} {unit}")
            ax.set_xlabel(f"median shift ({unit})")
        axes[0][0].set_ylabel("spread ratio (IQR / reference IQR)")
        figure.tight_layout()
    return figure


def control_pairs(sets: dict, control: str) -> dict:
    """{arm: the set it is compared with}, for every set but the controls.

    `control` names a set, the control of every other one, or a run: then each set
    <run>:<epoch> of another run is compared with <control>:<epoch>, the control
    at the same epoch, and a set whose epoch the control run lacks is left out.
    """
    if control in sets:
        return {label: control for label in sets if label != control}
    pairs = {}
    for label in sets:
        match = re.fullmatch(r"(.+):(\d+)", label)
        if match and match[1] != control and f"{control}:{match[2]}" in sets:
            pairs[label] = f"{control}:{match[2]}"
    return pairs


def delta_figure(sets: dict, pairs: dict, configs, seed_band: float, control: str):
    """Each arm's difference to its control (see control_pairs) in validity and the failure shares.

    Bars are ±1.96 sd; a hollow dot is an unpaired comparison. The shaded band on
    the validity panel is ±seed_band points, the spread of separately trained runs.
    """
    measures = [("RDKit valid", lambda e: e["rdkit_valid"].astype(bool))] + [
        (cause, lambda e, cause=cause: e["failure"] == cause)
        for cause in _causes(sets, configs)
    ]
    styles = set_styles(sets)
    arms = list(pairs)
    if not arms:
        return None
    offsets = (np.arange(len(arms)) - (len(arms) - 1) / 2) * 0.6 / max(len(arms), 1)
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(
            1, len(measures), sharey=True, squeeze=False,
            figsize=(2.3 * len(measures) + 1.5, 0.4 * len(configs) + 1.6),
        )
        for ax, (name, get) in zip(axes[0], measures):
            if ax is axes[0][0]:
                ax.axvspan(-seed_band, seed_band, color=GRID, alpha=0.7, linewidth=0)
            ax.axvline(0, color=AXIS, linewidth=1)
            for arm, offset in zip(arms, offsets):
                color, marker = styles[arm]["color"], styles[arm]["marker"]
                for row, config in enumerate(configs):
                    delta, sd, is_paired = difference(sets[pairs[arm]][config], sets[arm][config], get)
                    ax.errorbar(delta, row + offset, xerr=1.96 * sd, fmt=marker, markersize=6,
                                color=color, markerfacecolor=color if is_paired else SURFACE,
                                elinewidth=1.5, capsize=0,
                                label=arm if (row == 0 and ax is axes[0][0]) else None)
            ax.set_title(name)
            ax.grid(axis="y", visible=False)
        axes[0][0].set_yticks(range(len(configs)), configs)
        axes[0][0].invert_yaxis()
        figure.supxlabel(f"difference to {control} (points of the samples)", fontsize=9,
                         color=INK_SECONDARY)
        figure.legend(loc="upper center", ncols=min(len(arms), 4), bbox_to_anchor=(0.5, 1.02))
        figure.tight_layout(rect=(0, 0, 1, 0.88))
    return figure


def epoch_figure(sets: dict, configs, metric: str, scale: float, ylabel: str):
    """A results metric against the training epoch, one line per run.

    Sets labelled <run>:<epoch> make the points; a run needs two epochs for a line.
    The value is the mean over seeds.
    """
    runs: dict = {}
    for label in sets:
        match = re.fullmatch(r"(.+):(\d+)", label)
        if match:
            runs.setdefault(match[1], []).append((int(match[2]), label))
    runs = {run: sorted(points) for run, points in runs.items() if len(points) > 1}
    if not runs:
        return None
    styles = set_styles(sets)
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(
            1, len(configs), squeeze=False, figsize=(2.5 * len(configs) + 1, 3.2)
        )
        drawn = False
        for ax, config in zip(axes[0], configs):
            for run, points in runs.items():
                epochs, values = [], []
                for epoch, label in points:
                    seeds = sets[label][config].values()
                    found = [e["results"][metric] for e in seeds if metric in e["results"]]
                    if found:
                        epochs.append(epoch)
                        values.append(scale * np.mean(found))
                if epochs:
                    ax.plot(epochs, values, color=styles[points[0][1]]["color"], marker="o",
                            markersize=5, label=run)
                    drawn = True
            # milestones double from 100 to 10000, so a log axis with a tick at each
            ticks = sorted({epoch for points in runs.values() for epoch, _ in points})
            ax.set_xscale("log")
            ax.set_xticks(ticks, [str(epoch) for epoch in ticks])
            ax.minorticks_off()
            ax.set_title(config)
            ax.set_xlabel("epoch")
        if not drawn:
            plt.close(figure)
            return None
        axes[0][0].set_ylabel(ylabel)
        if len(runs) > 1:
            figure.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center",
                          ncols=min(len(runs), 4), bbox_to_anchor=(0.5, 1.02))
        figure.tight_layout(rect=(0, 0, 1, 0.9 if len(runs) > 1 else 1))
    return figure


def _legend_above(figure, axes, n_series: int) -> None:
    """One legend above the panels when there is more than one series."""
    if n_series > 1:
        figure.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center",
                      ncols=min(n_series, 4), bbox_to_anchor=(0.5, 1.02))
    figure.tight_layout(rect=(0, 0, 1, 0.9 if n_series > 1 else 1))


def _mean_result(evaluations: dict, key: str) -> float:
    values = [e["results"][key] for e in evaluations.values() if key in e["results"]]
    return float(np.mean(values)) if values else np.nan


def composition_figure(sets: dict, configs):
    """The heteroatoms' share of the atoms and of the molecules, per config and set.

    Only drawn when some set changed the composition of its test molecules
    (element_tv > 0), i.e. a model that generates the atom types; an untyped set
    next to it shows the composition the others should match. H and C are left
    out, they would dwarf the rest.
    """
    if not any(_mean_result(data[config], "element_tv") > 0
               for data in sets.values() for config in configs):
        return None
    elements = sorted(
        {key.removeprefix("element_fraction_") for data in sets.values() for config in configs
         for e in data[config].values() for key in e["results"]
         if key.startswith("element_fraction_")} - {"H", "C"},
        key=chemical_symbols.index,
    )
    width = 0.8 / len(sets)
    styles = set_styles(sets)
    panels = (("element_fraction_", "share of the atoms (%)"),
              ("molecule_fraction_", "share of the molecules (%)"))
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(2, len(configs), squeeze=False, sharey="row",
                                    figsize=(2.6 * len(configs) + 1, 5.4))
        for column, config in enumerate(configs):
            for row, (prefix, ylabel) in enumerate(panels):
                ax = axes[row][column]
                for i, (name, data) in enumerate(sets.items()):
                    values = [100 * _mean_result(data[config], prefix + symbol) for symbol in elements]
                    ax.bar(np.arange(len(elements)) + (i - (len(sets) - 1) / 2) * width, values,
                           width, color=styles[name]["color"], hatch=styles[name]["hatch"],
                           label=name, edgecolor=SURFACE, linewidth=1)
                ax.set_xticks(range(len(elements)), elements)
                ax.grid(axis="x", visible=False)
                if row == 0:
                    ax.set_title(config)
                if column == 0:
                    ax.set_ylabel(ylabel)
        _legend_above(figure, axes, len(sets))
    return figure


def elongation_figure(sets: dict, configs):
    """The cumulative distribution of the valid molecules' elongation, per config.

    The reference distribution elongation_w1 was computed against, when an
    evaluation recorded it in its provenance, is drawn dashed in every panel.
    """
    def valid_elongation(e):
        values = e["elongation"][e["rdkit_valid"].astype(bool)]
        return values[np.isfinite(values)]

    path = reference_path(sets, "elongation_w1", "reference_elongation_path")
    reference = np.sort(np.load(path)) if path is not None else None
    styles = set_styles(sets)
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, len(configs), sharey=True, squeeze=False,
                                    figsize=(2.5 * len(configs) + 1, 3.2))
        drawn = False
        for ax, config in zip(axes[0], configs):
            if reference is not None:
                ax.plot(reference, np.arange(1, len(reference) + 1) / len(reference),
                        color=INK_SECONDARY, linestyle="--", linewidth=1.5, label="reference")
            for name, data in sets.items():
                values = pooled(data[config], valid_elongation)
                if values is None or not values.size:
                    continue
                values = np.sort(values)
                ax.plot(values, np.arange(1, len(values) + 1) / len(values),
                        color=styles[name]["color"], linestyle=styles[name]["linestyle"],
                        label=name)
                drawn = True
            ax.set_title(config)
            ax.set_xlabel("elongation (Å)")
        if not drawn:
            plt.close(figure)
            return None
        axes[0][0].set_ylabel("share of the valid molecules")
        _legend_above(figure, axes, len(sets) + (reference is not None))
    return figure


def ring_figure(sets: dict, configs):
    """Rings of 3, 4, 5 and 6 atoms per valid molecule, per config and set."""
    def valid_rings(e):
        return e["ring_counts"][e["rdkit_valid"].astype(bool)]

    width = 0.8 / len(sets)
    styles = set_styles(sets)
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, len(configs), sharey=True, squeeze=False,
                                    figsize=(2.5 * len(configs) + 1, 3.2))
        drawn = False
        for ax, config in zip(axes[0], configs):
            for i, (name, data) in enumerate(sets.items()):
                rings = pooled(data[config], valid_rings)
                if rings is None or not len(rings):
                    continue
                ax.bar(np.arange(4) + (i - (len(sets) - 1) / 2) * width, np.nanmean(rings, axis=0),
                       width, color=styles[name]["color"], hatch=styles[name]["hatch"],
                       label=name, edgecolor=SURFACE, linewidth=1)
                drawn = True
            ax.set_xticks(range(4), ["3", "4", "5", "6"])
            ax.grid(axis="x", visible=False)
            ax.set_title(config)
            ax.set_xlabel("ring size (atoms)")
        if not drawn:
            plt.close(figure)
            return None
        axes[0][0].set_ylabel("rings per valid molecule")
        _legend_above(figure, axes, len(sets))
    return figure


def size_delta_figure(sets: dict, pairs: dict, configs, control: str):
    """Each arm's validity difference to its control per heavy-atom count, ±1.96 sd.

    Paired comparisons only (the same test molecules and seeds), so the sd comes
    from the samples of that size valid in one set and not the other. The points
    are those of _size_bins, binned like size_figure's.
    """
    styles = set_styles(sets)
    arms = list(pairs)
    if not arms:
        return None
    valid = lambda e: e["rdkit_valid"].astype(bool)  # noqa: E731
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, len(configs), sharey=True, squeeze=False,
                                    figsize=(2.5 * len(configs) + 1, 3.2))
        drawn = False
        for ax, config in zip(axes[0], configs):
            ax.axhline(0, color=AXIS, linewidth=1)
            for j, arm in enumerate(arms):
                a, b = sets[pairs[arm]][config], sets[arm][config]
                seeds = _paired_seeds(a, b)
                heavy = pooled(a, lambda e: e["heavy_atoms"], seeds) if seeds else None
                x, y = (pooled(a, valid, seeds), pooled(b, valid, seeds)) if seeds else (None, None)
                if heavy is None or x is None or y is None:
                    continue
                points = _size_bins(heavy)
                if not points:
                    continue
                sizes = np.array([size for size, _ in points])
                gaps = np.array([_mcnemar(x[mask], y[mask]) for _, mask in points])
                offset = (j - (len(arms) - 1) / 2) * 0.15
                style = styles[arm]
                ax.errorbar(sizes + offset, gaps[:, 0], yerr=1.96 * gaps[:, 1],
                            marker=style["marker"], linestyle=style["linestyle"], markersize=5,
                            color=style["color"], elinewidth=1.5, capsize=0, label=arm)
                drawn = True
            ax.set_title(config)
            ax.set_xlabel("heavy atoms")
        if not drawn:
            plt.close(figure)
            return None
        axes[0][0].set_ylabel(f"Δ RDKit valid vs {control} (points)")
        _legend_above(figure, axes, len(arms))
    return figure


def epoch_delta_figure(sets: dict, pairs: dict, configs, seed_band: float, control: str):
    """Each run's validity difference to the control run, per config and epoch.

    Only for a control that is a run (see control_pairs), and runs with two or
    more epochs: one panel per run, a cell per config and epoch, blue where the
    run is ahead of the control at that epoch and red where it is behind. A cell
    is starred when its difference lies outside both its 95% interval and
    ±seed_band, the spread of separately trained runs.
    """
    runs: dict = {}
    for arm, against in pairs.items():
        match = re.fullmatch(r"(.+):(\d+)", arm)
        if match and against != control:
            runs.setdefault(match[1], []).append((int(match[2]), arm))
    runs = {run: sorted(points) for run, points in runs.items() if len(points) > 1}
    if not runs:
        return None
    valid = lambda e: e["rdkit_valid"].astype(bool)  # noqa: E731
    grids = {}
    for run, points in runs.items():
        grid = np.full((len(configs), len(points), 2), np.nan)
        for column, (_, arm) in enumerate(points):
            for row, config in enumerate(configs):
                grid[row, column] = difference(sets[pairs[arm]][config], sets[arm][config], valid)[:2]
        grids[run] = grid
    largest = np.nanmax([np.nanmax(np.abs(grid[..., 0])) for grid in grids.values()])
    limit = max(float(largest) if np.isfinite(largest) else 0.0, 2 * seed_band)
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "diverging", ["#e34948", "#f0efec", "#2a78d6"]
    )
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(
            1, len(runs), squeeze=False, layout="constrained",
            figsize=(sum(0.8 * len(points) + 1 for points in runs.values()) + 2,
                     0.42 * len(configs) + 1.6),
            gridspec_kw={"width_ratios": [len(points) for points in runs.values()]},
        )
        for ax, (run, points) in zip(axes[0], runs.items()):
            grid = grids[run]
            image = ax.imshow(grid[..., 0], cmap=colormap, vmin=-limit, vmax=limit, aspect="auto")
            for row, column in np.ndindex(grid.shape[:2]):
                delta, sd = grid[row, column]
                if np.isfinite(delta):
                    star = "*" if abs(delta) > max(1.96 * sd, seed_band) else ""
                    ax.text(column, row, f"{delta:+.1f}{star}", ha="center", va="center",
                            fontsize=8, color=INK)
            ax.set_xticks(range(len(points)), [str(epoch) for epoch, _ in points])
            ax.set_yticks(range(len(configs)), configs if ax is axes[0][0] else [])
            ax.grid(False)
            ax.set_title(run)
            ax.set_xlabel("epoch")
        figure.colorbar(image, ax=axes[0].tolist(), shrink=0.9, label="Δ RDKit valid (points)")
    return figure


def _xyz(atoms) -> str:
    lines = [str(len(atoms)), atoms.get_chemical_formula()]
    lines += [f"{symbol} {x:.4f} {y:.4f} {z:.4f}"
              for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions)]
    return "\n".join(lines)


def gallery(data: dict, configs, n: int, seed: int = 0) -> list[dict]:
    """Up to n samples per failure cause (valid ones too) of every config's first seed.

    Drawn at random with a fixed seed. Each entry has the config, cause, sample
    index (its row in the samples database), formula, the cause's count and the
    number of samples, and the structure as XYZ text (None for non-finite
    coordinates, which no viewer can show).
    """
    rng = np.random.default_rng(seed)
    entries = []
    for config in configs:
        evaluation = next(iter(data[config].values()))
        if "failure" not in evaluation or "db" not in evaluation:
            continue
        failure = evaluation["failure"]
        with connect(evaluation["db"]) as db:
            for cause in FAILURES:
                rows = np.flatnonzero(failure == cause)
                for index in sorted(rng.choice(rows, min(n, len(rows)), replace=False).tolist()):
                    atoms = db.get(id=index + 1).toatoms()
                    finite = bool(np.isfinite(atoms.positions).all())
                    entries.append({
                        "config": config, "cause": cause, "index": index,
                        "formula": atoms.get_chemical_formula(),
                        "n_cause": len(rows), "n": len(failure),
                        "xyz": _xyz(atoms) if finite else None,
                    })
    return entries
