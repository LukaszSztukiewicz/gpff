"""Figures of evaluated sample sets, for ``gpff_report``.

A set is one model's repro folder, <run>/frozen/repro/<model>, holding
<config>/seed<N>/evaluation/ per sampler config. The figures read the
per-sample arrays of gpff_prepare_metadata (rdkit_valid, failure,
element_counts) and the results and breakdowns of gpff_calculate_metrics; what
an evaluation lacks is left out of the figures, never filled in.
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gpff.analysis.structure import FAILURES  # noqa: E402

__all__ = [
    "CONFIG_ORDER",
    "delta_figure",
    "difference",
    "environment_figure",
    "epoch_figure",
    "failure_figure",
    "load_set",
    "pooled",
    "size_figure",
    "wilson",
]

# categorical colors, used in this order: the i-th set (or run, or failure
# cause) of a figure always gets SERIES[i]
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
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
# fewest samples a heavy-atom count needs to get a point in size_figure
MIN_SAMPLES = 20


def config_key(config: str) -> tuple:
    if config in CONFIG_ORDER:
        return (0, CONFIG_ORDER.index(config), "")
    return (1, 0, config)


def load_evaluation(evaluation_dir: Path) -> dict:
    """The results and the per-sample arrays the figures use of one evaluation folder."""
    metadata = evaluation_dir / "metadata"
    evaluation = {"results": json.loads((evaluation_dir / "results.json").read_text())}
    for key in ("rdkit_valid", "failure"):
        if (metadata / f"{key}.npy").exists():
            evaluation[key] = np.load(metadata / f"{key}.npy")
    if (metadata / "element_counts.npz").exists():
        with np.load(metadata / "element_counts.npz") as table:
            evaluation["sizes"] = sum(table[symbol] for symbol in table.files)
            evaluation["heavy_atoms"] = evaluation["sizes"] - (
                table["H"] if "H" in table.files else 0
            )
    for name in ("bond_length_w1", "bond_angle_w1"):
        if (evaluation_dir / f"{name}.json").exists():
            evaluation[name] = json.loads((evaluation_dir / f"{name}.json").read_text())
    return evaluation


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


def difference(a: dict, b: dict, get) -> tuple[float, float, bool]:
    """b minus a in points of a per-sample boolean, its sd, and whether it is paired.

    a and b are {seed: evaluation} of one config. They are paired when they share
    seeds and every shared seed's samples have the same atom counts row by row,
    i.e. were generated from the same test molecules with the same prior draws;
    the sd then comes from the discordant samples (McNemar), otherwise from two
    independent proportions. nan if either lacks the array.
    """
    common = sorted(set(a) & set(b))
    if common and all(_same_rows(a[seed], b[seed]) for seed in common):
        x, y = pooled(a, get, common), pooled(b, get, common)
        if x is None or y is None:
            return np.nan, np.nan, True
        n10, n01 = int((x & ~y).sum()), int((~x & y).sum())
        return 100 * (n01 - n10) / len(x), 100 * np.sqrt(n10 + n01) / len(x), True
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
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(
            1, len(configs), sharey=True, squeeze=False, figsize=(2.5 * len(configs) + 1, 3.2)
        )
        drawn = False
        for ax, config in zip(axes[0], configs):
            for i, (label, data) in enumerate(sets.items()):
                heavy = pooled(data[config], lambda e: e["heavy_atoms"])
                valid = pooled(data[config], lambda e: e["rdkit_valid"])
                if heavy is None or valid is None:
                    continue
                sizes, n = np.unique(heavy, return_counts=True)
                sizes, n = sizes[n >= MIN_SAMPLES], n[n >= MIN_SAMPLES]
                k = np.array([valid[heavy == size].sum() for size in sizes])
                low, high = wilson(k, n)
                ax.fill_between(sizes, 100 * low, 100 * high, color=SERIES[i], alpha=0.15,
                                linewidth=0)
                ax.plot(sizes, 100 * k / n, color=SERIES[i], label=label)
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


def delta_figure(sets: dict, control: str, configs, seed_band: float):
    """Each set's difference to the control in validity and the failure shares.

    Bars are ±1.96 sd; a hollow dot is an unpaired comparison. The shaded band on
    the validity panel is ±seed_band points, the spread of separately trained runs.
    """
    measures = [("RDKit valid", lambda e: e["rdkit_valid"].astype(bool))] + [
        (cause, lambda e, cause=cause: e["failure"] == cause)
        for cause in _causes(sets, configs)
    ]
    labels = list(sets)
    arms = [label for label in labels if label != control]
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
                color = SERIES[labels.index(arm) % len(SERIES)]
                for row, config in enumerate(configs):
                    delta, sd, is_paired = difference(sets[control][config], sets[arm][config], get)
                    ax.errorbar(delta, row + offset, xerr=1.96 * sd, fmt="o", markersize=6,
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
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(
            1, len(configs), squeeze=False, figsize=(2.5 * len(configs) + 1, 3.2)
        )
        drawn = False
        for ax, config in zip(axes[0], configs):
            for i, (run, points) in enumerate(runs.items()):
                epochs, values = [], []
                for epoch, label in points:
                    seeds = sets[label][config].values()
                    found = [e["results"][metric] for e in seeds if metric in e["results"]]
                    if found:
                        epochs.append(epoch)
                        values.append(scale * np.mean(found))
                if epochs:
                    ax.plot(epochs, values, color=SERIES[i % len(SERIES)], marker="o",
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
