"""Summarize and compare GPFF sampler trajectory diagnostics."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_STEP_FIELDS = (
    "radius_gyration",
    "step_rms_displacement",
    "max_atom_displacement",
    "finite_position_fraction",
    "sigma_curr",
    "nfe",
    "centroid_drift",
    "condition_shape_error",
    "scaffold_rms_displacement_from_prior",
    "shape_ratio_1",
    "shape_ratio_2",
    "target_shape_ratio_1",
    "target_shape_ratio_2",
    "frozen",
)


def _as_finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, (dict, list, tuple, str, bool)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_records(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    records = []
    for input_path in paths:
        run_name = input_path.stem.removesuffix("_diagnostics")
        with input_path.open() as diagnostics_file:
            for line_number, line in enumerate(diagnostics_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {input_path} at line {line_number}: {error}"
                    ) from error
                if "sample_idx" not in record or "trajectory_step" not in record:
                    raise ValueError(
                        f"Missing sample_idx or trajectory_step in {input_path}:{line_number}"
                    )
                record.setdefault("run", run_name)
                shape_ratios = record.pop("shape_ratios", None)
                if isinstance(shape_ratios, list):
                    record["shape_ratio_1"] = shape_ratios[0] if shape_ratios else None
                    record["shape_ratio_2"] = shape_ratios[1] if len(shape_ratios) > 1 else None
                target_shape_ratios = record.pop("target_shape_ratios", None)
                if isinstance(target_shape_ratios, list):
                    record["target_shape_ratio_1"] = (
                        target_shape_ratios[0] if target_shape_ratios else None
                    )
                    record["target_shape_ratio_2"] = (
                        target_shape_ratios[1] if len(target_shape_ratios) > 1 else None
                    )
                sampler_log = record.get("sampler_log") or {}
                for key, value in sampler_log.items():
                    if isinstance(value, dict):
                        for statistic, statistic_value in value.items():
                            record[f"log_{key}_{statistic}"] = statistic_value
                    elif isinstance(value, (int, float)) and not isinstance(value, bool):
                        record[f"log_{key}"] = value
                records.append(record)
    if not records:
        raise ValueError("The input diagnostics files contain no records.")
    return records


def _distribution_summary(values: Iterable[Any]) -> Dict[str, float | None]:
    finite_values = [
        numeric for value in values if (numeric := _as_finite_float(value)) is not None
    ]
    if not finite_values:
        return {name: None for name in ("mean", "median", "p10", "p90")}
    distribution = np.asarray(finite_values, dtype=float)
    return {
        "mean": float(distribution.mean()),
        "median": float(np.median(distribution)),
        "p10": float(np.quantile(distribution, 0.10)),
        "p90": float(np.quantile(distribution, 0.90)),
    }


def summarize_steps(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, list] = defaultdict(list)
    for record in records:
        groups[(record["run"], int(record["trajectory_step"]))].append(record)

    summaries = []
    logged_fields = sorted(
        {
            key
            for record in records
            for key in record
            if key.startswith("log_")
        }
    )
    for (run_name, trajectory_step), group in sorted(groups.items()):
        summary: Dict[str, Any] = {
            "run": run_name,
            "trajectory_step": trajectory_step,
            "step": group[0].get("step"),
            "phase": group[0].get("phase"),
            "n_samples": len({record["sample_idx"] for record in group}),
        }
        for field in (*_STEP_FIELDS, *logged_fields):
            values = [record.get(field) for record in group]
            if field == "frozen":
                values = [float(value) if value is not None else None for value in values]
                output_name = "frozen_fraction"
            else:
                output_name = field
            for statistic, value in _distribution_summary(values).items():
                summary[f"{output_name}_{statistic}"] = value
        summaries.append(summary)
    return summaries


def summarize_samples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, list] = defaultdict(list)
    for record in records:
        groups[(record["run"], int(record["sample_idx"]))].append(record)

    summaries = []
    for (run_name, sample_idx), group in sorted(groups.items()):
        group.sort(key=lambda record: int(record["trajectory_step"]))
        initial = group[0]
        final = group[-1]
        displacements = [
            value
            for record in group
            if (value := _as_finite_float(record.get("step_rms_displacement"))) is not None
        ]
        atom_displacements = [
            value
            for record in group
            if (value := _as_finite_float(record.get("max_atom_displacement"))) is not None
        ]
        finite_fractions = [
            value
            for record in group
            if (value := _as_finite_float(record.get("finite_position_fraction"))) is not None
        ]
        initial_radius = _as_finite_float(initial.get("radius_gyration"))
        final_radius = _as_finite_float(final.get("radius_gyration"))
        summaries.append(
            {
                "run": run_name,
                "sample_idx": sample_idx,
                "seed": final.get("seed"),
                "sampler": final.get("sampler"),
                "prior": final.get("prior"),
                "n_atoms": final.get("n_atoms"),
                "trajectory_steps": max(int(record["trajectory_step"]) for record in group),
                "initial_radius_gyration": initial_radius,
                "final_radius_gyration": final_radius,
                "radius_gyration_change": (
                    final_radius - initial_radius
                    if initial_radius is not None and final_radius is not None
                    else None
                ),
                "max_step_rms_displacement": max(displacements) if displacements else None,
                "total_step_rms_displacement": sum(displacements) if displacements else None,
                "max_atom_displacement": max(atom_displacements) if atom_displacements else None,
                "minimum_finite_position_fraction": min(finite_fractions) if finite_fractions else None,
                "nonfinite_positions_seen": bool(finite_fractions and min(finite_fractions) < 1),
                "final_sigma": final.get("sigma_curr"),
                "final_nfe": final.get("nfe"),
                "final_frozen": final.get("frozen"),
                "final_shape_ratio_1": final.get("shape_ratio_1"),
                "final_shape_ratio_2": final.get("shape_ratio_2"),
                "final_condition_shape_error": final.get("condition_shape_error"),
                "final_scaffold_rms_displacement_from_prior": final.get(
                    "scaffold_rms_displacement_from_prior"
                ),
                "final_centroid_drift": final.get("centroid_drift"),
            }
        )
    return summaries


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, allow_nan=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _plot_trajectory(step_rows: List[Dict[str, Any]], output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    run_names = sorted({record["run"] for record in step_rows})
    panels = (
        (axes[0, 0], "radius_gyration", "Radius of gyration (Å)"),
        (axes[0, 1], "step_rms_displacement", "Per-step RMS displacement (Å)"),
        (axes[1, 0], "finite_position_fraction", "Finite-position fraction"),
        (axes[1, 1], "condition_shape_error", "Shape-condition ratio error"),
    )
    for axis, field, title in panels:
        for run_name in run_names:
            run_rows = sorted(
                (record for record in step_rows if record["run"] == run_name),
                key=lambda record: record["trajectory_step"],
            )
            x_values = [record["trajectory_step"] for record in run_rows]
            means = [record.get(f"{field}_mean") for record in run_rows]
            lower = [record.get(f"{field}_p10") for record in run_rows]
            upper = [record.get(f"{field}_p90") for record in run_rows]
            axis.plot(x_values, means, label=run_name)
            if all(value is not None for value in lower + upper):
                axis.fill_between(x_values, lower, upper, alpha=0.15)
        axis.set_title(title)
        axis.set_xlabel("Trajectory step (prior is 0)")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()

    axes[1, 1].set_ylabel("Mean absolute error of principal variance ratios")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _run_summaries(sample_rows: List[Dict[str, Any]], records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    samples_by_run: Dict[str, list] = defaultdict(list)
    records_by_run: Dict[str, list] = defaultdict(list)
    for row in sample_rows:
        samples_by_run[row["run"]].append(row)
    for record in records:
        records_by_run[record["run"]].append(record)

    summaries = []
    for run_name in sorted(samples_by_run):
        samples = samples_by_run[run_name]
        final_nfe = [row["final_nfe"] for row in samples]
        final_nfe = [
            numeric
            for value in final_nfe
            if (numeric := _as_finite_float(value)) is not None
        ]
        max_jumps = [row["max_step_rms_displacement"] for row in samples]
        max_jumps = [
            numeric
            for value in max_jumps
            if (numeric := _as_finite_float(value)) is not None
        ]
        nonfinite = sum(row["nonfinite_positions_seen"] for row in samples)
        summaries.append(
            {
                "run": run_name,
                "samples": len(samples),
                "trajectory_records": len(records_by_run[run_name]),
                "nonfinite_sample_count": nonfinite,
                "nonfinite_sample_fraction": nonfinite / len(samples),
                "median_final_nfe": float(np.median(final_nfe)) if final_nfe else None,
                "mean_max_step_rms_displacement": float(np.mean(max_jumps)) if max_jumps else None,
                "seeds": sorted({row["seed"] for row in samples if row.get("seed") is not None}),
                "samplers": sorted({row["sampler"] for row in samples if row.get("sampler") is not None}),
            }
        )
    return summaries


def create_diagnostics_report(input_paths: List[Path], output_dir: Path, top_samples: int = 50) -> None:
    records = load_records(input_paths)
    step_rows = summarize_steps(records)
    sample_rows = summarize_samples(records)
    run_rows = _run_summaries(sample_rows, records)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "trajectory_by_step.csv", step_rows)
    _write_csv(output_dir / "sample_diagnostics.csv", sample_rows)

    triage_rows = sorted(
        sample_rows,
        key=lambda row: (
            bool(row["nonfinite_positions_seen"]),
            _as_finite_float(row["max_step_rms_displacement"])
            if _as_finite_float(row["max_step_rms_displacement"]) is not None
            else -1.0,
        ),
        reverse=True,
    )[:top_samples]
    _write_csv(output_dir / "sample_triage.csv", triage_rows)
    _plot_trajectory(step_rows, output_dir / "trajectory_diagnostics.png")

    report = {
        "input_files": [str(path) for path in input_paths],
        "n_records": len(records),
        "runs": run_rows,
        "triage_order": "nonfinite positions first, then largest per-step RMS displacement",
        "interpretation": (
            "Frozen is the sampler's internal frozen mask, not a chemical-validity result. "
            "Large motion is surfaced for inspection and is not automatically a failure."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and compare GPFF trajectory diagnostics.")
    parser.add_argument("--input", nargs="+", required=True, type=Path, help="One or more diagnostics JSONL files.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory for CSV, JSON, and plot outputs.")
    parser.add_argument("--top_samples", type=int, default=50, help="Rows to include in the sample triage CSV.")
    args = parser.parse_args()
    if args.top_samples < 0:
        parser.error("--top_samples must be non-negative")
    create_diagnostics_report(args.input, args.output_dir, args.top_samples)
    print(f"Diagnostics report written to {args.output_dir}")


if __name__ == "__main__":
    main()
