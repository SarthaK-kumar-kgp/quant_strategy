#!/usr/bin/env python3
"""Audit nested-barrier ordering for retained Phase-3 validation candidates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_benchmarks import atomic_parquet, file_sha256


FOLDER = Path(__file__).resolve().parent
NESTED_FILE = FOLDER.parent / "phase1_registry" / "nested_barrier_groups.parquet"
METRICS_FILE = FOLDER / "nested_coherence_metrics.parquet"
VIOLATIONS_FILE = FOLDER / "nested_coherence_violations.parquet"
REPORT_FILE = FOLDER / "nested_coherence_report.md"
TOLERANCE = 1e-12

CANDIDATES = {
    "E03-PM-RAW-V1": {
        "label": "Polymarket raw",
        "manifest": "benchmark_prediction_manifest.parquet",
        "diagnostic_only": True,
    },
    "E03-GBM-PLATT-V1": {
        "label": "GBM + Platt",
        "manifest": "benchmark_prediction_manifest.parquet",
        "diagnostic_only": False,
    },
    "E04-EMP-PLATT-V1": {
        "label": "Pooled empirical + Platt",
        "manifest": "empirical_prediction_manifest.parquet",
        "diagnostic_only": False,
    },
    "E08-EMP-ASSET-PLATT-V1": {
        "label": "Asset-specific empirical + Platt",
        "manifest": "empirical_asset_prediction_manifest.parquet",
        "diagnostic_only": False,
    },
    "E06-HAR-PLATT-V1": {
        "label": "HAR/range + Platt",
        "manifest": "har_prediction_manifest.parquet",
        "diagnostic_only": False,
    },
    "E07-HAR-DVOL-PLATT-V1": {
        "label": "BTC/ETH HAR + DVOL + Platt",
        "manifest": "dvol_prediction_manifest.parquet",
        "diagnostic_only": False,
    },
}

PREDICTION_COLUMNS = [
    "condition_id",
    "decision_time",
    "fold_id",
    "calibrated_yes_probability",
]
METADATA_COLUMNS = [
    "condition_id",
    "barrier_group_id",
    "barrier_group_size",
    "barrier_rank_easiest_first",
    "asset",
    "direction",
    "barrier",
]


def audit_partition(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    tolerance: float = TOLERANCE,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    required_predictions = set(PREDICTION_COLUMNS)
    required_metadata = set(METADATA_COLUMNS)
    if not required_predictions <= set(predictions.columns):
        raise ValueError("prediction frame is missing required columns")
    if not required_metadata <= set(metadata.columns):
        raise ValueError("nested metadata is missing required columns")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    nested = predictions[PREDICTION_COLUMNS].merge(
        metadata[METADATA_COLUMNS], on="condition_id", how="inner", validate="many_to_one"
    )
    if nested.empty:
        return {
            "prediction_rows": len(predictions),
            "nested_rows": 0,
            "nested_contracts": 0,
            "comparable_group_times": 0,
            "adjacent_pairs": 0,
            "violations": 0,
            "violating_group_times": 0,
            "positive_excess_sum": 0.0,
            "mean_positive_excess": 0.0,
            "maximum_positive_excess": 0.0,
        }, pd.DataFrame()

    probability = pd.to_numeric(
        nested.calibrated_yes_probability, errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("invalid calibrated probabilities")
    nested = nested.sort_values(
        ["barrier_group_id", "decision_time", "barrier_rank_easiest_first"],
        ignore_index=True,
    )
    group_columns = ["barrier_group_id", "decision_time"]
    same_group_time = (
        nested.barrier_group_id.eq(nested.barrier_group_id.shift())
        & nested.decision_time.eq(nested.decision_time.shift())
    )
    nested["easier_condition_id"] = nested.condition_id.shift()
    nested["easier_rank"] = nested.barrier_rank_easiest_first.shift()
    nested["easier_barrier"] = nested.barrier.shift()
    nested["easier_probability"] = nested.calibrated_yes_probability.shift()
    nested["positive_excess"] = (
        nested.calibrated_yes_probability - nested.easier_probability
    )
    pair_mask = same_group_time
    violation_mask = pair_mask & (nested.positive_excess > tolerance)
    sizes = nested.groupby(group_columns, sort=False).size()
    violations = nested.loc[
        violation_mask,
        [
            "fold_id",
            "barrier_group_id",
            "decision_time",
            "asset",
            "direction",
            "easier_condition_id",
            "condition_id",
            "easier_rank",
            "barrier_rank_easiest_first",
            "easier_barrier",
            "barrier",
            "easier_probability",
            "calibrated_yes_probability",
            "positive_excess",
        ],
    ].copy()
    violation_count = int(violation_mask.sum())
    positive_excess_sum = float(nested.loc[violation_mask, "positive_excess"].sum())
    metrics = {
        "prediction_rows": len(predictions),
        "nested_rows": len(nested),
        "nested_contracts": nested.condition_id.nunique(),
        "comparable_group_times": int((sizes >= 2).sum()),
        "adjacent_pairs": int(pair_mask.sum()),
        "violations": violation_count,
        "violating_group_times": int(
            violations[["barrier_group_id", "decision_time"]]
            .drop_duplicates()
            .shape[0]
        ),
        "positive_excess_sum": positive_excess_sum,
        "mean_positive_excess": (
            positive_excess_sum / violation_count if violation_count else 0.0
        ),
        "maximum_positive_excess": (
            float(nested.loc[violation_mask, "positive_excess"].max())
            if violation_count
            else 0.0
        ),
    }
    return metrics, violations


def load_verified_manifest(experiment_id: str, manifest_name: str) -> pd.DataFrame:
    manifest = pd.read_parquet(FOLDER / manifest_name)
    selected = manifest[manifest.experiment_id == experiment_id].copy()
    if selected.empty:
        raise ValueError(f"no manifest rows for {experiment_id}")
    for row in selected.itertuples(index=False):
        path = FOLDER / row.path
        if not path.exists() or file_sha256(path) != row.sha256:
            raise ValueError(f"prediction checksum mismatch: {path}")
    return selected.sort_values(["fold_id", "asset"], ignore_index=True)


def aggregate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment_id, group in frame.groupby("experiment_id", sort=False):
        violations = int(group.violations.sum())
        positive_excess_sum = float(group.positive_excess_sum.sum())
        rows.append(
            {
                "experiment_id": experiment_id,
                "label": group.label.iloc[0],
                "fold_id": "overall",
                "diagnostic_only": bool(group.diagnostic_only.iloc[0]),
                "prediction_rows": int(group.prediction_rows.sum()),
                "nested_rows": int(group.nested_rows.sum()),
                "nested_contracts": int(group.nested_contracts.sum()),
                "comparable_group_times": int(group.comparable_group_times.sum()),
                "adjacent_pairs": int(group.adjacent_pairs.sum()),
                "violations": violations,
                "violating_group_times": int(group.violating_group_times.sum()),
                "positive_excess_sum": positive_excess_sum,
                "mean_positive_excess": (
                    positive_excess_sum / violations if violations else 0.0
                ),
                "maximum_positive_excess": float(
                    group.maximum_positive_excess.max()
                ),
                "violation_rate": (
                    violations / int(group.adjacent_pairs.sum())
                    if int(group.adjacent_pairs.sum())
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def make_report(metrics: pd.DataFrame, violations: pd.DataFrame) -> str:
    overall = metrics[metrics.fold_id == "overall"].set_index("experiment_id")
    model_triggers = [
        experiment_id
        for experiment_id, specification in CANDIDATES.items()
        if not specification["diagnostic_only"]
        and int(overall.loc[experiment_id, "violations"]) > 0
    ]
    lines = [
        "# Phase-3 nested-barrier coherence audit",
        "",
        "The audit compares simultaneously available validation predictions in",
        "frozen easiest-to-hardest barrier order. Evaluation and Phase-4 execution",
        "data were not read.",
        "",
        "## Combined audit",
        "",
        "| Candidate | Comparable group-times | Adjacent pairs | Violations | Rate | Mean excess | Maximum excess |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment_id, specification in CANDIDATES.items():
        row = overall.loc[experiment_id]
        lines.append(
            f"| {specification['label']} | {int(row.comparable_group_times):,} | "
            f"{int(row.adjacent_pairs):,} | {int(row.violations):,} | "
            f"{row.violation_rate:.8%} | {row.mean_positive_excess:.8f} | "
            f"{row.maximum_positive_excess:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Audit by fold",
            "",
            "| Candidate | Fold | Adjacent pairs | Violations | Rate | Maximum excess |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    fold_rows = metrics[metrics.fold_id != "overall"].set_index(
        ["experiment_id", "fold_id"]
    )
    for experiment_id, specification in CANDIDATES.items():
        for fold_id in ["fold_01", "fold_02", "fold_03", "fold_04"]:
            row = fold_rows.loc[(experiment_id, fold_id)]
            lines.append(
                f"| {specification['label']} | {fold_id} | "
                f"{int(row.adjacent_pairs):,} | {int(row.violations):,} | "
                f"{row.violation_rate:.8%} | {row.maximum_positive_excess:.8f} |"
            )
    lines.extend(["", "## Frozen decision", ""])
    if model_triggers:
        lines.append(
            "- Corrective projection is triggered for: " + ", ".join(model_triggers) + "."
        )
        lines.append(
            "- Each projected variant must be registered before its validation scores are generated."
        )
    else:
        lines.append("- No actual model candidate has a material coherence violation.")
        lines.append("- The conditional projection is therefore not triggered.")
    market_violations = int(overall.loc["E03-PM-RAW-V1", "violations"])
    lines.append(
        f"- The diagnostic raw market benchmark has {market_violations:,} material violations."
    )
    lines.extend(
        [
            "",
            "## Artifact coverage",
            "",
            f"- Audited candidate-fold rows: {len(metrics) - len(CANDIDATES):,}.",
            f"- Stored violation rows: {len(violations):,}.",
            f"- Numerical tolerance: {TOLERANCE:.0e}.",
            "",
            "Market-price violations are diagnostic and do not create an executable",
            "or arbitrage claim. This is a probability-structure audit, not a",
            "profitability result.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    metadata = pd.read_parquet(NESTED_FILE, columns=METADATA_COLUMNS)
    if metadata.condition_id.duplicated().any():
        raise ValueError("nested metadata contains duplicate contracts")
    metric_rows = []
    violation_parts = []
    for experiment_id, specification in CANDIDATES.items():
        manifest = load_verified_manifest(experiment_id, specification["manifest"])
        for fold_id, fold_manifest in manifest.groupby("fold_id", sort=True):
            fold_metrics = []
            for row in fold_manifest.itertuples(index=False):
                predictions = pd.read_parquet(FOLDER / row.path, columns=PREDICTION_COLUMNS)
                if len(predictions) != int(row.rows):
                    raise ValueError(f"manifest row mismatch: {row.path}")
                partition_metrics, partition_violations = audit_partition(
                    predictions, metadata
                )
                fold_metrics.append(partition_metrics)
                if not partition_violations.empty:
                    partition_violations.insert(0, "experiment_id", experiment_id)
                    violation_parts.append(partition_violations)
            fold_frame = pd.DataFrame(fold_metrics)
            violations = int(fold_frame.violations.sum())
            positive_excess_sum = float(fold_frame.positive_excess_sum.sum())
            adjacent_pairs = int(fold_frame.adjacent_pairs.sum())
            metric_rows.append(
                {
                    "experiment_id": experiment_id,
                    "label": specification["label"],
                    "fold_id": fold_id,
                    "diagnostic_only": specification["diagnostic_only"],
                    "prediction_rows": int(fold_frame.prediction_rows.sum()),
                    "nested_rows": int(fold_frame.nested_rows.sum()),
                    "nested_contracts": int(fold_frame.nested_contracts.sum()),
                    "comparable_group_times": int(
                        fold_frame.comparable_group_times.sum()
                    ),
                    "adjacent_pairs": adjacent_pairs,
                    "violations": violations,
                    "violating_group_times": int(
                        fold_frame.violating_group_times.sum()
                    ),
                    "positive_excess_sum": positive_excess_sum,
                    "mean_positive_excess": (
                        positive_excess_sum / violations if violations else 0.0
                    ),
                    "maximum_positive_excess": float(
                        fold_frame.maximum_positive_excess.max()
                    ),
                    "violation_rate": (
                        violations / adjacent_pairs if adjacent_pairs else 0.0
                    ),
                }
            )
            print(
                f"audited {experiment_id} {fold_id}: {adjacent_pairs:,} pairs, "
                f"{violations:,} violations",
                flush=True,
            )
    fold_metrics = pd.DataFrame(metric_rows)
    overall_metrics = aggregate_metrics(fold_metrics)
    metrics = pd.concat([fold_metrics, overall_metrics], ignore_index=True)
    violations = (
        pd.concat(violation_parts, ignore_index=True)
        if violation_parts
        else pd.DataFrame(
            columns=[
                "experiment_id",
                "fold_id",
                "barrier_group_id",
                "decision_time",
                "asset",
                "direction",
                "easier_condition_id",
                "condition_id",
                "easier_rank",
                "barrier_rank_easiest_first",
                "easier_barrier",
                "barrier",
                "easier_probability",
                "calibrated_yes_probability",
                "positive_excess",
            ]
        )
    )
    atomic_parquet(metrics, METRICS_FILE)
    atomic_parquet(violations, VIOLATIONS_FILE)
    REPORT_FILE.write_text(make_report(metrics, violations), encoding="utf-8")
    print(f"metrics: {METRICS_FILE}")
    print(f"violations: {VIOLATIONS_FILE}")
    print(f"report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
