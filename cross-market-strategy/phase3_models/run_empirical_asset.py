#!/usr/bin/env python3
"""Run the frozen asset-specific empirical-excursion pooling comparison."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from run_benchmarks import (
    PLATT_L2,
    PROBABILITY_CLIP,
    RELIABILITY_BINS,
    atomic_parquet,
    contract_metric,
    file_sha256,
    fit_platt,
)
from run_empirical import (
    ASSET_SYMBOLS,
    CALIBRATION_BLOCKS,
    CUTOFF_TEXT,
    FOLD_CALIBRATION_BLOCKS,
    FOLD_DISTRIBUTION_CUTOFF,
    FOLD_FILE,
    HORIZONS,
    REFERENCE_STRIDE_MINUTES,
    TAIL_PSEUDOCOUNT,
    build_reference_catalog,
    cutoff_timestamp,
    predict_panel,
    read_panel,
    unique_contract_calendar,
)
from run_har import metric_tables


FOLDER = Path(__file__).resolve().parent
OUTPUT_DIR = FOLDER / "empirical_asset_predictions"
DISTRIBUTION_DIR = FOLDER / "empirical_asset_distributions"
PARAMETERS_FILE = FOLDER / "empirical_asset_calibrators.parquet"
DISTRIBUTION_MANIFEST_FILE = FOLDER / "empirical_asset_distribution_manifest.parquet"
METRICS_FILE = FOLDER / "empirical_asset_contract_metrics.parquet"
RELIABILITY_FILE = FOLDER / "empirical_asset_reliability.parquet"
PREDICTION_MANIFEST_FILE = FOLDER / "empirical_asset_prediction_manifest.parquet"
REPORT_FILE = FOLDER / "empirical_pooling_report.md"

MODEL_KEYS = ("empirical_asset_raw", "empirical_asset_platt")
EXPERIMENTS = {
    "empirical_asset_raw": {
        "experiment_id": "E08-EMP-ASSET-RAW-V1",
        "calibration_method": "raw",
    },
    "empirical_asset_platt": {
        "experiment_id": "E08-EMP-ASSET-PLATT-V1",
        "calibration_method": "platt",
    },
}


def select_asset_values(
    catalog: dict[int, dict[str, np.ndarray]],
    horizon: int,
    asset: str,
    direction: str,
    cutoff_epoch: int,
) -> tuple[np.ndarray, np.ndarray]:
    data = catalog[int(horizon)]
    eligible = (data["end"] <= cutoff_epoch) & (data["asset"] == asset)
    values = np.sort(data[direction][eligible])
    ends = data["end"][eligible]
    if not len(values):
        raise ValueError(f"no references for {asset} {horizon} {direction}")
    return values, ends


def build_asset_distributions(
    catalog: dict[int, dict[str, np.ndarray]],
) -> tuple[
    dict[str, dict[str, dict[tuple[int, str], np.ndarray]]], pd.DataFrame
]:
    distributions = {}
    manifest_rows = []
    DISTRIBUTION_DIR.mkdir(parents=True, exist_ok=True)
    for cutoff_id, cutoff_text_value in CUTOFF_TEXT.items():
        cutoff = cutoff_timestamp(cutoff_text_value)
        cutoff_epoch = int(cutoff.timestamp())
        distributions[cutoff_id] = {}
        for asset in ASSET_SYMBOLS:
            fitted = {}
            save_values = {
                "cutoff_epoch": np.array([cutoff_epoch], dtype=np.int64),
                "asset": np.array([asset]),
                "horizons": HORIZONS,
            }
            asset_rows = []
            for horizon_value in HORIZONS:
                horizon = int(horizon_value)
                for direction in ["up", "down"]:
                    values, ends = select_asset_values(
                        catalog, horizon, asset, direction, cutoff_epoch
                    )
                    fitted[(horizon, direction)] = values
                    save_values[f"h{horizon}_{direction}"] = values
                    asset_rows.append(
                        {
                            "cutoff_id": cutoff_id,
                            "cutoff_time": cutoff,
                            "asset": asset,
                            "horizon_minutes": horizon,
                            "direction": direction,
                            "reference_count": len(values),
                            "latest_reference_end": pd.to_datetime(
                                int(ends.max()), unit="s", utc=True
                            ),
                            "minimum": float(values[0]),
                            "median": float(np.median(values)),
                            "p95": float(np.quantile(values, 0.95)),
                            "p99": float(np.quantile(values, 0.99)),
                            "maximum": float(values[-1]),
                        }
                    )
            path = DISTRIBUTION_DIR / f"{cutoff_id}_{asset}.npz"
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **save_values)
            temporary.replace(path)
            checksum = file_sha256(path)
            for row in asset_rows:
                row["distribution_path"] = str(path.relative_to(FOLDER))
                row["distribution_sha256"] = checksum
            manifest_rows.extend(asset_rows)
            distributions[cutoff_id][asset] = fitted
        print(f"froze asset-specific empirical distributions: {cutoff_id}", flush=True)
    manifest = pd.DataFrame(manifest_rows)
    atomic_parquet(manifest, DISTRIBUTION_MANIFEST_FILE)
    return distributions, manifest


def load_calibration_predictions(
    calendar: pd.DataFrame,
    distributions: dict[str, dict[str, dict[tuple[int, str], np.ndarray]]],
) -> dict[str, dict[str, object]]:
    blocks = {}
    for block_id, specification in CALIBRATION_BLOCKS.items():
        members = calendar[
            (calendar.resolution_date_et >= specification["start"])
            & (calendar.resolution_date_et <= specification["end"])
        ]
        probability_parts = []
        outcome_parts = []
        weight_parts = []
        cutoff_id = specification["cutoff_id"]
        for member in members.itertuples(index=False):
            panel = read_panel(member.condition_id)
            probability = predict_panel(
                panel, distributions[cutoff_id][member.asset]
            )
            if not np.isfinite(probability).all():
                raise ValueError(f"invalid asset empirical probability")
            probability_parts.append(probability)
            outcome_parts.append(panel.y.to_numpy(dtype=float))
            weight_parts.append(panel.contract_row_weight.to_numpy(dtype=float))
        blocks[block_id] = {
            "probability": np.concatenate(probability_parts),
            "outcome": np.concatenate(outcome_parts),
            "weight": np.concatenate(weight_parts),
            "contracts": len(members),
            "cutoff_id": cutoff_id,
        }
        print(
            f"generated asset empirical OOF {block_id}: {len(members):,} contracts, "
            f"{len(blocks[block_id]['probability']):,} rows",
            flush=True,
        )
    return blocks


def fit_fold_calibrators(blocks: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for fold_id, block_ids in FOLD_CALIBRATION_BLOCKS.items():
        probability = np.concatenate([blocks[item]["probability"] for item in block_ids])
        outcome = np.concatenate([blocks[item]["outcome"] for item in block_ids])
        weight = np.concatenate([blocks[item]["weight"] for item in block_ids])
        fitted = fit_platt(probability, outcome, weight)
        rows.append(
            {
                "fold_id": fold_id,
                "model_distribution_cutoff": FOLD_DISTRIBUTION_CUTOFF[fold_id],
                "calibration_version": f"{fold_id}_empirical_asset_platt_v1",
                "calibration_blocks": "|".join(block_ids),
                "calibration_distribution_cutoffs": "|".join(
                    str(blocks[item]["cutoff_id"]) for item in block_ids
                ),
                "calibration_contracts": int(
                    sum(int(blocks[item]["contracts"]) for item in block_ids)
                ),
                "calibration_rows": len(probability),
                "calibration_weight": float(weight.sum()),
                "horizons_minutes": "|".join(map(str, HORIZONS)),
                "reference_stride_minutes": REFERENCE_STRIDE_MINUTES,
                "tail_pseudocount": TAIL_PSEUDOCOUNT,
                "asset_pooling": "asset_specific",
                "direction_pooling": "separate",
                "time_interpolation": "linear_probability_in_log_minutes",
                "platt_probability_clip": PROBABILITY_CLIP,
                "platt_l2_slope": PLATT_L2,
                **fitted,
            }
        )
        print(
            f"fitted asset empirical {fold_id} Platt: "
            f"slope={fitted['slope']:.6f} intercept={fitted['intercept']:.6f}",
            flush=True,
        )
    calibrators = pd.DataFrame(rows)
    atomic_parquet(calibrators, PARAMETERS_FILE)
    return calibrators


def update_reliability(
    accumulator: dict[tuple[str, str, int], np.ndarray],
    fold_id: str,
    model_key: str,
    probability: np.ndarray,
    outcome: np.ndarray,
    weight: np.ndarray,
) -> None:
    bins = np.minimum((probability * RELIABILITY_BINS).astype(int), RELIABILITY_BINS - 1)
    for bin_id in np.unique(bins):
        selected = bins == bin_id
        accumulator[(fold_id, model_key, int(bin_id))] += np.array(
            [
                weight[selected].sum(),
                np.dot(weight[selected], probability[selected]),
                np.dot(weight[selected], outcome[selected]),
                selected.sum(),
            ]
        )


def prediction_frame(
    base: pd.DataFrame,
    model_key: str,
    fold_id: str,
    probability_column: str,
) -> pd.DataFrame:
    experiment = EXPERIMENTS[model_key]
    calibrated = experiment["calibration_method"] == "platt"
    return pd.DataFrame(
        {
            "schema_version": "phase3_prediction_v1",
            "experiment_id": experiment["experiment_id"],
            "condition_id": base.condition_id,
            "decision_time": base.decision_time,
            "information_timestamp": base.spot_available_time,
            "fold_id": fold_id,
            "role": "validation",
            "prediction_kind": "forward",
            "model_id": "empirical_asset_excursion_v1",
            "model_version": "v1",
            "feature_set_id": "empirical_spot_barrier_rv24h_asset_v1",
            "calibration_method": experiment["calibration_method"],
            "calibration_version": (
                f"{fold_id}_empirical_asset_platt_v1" if calibrated else "none"
            ),
            "raw_yes_probability": base.empirical_asset_probability.to_numpy(
                dtype=float
            ),
            "calibrated_yes_probability": base[probability_column].to_numpy(
                dtype=float
            ),
            "contract_row_weight": base.contract_row_weight.to_numpy(dtype=float),
        }
    )


def generate_validation_outputs(
    folds: pd.DataFrame,
    distributions: dict[str, dict[str, dict[tuple[int, str], np.ndarray]]],
    calibrators: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calibrator_map = calibrators.set_index("fold_id").to_dict("index")
    metrics = []
    manifests = []
    reliability = defaultdict(lambda: np.zeros(4, dtype=float))
    for fold_id in FOLD_CALIBRATION_BLOCKS:
        validation = folds[(folds.fold_id == fold_id) & (folds.role == "validation")]
        fitted = calibrator_map[fold_id]
        cutoff_id = FOLD_DISTRIBUTION_CUTOFF[fold_id]
        for asset in ASSET_SYMBOLS:
            members = validation[validation.asset == asset]
            parts = []
            distribution = distributions[cutoff_id][asset]
            for member in members.itertuples(index=False):
                panel = read_panel(member.condition_id)
                raw = predict_panel(panel, distribution)
                if not np.isfinite(raw).all():
                    raise ValueError(f"invalid asset empirical probability")
                clipped = np.clip(raw, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
                logit = np.log(clipped) - np.log1p(-clipped)
                platt = expit(fitted["slope"] * logit + fitted["intercept"])
                panel = panel.copy()
                panel["empirical_asset_probability"] = raw
                panel["empirical_asset_platt_probability"] = platt
                parts.append(panel)
                outcome = panel.y.to_numpy(dtype=float)
                weight = panel.contract_row_weight.to_numpy(dtype=float)
                for model_key, probability in {
                    "empirical_asset_raw": raw,
                    "empirical_asset_platt": platt,
                }.items():
                    brier, log_loss, mean_probability = contract_metric(
                        probability, outcome, weight
                    )
                    metrics.append(
                        {
                            "fold_id": fold_id,
                            "role": "validation",
                            "model_key": model_key,
                            "experiment_id": EXPERIMENTS[model_key]["experiment_id"],
                            "condition_id": member.condition_id,
                            "asset": member.asset,
                            "direction": member.direction,
                            "n_rows": len(panel),
                            "outcome": int(outcome[0]),
                            "mean_probability": mean_probability,
                            "brier": brier,
                            "log_loss": log_loss,
                        }
                    )
                    update_reliability(
                        reliability,
                        fold_id,
                        model_key,
                        probability,
                        outcome,
                        weight,
                    )
            base = pd.concat(parts, ignore_index=True)
            for model_key, probability_column in [
                ("empirical_asset_raw", "empirical_asset_probability"),
                ("empirical_asset_platt", "empirical_asset_platt_probability"),
            ]:
                output = prediction_frame(base, model_key, fold_id, probability_column)
                path = (
                    OUTPUT_DIR
                    / EXPERIMENTS[model_key]["experiment_id"]
                    / fold_id
                    / "validation"
                    / f"{asset}.parquet"
                )
                atomic_parquet(output, path)
                manifests.append(
                    {
                        "experiment_id": EXPERIMENTS[model_key]["experiment_id"],
                        "fold_id": fold_id,
                        "role": "validation",
                        "asset": asset,
                        "path": str(path.relative_to(FOLDER)),
                        "rows": len(output),
                        "contracts": output.condition_id.nunique(),
                        "first_decision": output.decision_time.min(),
                        "last_decision": output.decision_time.max(),
                        "bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                )
            print(
                f"wrote asset empirical {fold_id} {asset}: {len(base):,} rows, "
                f"{len(members):,} contracts",
                flush=True,
            )

    reliability_rows = []
    for (fold_id, model_key, bin_id), values in sorted(reliability.items()):
        total_weight, weighted_probability, weighted_outcome, row_count = values
        reliability_rows.append(
            {
                "fold_id": fold_id,
                "role": "validation",
                "model_key": model_key,
                "experiment_id": EXPERIMENTS[model_key]["experiment_id"],
                "bin_id": bin_id,
                "bin_lower": bin_id / RELIABILITY_BINS,
                "bin_upper": (bin_id + 1) / RELIABILITY_BINS,
                "contract_weight": total_weight,
                "rows": int(row_count),
                "mean_probability": weighted_probability / total_weight,
                "outcome_rate": weighted_outcome / total_weight,
                "absolute_gap": abs(weighted_probability - weighted_outcome)
                / total_weight,
            }
        )
    metric_frame = pd.DataFrame(metrics)
    reliability_frame = pd.DataFrame(reliability_rows)
    manifest_frame = pd.DataFrame(manifests)
    atomic_parquet(metric_frame, METRICS_FILE)
    atomic_parquet(reliability_frame, RELIABILITY_FILE)
    atomic_parquet(manifest_frame, PREDICTION_MANIFEST_FILE)
    return metric_frame, reliability_frame, manifest_frame


def make_report(
    calibrators: pd.DataFrame,
    distribution_manifest: pd.DataFrame,
    asset_metrics: pd.DataFrame,
    asset_reliability: pd.DataFrame,
    prediction_manifest: pd.DataFrame,
) -> str:
    pooled_metrics = pd.read_parquet(FOLDER / "empirical_contract_metrics.parquet")
    pooled_reliability = pd.read_parquet(FOLDER / "empirical_reliability.parquet")
    metrics = pd.concat([pooled_metrics, asset_metrics], ignore_index=True)
    reliability = pd.concat([pooled_reliability, asset_reliability], ignore_index=True)
    by_fold, overall, by_asset, by_direction = metric_tables(metrics, reliability)
    labels = {
        "empirical_raw": "Pooled empirical raw",
        "empirical_platt": "Pooled empirical + Platt",
        "empirical_asset_raw": "Asset-specific empirical raw",
        "empirical_asset_platt": "Asset-specific empirical + Platt",
    }
    order = list(labels)
    indexed = overall.set_index("model_key")
    winner = min(
        order,
        key=lambda key: (indexed.loc[key, "brier"], indexed.loc[key, "log_loss"]),
    )
    pooled_fold = by_fold[by_fold.model_key == "empirical_platt"].set_index("fold_id")
    asset_fold = by_fold[
        by_fold.model_key == "empirical_asset_platt"
    ].set_index("fold_id")

    lines = [
        "# Phase-3 empirical pooling comparison",
        "",
        "This report compares pooled and asset-specific empirical excursion",
        "distributions on identical validation rows. Evaluation and Phase-4",
        "execution data were not read.",
        "",
        "## Reference coverage",
        "",
        "| Asset | Minimum references | Maximum references |",
        "|---|---:|---:|",
    ]
    coverage = distribution_manifest.groupby("asset").reference_count.agg(
        ["min", "max"]
    )
    for asset in ASSET_SYMBOLS:
        lines.append(
            f"| {asset} | {int(coverage.loc[asset, 'min']):,} | "
            f"{int(coverage.loc[asset, 'max']):,} |"
        )
    lines.extend(
        [
            "",
            "## Asset-specific Platt parameters",
            "",
            "| Fold | Model cutoff | Contracts | Rows | Slope | Intercept |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in calibrators.itertuples(index=False):
        lines.append(
            f"| {row.fold_id} | {row.model_distribution_cutoff} | "
            f"{row.calibration_contracts:,} | {row.calibration_rows:,} | "
            f"{row.slope:.6f} | {row.intercept:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Validation metrics by fold",
            "",
            "| Fold | Model | Contracts | Rows | Brier | Log loss | ECE-10 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for fold_id in FOLD_CALIBRATION_BLOCKS:
        for model_key in order:
            row = by_fold[
                (by_fold.fold_id == fold_id) & (by_fold.model_key == model_key)
            ].iloc[0]
            lines.append(
                f"| {fold_id} | {labels[model_key]} | {int(row.contracts):,} | "
                f"{int(row.decision_rows):,} | {row.brier:.6f} | "
                f"{row.log_loss:.6f} | {row.ece_10:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Combined validation comparison",
            "",
            "| Model | Fold-contract appearances | Brier | Log loss | ECE-10 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model_key in order:
        row = indexed.loc[model_key]
        lines.append(
            f"| {labels[model_key]} | {int(row.fold_contract_appearances):,} | "
            f"{row.brier:.6f} | {row.log_loss:.6f} | {row.ece_10:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen pooling decision",
            "",
            f"- Registered four-way winner: **{labels[winner]}**.",
            f"- Asset-specific raw minus pooled raw Brier: {indexed.loc['empirical_asset_raw', 'brier'] - indexed.loc['empirical_raw', 'brier']:+.6f}.",
            f"- Asset-specific Platt minus pooled Platt Brier: {indexed.loc['empirical_asset_platt', 'brier'] - indexed.loc['empirical_platt', 'brier']:+.6f}.",
            f"- Asset-specific Platt minus pooled Platt log loss: {indexed.loc['empirical_asset_platt', 'log_loss'] - indexed.loc['empirical_platt', 'log_loss']:+.6f}.",
            f"- Asset-specific Platt beats pooled Platt on Brier in {int((asset_fold.brier < pooled_fold.brier).sum())}/4 folds and log loss in {int((asset_fold.log_loss < pooled_fold.log_loss).sum())}/4 folds.",
            "",
            "## Breakdown by asset",
            "",
            "| Model | Asset | Contracts | Brier | Log loss |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for model_key in order:
        for asset in ASSET_SYMBOLS:
            row = by_asset[
                (by_asset.model_key == model_key) & (by_asset.asset == asset)
            ].iloc[0]
            lines.append(
                f"| {labels[model_key]} | {asset} | {int(row.contracts):,} | "
                f"{row.brier:.6f} | {row.log_loss:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Breakdown by direction",
            "",
            "| Model | Direction | Contracts | Brier | Log loss |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for model_key in order:
        for direction in ["up", "down"]:
            row = by_direction[
                (by_direction.model_key == model_key)
                & (by_direction.direction == direction)
            ].iloc[0]
            lines.append(
                f"| {labels[model_key]} | {direction} | {int(row.contracts):,} | "
                f"{row.brier:.6f} | {row.log_loss:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Artifact coverage",
            "",
            f"- Distribution audit rows: {len(distribution_manifest):,}.",
            f"- Prediction partitions: {len(prediction_manifest):,}.",
            f"- Prediction rows: {int(prediction_manifest.rows.sum()):,}.",
            "",
            "These are probability-validation results, not profitability results.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    folds = pd.read_parquet(FOLD_FILE)
    calendar = unique_contract_calendar(folds)
    catalog = build_reference_catalog()
    distributions, distribution_manifest = build_asset_distributions(catalog)
    del catalog
    blocks = load_calibration_predictions(calendar, distributions)
    calibrators = fit_fold_calibrators(blocks)
    del blocks
    metrics, reliability, manifest = generate_validation_outputs(
        folds, distributions, calibrators
    )
    REPORT_FILE.write_text(
        make_report(
            calibrators,
            distribution_manifest,
            metrics,
            reliability,
            manifest,
        ),
        encoding="utf-8",
    )
    print(f"distribution manifest: {DISTRIBUTION_MANIFEST_FILE}")
    print(f"metrics: {METRICS_FILE}")
    print(f"prediction manifest: {PREDICTION_MANIFEST_FILE}")
    print(f"report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
