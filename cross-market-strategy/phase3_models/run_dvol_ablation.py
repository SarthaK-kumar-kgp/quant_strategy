#!/usr/bin/env python3
"""Run the frozen matched BTC/ETH HAR DVOL ablation on validation only."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from run_benchmarks import (
    MINUTES_PER_YEAR,
    PLATT_L2,
    PROBABILITY_CLIP,
    RELIABILITY_BINS,
    atomic_parquet,
    contract_metric,
    file_sha256,
    fit_platt,
    weighted_ece,
)
from run_empirical import (
    CALIBRATION_BLOCKS,
    CUTOFF_TEXT,
    FOLD_CALIBRATION_BLOCKS,
    FOLD_DISTRIBUTION_CUTOFF,
    FOLD_FILE,
    HORIZONS,
    PANEL_DIR,
    cutoff_timestamp,
    unique_contract_calendar,
)
from run_har import (
    AssetData,
    FEATURE_NAMES,
    REFERENCE_STRIDE_MINUTES,
    RIDGE_ALPHA,
    TARGET_CLIP_LOWER,
    TARGET_CLIP_UPPER,
    VARIANCE_FLOOR,
    features_for_panel,
    fit_ridge_log_har,
    forecast_integrated_variance,
    har_touch_probability,
    metric_tables,
    prepare_asset_data,
)


FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parent.parent
VOL_DIR = PROJECT / "data" / "vol"
OUTPUT_DIR = FOLDER / "dvol_predictions"
PARAMETERS_FILE = FOLDER / "dvol_parameters.parquet"
CALIBRATORS_FILE = FOLDER / "dvol_calibrators.parquet"
METRICS_FILE = FOLDER / "dvol_contract_metrics.parquet"
RELIABILITY_FILE = FOLDER / "dvol_reliability.parquet"
PREDICTION_MANIFEST_FILE = FOLDER / "dvol_prediction_manifest.parquet"
COVERAGE_FILE = FOLDER / "dvol_coverage.parquet"
REPORT_FILE = FOLDER / "dvol_report.md"

ASSETS = ("BTC", "ETH")
DVOL_DELAY_SECONDS = 3600
DVOL_MAX_AGE_SECONDS = 7200
DVOL_FEATURE_NAME = "dvol_implied_variance_per_minute"
CONTROL_FEATURE_NAMES = FEATURE_NAMES
DVOL_FEATURE_NAMES = FEATURE_NAMES + (DVOL_FEATURE_NAME,)
VARIANTS = ("btceth_control", "btceth_dvol")

EXPERIMENTS = {
    "btceth_control_raw": {
        "experiment_id": "E07-HAR-BE-RAW-V1",
        "variant": "btceth_control",
        "calibration_method": "raw",
        "model_id": "har_range_btceth_control_v1",
        "feature_set_id": "har_d1_w7_m30_pk24_gk24_btceth_v1",
    },
    "btceth_control_platt": {
        "experiment_id": "E07-HAR-BE-PLATT-V1",
        "variant": "btceth_control",
        "calibration_method": "platt",
        "model_id": "har_range_btceth_control_v1",
        "feature_set_id": "har_d1_w7_m30_pk24_gk24_btceth_v1",
    },
    "btceth_dvol_raw": {
        "experiment_id": "E07-HAR-DVOL-RAW-V1",
        "variant": "btceth_dvol",
        "calibration_method": "raw",
        "model_id": "har_range_btceth_dvol_v1",
        "feature_set_id": "har_d1_w7_m30_pk24_gk24_dvol_v1",
    },
    "btceth_dvol_platt": {
        "experiment_id": "E07-HAR-DVOL-PLATT-V1",
        "variant": "btceth_dvol",
        "calibration_method": "platt",
        "model_id": "har_range_btceth_dvol_v1",
        "feature_set_id": "har_d1_w7_m30_pk24_gk24_dvol_v1",
    },
}

PANEL_COLUMNS = [
    "condition_id",
    "asset",
    "direction",
    "decision_time",
    "spot_available_time",
    "log_distance_to_barrier",
    "minutes_to_expiry",
    "dvol_available_time",
    "dvol_age_minutes",
    "dvol_sigma",
    "dvol_available",
    "y",
    "contract_row_weight",
]


def load_dvol(asset: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(
        VOL_DIR / f"DVOL_{asset}.parquet", columns=["ts", "sigma"]
    ).sort_values("ts", ignore_index=True)
    source_time = frame["ts"].to_numpy(dtype=np.int64)
    sigma = frame["sigma"].to_numpy(dtype=float)
    if not np.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError(f"invalid {asset} DVOL sigma")
    return source_time, sigma


def causal_dvol(
    decision_time: np.ndarray,
    source_time: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    decision = np.asarray(decision_time, dtype=np.int64)
    available_source = np.asarray(source_time, dtype=np.int64) + DVOL_DELAY_SECONDS
    indices = np.searchsorted(available_source, decision, side="right") - 1
    safe = np.maximum(indices, 0)
    age = decision - available_source[safe]
    valid = (
        (indices >= 0)
        & (age >= 0)
        & (age <= DVOL_MAX_AGE_SECONDS)
    )
    selected_sigma = np.full(len(decision), np.nan, dtype=float)
    selected_available = np.full(len(decision), -1, dtype=np.int64)
    selected_age = np.full(len(decision), np.nan, dtype=float)
    selected_sigma[valid] = np.asarray(sigma, dtype=float)[indices[valid]]
    selected_available[valid] = available_source[indices[valid]]
    selected_age[valid] = age[valid]
    return selected_sigma, selected_available, selected_age


def augment_with_dvol(
    base: AssetData,
    source_time: np.ndarray,
    sigma: np.ndarray,
) -> tuple[AssetData, dict[str, int | float]]:
    minute_sigma, _, minute_age = causal_dvol(
        base.available_time, source_time, sigma
    )
    dvol_log_variance = np.full(len(minute_sigma), np.nan, dtype=float)
    valid_minute = np.isfinite(minute_sigma)
    dvol_log_variance[valid_minute] = np.log(
        np.maximum(minute_sigma[valid_minute] ** 2 / MINUTES_PER_YEAR, VARIANCE_FLOOR)
    )
    log_features = np.column_stack([base.log_features, dvol_log_variance])

    reference_sigma, _, reference_age = causal_dvol(
        base.reference_time, source_time, sigma
    )
    reference_log_variance = np.full(len(reference_sigma), np.nan, dtype=float)
    valid_reference = np.isfinite(reference_sigma)
    reference_log_variance[valid_reference] = np.log(
        np.maximum(
            reference_sigma[valid_reference] ** 2 / MINUTES_PER_YEAR,
            VARIANCE_FLOOR,
        )
    )
    reference_features = np.column_stack(
        [base.reference_features, reference_log_variance]
    )
    coverage = {
        "minute_rows": int(len(minute_sigma)),
        "minute_dvol_rows": int(valid_minute.sum()),
        "minute_coverage": float(valid_minute.mean()),
        "reference_rows": int(len(reference_sigma)),
        "reference_dvol_rows": int(valid_reference.sum()),
        "reference_coverage": float(valid_reference.mean()),
        "maximum_minute_age_seconds": int(np.nanmax(minute_age)),
        "maximum_reference_age_seconds": int(np.nanmax(reference_age)),
    }
    return (
        AssetData(
            available_time=base.available_time,
            log_features=log_features,
            reference_time=base.reference_time,
            reference_features=reference_features,
            target_end_time=base.target_end_time,
            target_variance=base.target_variance,
        ),
        coverage,
    )


def fit_variant_models(
    assets: dict[str, AssetData],
    feature_names: tuple[str, ...],
    variant: str,
) -> tuple[dict[str, dict[int, dict[str, object]]], list[dict[str, object]]]:
    models = {}
    rows = []
    for cutoff_id, cutoff_text_value in CUTOFF_TEXT.items():
        cutoff = cutoff_timestamp(cutoff_text_value)
        cutoff_epoch = int(cutoff.timestamp())
        models[cutoff_id] = {}
        for horizon_value in HORIZONS:
            horizon = int(horizon_value)
            feature_parts = []
            target_parts = []
            asset_counts = {}
            latest_target = 0
            for asset, data in assets.items():
                eligible = (
                    (data.target_end_time[horizon] <= cutoff_epoch)
                    & np.isfinite(data.target_variance[horizon])
                    & np.isfinite(data.reference_features).all(axis=1)
                )
                feature_parts.append(data.reference_features[eligible])
                target_parts.append(data.target_variance[horizon][eligible])
                asset_counts[asset] = int(eligible.sum())
                if eligible.any():
                    latest_target = max(
                        latest_target,
                        int(data.target_end_time[horizon][eligible].max()),
                    )
            fitted = fit_ridge_log_har(
                np.concatenate(feature_parts), np.concatenate(target_parts)
            )
            fitted["latest_target_available"] = latest_target
            models[cutoff_id][horizon] = fitted
            row = {
                "variant": variant,
                "cutoff_id": cutoff_id,
                "cutoff_time": cutoff,
                "horizon_minutes": horizon,
                "latest_target_available": latest_target,
                "reference_stride_minutes": REFERENCE_STRIDE_MINUTES,
                "ridge_alpha": RIDGE_ALPHA,
                "variance_floor": VARIANCE_FLOOR,
                "target_clip_lower_quantile": TARGET_CLIP_LOWER,
                "target_clip_upper_quantile": TARGET_CLIP_UPPER,
                "asset_pooling": "BTC|ETH",
                **{
                    f"observations_{asset.lower()}": count
                    for asset, count in asset_counts.items()
                },
                **{
                    key: value
                    for key, value in fitted.items()
                    if key not in {"feature_mean", "feature_scale", "coefficient"}
                },
            }
            for index, feature_name in enumerate(feature_names):
                row[f"mean_{feature_name}"] = fitted["feature_mean"][index]
                row[f"scale_{feature_name}"] = fitted["feature_scale"][index]
                row[f"coefficient_{feature_name}"] = fitted["coefficient"][index]
            rows.append(row)
            if latest_target > cutoff_epoch:
                raise ValueError("DVOL HAR target cutoff violation")
        print(f"fitted {variant}: {cutoff_id}", flush=True)
    return models, rows


def read_panel(condition_id: str) -> pd.DataFrame:
    return pd.read_parquet(PANEL_DIR / f"{condition_id}.parquet", columns=PANEL_COLUMNS)


def verify_panel_dvol(panel: pd.DataFrame, dvol_features: np.ndarray) -> None:
    if not panel["dvol_available"].all():
        raise ValueError("BTC/ETH validation panel has unavailable DVOL")
    decision = pd.to_datetime(panel["decision_time"], utc=True)
    available = pd.to_datetime(panel["dvol_available_time"], utc=True)
    if available.isna().any() or (available > decision).any():
        raise ValueError("DVOL availability exceeds decision time")
    age = panel["dvol_age_minutes"].to_numpy(dtype=float) * 60
    if (age < 0).any() or (age > DVOL_MAX_AGE_SECONDS).any():
        raise ValueError("DVOL exceeds frozen age bound")
    sigma = panel["dvol_sigma"].to_numpy(dtype=float)
    expected = np.log(np.maximum(sigma**2 / MINUTES_PER_YEAR, VARIANCE_FLOOR))
    if not np.allclose(dvol_features[:, -1], expected, rtol=0, atol=1e-12):
        raise ValueError("raw DVOL alignment disagrees with causal panel")


def predict_panel(
    panel: pd.DataFrame,
    data: AssetData,
    models: dict[int, dict[str, object]],
    verify_dvol: bool,
) -> np.ndarray:
    features = features_for_panel(panel, data)
    if verify_dvol:
        verify_panel_dvol(panel, features)
    variance = forecast_integrated_variance(
        features,
        panel["minutes_to_expiry"].to_numpy(dtype=float),
        models,
    )
    return har_touch_probability(
        panel["log_distance_to_barrier"].to_numpy(dtype=float),
        variance,
        panel["direction"].to_numpy(),
    )


def load_calibration_predictions(
    calendar: pd.DataFrame,
    assets_by_variant: dict[str, dict[str, AssetData]],
    models_by_variant: dict[str, dict[str, dict[int, dict[str, object]]]],
) -> dict[str, dict[str, dict[str, object]]]:
    result = {variant: {} for variant in VARIANTS}
    for block_id, specification in CALIBRATION_BLOCKS.items():
        members = calendar[
            calendar.asset.isin(ASSETS)
            & (calendar.resolution_date_et >= specification["start"])
            & (calendar.resolution_date_et <= specification["end"])
        ]
        accumulators = {
            variant: {"probability": [], "outcome": [], "weight": []}
            for variant in VARIANTS
        }
        cutoff_id = specification["cutoff_id"]
        for member in members.itertuples(index=False):
            panel = read_panel(member.condition_id)
            for variant in VARIANTS:
                probability = predict_panel(
                    panel,
                    assets_by_variant[variant][member.asset],
                    models_by_variant[variant][cutoff_id],
                    verify_dvol=variant == "btceth_dvol",
                )
                if not np.isfinite(probability).all():
                    raise ValueError(f"invalid {variant} probability")
                accumulators[variant]["probability"].append(probability)
                accumulators[variant]["outcome"].append(
                    panel.y.to_numpy(dtype=float)
                )
                accumulators[variant]["weight"].append(
                    panel.contract_row_weight.to_numpy(dtype=float)
                )
        for variant in VARIANTS:
            result[variant][block_id] = {
                key: np.concatenate(parts)
                for key, parts in accumulators[variant].items()
            }
            result[variant][block_id].update(
                {"contracts": len(members), "cutoff_id": cutoff_id}
            )
        print(
            f"generated matched BTC/ETH OOF {block_id}: {len(members):,} contracts",
            flush=True,
        )
    return result


def fit_calibrators(
    blocks_by_variant: dict[str, dict[str, dict[str, object]]]
) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        blocks = blocks_by_variant[variant]
        for fold_id, block_ids in FOLD_CALIBRATION_BLOCKS.items():
            probability = np.concatenate(
                [blocks[item]["probability"] for item in block_ids]
            )
            outcome = np.concatenate([blocks[item]["outcome"] for item in block_ids])
            weight = np.concatenate([blocks[item]["weight"] for item in block_ids])
            fitted = fit_platt(probability, outcome, weight)
            rows.append(
                {
                    "variant": variant,
                    "fold_id": fold_id,
                    "model_parameter_cutoff": FOLD_DISTRIBUTION_CUTOFF[fold_id],
                    "calibration_version": f"{fold_id}_{variant}_platt_v1",
                    "calibration_blocks": "|".join(block_ids),
                    "calibration_parameter_cutoffs": "|".join(
                        str(blocks[item]["cutoff_id"]) for item in block_ids
                    ),
                    "calibration_contracts": int(
                        sum(int(blocks[item]["contracts"]) for item in block_ids)
                    ),
                    "calibration_rows": len(probability),
                    "calibration_weight": float(weight.sum()),
                    "platt_probability_clip": PROBABILITY_CLIP,
                    "platt_l2_slope": PLATT_L2,
                    **fitted,
                }
            )
            print(
                f"fitted {variant} {fold_id} Platt: "
                f"slope={fitted['slope']:.6f} intercept={fitted['intercept']:.6f}",
                flush=True,
            )
    calibrators = pd.DataFrame(rows)
    atomic_parquet(calibrators, CALIBRATORS_FILE)
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
    raw_column: str,
    probability_column: str,
) -> pd.DataFrame:
    experiment = EXPERIMENTS[model_key]
    calibrated = experiment["calibration_method"] == "platt"
    if experiment["variant"] == "btceth_dvol":
        information_timestamp = pd.concat(
            [base.spot_available_time, base.dvol_available_time], axis=1
        ).max(axis=1)
    else:
        information_timestamp = base.spot_available_time
    return pd.DataFrame(
        {
            "schema_version": "phase3_prediction_v1",
            "experiment_id": experiment["experiment_id"],
            "condition_id": base.condition_id,
            "decision_time": base.decision_time,
            "information_timestamp": information_timestamp,
            "fold_id": fold_id,
            "role": "validation",
            "prediction_kind": "forward",
            "model_id": experiment["model_id"],
            "model_version": "v1",
            "feature_set_id": experiment["feature_set_id"],
            "calibration_method": experiment["calibration_method"],
            "calibration_version": (
                f"{fold_id}_{experiment['variant']}_platt_v1"
                if calibrated
                else "none"
            ),
            "raw_yes_probability": base[raw_column].to_numpy(dtype=float),
            "calibrated_yes_probability": base[probability_column].to_numpy(
                dtype=float
            ),
            "contract_row_weight": base.contract_row_weight.to_numpy(dtype=float),
        }
    )


def generate_validation_outputs(
    folds: pd.DataFrame,
    assets_by_variant: dict[str, dict[str, AssetData]],
    models_by_variant: dict[str, dict[str, dict[int, dict[str, object]]]],
    calibrators: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calibrator_map = calibrators.set_index(["variant", "fold_id"]).to_dict("index")
    metrics = []
    manifests = []
    reliability = defaultdict(lambda: np.zeros(4, dtype=float))
    for fold_id in FOLD_CALIBRATION_BLOCKS:
        validation = folds[
            (folds.fold_id == fold_id)
            & (folds.role == "validation")
            & folds.asset.isin(ASSETS)
        ]
        cutoff_id = FOLD_DISTRIBUTION_CUTOFF[fold_id]
        for asset in ASSETS:
            members = validation[validation.asset == asset]
            parts = []
            for member in members.itertuples(index=False):
                panel = read_panel(member.condition_id)
                raw_by_variant = {}
                platt_by_variant = {}
                for variant in VARIANTS:
                    raw = predict_panel(
                        panel,
                        assets_by_variant[variant][asset],
                        models_by_variant[variant][cutoff_id],
                        verify_dvol=variant == "btceth_dvol",
                    )
                    fitted = calibrator_map[(variant, fold_id)]
                    clipped = np.clip(raw, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
                    logit = np.log(clipped) - np.log1p(-clipped)
                    raw_by_variant[variant] = raw
                    platt_by_variant[variant] = expit(
                        fitted["slope"] * logit + fitted["intercept"]
                    )
                panel = panel.copy()
                panel["control_probability"] = raw_by_variant["btceth_control"]
                panel["control_platt_probability"] = platt_by_variant[
                    "btceth_control"
                ]
                panel["dvol_probability"] = raw_by_variant["btceth_dvol"]
                panel["dvol_platt_probability"] = platt_by_variant["btceth_dvol"]
                parts.append(panel)
                outcome = panel.y.to_numpy(dtype=float)
                weight = panel.contract_row_weight.to_numpy(dtype=float)
                model_probability = {
                    "btceth_control_raw": raw_by_variant["btceth_control"],
                    "btceth_control_platt": platt_by_variant["btceth_control"],
                    "btceth_dvol_raw": raw_by_variant["btceth_dvol"],
                    "btceth_dvol_platt": platt_by_variant["btceth_dvol"],
                }
                for model_key, probability in model_probability.items():
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
            columns = {
                "btceth_control_raw": ("control_probability", "control_probability"),
                "btceth_control_platt": (
                    "control_probability",
                    "control_platt_probability",
                ),
                "btceth_dvol_raw": ("dvol_probability", "dvol_probability"),
                "btceth_dvol_platt": (
                    "dvol_probability",
                    "dvol_platt_probability",
                ),
            }
            for model_key, (raw_column, probability_column) in columns.items():
                output = prediction_frame(
                    base, model_key, fold_id, raw_column, probability_column
                )
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
                f"wrote DVOL ablation {fold_id} {asset}: {len(base):,} rows, "
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
    parameters: pd.DataFrame,
    coverage: pd.DataFrame,
    calibrators: pd.DataFrame,
    metrics: pd.DataFrame,
    reliability: pd.DataFrame,
    manifest: pd.DataFrame,
) -> str:
    by_fold, overall, by_asset, by_direction = metric_tables(metrics, reliability)
    labels = {
        "btceth_control_raw": "BTC/ETH control raw",
        "btceth_control_platt": "BTC/ETH control + Platt",
        "btceth_dvol_raw": "BTC/ETH DVOL raw",
        "btceth_dvol_platt": "BTC/ETH DVOL + Platt",
    }
    order = list(labels)
    indexed = overall.set_index("model_key")
    winner = min(
        order,
        key=lambda key: (indexed.loc[key, "brier"], indexed.loc[key, "log_loss"]),
    )
    empirical = pd.read_parquet(FOLDER / "empirical_contract_metrics.parquet")
    empirical = empirical[
        (empirical.model_key == "empirical_platt") & empirical.asset.isin(ASSETS)
    ]
    empirical_brier = empirical.brier.mean()
    empirical_log_loss = empirical.log_loss.mean()

    lines = [
        "# Phase-3 matched BTC/ETH DVOL ablation",
        "",
        "This report uses validation roles only. SOL and XRP were not assigned",
        "DVOL values, and evaluation and Phase-4 execution data were not read.",
        "",
        "## Causal DVOL coverage",
        "",
        "| Asset | Minute coverage | Hourly-origin coverage | Maximum age |",
        "|---|---:|---:|---:|",
    ]
    for row in coverage.itertuples(index=False):
        lines.append(
            f"| {row.asset} | {row.minute_coverage:.2%} | "
            f"{row.reference_coverage:.2%} | {row.maximum_reference_age_seconds / 60:.0f} min |"
        )
    lines.extend(
        [
            "",
            "Coverage percentages include the full spot history, including the",
            "initial period before rolling 30-day HAR features become available.",
            "All fitted origins and BTC/ETH decision rows have valid delayed DVOL.",
            "",
            "## Platt parameters",
            "",
            "| Variant | Fold | Contracts | Rows | Slope | Intercept |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in calibrators.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.fold_id} | {row.calibration_contracts:,} | "
            f"{row.calibration_rows:,} | {row.slope:.6f} | {row.intercept:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Matched validation metrics by fold",
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
            "## Combined matched BTC/ETH comparison",
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
            "## Frozen decision",
            "",
            f"- Registered four-way winner: **{labels[winner]}**.",
            f"- Raw DVOL minus raw control Brier: {indexed.loc['btceth_dvol_raw', 'brier'] - indexed.loc['btceth_control_raw', 'brier']:+.6f}.",
            f"- Platt DVOL minus Platt control Brier: {indexed.loc['btceth_dvol_platt', 'brier'] - indexed.loc['btceth_control_platt', 'brier']:+.6f}.",
            f"- Platt DVOL minus Platt control log loss: {indexed.loc['btceth_dvol_platt', 'log_loss'] - indexed.loc['btceth_control_platt', 'log_loss']:+.6f}.",
            f"- Empirical + Platt on the same BTC/ETH fold-contract appearances: Brier {empirical_brier:.6f}; log loss {empirical_log_loss:.6f}.",
            "- Any retained DVOL result applies only to BTC and ETH; SOL and XRP",
            "  continue on the common no-DVOL specification.",
            "",
            "## Breakdown by asset",
            "",
            "| Model | Asset | Contracts | Brier | Log loss |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for model_key in order:
        for asset in ASSETS:
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
            f"- Parameter fits: {len(parameters):,}.",
            f"- Prediction partitions: {len(manifest):,}.",
            f"- Prediction rows: {int(manifest.rows.sum()):,}.",
            "",
            "These are probability-validation results, not profitability results.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    folds = pd.read_parquet(FOLD_FILE)
    calendar = unique_contract_calendar(folds)
    control_assets = {asset: prepare_asset_data(asset) for asset in ASSETS}
    dvol_assets = {}
    coverage_rows = []
    for asset in ASSETS:
        source_time, sigma = load_dvol(asset)
        dvol_assets[asset], coverage = augment_with_dvol(
            control_assets[asset], source_time, sigma
        )
        coverage_rows.append(
            {
                "asset": asset,
                "source_rows": len(source_time),
                "first_source_time": pd.to_datetime(source_time[0], unit="s", utc=True),
                "last_source_time": pd.to_datetime(source_time[-1], unit="s", utc=True),
                "delay_seconds": DVOL_DELAY_SECONDS,
                "maximum_age_seconds": DVOL_MAX_AGE_SECONDS,
                **coverage,
            }
        )
    coverage_frame = pd.DataFrame(coverage_rows)
    atomic_parquet(coverage_frame, COVERAGE_FILE)

    control_models, control_rows = fit_variant_models(
        control_assets, CONTROL_FEATURE_NAMES, "btceth_control"
    )
    dvol_models, dvol_rows = fit_variant_models(
        dvol_assets, DVOL_FEATURE_NAMES, "btceth_dvol"
    )
    parameters = pd.DataFrame(control_rows + dvol_rows)
    atomic_parquet(parameters, PARAMETERS_FILE)
    assets_by_variant = {
        "btceth_control": control_assets,
        "btceth_dvol": dvol_assets,
    }
    models_by_variant = {
        "btceth_control": control_models,
        "btceth_dvol": dvol_models,
    }
    blocks = load_calibration_predictions(
        calendar, assets_by_variant, models_by_variant
    )
    calibrators = fit_calibrators(blocks)
    del blocks
    metrics, reliability, manifest = generate_validation_outputs(
        folds, assets_by_variant, models_by_variant, calibrators
    )
    REPORT_FILE.write_text(
        make_report(
            parameters,
            coverage_frame,
            calibrators,
            metrics,
            reliability,
            manifest,
        ),
        encoding="utf-8",
    )
    print(f"parameters: {PARAMETERS_FILE}")
    print(f"metrics: {METRICS_FILE}")
    print(f"prediction manifest: {PREDICTION_MANIFEST_FILE}")
    print(f"report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
