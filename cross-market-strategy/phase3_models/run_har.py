#!/usr/bin/env python3
"""Run the frozen validation-only zero-price-drift HAR/range model."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, ndtr

from run_benchmarks import (
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
    ASSET_SYMBOLS,
    CALIBRATION_BLOCKS,
    CUTOFF_TEXT,
    FOLD_CALIBRATION_BLOCKS,
    FOLD_DISTRIBUTION_CUTOFF,
    FOLD_FILE,
    HORIZONS,
    PANEL_DIR,
    SPOT_DIR,
    cutoff_timestamp,
    unique_contract_calendar,
)


FOLDER = Path(__file__).resolve().parent
OUTPUT_DIR = FOLDER / "har_predictions"
PARAMETERS_FILE = FOLDER / "har_parameters.parquet"
CALIBRATORS_FILE = FOLDER / "har_calibrators.parquet"
METRICS_FILE = FOLDER / "har_contract_metrics.parquet"
RELIABILITY_FILE = FOLDER / "har_reliability.parquet"
PREDICTION_MANIFEST_FILE = FOLDER / "har_prediction_manifest.parquet"
REPORT_FILE = FOLDER / "har_report.md"

FEATURE_NAMES = (
    "variance_1d",
    "variance_7d",
    "variance_30d",
    "parkinson_variance_24h",
    "garman_klass_variance_24h",
)
FEATURE_WINDOWS = (1440, 10080, 43200)
REFERENCE_STRIDE_MINUTES = 60
VARIANCE_FLOOR = 1e-16
RIDGE_ALPHA = 1e-3
TARGET_CLIP_LOWER = 0.001
TARGET_CLIP_UPPER = 0.999

MODEL_KEYS = ("har_raw", "har_platt")
EXPERIMENTS = {
    "har_raw": {
        "experiment_id": "E06-HAR-RAW-V1",
        "calibration_method": "raw",
    },
    "har_platt": {
        "experiment_id": "E06-HAR-PLATT-V1",
        "calibration_method": "platt",
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
    "y",
    "contract_row_weight",
]


@dataclass
class AssetData:
    available_time: np.ndarray
    log_features: np.ndarray
    reference_time: np.ndarray
    reference_features: np.ndarray
    target_end_time: dict[int, np.ndarray]
    target_variance: dict[int, np.ndarray]


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values, dtype=float)
        .rolling(window, min_periods=window)
        .mean()
        .to_numpy()
    )


def future_average_variance(squared_returns: np.ndarray, horizon: int) -> np.ndarray:
    """Mean of returns i+1 through i+h, excluding the origin return."""

    values = np.asarray(squared_returns, dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    if horizon <= 0 or len(values) <= horizon:
        return result
    clean = np.where(np.isfinite(values), values, 0.0)
    finite = np.isfinite(values).astype(np.int64)
    cumulative = np.concatenate(([0.0], np.cumsum(clean)))
    finite_cumulative = np.concatenate(([0], np.cumsum(finite)))
    origins = np.arange(len(values) - horizon)
    starts = origins + 1
    ends = origins + horizon + 1
    counts = finite_cumulative[ends] - finite_cumulative[starts]
    valid = counts == horizon
    result[origins[valid]] = (
        cumulative[ends[valid]] - cumulative[starts[valid]]
    ) / horizon
    return result


def prepare_asset_data(asset: str) -> AssetData:
    symbol = ASSET_SYMBOLS[asset]
    spot = pd.read_parquet(
        SPOT_DIR / f"{symbol}_1m.parquet",
        columns=["ts", "open", "high", "low", "close"],
    ).sort_values("ts", ignore_index=True)
    timestamp = spot["ts"].to_numpy(dtype=np.int64)
    if len(timestamp) < max(FEATURE_WINDOWS) or not np.all(np.diff(timestamp) == 60):
        raise ValueError(f"{asset} spot data is not a complete one-minute grid")

    log_close = np.log(spot["close"].to_numpy(dtype=float))
    returns = np.empty(len(log_close), dtype=float)
    returns[0] = np.nan
    returns[1:] = np.diff(log_close)
    squared_returns = returns**2

    return_variances = [
        rolling_mean(squared_returns, window) for window in FEATURE_WINDOWS
    ]
    log_range = np.log(
        spot["high"].to_numpy(dtype=float) / spot["low"].to_numpy(dtype=float)
    )
    parkinson = log_range**2 / (4 * np.log(2))
    log_open_close = np.log(
        spot["close"].to_numpy(dtype=float) / spot["open"].to_numpy(dtype=float)
    )
    garman_klass = (
        0.5 * log_range**2 - (2 * np.log(2) - 1) * log_open_close**2
    )
    parkinson_24h = np.clip(rolling_mean(parkinson, 1440), 0, None)
    garman_klass_24h = np.clip(rolling_mean(garman_klass, 1440), 0, None)
    raw_features = np.column_stack(
        [*return_variances, parkinson_24h, garman_klass_24h]
    )
    log_features = np.log(np.maximum(raw_features, VARIANCE_FLOOR))
    available_time = timestamp + 60

    hourly = (timestamp % (REFERENCE_STRIDE_MINUTES * 60)) == 0
    eligible_feature = hourly & np.isfinite(log_features).all(axis=1)
    reference_index = np.flatnonzero(eligible_feature)
    reference_time = available_time[reference_index]
    reference_features = log_features[reference_index]
    target_end_time = {}
    target_variance = {}
    for horizon_value in HORIZONS:
        horizon = int(horizon_value)
        target = future_average_variance(squared_returns, horizon)
        end = np.full(len(timestamp), np.iinfo(np.int64).max, dtype=np.int64)
        valid_end = np.arange(len(timestamp)) + horizon < len(timestamp)
        indices = np.flatnonzero(valid_end)
        end[indices] = timestamp[indices + horizon] + 60
        target_end_time[horizon] = end[reference_index]
        target_variance[horizon] = target[reference_index]

    print(
        f"prepared HAR inputs: {asset} {len(timestamp):,} minutes, "
        f"{len(reference_index):,} hourly origins",
        flush=True,
    )
    return AssetData(
        available_time=available_time,
        log_features=log_features,
        reference_time=reference_time,
        reference_features=reference_features,
        target_end_time=target_end_time,
        target_variance=target_variance,
    )


def fit_ridge_log_har(
    features: np.ndarray,
    target_variance: np.ndarray,
) -> dict[str, object]:
    features = np.asarray(features, dtype=float)
    target = np.asarray(target_variance, dtype=float)
    valid = (
        np.isfinite(features).all(axis=1)
        & np.isfinite(target)
        & (target >= 0)
    )
    features = features[valid]
    target = target[valid]
    if len(target) < 1000:
        raise ValueError("insufficient observations for HAR fit")

    y = np.log(np.maximum(target, VARIANCE_FLOOR))
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0, ddof=0)
    if not np.isfinite(feature_scale).all() or (feature_scale <= 0).any():
        raise ValueError("invalid HAR feature scale")
    standardized = (features - feature_mean) / feature_scale
    intercept = float(y.mean())
    centred_y = y - intercept
    gram = standardized.T @ standardized / len(y)
    rhs = standardized.T @ centred_y / len(y)
    coefficient = np.linalg.solve(
        gram + RIDGE_ALPHA * np.eye(standardized.shape[1]), rhs
    )
    fitted = intercept + np.einsum("ij,j->i", standardized, coefficient)
    residual = y - fitted
    residual_mse = float(np.mean(residual**2))
    correction = float(np.exp(0.5 * residual_mse))
    lower = float(np.quantile(target, TARGET_CLIP_LOWER))
    upper = float(np.quantile(target, TARGET_CLIP_UPPER))
    if not (0 <= lower < upper and np.isfinite(correction)):
        raise ValueError("invalid HAR target bounds or lognormal correction")
    return {
        "observations": int(len(target)),
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "coefficient": coefficient,
        "intercept": intercept,
        "residual_mse": residual_mse,
        "lognormal_correction": correction,
        "target_variance_lower": lower,
        "target_variance_upper": upper,
        "target_variance_mean": float(target.mean()),
        "target_variance_median": float(np.median(target)),
    }


def predict_average_variance(features: np.ndarray, model: dict[str, object]) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    standardized = (
        features - np.asarray(model["feature_mean"])
    ) / np.asarray(model["feature_scale"])
    log_prediction = float(model["intercept"]) + np.einsum(
        "ij,j->i", standardized, np.asarray(model["coefficient"])
    )
    prediction = np.exp(log_prediction) * float(model["lognormal_correction"])
    return np.clip(
        prediction,
        float(model["target_variance_lower"]),
        float(model["target_variance_upper"]),
    )


def fit_all_models(
    assets: dict[str, AssetData],
) -> tuple[dict[str, dict[int, dict[str, object]]], pd.DataFrame]:
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
            latest_target_end = 0
            for asset, data in assets.items():
                eligible = (
                    (data.target_end_time[horizon] <= cutoff_epoch)
                    & np.isfinite(data.target_variance[horizon])
                )
                feature_parts.append(data.reference_features[eligible])
                target_parts.append(data.target_variance[horizon][eligible])
                asset_counts[asset] = int(eligible.sum())
                if eligible.any():
                    latest_target_end = max(
                        latest_target_end,
                        int(data.target_end_time[horizon][eligible].max()),
                    )
            fitted = fit_ridge_log_har(
                np.concatenate(feature_parts), np.concatenate(target_parts)
            )
            fitted["latest_target_available"] = latest_target_end
            models[cutoff_id][horizon] = fitted
            row = {
                "cutoff_id": cutoff_id,
                "cutoff_time": cutoff,
                "horizon_minutes": horizon,
                "latest_target_available": latest_target_end,
                "reference_stride_minutes": REFERENCE_STRIDE_MINUTES,
                "ridge_alpha": RIDGE_ALPHA,
                "variance_floor": VARIANCE_FLOOR,
                "target_clip_lower_quantile": TARGET_CLIP_LOWER,
                "target_clip_upper_quantile": TARGET_CLIP_UPPER,
                "asset_pooling": "BTC|ETH|SOL|XRP",
                **{
                    f"observations_{asset.lower()}": count
                    for asset, count in asset_counts.items()
                },
                **{
                    key: value
                    for key, value in fitted.items()
                    if key
                    not in {"feature_mean", "feature_scale", "coefficient"}
                },
            }
            for index, feature_name in enumerate(FEATURE_NAMES):
                row[f"mean_{feature_name}"] = fitted["feature_mean"][index]
                row[f"scale_{feature_name}"] = fitted["feature_scale"][index]
                row[f"coefficient_{feature_name}"] = fitted["coefficient"][index]
            rows.append(row)
            if latest_target_end > cutoff_epoch:
                raise ValueError("HAR target cutoff violation")
        print(f"fitted pooled HAR models: {cutoff_id}", flush=True)

    parameters = pd.DataFrame(rows)
    atomic_parquet(parameters, PARAMETERS_FILE)
    return models, parameters


def features_for_panel(panel: pd.DataFrame, data: AssetData) -> np.ndarray:
    requested = (
        pd.to_datetime(panel["spot_available_time"], utc=True)
        .astype("int64")
        .to_numpy()
        // 1_000_000_000
    )
    indices = np.searchsorted(data.available_time, requested)
    valid = indices < len(data.available_time)
    if not valid.all() or not np.array_equal(data.available_time[indices], requested):
        raise ValueError("panel spot timestamps do not match HAR feature grid")
    features = data.log_features[indices]
    if not np.isfinite(features).all():
        raise ValueError("panel contains unavailable HAR features")
    return features


def forecast_integrated_variance(
    features: np.ndarray,
    minutes_to_expiry: np.ndarray,
    models: dict[int, dict[str, object]],
) -> np.ndarray:
    horizon = np.asarray(minutes_to_expiry, dtype=float)
    valid = np.isfinite(horizon) & (horizon > 0)
    result = np.full(len(horizon), np.nan, dtype=float)
    if not valid.any():
        return result
    feature_values = np.asarray(features, dtype=float)[valid]
    forecasts = np.column_stack(
        [predict_average_variance(feature_values, models[int(item)]) for item in HORIZONS]
    )
    clipped_horizon = np.clip(horizon[valid], HORIZONS[0], HORIZONS[-1])
    upper_index = np.searchsorted(HORIZONS, clipped_horizon, side="left")
    upper_index = np.minimum(upper_index, len(HORIZONS) - 1)
    lower_index = np.maximum(upper_index - 1, 0)
    exact = HORIZONS[upper_index] == clipped_horizon
    lower_index[exact] = upper_index[exact]
    row_index = np.arange(len(feature_values))
    low = forecasts[row_index, lower_index]
    high = forecasts[row_index, upper_index]
    denominator = np.log(HORIZONS[upper_index]) - np.log(HORIZONS[lower_index])
    weight = np.zeros(len(feature_values), dtype=float)
    different = upper_index != lower_index
    weight[different] = (
        np.log(clipped_horizon[different]) - np.log(HORIZONS[lower_index[different]])
    ) / denominator[different]
    log_average_variance = (1 - weight) * np.log(low) + weight * np.log(high)
    result[valid] = np.exp(log_average_variance) * horizon[valid]
    return result


def har_touch_probability(
    distance: np.ndarray,
    integrated_variance: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    variance = np.asarray(integrated_variance, dtype=float)
    direction = np.asarray(direction)
    valid = (
        np.isfinite(distance)
        & np.isfinite(variance)
        & (variance > 0)
        & np.isin(direction, ["up", "down"])
    )
    probability = np.full(len(distance), np.nan, dtype=float)
    if not valid.any():
        return probability
    d = distance[valid]
    total_variance = variance[valid]
    root_variance = np.sqrt(total_variance)
    log_mean = -0.5 * total_variance
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        up = ndtr((log_mean - d) / root_variance) + np.exp(
            2 * log_mean * d / total_variance
        ) * ndtr((-log_mean - d) / root_variance)
        down = ndtr((-log_mean - d) / root_variance) + np.exp(
            -2 * log_mean * d / total_variance
        ) * ndtr((log_mean - d) / root_variance)
    selected = np.where(direction[valid] == "up", up, down)
    selected = np.where(d <= 0, 1.0, selected)
    probability[valid] = np.clip(selected, 0.0, 1.0)
    return probability


def predict_panel(
    panel: pd.DataFrame,
    asset_data: AssetData,
    models: dict[int, dict[str, object]],
) -> np.ndarray:
    features = features_for_panel(panel, asset_data)
    integrated_variance = forecast_integrated_variance(
        features, panel["minutes_to_expiry"].to_numpy(dtype=float), models
    )
    return har_touch_probability(
        panel["log_distance_to_barrier"].to_numpy(dtype=float),
        integrated_variance,
        panel["direction"].to_numpy(),
    )


def read_panel(condition_id: str) -> pd.DataFrame:
    return pd.read_parquet(PANEL_DIR / f"{condition_id}.parquet", columns=PANEL_COLUMNS)


def load_calibration_predictions(
    calendar: pd.DataFrame,
    assets: dict[str, AssetData],
    models: dict[str, dict[int, dict[str, object]]],
) -> dict[str, dict[str, object]]:
    blocks = {}
    for block_id, specification in CALIBRATION_BLOCKS.items():
        members = calendar[
            (calendar["resolution_date_et"] >= specification["start"])
            & (calendar["resolution_date_et"] <= specification["end"])
        ]
        probability_parts = []
        outcome_parts = []
        weight_parts = []
        cutoff_id = specification["cutoff_id"]
        for member in members.itertuples(index=False):
            panel = read_panel(member.condition_id)
            probability = predict_panel(
                panel, assets[member.asset], models[cutoff_id]
            )
            if not np.isfinite(probability).all():
                raise ValueError(f"invalid HAR probability in {member.condition_id}")
            probability_parts.append(probability)
            outcome_parts.append(panel["y"].to_numpy(dtype=float))
            weight_parts.append(panel["contract_row_weight"].to_numpy(dtype=float))
        blocks[block_id] = {
            "probability": np.concatenate(probability_parts),
            "outcome": np.concatenate(outcome_parts),
            "weight": np.concatenate(weight_parts),
            "contracts": len(members),
            "cutoff_id": cutoff_id,
        }
        print(
            f"generated HAR OOF {block_id}: {len(members):,} contracts, "
            f"{len(blocks[block_id]['probability']):,} rows",
            flush=True,
        )
    return blocks


def fit_fold_calibrators(
    blocks: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for fold_id, block_ids in FOLD_CALIBRATION_BLOCKS.items():
        probability = np.concatenate([blocks[item]["probability"] for item in block_ids])
        outcome = np.concatenate([blocks[item]["outcome"] for item in block_ids])
        weight = np.concatenate([blocks[item]["weight"] for item in block_ids])
        fitted = fit_platt(probability, outcome, weight)
        rows.append(
            {
                "fold_id": fold_id,
                "model_parameter_cutoff": FOLD_DISTRIBUTION_CUTOFF[fold_id],
                "calibration_version": f"{fold_id}_har_platt_v1",
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
            f"fitted HAR {fold_id} Platt: slope={fitted['slope']:.6f}, "
            f"intercept={fitted['intercept']:.6f}",
            flush=True,
        )
    calibrators = pd.DataFrame(rows)
    atomic_parquet(calibrators, CALIBRATORS_FILE)
    return calibrators


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
            "condition_id": base["condition_id"],
            "decision_time": base["decision_time"],
            "information_timestamp": base["spot_available_time"],
            "fold_id": fold_id,
            "role": "validation",
            "prediction_kind": "forward",
            "model_id": "har_range_zero_drift_v1",
            "model_version": "v1",
            "feature_set_id": "har_d1_w7_m30_pk24_gk24_v1",
            "calibration_method": experiment["calibration_method"],
            "calibration_version": (
                f"{fold_id}_har_platt_v1" if calibrated else "none"
            ),
            "raw_yes_probability": base["har_probability"].to_numpy(dtype=float),
            "calibrated_yes_probability": base[probability_column].to_numpy(dtype=float),
            "contract_row_weight": base["contract_row_weight"].to_numpy(dtype=float),
        }
    )


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


def generate_validation_outputs(
    folds: pd.DataFrame,
    assets: dict[str, AssetData],
    models: dict[str, dict[int, dict[str, object]]],
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
            for member in members.itertuples(index=False):
                panel = read_panel(member.condition_id)
                raw = predict_panel(panel, assets[asset], models[cutoff_id])
                if not np.isfinite(raw).all():
                    raise ValueError(f"invalid HAR probability in {member.condition_id}")
                clipped = np.clip(raw, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
                logit = np.log(clipped) - np.log1p(-clipped)
                platt = expit(fitted["slope"] * logit + fitted["intercept"])
                panel = panel.copy()
                panel["har_probability"] = raw
                panel["har_platt_probability"] = platt
                parts.append(panel)
                outcome = panel["y"].to_numpy(dtype=float)
                weight = panel["contract_row_weight"].to_numpy(dtype=float)
                for model_key, probability in {
                    "har_raw": raw,
                    "har_platt": platt,
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
                ("har_raw", "har_probability"),
                ("har_platt", "har_platt_probability"),
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
                f"wrote HAR {fold_id} validation {asset}: {len(base):,} rows, "
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


def metric_tables(
    metrics: pd.DataFrame, reliability: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_fold = metrics.groupby(["fold_id", "model_key"], as_index=False).agg(
        contracts=("condition_id", "size"),
        decision_rows=("n_rows", "sum"),
        brier=("brier", "mean"),
        log_loss=("log_loss", "mean"),
    )
    fold_ece = (
        reliability.groupby(["fold_id", "model_key"])
        .apply(weighted_ece, include_groups=False)
        .rename("ece_10")
        .reset_index()
    )
    by_fold = by_fold.merge(fold_ece, on=["fold_id", "model_key"])
    overall = metrics.groupby("model_key", as_index=False).agg(
        fold_contract_appearances=("condition_id", "size"),
        brier=("brier", "mean"),
        log_loss=("log_loss", "mean"),
    )
    overall_ece = (
        reliability.groupby("model_key")
        .apply(weighted_ece, include_groups=False)
        .rename("ece_10")
        .reset_index()
    )
    overall = overall.merge(overall_ece, on="model_key")
    by_asset = metrics.groupby(["model_key", "asset"], as_index=False).agg(
        contracts=("condition_id", "size"),
        brier=("brier", "mean"),
        log_loss=("log_loss", "mean"),
    )
    by_direction = metrics.groupby(["model_key", "direction"], as_index=False).agg(
        contracts=("condition_id", "size"),
        brier=("brier", "mean"),
        log_loss=("log_loss", "mean"),
    )
    return by_fold, overall, by_asset, by_direction


def make_report(
    parameters: pd.DataFrame,
    calibrators: pd.DataFrame,
    har_metrics: pd.DataFrame,
    har_reliability: pd.DataFrame,
    prediction_manifest: pd.DataFrame,
) -> str:
    metric_files = [
        "benchmark_contract_metrics.parquet",
        "empirical_contract_metrics.parquet",
        "kou_contract_metrics.parquet",
        "kou_local_contract_metrics.parquet",
    ]
    reliability_files = [
        "benchmark_reliability.parquet",
        "empirical_reliability.parquet",
        "kou_reliability.parquet",
        "kou_local_reliability.parquet",
    ]
    metrics = pd.concat(
        [pd.read_parquet(FOLDER / path) for path in metric_files] + [har_metrics],
        ignore_index=True,
    )
    reliability = pd.concat(
        [pd.read_parquet(FOLDER / path) for path in reliability_files]
        + [har_reliability],
        ignore_index=True,
    )
    by_fold, overall, by_asset, by_direction = metric_tables(metrics, reliability)
    labels = {
        "market": "Polymarket raw",
        "gbm_raw": "GBM raw",
        "gbm_platt": "GBM + Platt",
        "empirical_raw": "Empirical raw",
        "empirical_platt": "Empirical + Platt",
        "kou_raw": "Kou 4-MAD raw",
        "kou_platt": "Kou 4-MAD + Platt",
        "kou_local_raw": "Kou local-z raw",
        "kou_local_platt": "Kou local-z + Platt",
        "har_raw": "HAR/range raw",
        "har_platt": "HAR/range + Platt",
    }
    indexed = overall.set_index("model_key")
    family_winner = min(
        MODEL_KEYS,
        key=lambda key: (indexed.loc[key, "brier"], indexed.loc[key, "log_loss"]),
    )
    tournament_winner = min(
        labels,
        key=lambda key: (indexed.loc[key, "brier"], indexed.loc[key, "log_loss"]),
    )
    har_fold = {
        key: by_fold[by_fold.model_key == key].set_index("fold_id") for key in MODEL_KEYS
    }
    brier_wins = int((har_fold["har_platt"].brier < har_fold["har_raw"].brier).sum())
    log_wins = int(
        (har_fold["har_platt"].log_loss < har_fold["har_raw"].log_loss).sum()
    )

    lines = [
        "# Phase-3 zero-price-drift HAR/range validation report",
        "",
        "This report contains validation results only. Evaluation roles and the",
        "Phase-4 execution proxy were not read.",
        "",
        "## Frozen specification",
        "",
        "- Pooled BTC/ETH/SOL/XRP multi-horizon log-HAR coefficients.",
        "- Features: 1-day, 7-day, and 30-day realized variance plus 24-hour",
        "  Parkinson and Garman--Klass variance.",
        f"- Horizons: `{list(map(int, HORIZONS))}` minutes; hourly training origins.",
        f"- Fixed standardized ridge penalty `{RIDGE_ALPHA}` and training target",
        "  0.1%/99.9% forecast clipping.",
        "- Zero price drift; non-zero drift is deferred to Phase 3.5.",
        "",
        "## Parameter diagnostics",
        "",
        "| Cutoff | Horizon | Observations | Residual MSE | Correction | Mean target variance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in parameters.itertuples(index=False):
        lines.append(
            f"| {row.cutoff_id} | {row.horizon_minutes} | {row.observations:,} | "
            f"{row.residual_mse:.6f} | {row.lognormal_correction:.4f} | "
            f"{row.target_variance_mean:.10g} |"
        )

    lines.extend(
        [
            "",
            "## Platt parameters",
            "",
            "| Fold | Model cutoff | OOF blocks | Contracts | Rows | Slope | Intercept |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in calibrators.itertuples(index=False):
        lines.append(
            f"| {row.fold_id} | {row.model_parameter_cutoff} | "
            f"{row.calibration_blocks} | {row.calibration_contracts:,} | "
            f"{row.calibration_rows:,} | {row.slope:.6f} | {row.intercept:.6f} |"
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
        for model_key in MODEL_KEYS:
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
            "Fold-contract appearances overlap through time and are not independent.",
            "",
            "| Model | Fold-contract appearances | Brier | Log loss | ECE-10 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model_key in labels:
        row = indexed.loc[model_key]
        lines.append(
            f"| {labels[model_key]} | {int(row.fold_contract_appearances):,} | "
            f"{row.brier:.6f} | {row.log_loss:.6f} | {row.ece_10:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Frozen validation decision",
            "",
            f"- Within HAR, **{labels[family_winner]}** wins by the registered Brier-then-log-loss rule.",
            f"- HAR + Platt beats HAR raw on Brier in {brier_wins}/4 folds and log loss in {log_wins}/4 folds.",
            f"- Across all implemented probability models, **{labels[tournament_winner]}** has the lowest combined Brier.",
            f"- HAR family winner versus Empirical + Platt: Brier {indexed.loc[family_winner, 'brier']:.6f} versus {indexed.loc['empirical_platt', 'brier']:.6f}; log loss {indexed.loc[family_winner, 'log_loss']:.6f} versus {indexed.loc['empirical_platt', 'log_loss']:.6f}.",
            "- The one-minute target has many zero or near-zero close returns,",
            "  producing a very large lognormal correction. The frozen 0.1%/99.9%",
            "  training-target cap bounds the resulting forecasts; this is retained",
            "  as a documented limitation rather than retuned after validation.",
            "",
            "## HAR breakdown by asset",
            "",
            "| Model | Asset | Contracts | Brier | Log loss |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for model_key in MODEL_KEYS:
        for asset in ASSET_SYMBOLS:
            row = by_asset[(by_asset.model_key == model_key) & (by_asset.asset == asset)].iloc[0]
            lines.append(
                f"| {labels[model_key]} | {asset} | {int(row.contracts):,} | "
                f"{row.brier:.6f} | {row.log_loss:.6f} |"
            )
    lines.extend(
        [
            "",
            "## HAR breakdown by direction",
            "",
            "| Model | Direction | Contracts | Brier | Log loss |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for model_key in MODEL_KEYS:
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
            f"- Prediction partitions: {len(prediction_manifest):,}.",
            f"- Prediction rows: {int(prediction_manifest.rows.sum()):,}.",
            f"- Causal horizon/cutoff parameter fits: {len(parameters):,}.",
            "",
            "These are probability-validation results, not profitability results.",
            "No execution, P&L, evaluation, or portfolio metric was inspected.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    folds = pd.read_parquet(FOLD_FILE)
    calendar = unique_contract_calendar(folds)
    assets = {asset: prepare_asset_data(asset) for asset in ASSET_SYMBOLS}
    models, parameters = fit_all_models(assets)
    calibration_blocks = load_calibration_predictions(calendar, assets, models)
    calibrators = fit_fold_calibrators(calibration_blocks)
    del calibration_blocks
    metrics, reliability, manifest = generate_validation_outputs(
        folds, assets, models, calibrators
    )
    REPORT_FILE.write_text(
        make_report(parameters, calibrators, metrics, reliability, manifest),
        encoding="utf-8",
    )
    print(f"parameters: {PARAMETERS_FILE}")
    print(f"metrics: {METRICS_FILE}")
    print(f"prediction manifest: {PREDICTION_MANIFEST_FILE}")
    print(f"report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
