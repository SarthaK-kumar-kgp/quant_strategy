#!/usr/bin/env python3
"""Fit frozen causal drift parameters and training-only Platt calibrators.

This script deliberately does not generate or score outer-validation predictions.
It reuses the already-fitted Phase-3 HAR and DVOL variance parameters unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import ndtr


FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parent
PHASE3 = PROJECT / "phase3_models"
if str(PHASE3) not in sys.path:
    sys.path.insert(0, str(PHASE3))

from run_benchmarks import (  # noqa: E402
    MINUTES_PER_YEAR,
    PLATT_L2,
    PROBABILITY_CLIP,
    atomic_parquet,
    fit_platt,
)
from run_dvol_ablation import (  # noqa: E402
    DVOL_FEATURE_NAMES,
    augment_with_dvol,
    load_dvol,
    read_panel as read_dvol_panel,
    verify_panel_dvol,
)
from run_empirical import (  # noqa: E402
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
from run_har import (  # noqa: E402
    FEATURE_NAMES,
    features_for_panel,
    forecast_integrated_variance,
    prepare_asset_data,
)


ESTIMATOR_VERSION = "drift_daily365_huber_hac5_shrink_v1"
WINDOW_DAYS = 365
HUBER_C = 1.345
HUBER_TOLERANCE = 1e-12
HUBER_MAX_ITERATIONS = 100
HAC_LAGS = 5
ANNUALIZATION_DAYS = 365
DRIFT_LOWER = -1.0
DRIFT_UPPER = 1.0

PARAMETERS_FILE = FOLDER / "drift_parameters.parquet"
CALIBRATORS_FILE = FOLDER / "drift_calibrators.parquet"
REPORT_FILE = FOLDER / "drift_fit_report.md"

BRANCH_ASSETS = {
    "gbm_drift": tuple(ASSET_SYMBOLS),
    "har_drift": tuple(ASSET_SYMBOLS),
    "har_dvol_drift": ("BTC", "ETH"),
}

BASE_PANEL_COLUMNS = [
    "condition_id",
    "asset",
    "direction",
    "barrier",
    "decision_time",
    "spot_available_time",
    "spot_close",
    "log_distance_to_barrier",
    "minutes_to_expiry",
    "rv_24h",
    "y",
    "contract_row_weight",
]


def huber_location(values: np.ndarray) -> tuple[float, float, int, bool]:
    """Return fixed-scale Huber location, robust scale, iterations, fallback."""

    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or len(sample) == 0 or not np.isfinite(sample).all():
        raise ValueError("Huber input must be a non-empty finite vector")
    median = float(np.median(sample))
    scale = float(1.4826 * np.median(np.abs(sample - median)))
    scale_fallback = False
    if scale == 0:
        if np.all(sample == sample[0]):
            return float(np.mean(sample)), 0.0, 0, True
        scale = float(np.std(sample, ddof=0))
        scale_fallback = True
    if scale == 0:
        return float(np.mean(sample)), 0.0, 0, True

    location = median
    for iteration in range(1, HUBER_MAX_ITERATIONS + 1):
        standardized = (sample - location) / scale
        absolute = np.abs(standardized)
        weight = np.ones(len(sample), dtype=float)
        outside = absolute > HUBER_C
        weight[outside] = HUBER_C / absolute[outside]
        updated = float(np.dot(weight, sample) / weight.sum())
        if abs(updated - location) <= HUBER_TOLERANCE:
            return updated, scale, iteration, scale_fallback
        location = updated
    raise RuntimeError("Huber location failed to converge in 100 iterations")


def newey_west_standard_error(
    values: np.ndarray, location: float, scale: float
) -> tuple[float, float]:
    """Return daily mean SE and long-run variance for Huber-limited returns."""

    sample = np.asarray(values, dtype=float)
    if scale == 0:
        limited = sample.copy()
    else:
        standardized = (sample - location) / scale
        limited = location + scale * np.clip(standardized, -HUBER_C, HUBER_C)
    centred = limited - limited.mean()
    n = len(centred)
    gamma_zero = float(np.dot(centred, centred) / n)
    omega = gamma_zero
    for lag in range(1, HAC_LAGS + 1):
        gamma = float(np.dot(centred[lag:], centred[:-lag]) / n)
        omega += 2 * (1 - lag / (HAC_LAGS + 1)) * gamma
    omega = max(omega, 0.0)
    return float(np.sqrt(omega / n)), omega


def estimate_drift(returns: np.ndarray) -> dict[str, float | int | bool]:
    sample = np.asarray(returns, dtype=float)
    if len(sample) != WINDOW_DAYS:
        raise ValueError(f"drift estimator requires exactly {WINDOW_DAYS} returns")
    location, scale, iterations, scale_fallback = huber_location(sample)
    daily_se, long_run_variance = newey_west_standard_error(
        sample, location, scale
    )
    raw = ANNUALIZATION_DAYS * location
    annual_se = ANNUALIZATION_DAYS * daily_se
    if raw == 0 or abs(raw) <= annual_se:
        shrinkage = 0.0
    else:
        shrinkage = max(0.0, 1 - annual_se**2 / raw**2)
    shrunk = 0.0 if shrinkage == 0 else shrinkage * raw
    bounded = float(np.clip(shrunk, DRIFT_LOWER, DRIFT_UPPER))
    return {
        "huber_location_daily": location,
        "robust_scale_daily": scale,
        "huber_iterations": iterations,
        "scale_fallback": scale_fallback,
        "newey_west_lags": HAC_LAGS,
        "newey_west_long_run_variance": long_run_variance,
        "raw_annual_drift": raw,
        "newey_west_annual_se": annual_se,
        "shrinkage_factor": shrinkage,
        "shrunk_annual_drift": shrunk,
        "final_annual_drift": bounded,
        "cap_activated": not np.isclose(bounded, shrunk, rtol=0, atol=1e-15),
    }


def completed_daily_returns(
    spot: pd.DataFrame, cutoff_epoch: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return values, return-day IDs, and availability seconds before cutoff."""

    ordered = spot.sort_values("ts", ignore_index=True)
    timestamp = ordered["ts"].to_numpy(dtype=np.int64)
    close = ordered["close"].to_numpy(dtype=float)
    if len(timestamp) == 0 or not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("invalid spot close series")
    day = timestamp // 86400
    frame = pd.DataFrame({"day": day, "ts": timestamp, "close": close})
    daily = frame.groupby("day", sort=True, as_index=False).tail(1).reset_index(drop=True)
    daily_day = daily["day"].to_numpy(dtype=np.int64)
    daily_ts = daily["ts"].to_numpy(dtype=np.int64)
    completed = (daily_ts % 86400) == 86340
    daily = daily.loc[completed].reset_index(drop=True)
    daily_day = daily["day"].to_numpy(dtype=np.int64)
    daily_ts = daily["ts"].to_numpy(dtype=np.int64)
    if len(daily) < 2:
        return (
            np.array([], dtype=float),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )
    availability = daily_ts + 60
    daily_close = daily["close"].to_numpy(dtype=float)
    values = daily_close[1:] / daily_close[:-1] - 1
    return_day = daily_day[1:]
    return_availability = availability[1:]
    eligible = return_availability <= cutoff_epoch
    return values[eligible], return_day[eligible], return_availability[eligible]


def fit_drift_parameters() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for asset, symbol in ASSET_SYMBOLS.items():
        spot = pd.read_parquet(
            SPOT_DIR / f"{symbol}_1m.parquet", columns=["ts", "close"]
        )
        for cutoff_id, cutoff_text_value in CUTOFF_TEXT.items():
            cutoff = cutoff_timestamp(cutoff_text_value)
            cutoff_epoch = int(cutoff.timestamp())
            values, days, availability = completed_daily_returns(spot, cutoff_epoch)
            fallback = len(values) < WINDOW_DAYS
            consecutive = False
            if fallback:
                selected = np.array([], dtype=float)
                selected_days = np.array([], dtype=np.int64)
                selected_availability = np.array([], dtype=np.int64)
                estimate = {
                    "huber_location_daily": 0.0,
                    "robust_scale_daily": 0.0,
                    "huber_iterations": 0,
                    "scale_fallback": True,
                    "newey_west_lags": HAC_LAGS,
                    "newey_west_long_run_variance": 0.0,
                    "raw_annual_drift": 0.0,
                    "newey_west_annual_se": 0.0,
                    "shrinkage_factor": 0.0,
                    "shrunk_annual_drift": 0.0,
                    "final_annual_drift": 0.0,
                    "cap_activated": False,
                }
            else:
                selected = values[-WINDOW_DAYS:]
                selected_days = days[-WINDOW_DAYS:]
                selected_availability = availability[-WINDOW_DAYS:]
                consecutive = bool(np.all(np.diff(selected_days) == 1))
                if not consecutive:
                    fallback = True
                    estimate = {
                        "huber_location_daily": 0.0,
                        "robust_scale_daily": 0.0,
                        "huber_iterations": 0,
                        "scale_fallback": True,
                        "newey_west_lags": HAC_LAGS,
                        "newey_west_long_run_variance": 0.0,
                        "raw_annual_drift": 0.0,
                        "newey_west_annual_se": 0.0,
                        "shrinkage_factor": 0.0,
                        "shrunk_annual_drift": 0.0,
                        "final_annual_drift": 0.0,
                        "cap_activated": False,
                    }
                else:
                    estimate = estimate_drift(selected)
            first_day = (
                pd.Timestamp(int(selected_days[0]) * 86400, unit="s", tz="UTC").date()
                if len(selected_days)
                else pd.NaT
            )
            last_day = (
                pd.Timestamp(int(selected_days[-1]) * 86400, unit="s", tz="UTC").date()
                if len(selected_days)
                else pd.NaT
            )
            latest_available = (
                pd.Timestamp(int(selected_availability[-1]), unit="s", tz="UTC")
                if len(selected_availability)
                else pd.NaT
            )
            if len(selected_availability) and selected_availability[-1] > cutoff_epoch:
                raise ValueError("drift source availability exceeds cutoff")
            rows.append(
                {
                    "estimator_version": ESTIMATOR_VERSION,
                    "asset": asset,
                    "cutoff_id": cutoff_id,
                    "cutoff_time": cutoff,
                    "first_return_date_utc": first_day,
                    "last_return_date_utc": last_day,
                    "observations": len(selected),
                    "consecutive_window": consecutive,
                    "fallback_to_zero": fallback,
                    "latest_source_availability": latest_available,
                    **estimate,
                }
            )
    parameters = pd.DataFrame(rows).sort_values(
        ["cutoff_time", "asset"], ignore_index=True
    )
    atomic_parquet(parameters, PARAMETERS_FILE)
    return parameters


def drift_touch_probability_from_variance(
    distance: np.ndarray,
    integrated_variance: np.ndarray,
    minutes_to_expiry: np.ndarray,
    direction: np.ndarray,
    annual_price_drift: float,
) -> np.ndarray:
    """First-passage probability using constant-effective integrated variance."""

    distance = np.asarray(distance, dtype=float)
    variance = np.asarray(integrated_variance, dtype=float)
    minutes = np.asarray(minutes_to_expiry, dtype=float)
    direction = np.asarray(direction)
    valid = (
        np.isfinite(distance)
        & np.isfinite(variance)
        & np.isfinite(minutes)
        & (variance > 0)
        & (minutes > 0)
        & np.isin(direction, ["up", "down"])
    )
    probability = np.full(len(distance), np.nan, dtype=float)
    if not valid.any():
        return probability
    d = distance[valid]
    total_variance = variance[valid]
    horizon = minutes[valid] / MINUTES_PER_YEAR
    root_variance = np.sqrt(total_variance)
    log_mean = annual_price_drift * horizon - 0.5 * total_variance
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


def gbm_touch_probability_drift(
    spot: np.ndarray,
    barrier: np.ndarray,
    minutes_to_expiry: np.ndarray,
    annualized_sigma: np.ndarray,
    direction: np.ndarray,
    annual_price_drift: float,
) -> np.ndarray:
    spot = np.asarray(spot, dtype=float)
    barrier = np.asarray(barrier, dtype=float)
    minutes = np.asarray(minutes_to_expiry, dtype=float)
    sigma = np.asarray(annualized_sigma, dtype=float)
    direction = np.asarray(direction)
    valid = (
        np.isfinite(spot)
        & np.isfinite(barrier)
        & np.isfinite(minutes)
        & np.isfinite(sigma)
        & (spot > 0)
        & (barrier > 0)
        & (minutes > 0)
        & (sigma > 0)
        & np.isin(direction, ["up", "down"])
    )
    probability = np.full(len(spot), np.nan, dtype=float)
    if not valid.any():
        return probability
    log_barrier = np.log(barrier[valid] / spot[valid])
    distance = np.abs(log_barrier)
    horizon = minutes[valid] / MINUTES_PER_YEAR
    integrated_variance = sigma[valid] ** 2 * horizon
    selected = drift_touch_probability_from_variance(
        distance,
        integrated_variance,
        minutes[valid],
        direction[valid],
        annual_price_drift,
    )
    already_touched = (
        ((direction[valid] == "up") & (spot[valid] >= barrier[valid]))
        | ((direction[valid] == "down") & (spot[valid] <= barrier[valid]))
    )
    probability[valid] = np.where(already_touched, 1.0, selected)
    return probability


def models_from_parameters(
    parameters: pd.DataFrame, feature_names: tuple[str, ...]
) -> dict[str, dict[int, dict[str, object]]]:
    models: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for row in parameters.itertuples(index=False):
        model = {
            "feature_mean": np.array(
                [getattr(row, f"mean_{name}") for name in feature_names]
            ),
            "feature_scale": np.array(
                [getattr(row, f"scale_{name}") for name in feature_names]
            ),
            "coefficient": np.array(
                [getattr(row, f"coefficient_{name}") for name in feature_names]
            ),
            "intercept": row.intercept,
            "lognormal_correction": row.lognormal_correction,
            "target_variance_lower": row.target_variance_lower,
            "target_variance_upper": row.target_variance_upper,
        }
        models[row.cutoff_id][int(row.horizon_minutes)] = model
    expected = set(map(int, HORIZONS))
    for cutoff_id in CUTOFF_TEXT:
        if set(models[cutoff_id]) != expected:
            raise ValueError(f"incomplete frozen variance parameters for {cutoff_id}")
    return dict(models)


def drift_map(parameters: pd.DataFrame) -> dict[tuple[str, str], float]:
    if parameters["fallback_to_zero"].any():
        print("warning: at least one drift cutoff used the declared zero fallback", flush=True)
    return {
        (row.cutoff_id, row.asset): float(row.final_annual_drift)
        for row in parameters.itertuples(index=False)
    }


def predict_har_drift(
    panel: pd.DataFrame,
    asset_data,
    models: dict[int, dict[str, object]],
    annual_drift: float,
    verify_dvol_features: bool = False,
) -> np.ndarray:
    features = features_for_panel(panel, asset_data)
    if verify_dvol_features:
        verify_panel_dvol(panel, features)
    variance = forecast_integrated_variance(
        features,
        panel["minutes_to_expiry"].to_numpy(dtype=float),
        models,
    )
    return drift_touch_probability_from_variance(
        panel["log_distance_to_barrier"].to_numpy(dtype=float),
        variance,
        panel["minutes_to_expiry"].to_numpy(dtype=float),
        panel["direction"].to_numpy(),
        annual_drift,
    )


def fit_branch_calibrators(
    branch: str, blocks: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    rows = []
    for fold_id, block_ids in FOLD_CALIBRATION_BLOCKS.items():
        probability = np.concatenate([blocks[item]["probability"] for item in block_ids])
        outcome = np.concatenate([blocks[item]["outcome"] for item in block_ids])
        weight = np.concatenate([blocks[item]["weight"] for item in block_ids])
        fitted = fit_platt(probability, outcome, weight)
        rows.append(
            {
                "branch": branch,
                "fold_id": fold_id,
                "model_parameter_cutoff": FOLD_DISTRIBUTION_CUTOFF[fold_id],
                "estimator_version": ESTIMATOR_VERSION,
                "calibration_version": f"{fold_id}_{branch}_platt_v1",
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
            f"fitted {branch} {fold_id} Platt: "
            f"slope={fitted['slope']:.6f} intercept={fitted['intercept']:.6f}",
            flush=True,
        )
    return rows


def load_base_calibration_blocks(
    calendar: pd.DataFrame,
    assets: dict[str, object],
    har_models: dict[str, dict[int, dict[str, object]]],
    drifts: dict[tuple[str, str], float],
) -> dict[str, dict[str, dict[str, object]]]:
    result = {"gbm_drift": {}, "har_drift": {}}
    for block_id, specification in CALIBRATION_BLOCKS.items():
        members = calendar[
            (calendar["resolution_date_et"] >= specification["start"])
            & (calendar["resolution_date_et"] <= specification["end"])
        ]
        accumulators = {
            branch: {"probability": [], "outcome": [], "weight": []}
            for branch in result
        }
        cutoff_id = specification["cutoff_id"]
        for member in members.itertuples(index=False):
            panel = pd.read_parquet(
                PANEL_DIR / f"{member.condition_id}.parquet",
                columns=BASE_PANEL_COLUMNS,
            )
            annual_drift = drifts[(cutoff_id, member.asset)]
            probabilities = {
                "gbm_drift": gbm_touch_probability_drift(
                    panel["spot_close"].to_numpy(dtype=float),
                    panel["barrier"].to_numpy(dtype=float),
                    panel["minutes_to_expiry"].to_numpy(dtype=float),
                    panel["rv_24h"].to_numpy(dtype=float),
                    panel["direction"].to_numpy(),
                    annual_drift,
                ),
                "har_drift": predict_har_drift(
                    panel,
                    assets[member.asset],
                    har_models[cutoff_id],
                    annual_drift,
                ),
            }
            for branch, probability in probabilities.items():
                if not np.isfinite(probability).all():
                    raise ValueError(f"invalid {branch} OOF probability")
                accumulators[branch]["probability"].append(probability)
                accumulators[branch]["outcome"].append(panel["y"].to_numpy(float))
                accumulators[branch]["weight"].append(
                    panel["contract_row_weight"].to_numpy(float)
                )
        for branch in result:
            result[branch][block_id] = {
                key: np.concatenate(parts)
                for key, parts in accumulators[branch].items()
            }
            result[branch][block_id].update(
                {"contracts": len(members), "cutoff_id": cutoff_id}
            )
        print(
            f"generated GBM/HAR drift OOF {block_id}: {len(members):,} contracts",
            flush=True,
        )
    return result


def load_dvol_calibration_blocks(
    calendar: pd.DataFrame,
    assets: dict[str, object],
    models: dict[str, dict[int, dict[str, object]]],
    drifts: dict[tuple[str, str], float],
) -> dict[str, dict[str, object]]:
    blocks = {}
    for block_id, specification in CALIBRATION_BLOCKS.items():
        members = calendar[
            calendar["asset"].isin(BRANCH_ASSETS["har_dvol_drift"])
            & (calendar["resolution_date_et"] >= specification["start"])
            & (calendar["resolution_date_et"] <= specification["end"])
        ]
        probability_parts = []
        outcome_parts = []
        weight_parts = []
        cutoff_id = specification["cutoff_id"]
        for member in members.itertuples(index=False):
            panel = read_dvol_panel(member.condition_id)
            probability = predict_har_drift(
                panel,
                assets[member.asset],
                models[cutoff_id],
                drifts[(cutoff_id, member.asset)],
                verify_dvol_features=True,
            )
            if not np.isfinite(probability).all():
                raise ValueError("invalid HAR DVOL drift OOF probability")
            probability_parts.append(probability)
            outcome_parts.append(panel["y"].to_numpy(float))
            weight_parts.append(panel["contract_row_weight"].to_numpy(float))
        blocks[block_id] = {
            "probability": np.concatenate(probability_parts),
            "outcome": np.concatenate(outcome_parts),
            "weight": np.concatenate(weight_parts),
            "contracts": len(members),
            "cutoff_id": cutoff_id,
        }
        print(
            f"generated HAR DVOL drift OOF {block_id}: {len(members):,} contracts",
            flush=True,
        )
    return blocks


def fit_calibrators(parameters: pd.DataFrame) -> pd.DataFrame:
    folds = pd.read_parquet(FOLD_FILE)
    calendar = unique_contract_calendar(folds)
    drifts = drift_map(parameters)

    assets = {asset: prepare_asset_data(asset) for asset in ASSET_SYMBOLS}
    har_parameter_frame = pd.read_parquet(PHASE3 / "har_parameters.parquet")
    har_models = models_from_parameters(har_parameter_frame, FEATURE_NAMES)
    base_blocks = load_base_calibration_blocks(calendar, assets, har_models, drifts)
    calibrator_rows = []
    for branch in ("gbm_drift", "har_drift"):
        calibrator_rows.extend(fit_branch_calibrators(branch, base_blocks[branch]))
    del base_blocks

    dvol_assets = {}
    for asset in BRANCH_ASSETS["har_dvol_drift"]:
        source_time, sigma = load_dvol(asset)
        dvol_assets[asset], _ = augment_with_dvol(assets[asset], source_time, sigma)
    dvol_parameter_frame = pd.read_parquet(PHASE3 / "dvol_parameters.parquet")
    dvol_parameter_frame = dvol_parameter_frame[
        dvol_parameter_frame["variant"] == "btceth_dvol"
    ]
    dvol_models = models_from_parameters(dvol_parameter_frame, DVOL_FEATURE_NAMES)
    dvol_blocks = load_dvol_calibration_blocks(
        calendar, dvol_assets, dvol_models, drifts
    )
    calibrator_rows.extend(
        fit_branch_calibrators("har_dvol_drift", dvol_blocks)
    )
    calibrators = pd.DataFrame(calibrator_rows).sort_values(
        ["branch", "fold_id"], ignore_index=True
    )
    atomic_parquet(calibrators, CALIBRATORS_FILE)
    return calibrators


def write_report(parameters: pd.DataFrame, calibrators: pd.DataFrame) -> None:
    lines = [
        "# Phase 3.5 drift fitting report",
        "",
        "**Status:** Step 2 complete. The six matched trials were registered, the "
        "frozen drift estimator was fitted, and each Platt variant received its own "
        "training-only OOF map. Outer-validation predictions and scores have not "
        "been generated; evaluation and execution-proxy data remain closed.",
        "",
        "## Drift estimates",
        "",
        "Annual price drift after robust estimation, HAC uncertainty shrinkage, and "
        "the frozen [-1, 1] cap:",
        "",
        "| Cutoff | Asset | Raw drift | Annual SE | Shrinkage | Final drift | Fallback | Cap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in parameters.itertuples(index=False):
        lines.append(
            f"| {row.cutoff_id.removeprefix('cutoff_')} | {row.asset} | "
            f"{row.raw_annual_drift:.4f} | {row.newey_west_annual_se:.4f} | "
            f"{row.shrinkage_factor:.4f} | {row.final_annual_drift:.4f} | "
            f"{row.fallback_to_zero} | {row.cap_activated} |"
        )
    lines.extend(
        [
            "",
            "## Fitted training-only Platt maps",
            "",
            "| Branch | Fold | OOF contracts | OOF rows | Slope | Intercept |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in calibrators.itertuples(index=False):
        lines.append(
            f"| {row.branch} | {row.fold_id} | {row.calibration_contracts:,} | "
            f"{row.calibration_rows:,} | {row.slope:.6f} | {row.intercept:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Audit",
            "",
            f"- Estimator: `{ESTIMATOR_VERSION}`.",
            "- 20 asset/cutoff parameter rows were expected and produced.",
            "- The existing Phase-3 HAR and HAR+DVOL variance parameters were read "
            "unchanged rather than refitted.",
            "- Only the predeclared April-July OOF calibration blocks were opened "
            "for Platt fitting.",
            "- No outer-validation/evaluation prediction or metric, P&L metric, "
            "or execution-proxy row was read or produced.",
            "",
            "## Next action",
            "",
            "Generate the six matched outer-validation prediction sets, validate "
            "their schemas and causal timestamps, then compare each drift variant "
            "only with its corresponding zero-drift control.",
            "",
        ]
    )
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parameters = fit_drift_parameters()
    if len(parameters) != len(ASSET_SYMBOLS) * len(CUTOFF_TEXT):
        raise ValueError("unexpected drift parameter row count")
    calibrators = fit_calibrators(parameters)
    if len(calibrators) != len(BRANCH_ASSETS) * len(FOLD_CALIBRATION_BLOCKS):
        raise ValueError("unexpected drift calibrator row count")
    write_report(parameters, calibrators)
    print(f"wrote {PARAMETERS_FILE}", flush=True)
    print(f"wrote {CALIBRATORS_FILE}", flush=True)
    print(f"wrote {REPORT_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
