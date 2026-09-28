#!/usr/bin/env python3
"""Run the frozen validation-only empirical excursion model and Platt variant."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from run_benchmarks import (
    LOG_LOSS_CLIP,
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


FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parent.parent
PHASE2 = FOLDER.parent / "phase2_dataset"
PANEL_DIR = PHASE2 / "causal_panel_1m"
FOLD_FILE = PHASE2 / "walk_forward_folds.parquet"
SPOT_DIR = PROJECT / "data" / "spot"
OUTPUT_DIR = FOLDER / "empirical_predictions"
DISTRIBUTION_DIR = FOLDER / "empirical_distributions"
PARAMETERS_FILE = FOLDER / "empirical_parameters.parquet"
DISTRIBUTION_MANIFEST_FILE = FOLDER / "empirical_distribution_manifest.parquet"
METRICS_FILE = FOLDER / "empirical_contract_metrics.parquet"
RELIABILITY_FILE = FOLDER / "empirical_reliability.parquet"
PREDICTION_MANIFEST_FILE = FOLDER / "empirical_prediction_manifest.parquet"
REPORT_FILE = FOLDER / "empirical_report.md"

ASSET_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}
HORIZONS = np.array([1, 5, 15, 30, 60, 120, 240, 480, 720, 1440], dtype=int)
REFERENCE_STRIDE_MINUTES = 60
TAIL_PSEUDOCOUNT = 0.5

EXPERIMENTS = {
    "empirical_raw": {
        "experiment_id": "E04-EMP-RAW-V1",
        "calibration_method": "raw",
    },
    "empirical_platt": {
        "experiment_id": "E04-EMP-PLATT-V1",
        "calibration_method": "platt",
    },
}

CUTOFF_TEXT = {
    "cutoff_2026_04_01": "2026-04-01",
    "cutoff_2026_05_01": "2026-05-01",
    "cutoff_2026_06_01": "2026-06-01",
    "cutoff_2026_07_01": "2026-07-01",
    "cutoff_2026_08_01": "2026-08-01",
}

CALIBRATION_BLOCKS = {
    "cal_2026_04": {
        "start": date(2026, 4, 2),
        "end": date(2026, 4, 30),
        "cutoff_id": "cutoff_2026_04_01",
    },
    "cal_2026_05": {
        "start": date(2026, 5, 2),
        "end": date(2026, 5, 31),
        "cutoff_id": "cutoff_2026_05_01",
    },
    "cal_2026_06": {
        "start": date(2026, 6, 2),
        "end": date(2026, 6, 30),
        "cutoff_id": "cutoff_2026_06_01",
    },
    "cal_2026_07": {
        "start": date(2026, 7, 2),
        "end": date(2026, 7, 31),
        "cutoff_id": "cutoff_2026_07_01",
    },
}

FOLD_CALIBRATION_BLOCKS = {
    "fold_01": ["cal_2026_04"],
    "fold_02": ["cal_2026_04", "cal_2026_05"],
    "fold_03": ["cal_2026_04", "cal_2026_05", "cal_2026_06"],
    "fold_04": [
        "cal_2026_04",
        "cal_2026_05",
        "cal_2026_06",
        "cal_2026_07",
    ],
}

FOLD_DISTRIBUTION_CUTOFF = {
    "fold_01": "cutoff_2026_05_01",
    "fold_02": "cutoff_2026_06_01",
    "fold_03": "cutoff_2026_07_01",
    "fold_04": "cutoff_2026_08_01",
}

PANEL_COLUMNS = [
    "condition_id",
    "asset",
    "direction",
    "decision_time",
    "spot_available_time",
    "log_distance_to_barrier",
    "minutes_to_expiry",
    "rv_24h",
    "y",
    "contract_row_weight",
]


def cutoff_timestamp(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="America/New_York").tz_convert("UTC")


def read_panel(condition_id: str) -> pd.DataFrame:
    return pd.read_parquet(
        PANEL_DIR / f"{condition_id}.parquet", columns=PANEL_COLUMNS
    )


def unique_contract_calendar(folds: pd.DataFrame) -> pd.DataFrame:
    columns = ["condition_id", "asset", "direction", "resolution_date_et"]
    calendar = folds[columns].drop_duplicates().copy()
    if (calendar.groupby("condition_id").size() != 1).any():
        raise ValueError("contract metadata changes between outer folds")
    return calendar


def future_extreme(values: np.ndarray, horizon: int, kind: str) -> np.ndarray:
    series = pd.Series(values).shift(-1)
    rolling = series.rolling(horizon, min_periods=horizon)
    result = rolling.max() if kind == "max" else rolling.min()
    return result.shift(-(horizon - 1)).to_numpy()


def build_reference_catalog() -> dict[int, dict[str, np.ndarray]]:
    pooled = {
        int(horizon): {"end": [], "up": [], "down": [], "asset": []}
        for horizon in HORIZONS
    }
    maximum_cutoff = max(cutoff_timestamp(text) for text in CUTOFF_TEXT.values())
    maximum_cutoff_epoch = int(maximum_cutoff.timestamp())

    for asset, symbol in ASSET_SYMBOLS.items():
        spot = pd.read_parquet(
            SPOT_DIR / f"{symbol}_1m.parquet",
            columns=["ts", "high", "low", "close"],
        ).sort_values("ts", ignore_index=True)
        timestamp = spot["ts"].to_numpy(dtype=np.int64)
        close = spot["close"].to_numpy(dtype=float)
        high = spot["high"].to_numpy(dtype=float)
        low = spot["low"].to_numpy(dtype=float)
        returns = pd.Series(np.log(close)).diff()
        rv_24h = np.sqrt(
            returns.pow(2).rolling(1440, min_periods=1440).mean()
            * MINUTES_PER_YEAR
        ).to_numpy()
        hourly_origin = (timestamp % (REFERENCE_STRIDE_MINUTES * 60)) == 0

        for horizon in HORIZONS:
            horizon = int(horizon)
            future_high = future_extreme(high, horizon, "max")
            future_low = future_extreme(low, horizon, "min")
            end_index = np.arange(len(timestamp)) + horizon
            valid_end = end_index < len(timestamp)
            end_available = np.full(len(timestamp), np.iinfo(np.int64).max, dtype=np.int64)
            end_available[valid_end] = timestamp[end_index[valid_end]] + 60
            scale = rv_24h * np.sqrt(horizon / MINUTES_PER_YEAR)
            valid = (
                hourly_origin
                & valid_end
                & (end_available <= maximum_cutoff_epoch)
                & np.isfinite(scale)
                & (scale > 0)
                & np.isfinite(future_high)
                & np.isfinite(future_low)
            )
            up = np.log(future_high[valid] / close[valid]) / scale[valid]
            down = np.log(close[valid] / future_low[valid]) / scale[valid]
            # A future window can remain entirely below (or above) its starting
            # close, so one directional maximum excursion may be negative. Those
            # are valid non-exceedances for a positive barrier distance and must
            # remain in the empirical distribution.
            finite = np.isfinite(up) & np.isfinite(down)
            pooled[horizon]["end"].append(end_available[valid][finite])
            pooled[horizon]["up"].append(up[finite])
            pooled[horizon]["down"].append(down[finite])
            pooled[horizon]["asset"].append(
                np.full(int(finite.sum()), asset, dtype="U3")
            )
        print(f"prepared empirical reference paths: {asset}", flush=True)

    catalog = {}
    for horizon in HORIZONS:
        horizon = int(horizon)
        catalog[horizon] = {
            key: np.concatenate(parts) for key, parts in pooled[horizon].items()
        }
    return catalog


def build_distributions(
    catalog: dict[int, dict[str, np.ndarray]]
) -> tuple[dict[str, dict[tuple[int, str], np.ndarray]], pd.DataFrame]:
    distributions = {}
    manifest_rows = []
    DISTRIBUTION_DIR.mkdir(parents=True, exist_ok=True)
    for cutoff_id, cutoff_text_value in CUTOFF_TEXT.items():
        cutoff_manifest_start = len(manifest_rows)
        cutoff = cutoff_timestamp(cutoff_text_value)
        cutoff_epoch = int(cutoff.timestamp())
        fitted = {}
        save_values = {
            "cutoff_epoch": np.array([cutoff_epoch], dtype=np.int64),
            "horizons": HORIZONS,
        }
        for horizon in HORIZONS:
            horizon = int(horizon)
            eligible = catalog[horizon]["end"] <= cutoff_epoch
            if not eligible.any():
                raise ValueError(f"no empirical references for {cutoff_id} {horizon}")
            for direction in ["up", "down"]:
                values = np.sort(catalog[horizon][direction][eligible])
                fitted[(horizon, direction)] = values
                save_values[f"h{horizon}_{direction}"] = values
                asset_values = catalog[horizon]["asset"][eligible]
                counts = {
                    asset: int((asset_values == asset).sum()) for asset in ASSET_SYMBOLS
                }
                manifest_rows.append(
                    {
                        "cutoff_id": cutoff_id,
                        "cutoff_time": cutoff,
                        "horizon_minutes": horizon,
                        "direction": direction,
                        "reference_count": len(values),
                        "latest_reference_end": pd.to_datetime(
                            int(catalog[horizon]["end"][eligible].max()),
                            unit="s",
                            utc=True,
                        ),
                        "btc_count": counts["BTC"],
                        "eth_count": counts["ETH"],
                        "sol_count": counts["SOL"],
                        "xrp_count": counts["XRP"],
                        "minimum": float(values[0]),
                        "median": float(np.median(values)),
                        "p95": float(np.quantile(values, 0.95)),
                        "p99": float(np.quantile(values, 0.99)),
                        "maximum": float(values[-1]),
                    }
                )
        path = DISTRIBUTION_DIR / f"{cutoff_id}.npz"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **save_values)
        temporary.replace(path)
        checksum = file_sha256(path)
        for row in manifest_rows[cutoff_manifest_start:]:
            row["distribution_path"] = str(path.relative_to(FOLDER))
            row["distribution_sha256"] = checksum
        distributions[cutoff_id] = fitted
        print(f"froze empirical distribution: {cutoff_id}", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    atomic_parquet(manifest, DISTRIBUTION_MANIFEST_FILE)
    return distributions, manifest


def smoothed_survival(sorted_values: np.ndarray, standardized_distance: np.ndarray) -> np.ndarray:
    exceed = len(sorted_values) - np.searchsorted(
        sorted_values, standardized_distance, side="left"
    )
    return (exceed + TAIL_PSEUDOCOUNT) / (len(sorted_values) + 2 * TAIL_PSEUDOCOUNT)


def empirical_touch_probability(
    distance: np.ndarray,
    annualized_sigma: np.ndarray,
    minutes_to_expiry: np.ndarray,
    direction: str,
    distribution: dict[tuple[int, str], np.ndarray],
) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    sigma = np.asarray(annualized_sigma, dtype=float)
    horizon = np.asarray(minutes_to_expiry, dtype=float)
    valid = (
        np.isfinite(distance)
        & np.isfinite(sigma)
        & np.isfinite(horizon)
        & (sigma > 0)
        & (horizon > 0)
    )
    probability = np.full(len(distance), np.nan, dtype=float)
    if not valid.any():
        return probability

    actual_horizon = horizon[valid]
    standardized_distance = distance[valid] / (
        sigma[valid] * np.sqrt(actual_horizon / MINUTES_PER_YEAR)
    )
    clipped_horizon = np.clip(actual_horizon, HORIZONS[0], HORIZONS[-1])
    upper_index = np.searchsorted(HORIZONS, clipped_horizon, side="left")
    upper_index = np.minimum(upper_index, len(HORIZONS) - 1)
    lower_index = np.maximum(upper_index - 1, 0)
    exact = HORIZONS[upper_index] == clipped_horizon
    lower_index[exact] = upper_index[exact]
    selected = np.empty(valid.sum(), dtype=float)

    for low_index, high_index in set(zip(lower_index, upper_index)):
        rows = (lower_index == low_index) & (upper_index == high_index)
        low_horizon = int(HORIZONS[low_index])
        high_horizon = int(HORIZONS[high_index])
        low_probability = smoothed_survival(
            distribution[(low_horizon, direction)], standardized_distance[rows]
        )
        if low_horizon == high_horizon:
            selected[rows] = low_probability
            continue
        high_probability = smoothed_survival(
            distribution[(high_horizon, direction)], standardized_distance[rows]
        )
        interpolation = (
            np.log(clipped_horizon[rows]) - np.log(low_horizon)
        ) / (np.log(high_horizon) - np.log(low_horizon))
        selected[rows] = (
            (1 - interpolation) * low_probability + interpolation * high_probability
        )

    selected = np.where(distance[valid] <= 0, 1.0, selected)
    probability[valid] = np.clip(selected, 0.0, 1.0)
    return probability


def predict_panel(
    panel: pd.DataFrame, distribution: dict[tuple[int, str], np.ndarray]
) -> np.ndarray:
    directions = panel["direction"].unique()
    if len(directions) != 1 or directions[0] not in {"up", "down"}:
        raise ValueError("panel must contain exactly one valid direction")
    return empirical_touch_probability(
        panel["log_distance_to_barrier"].to_numpy(),
        panel["rv_24h"].to_numpy(),
        panel["minutes_to_expiry"].to_numpy(),
        directions[0],
        distribution,
    )


def load_calibration_predictions(
    calendar: pd.DataFrame,
    distributions: dict[str, dict[tuple[int, str], np.ndarray]],
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
        distribution = distributions[specification["cutoff_id"]]
        for condition_id in members["condition_id"]:
            panel = read_panel(condition_id)
            probability = predict_panel(panel, distribution)
            if not np.isfinite(probability).all():
                raise ValueError(f"invalid empirical probability in {condition_id}")
            probability_parts.append(probability)
            outcome_parts.append(panel["y"].to_numpy(dtype=float))
            weight_parts.append(panel["contract_row_weight"].to_numpy(dtype=float))
        blocks[block_id] = {
            "probability": np.concatenate(probability_parts),
            "outcome": np.concatenate(outcome_parts),
            "weight": np.concatenate(weight_parts),
            "contracts": len(members),
            "cutoff_id": specification["cutoff_id"],
        }
        print(
            f"generated empirical OOF {block_id}: {len(members):,} contracts, "
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
                "calibration_version": f"{fold_id}_empirical_platt_v1",
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
                "asset_pooling": "BTC|ETH|SOL|XRP",
                "direction_pooling": "separate",
                "time_interpolation": "linear_probability_in_log_minutes",
                "platt_probability_clip": PROBABILITY_CLIP,
                "platt_l2_slope": PLATT_L2,
                **fitted,
            }
        )
        print(
            f"fitted empirical {fold_id} Platt: slope={fitted['slope']:.6f}, "
            f"intercept={fitted['intercept']:.6f}",
            flush=True,
        )
    parameters = pd.DataFrame(rows)
    atomic_parquet(parameters, PARAMETERS_FILE)
    return parameters


def prediction_frame(
    base: pd.DataFrame,
    model_key: str,
    fold_id: str,
    probability_column: str,
) -> pd.DataFrame:
    experiment = EXPERIMENTS[model_key]
    raw_column = (
        "empirical_probability" if model_key == "empirical_platt" else probability_column
    )
    calibration_version = (
        f"{fold_id}_empirical_platt_v1" if model_key == "empirical_platt" else "none"
    )
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
            "model_id": "empirical_pooled_excursion",
            "model_version": "v1",
            "feature_set_id": "empirical_spot_barrier_rv24h_v1",
            "calibration_method": experiment["calibration_method"],
            "calibration_version": calibration_version,
            "raw_yes_probability": base[raw_column].to_numpy(dtype=float),
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
    distributions: dict[str, dict[tuple[int, str], np.ndarray]],
    parameters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parameter_map = parameters.set_index("fold_id").to_dict("index")
    metrics = []
    manifests = []
    reliability = defaultdict(lambda: np.zeros(4, dtype=float))

    for fold_id in FOLD_CALIBRATION_BLOCKS:
        fold_validation = folds[
            (folds["fold_id"] == fold_id) & (folds["role"] == "validation")
        ]
        fitted = parameter_map[fold_id]
        distribution = distributions[FOLD_DISTRIBUTION_CUTOFF[fold_id]]
        for asset in ASSET_SYMBOLS:
            members = fold_validation[fold_validation["asset"] == asset]
            contract_frames = []
            for member in members.itertuples(index=False):
                panel = read_panel(member.condition_id)
                empirical = predict_panel(panel, distribution)
                if not np.isfinite(empirical).all():
                    raise ValueError(f"invalid empirical probability in {member.condition_id}")
                clipped = np.clip(empirical, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
                raw_logit = np.log(clipped) - np.log1p(-clipped)
                platt = expit(fitted["slope"] * raw_logit + fitted["intercept"])
                panel = panel.copy()
                panel["empirical_probability"] = empirical
                panel["empirical_platt_probability"] = platt
                contract_frames.append(panel)

                outcomes = panel["y"].to_numpy(dtype=float)
                weights = panel["contract_row_weight"].to_numpy(dtype=float)
                probability_map = {
                    "empirical_raw": empirical,
                    "empirical_platt": platt,
                }
                for model_key, probability in probability_map.items():
                    brier, log_loss, mean_probability = contract_metric(
                        probability, outcomes, weights
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
                            "outcome": int(outcomes[0]),
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
                        outcomes,
                        weights,
                    )

            base = pd.concat(contract_frames, ignore_index=True)
            for model_key, probability_column in [
                ("empirical_raw", "empirical_probability"),
                ("empirical_platt", "empirical_platt_probability"),
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
                        "contracts": output["condition_id"].nunique(),
                        "first_decision": output["decision_time"].min(),
                        "last_decision": output["decision_time"].max(),
                        "bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                )
                del output
            print(
                f"wrote empirical {fold_id} validation {asset}: "
                f"{len(base):,} rows, {len(members):,} contracts",
                flush=True,
            )
            del base, contract_frames

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
    distribution_manifest: pd.DataFrame,
    empirical_metrics: pd.DataFrame,
    empirical_reliability: pd.DataFrame,
    prediction_manifest: pd.DataFrame,
) -> str:
    benchmark_metrics = pd.read_parquet(FOLDER / "benchmark_contract_metrics.parquet")
    benchmark_reliability = pd.read_parquet(FOLDER / "benchmark_reliability.parquet")
    metrics = pd.concat([benchmark_metrics, empirical_metrics], ignore_index=True)
    reliability = pd.concat(
        [benchmark_reliability, empirical_reliability], ignore_index=True
    )
    summary = (
        metrics.groupby(["fold_id", "model_key"], as_index=False)
        .agg(
            contracts=("condition_id", "size"),
            decision_rows=("n_rows", "sum"),
            brier=("brier", "mean"),
            log_loss=("log_loss", "mean"),
        )
    )
    ece = (
        reliability.groupby(["fold_id", "model_key"])
        .apply(weighted_ece, include_groups=False)
        .rename("ece_10")
        .reset_index()
    )
    summary = summary.merge(ece, on=["fold_id", "model_key"], validate="one_to_one")
    overall = (
        metrics.groupby("model_key", as_index=False)
        .agg(
            fold_contract_appearances=("condition_id", "size"),
            brier=("brier", "mean"),
            log_loss=("log_loss", "mean"),
        )
    )
    overall_ece = (
        reliability.groupby("model_key")
        .apply(weighted_ece, include_groups=False)
        .rename("ece_10")
        .reset_index()
    )
    overall = overall.merge(overall_ece, on="model_key", validate="one_to_one")
    by_asset = (
        metrics.groupby(["model_key", "asset"], as_index=False)
        .agg(contracts=("condition_id", "size"), brier=("brier", "mean"), log_loss=("log_loss", "mean"))
    )
    by_direction = (
        metrics.groupby(["model_key", "direction"], as_index=False)
        .agg(contracts=("condition_id", "size"), brier=("brier", "mean"), log_loss=("log_loss", "mean"))
    )
    labels = {
        "market": "Polymarket raw",
        "gbm_raw": "GBM raw",
        "gbm_platt": "GBM + Platt",
        "empirical_raw": "Empirical raw",
        "empirical_platt": "Empirical + Platt",
    }
    model_order = ["market", "gbm_raw", "gbm_platt", "empirical_raw", "empirical_platt"]

    lines = [
        "# Phase-3 empirical-model validation report",
        "",
        "This report contains validation results only. Evaluation roles were not",
        "generated or scored. The historical execution proxy was not read.",
        "",
        "## Frozen empirical specification",
        "",
        "- Assets pooled after volatility standardization: BTC, ETH, SOL, XRP.",
        "- Directions estimated separately.",
        f"- Reference horizons: `{list(map(int, HORIZONS))}` minutes.",
        f"- Reference origins sampled every {REFERENCE_STRIDE_MINUTES} minutes.",
        "- Future one-minute highs/lows preserve barrier excursions.",
        f"- Tail smoothing: `{TAIL_PSEUDOCOUNT}` pseudocount in each tail.",
        "- Probabilities interpolated linearly in log remaining time.",
        "- Every reference path finishes before its frozen training cutoff.",
        "",
        "## Platt parameters",
        "",
        "| Fold | Model cutoff | OOF blocks | Contracts | Rows | Slope | Intercept |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in parameters.itertuples(index=False):
        lines.append(
            f"| {row.fold_id} | {row.model_distribution_cutoff} | "
            f"{row.calibration_blocks} | {row.calibration_contracts:,} | "
            f"{row.calibration_rows:,} | {row.slope:.6f} | {row.intercept:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Validation metrics by outer fold",
            "",
            "| Fold | Model | Contracts | Rows | Brier | Log loss | ECE-10 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for fold_id in FOLD_CALIBRATION_BLOCKS:
        for model_key in model_order:
            row = summary[
                (summary["fold_id"] == fold_id) & (summary["model_key"] == model_key)
            ].iloc[0]
            lines.append(
                f"| {fold_id} | {labels[model_key]} | {int(row.contracts):,} | "
                f"{int(row.decision_rows):,} | {row.brier:.6f} | "
                f"{row.log_loss:.6f} | {row.ece_10:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Combined validation summary",
            "",
            "Fold-contract appearances overlap through time and are not independent.",
            "",
            "| Model | Fold-contract appearances | Brier | Log loss | ECE-10 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model_key in model_order:
        row = overall[overall["model_key"] == model_key].iloc[0]
        lines.append(
            f"| {labels[model_key]} | {int(row.fold_contract_appearances):,} | "
            f"{row.brier:.6f} | {row.log_loss:.6f} | {row.ece_10:.6f} |"
        )

    best_brier_key = overall.loc[overall["brier"].idxmin(), "model_key"]
    best_log_loss_key = overall.loc[overall["log_loss"].idxmin(), "model_key"]
    empirical_raw = summary[summary["model_key"] == "empirical_raw"].set_index("fold_id")
    empirical_platt = summary[summary["model_key"] == "empirical_platt"].set_index("fold_id")
    gbm_platt = summary[summary["model_key"] == "gbm_platt"].set_index("fold_id")
    lines.extend(
        [
            "",
            "## Frozen validation decision",
            "",
            f"- Lowest combined Brier: **{labels[best_brier_key]}**.",
            f"- Lowest combined log loss: **{labels[best_log_loss_key]}**.",
            "- The empirical variant is selected only from the registered raw-versus-Platt",
            "  comparison; all benchmark references remain in the tournament.",
            f"- Empirical + Platt beats empirical raw on Brier in "
            f"{int((empirical_platt['brier'] < empirical_raw['brier']).sum())}/4 folds",
            f"  and log loss in {int((empirical_platt['log_loss'] < empirical_raw['log_loss']).sum())}/4 folds.",
            f"- Empirical + Platt beats GBM + Platt on Brier in "
            f"{int((empirical_platt['brier'] < gbm_platt['brier']).sum())}/4 folds",
            f"  and log loss in {int((empirical_platt['log_loss'] < gbm_platt['log_loss']).sum())}/4 folds.",
        ]
    )

    lines.extend(["", "## Breakdown by asset", "", "| Model | Asset | Contracts | Brier | Log loss |", "|---|---|---:|---:|---:|"])
    for model_key in model_order:
        for asset in ASSET_SYMBOLS:
            row = by_asset[(by_asset["model_key"] == model_key) & (by_asset["asset"] == asset)].iloc[0]
            lines.append(
                f"| {labels[model_key]} | {asset} | {int(row.contracts):,} | "
                f"{row.brier:.6f} | {row.log_loss:.6f} |"
            )

    lines.extend(["", "## Breakdown by direction", "", "| Model | Direction | Contracts | Brier | Log loss |", "|---|---|---:|---:|---:|"])
    for model_key in model_order:
        for direction in ["up", "down"]:
            row = by_direction[
                (by_direction["model_key"] == model_key)
                & (by_direction["direction"] == direction)
            ].iloc[0]
            lines.append(
                f"| {labels[model_key]} | {direction} | {int(row.contracts):,} | "
                f"{row.brier:.6f} | {row.log_loss:.6f} |"
            )

    minimum_reference = int(distribution_manifest["reference_count"].min())
    maximum_reference = int(distribution_manifest["reference_count"].max())
    lines.extend(
        [
            "",
            "## Artifact and reference coverage",
            "",
            f"- Empirical prediction partitions: {len(prediction_manifest):,}",
            f"- Empirical prediction rows: {int(prediction_manifest['rows'].sum()):,}",
            f"- Compressed empirical prediction size: {prediction_manifest['bytes'].sum() / 1024**3:.2f} GiB",
            f"- Reference samples per cutoff/horizon/direction: {minimum_reference:,} to {maximum_reference:,}.",
            "- Five frozen distribution files store the actual sorted empirical parameters",
            "  for both directions and all ten horizons at each cutoff.",
            "",
            "## Interpretation boundary",
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
    catalog = build_reference_catalog()
    distributions, distribution_manifest = build_distributions(catalog)
    del catalog
    calibration_blocks = load_calibration_predictions(calendar, distributions)
    parameters = fit_fold_calibrators(calibration_blocks)
    del calibration_blocks
    metrics, reliability, prediction_manifest = generate_validation_outputs(
        folds, distributions, parameters
    )
    REPORT_FILE.write_text(
        make_report(
            parameters,
            distribution_manifest,
            metrics,
            reliability,
            prediction_manifest,
        ),
        encoding="utf-8",
    )
    print(f"parameters: {PARAMETERS_FILE}")
    print(f"metrics: {METRICS_FILE}")
    print(f"prediction manifest: {PREDICTION_MANIFEST_FILE}")
    print(f"report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
