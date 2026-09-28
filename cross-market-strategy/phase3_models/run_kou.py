#!/usr/bin/env python3
"""Run the frozen validation-only Kou jump-diffusion model and Platt variant."""

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
    read_panel,
    unique_contract_calendar,
)


FOLDER = Path(__file__).resolve().parent
OUTPUT_DIR = FOLDER / "kou_predictions"
DISTRIBUTION_DIR = FOLDER / "kou_distributions"
PARAMETERS_FILE = FOLDER / "kou_parameters.parquet"
DISTRIBUTION_MANIFEST_FILE = FOLDER / "kou_distribution_manifest.parquet"
METRICS_FILE = FOLDER / "kou_contract_metrics.parquet"
RELIABILITY_FILE = FOLDER / "kou_reliability.parquet"
PREDICTION_MANIFEST_FILE = FOLDER / "kou_prediction_manifest.parquet"
REPORT_FILE = FOLDER / "kou_report.md"

JUMP_THRESHOLD = 4.0
MINIMUM_HISTORY_MINUTES = 30 * 24 * 60
SIMULATION_PATHS = 8192
SIMULATION_STEP_MINUTES = 1
TAIL_PSEUDOCOUNT = 0.5
ASSET_SEEDS = {"BTC": 11, "ETH": 23, "SOL": 37, "XRP": 53}

EXPERIMENTS = {
    "kou_raw": {
        "experiment_id": "E05-KOU-RAW-V1",
        "calibration_method": "raw",
    },
    "kou_platt": {
        "experiment_id": "E05-KOU-PLATT-V1",
        "calibration_method": "platt",
    },
}


def load_returns(asset: str) -> tuple[np.ndarray, np.ndarray]:
    symbol = ASSET_SYMBOLS[asset]
    data = pd.read_parquet(
        SPOT_DIR / f"{symbol}_1m.parquet", columns=["ts", "close"]
    ).sort_values("ts", ignore_index=True)
    timestamp = data["ts"].to_numpy(dtype=np.int64)
    returns = np.diff(np.log(data["close"].to_numpy(dtype=float)))
    # Return i is known when the later candle has closed.
    available_time = timestamp[1:] + 60
    return available_time, returns


def fit_kou_parameters(
    available_time: np.ndarray,
    returns: np.ndarray,
    cutoff_epoch: int,
) -> dict[str, float | int]:
    end = np.searchsorted(available_time, cutoff_epoch, side="right")
    history = returns[:end]
    if len(history) < MINIMUM_HISTORY_MINUTES:
        raise ValueError("insufficient return history for Kou fit")

    centre = float(np.median(history))
    robust_scale = float(1.4826 * np.median(np.abs(history - centre)))
    if not np.isfinite(robust_scale) or robust_scale <= 0:
        robust_scale = float(np.std(history, ddof=1))
    jump_mask = np.abs(history - centre) > JUMP_THRESHOLD * robust_scale
    core = history[~jump_mask]
    jumps = history[jump_mask]
    positive = jumps[jumps > 0]
    negative_magnitude = -jumps[jumps < 0]
    if len(core) < 100 or np.std(core, ddof=1) <= 0:
        raise ValueError("invalid diffusion core for Kou fit")
    if len(positive) == 0 or len(negative_magnitude) == 0:
        raise ValueError("both jump directions are required for Kou fit")

    return {
        "history_minutes": int(len(history)),
        "latest_return_available": int(available_time[end - 1]),
        "return_median": centre,
        "robust_scale": robust_scale,
        "jump_threshold_scales": JUMP_THRESHOLD,
        "jump_count": int(len(jumps)),
        "positive_jump_count": int(len(positive)),
        "negative_jump_count": int(len(negative_magnitude)),
        "drift_per_minute": float(np.mean(core)),
        "sigma_per_sqrt_minute": float(np.std(core, ddof=1)),
        "jump_intensity_per_minute": float(len(jumps) / len(history)),
        "up_jump_share": float(len(positive) / len(jumps)),
        "up_eta": float(1 / np.mean(positive)),
        "down_eta": float(1 / np.mean(negative_magnitude)),
    }


def simulation_seed(asset: str, cutoff_id: str) -> int:
    day_number = int(CUTOFF_TEXT[cutoff_id].replace("-", ""))
    return ASSET_SEEDS[asset] * 100_000_000 + day_number


def simulate_extrema(
    parameters: dict[str, float | int],
    seed: int,
    path_count: int = SIMULATION_PATHS,
) -> dict[tuple[int, str], np.ndarray]:
    rng = np.random.default_rng(seed)
    location = np.zeros(path_count, dtype=float)
    maximum = np.zeros(path_count, dtype=float)
    minimum = np.zeros(path_count, dtype=float)
    result = {}
    sigma = float(parameters["sigma_per_sqrt_minute"])
    drift = float(parameters["drift_per_minute"])
    intensity = float(parameters["jump_intensity_per_minute"])
    up_share = float(parameters["up_jump_share"])
    up_eta = float(parameters["up_eta"])
    down_eta = float(parameters["down_eta"])
    horizon_set = set(map(int, HORIZONS))

    for minute in range(1, int(HORIZONS.max()) + 1):
        start = location
        diffusion = drift + sigma * rng.standard_normal(path_count)
        before_jump = start + diffusion

        # Exact conditional Brownian-bridge extremum sample for the diffusion
        # segment between start and before_jump.
        bridge_term = -0.5 * sigma**2 * np.log(rng.random(path_count))
        spread = np.sqrt((start - before_jump) ** 2 + 4 * bridge_term)
        bridge_high = (start + before_jump + spread) / 2
        bridge_low = (start + before_jump - spread) / 2

        jump_counts = rng.poisson(intensity, path_count)
        up_counts = rng.binomial(jump_counts, up_share)
        down_counts = jump_counts - up_counts
        jump = rng.gamma(up_counts, 1 / up_eta) - rng.gamma(
            down_counts, 1 / down_eta
        )
        location = before_jump + jump
        maximum = np.maximum(maximum, np.maximum(bridge_high, location))
        minimum = np.minimum(minimum, np.minimum(bridge_low, location))

        if minute in horizon_set:
            result[(minute, "up")] = np.sort(maximum.copy())
            result[(minute, "down")] = np.sort((-minimum).copy())
    return result


def build_fits_and_distributions(
    returns_by_asset: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[
    dict[str, dict[str, dict[tuple[int, str], np.ndarray]]],
    pd.DataFrame,
    pd.DataFrame,
]:
    all_distributions = {}
    parameter_rows = []
    distribution_rows = []
    DISTRIBUTION_DIR.mkdir(parents=True, exist_ok=True)

    for cutoff_id, cutoff_text_value in CUTOFF_TEXT.items():
        cutoff = cutoff_timestamp(cutoff_text_value)
        cutoff_epoch = int(cutoff.timestamp())
        all_distributions[cutoff_id] = {}
        for asset in ASSET_SYMBOLS:
            fitted = fit_kou_parameters(*returns_by_asset[asset], cutoff_epoch)
            if fitted["latest_return_available"] > cutoff_epoch:
                raise ValueError("Kou parameter cutoff violation")
            seed = simulation_seed(asset, cutoff_id)
            distribution = simulate_extrema(fitted, seed)
            all_distributions[cutoff_id][asset] = distribution
            parameter_rows.append(
                {
                    "cutoff_id": cutoff_id,
                    "cutoff_time": cutoff,
                    "asset": asset,
                    "simulation_seed": seed,
                    "simulation_paths": SIMULATION_PATHS,
                    "simulation_step_minutes": SIMULATION_STEP_MINUTES,
                    **fitted,
                }
            )

            path = DISTRIBUTION_DIR / f"{cutoff_id}_{asset}.npz"
            save_values = {
                "cutoff_epoch": np.array([cutoff_epoch], dtype=np.int64),
                "simulation_seed": np.array([seed], dtype=np.int64),
                "horizons": HORIZONS,
            }
            for (horizon, direction), values in distribution.items():
                save_values[f"h{horizon}_{direction}"] = values
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **save_values)
            temporary.replace(path)
            checksum = file_sha256(path)

            for horizon in HORIZONS:
                for direction in ["up", "down"]:
                    values = distribution[(int(horizon), direction)]
                    distribution_rows.append(
                        {
                            "cutoff_id": cutoff_id,
                            "cutoff_time": cutoff,
                            "asset": asset,
                            "horizon_minutes": int(horizon),
                            "direction": direction,
                            "simulation_paths": len(values),
                            "minimum": float(values[0]),
                            "median": float(np.median(values)),
                            "p95": float(np.quantile(values, 0.95)),
                            "p99": float(np.quantile(values, 0.99)),
                            "maximum": float(values[-1]),
                            "distribution_path": str(path.relative_to(FOLDER)),
                            "distribution_sha256": checksum,
                        }
                    )
            print(
                f"fitted and simulated Kou: {cutoff_id} {asset} "
                f"jumps={fitted['jump_count']:,}",
                flush=True,
            )

    parameters = pd.DataFrame(parameter_rows)
    distribution_manifest = pd.DataFrame(distribution_rows)
    atomic_parquet(parameters, PARAMETERS_FILE)
    atomic_parquet(distribution_manifest, DISTRIBUTION_MANIFEST_FILE)
    return all_distributions, parameters, distribution_manifest


def smoothed_probability(
    sorted_values: np.ndarray, distance: np.ndarray
) -> np.ndarray:
    exceed = len(sorted_values) - np.searchsorted(sorted_values, distance, side="left")
    return (exceed + TAIL_PSEUDOCOUNT) / (
        len(sorted_values) + 2 * TAIL_PSEUDOCOUNT
    )


def kou_touch_probability(
    distance: np.ndarray,
    minutes_to_expiry: np.ndarray,
    direction: str,
    distribution: dict[tuple[int, str], np.ndarray],
) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    horizon = np.asarray(minutes_to_expiry, dtype=float)
    valid = np.isfinite(distance) & np.isfinite(horizon) & (horizon > 0)
    probability = np.full(len(distance), np.nan, dtype=float)
    if not valid.any():
        return probability

    clipped_horizon = np.clip(horizon[valid], HORIZONS[0], HORIZONS[-1])
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
        low_probability = smoothed_probability(
            distribution[(low_horizon, direction)], distance[valid][rows]
        )
        if low_horizon == high_horizon:
            selected[rows] = low_probability
            continue
        high_probability = smoothed_probability(
            distribution[(high_horizon, direction)], distance[valid][rows]
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
    panel: pd.DataFrame,
    distribution: dict[tuple[int, str], np.ndarray],
) -> np.ndarray:
    directions = panel["direction"].unique()
    if len(directions) != 1 or directions[0] not in {"up", "down"}:
        raise ValueError("panel must contain exactly one valid direction")
    return kou_touch_probability(
        panel["log_distance_to_barrier"].to_numpy(),
        panel["minutes_to_expiry"].to_numpy(),
        directions[0],
        distribution,
    )


def load_calibration_predictions(
    calendar: pd.DataFrame,
    distributions: dict[str, dict[str, dict[tuple[int, str], np.ndarray]]],
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
            probability = predict_panel(panel, distributions[cutoff_id][member.asset])
            if not np.isfinite(probability).all():
                raise ValueError(f"invalid Kou probability in {member.condition_id}")
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
            f"generated Kou OOF {block_id}: {len(members):,} contracts, "
            f"{len(blocks[block_id]['probability']):,} rows",
            flush=True,
        )
    return blocks


def fit_fold_calibrators(
    blocks: dict[str, dict[str, object]],
    calibration_tag: str = "kou_platt_v1",
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
                "model_distribution_cutoff": FOLD_DISTRIBUTION_CUTOFF[fold_id],
                "calibration_version": f"{fold_id}_{calibration_tag}",
                "calibration_blocks": "|".join(block_ids),
                "calibration_distribution_cutoffs": "|".join(
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
            f"fitted Kou {fold_id} Platt: slope={fitted['slope']:.6f}, "
            f"intercept={fitted['intercept']:.6f}",
            flush=True,
        )
    calibrators = pd.DataFrame(rows)
    return calibrators


def prediction_frame(
    base: pd.DataFrame,
    model_key: str,
    fold_id: str,
    probability_column: str,
    experiments: dict[str, dict[str, str]] | None = None,
    model_id: str = "kou_asset_specific_v1",
    feature_set_id: str = "spot_distance_time_kou_v1",
    calibration_tag: str = "kou_platt_v1",
) -> pd.DataFrame:
    experiments = EXPERIMENTS if experiments is None else experiments
    experiment = experiments[model_key]
    is_calibrated = experiment["calibration_method"] != "raw"
    raw_column = "kou_probability" if is_calibrated else probability_column
    calibration_version = (
        f"{fold_id}_{calibration_tag}" if is_calibrated else "none"
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
            "model_id": model_id,
            "model_version": "v1",
            "feature_set_id": feature_set_id,
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
    distributions: dict[str, dict[str, dict[tuple[int, str], np.ndarray]]],
    calibrators: pd.DataFrame,
    experiments: dict[str, dict[str, str]] | None = None,
    model_keys: tuple[str, str] = ("kou_raw", "kou_platt"),
    output_dir: Path = OUTPUT_DIR,
    metrics_file: Path = METRICS_FILE,
    reliability_file: Path = RELIABILITY_FILE,
    prediction_manifest_file: Path = PREDICTION_MANIFEST_FILE,
    model_id: str = "kou_asset_specific_v1",
    feature_set_id: str = "spot_distance_time_kou_v1",
    calibration_tag: str = "kou_platt_v1",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    experiments = EXPERIMENTS if experiments is None else experiments
    raw_key, calibrated_key = model_keys
    calibrator_map = calibrators.set_index("fold_id").to_dict("index")
    metrics = []
    manifests = []
    reliability = defaultdict(lambda: np.zeros(4, dtype=float))

    for fold_id in FOLD_CALIBRATION_BLOCKS:
        fold_validation = folds[
            (folds["fold_id"] == fold_id) & (folds["role"] == "validation")
        ]
        fitted = calibrator_map[fold_id]
        cutoff_id = FOLD_DISTRIBUTION_CUTOFF[fold_id]
        for asset in ASSET_SYMBOLS:
            members = fold_validation[fold_validation["asset"] == asset]
            contract_frames = []
            distribution = distributions[cutoff_id][asset]
            for member in members.itertuples(index=False):
                panel = read_panel(member.condition_id)
                kou = predict_panel(panel, distribution)
                if not np.isfinite(kou).all():
                    raise ValueError(f"invalid Kou probability in {member.condition_id}")
                clipped = np.clip(kou, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
                raw_logit = np.log(clipped) - np.log1p(-clipped)
                platt = expit(fitted["slope"] * raw_logit + fitted["intercept"])
                panel = panel.copy()
                panel["kou_probability"] = kou
                panel["kou_platt_probability"] = platt
                contract_frames.append(panel)

                outcomes = panel["y"].to_numpy(dtype=float)
                weights = panel["contract_row_weight"].to_numpy(dtype=float)
                for model_key, probability in {
                    raw_key: kou,
                    calibrated_key: platt,
                }.items():
                    brier, log_loss, mean_probability = contract_metric(
                        probability, outcomes, weights
                    )
                    metrics.append(
                        {
                            "fold_id": fold_id,
                            "role": "validation",
                            "model_key": model_key,
                            "experiment_id": experiments[model_key]["experiment_id"],
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
                (raw_key, "kou_probability"),
                (calibrated_key, "kou_platt_probability"),
            ]:
                output = prediction_frame(
                    base,
                    model_key,
                    fold_id,
                    probability_column,
                    experiments=experiments,
                    model_id=model_id,
                    feature_set_id=feature_set_id,
                    calibration_tag=calibration_tag,
                )
                path = (
                    output_dir
                    / experiments[model_key]["experiment_id"]
                    / fold_id
                    / "validation"
                    / f"{asset}.parquet"
                )
                atomic_parquet(output, path)
                manifests.append(
                    {
                        "experiment_id": experiments[model_key]["experiment_id"],
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
                f"wrote Kou {fold_id} validation {asset}: "
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
                "experiment_id": experiments[model_key]["experiment_id"],
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
    atomic_parquet(metric_frame, metrics_file)
    atomic_parquet(reliability_frame, reliability_file)
    atomic_parquet(manifest_frame, prediction_manifest_file)
    return metric_frame, reliability_frame, manifest_frame


def make_report(
    parameters: pd.DataFrame,
    calibrators: pd.DataFrame,
    distribution_manifest: pd.DataFrame,
    kou_metrics: pd.DataFrame,
    kou_reliability: pd.DataFrame,
    prediction_manifest: pd.DataFrame,
) -> str:
    benchmark_metrics = pd.read_parquet(FOLDER / "benchmark_contract_metrics.parquet")
    empirical_metrics = pd.read_parquet(FOLDER / "empirical_contract_metrics.parquet")
    benchmark_reliability = pd.read_parquet(FOLDER / "benchmark_reliability.parquet")
    empirical_reliability = pd.read_parquet(FOLDER / "empirical_reliability.parquet")
    metrics = pd.concat(
        [benchmark_metrics, empirical_metrics, kou_metrics], ignore_index=True
    )
    reliability = pd.concat(
        [benchmark_reliability, empirical_reliability, kou_reliability],
        ignore_index=True,
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
        .agg(
            contracts=("condition_id", "size"),
            brier=("brier", "mean"),
            log_loss=("log_loss", "mean"),
        )
    )
    by_direction = (
        metrics.groupby(["model_key", "direction"], as_index=False)
        .agg(
            contracts=("condition_id", "size"),
            brier=("brier", "mean"),
            log_loss=("log_loss", "mean"),
        )
    )
    labels = {
        "market": "Polymarket raw",
        "gbm_raw": "GBM raw",
        "gbm_platt": "GBM + Platt",
        "empirical_raw": "Empirical raw",
        "empirical_platt": "Empirical + Platt",
        "kou_raw": "Kou raw",
        "kou_platt": "Kou + Platt",
    }
    model_order = list(labels)

    lines = [
        "# Phase-3 Kou jump-diffusion validation report",
        "",
        "This report contains validation results only. Evaluation roles and the",
        "historical execution proxy were not read.",
        "",
        "## Frozen Kou specification",
        "",
        "- Asset-specific parameters from returns available by each training cutoff.",
        f"- Jumps: absolute median-centred return above {JUMP_THRESHOLD:g} robust scales.",
        f"- Monte Carlo: {SIMULATION_PATHS:,} fixed-seed paths at one-minute steps.",
        "- Brownian-bridge diffusion extrema plus endpoint compound Kou jumps.",
        f"- Horizons: `{list(map(int, HORIZONS))}` minutes with log-time interpolation.",
        "- Bounded 0.5-tail smoothing and training-only Platt calibration.",
        "",
        "## Parameter summary",
        "",
        "| Asset | Fits | Median history | Median jumps | Median sigma/minute | Median lambda/minute |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    parameter_summary = parameters.groupby("asset").agg(
        fits=("cutoff_id", "size"),
        history=("history_minutes", "median"),
        jumps=("jump_count", "median"),
        sigma=("sigma_per_sqrt_minute", "median"),
        intensity=("jump_intensity_per_minute", "median"),
    )
    for asset in ASSET_SYMBOLS:
        row = parameter_summary.loc[asset]
        lines.append(
            f"| {asset} | {int(row.fits)} | {int(row.history):,} | "
            f"{int(row.jumps):,} | {row.sigma:.8f} | {row.intensity:.6f} |"
        )

    lines.extend(["", "## Platt parameters", "", "| Fold | Model cutoff | OOF blocks | Contracts | Rows | Slope | Intercept |", "|---|---|---|---:|---:|---:|---:|"])
    for row in calibrators.itertuples(index=False):
        lines.append(
            f"| {row.fold_id} | {row.model_distribution_cutoff} | "
            f"{row.calibration_blocks} | {row.calibration_contracts:,} | "
            f"{row.calibration_rows:,} | {row.slope:.6f} | {row.intercept:.6f} |"
        )

    lines.extend(["", "## Validation metrics by outer fold", "", "| Fold | Model | Contracts | Rows | Brier | Log loss | ECE-10 |", "|---|---|---:|---:|---:|---:|---:|"])
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

    lines.extend(["", "## Combined validation summary", "", "Fold-contract appearances overlap through time and are not independent.", "", "| Model | Fold-contract appearances | Brier | Log loss | ECE-10 |", "|---|---:|---:|---:|---:|"])
    for model_key in model_order:
        row = overall[overall["model_key"] == model_key].iloc[0]
        lines.append(
            f"| {labels[model_key]} | {int(row.fold_contract_appearances):,} | "
            f"{row.brier:.6f} | {row.log_loss:.6f} | {row.ece_10:.6f} |"
        )

    best_brier_key = overall.loc[overall["brier"].idxmin(), "model_key"]
    best_log_loss_key = overall.loc[overall["log_loss"].idxmin(), "model_key"]
    raw_fold = summary[summary["model_key"] == "kou_raw"].set_index("fold_id")
    platt_fold = summary[summary["model_key"] == "kou_platt"].set_index("fold_id")
    empirical_fold = summary[summary["model_key"] == "empirical_platt"].set_index("fold_id")
    lines.extend(
        [
            "",
            "## Frozen validation decision",
            "",
            f"- Lowest combined Brier: **{labels[best_brier_key]}**.",
            f"- Lowest combined log loss: **{labels[best_log_loss_key]}**.",
            f"- Kou + Platt beats Kou raw on Brier in {int((platt_fold['brier'] < raw_fold['brier']).sum())}/4 folds",
            f"  and log loss in {int((platt_fold['log_loss'] < raw_fold['log_loss']).sum())}/4 folds.",
            f"- Kou + Platt beats Empirical + Platt on Brier in {int((platt_fold['brier'] < empirical_fold['brier']).sum())}/4 folds",
            f"  and log loss in {int((platt_fold['log_loss'] < empirical_fold['log_loss']).sum())}/4 folds.",
            "- Under the registered within-family rule, Kou + Platt is retained as",
            "  the calibrated Kou representative, but Empirical + Platt remains the",
            "  leading implemented model.",
            "- The 4-MAD estimator labels roughly 2.2%–3.8% of minutes as jumps",
            "  across the median asset fits. This weakens the interpretation of",
            "  these events as rare Poisson jumps and is a documented limitation,",
            "  not a parameter to retune after viewing validation results.",
        ]
    )

    lines.extend(["", "## Breakdown by asset", "", "| Model | Asset | Contracts | Brier | Log loss |", "|---|---|---:|---:|---:|"])
    for model_key in model_order:
        for asset in ASSET_SYMBOLS:
            row = by_asset[
                (by_asset["model_key"] == model_key)
                & (by_asset["asset"] == asset)
            ].iloc[0]
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

    lines.extend(
        [
            "",
            "## Artifact coverage",
            "",
            f"- Kou prediction partitions: {len(prediction_manifest):,}",
            f"- Kou prediction rows: {int(prediction_manifest['rows'].sum()):,}",
            f"- Compressed Kou prediction size: {prediction_manifest['bytes'].sum() / 1024**3:.2f} GiB",
            f"- Parameter fits: {len(parameters):,}.",
            f"- Simulated distribution audit rows: {len(distribution_manifest):,}.",
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
    returns_by_asset = {asset: load_returns(asset) for asset in ASSET_SYMBOLS}
    distributions, parameters, distribution_manifest = build_fits_and_distributions(
        returns_by_asset
    )
    calibration_blocks = load_calibration_predictions(calendar, distributions)
    calibrators = fit_fold_calibrators(calibration_blocks)
    del calibration_blocks
    # Keep the fitted structural parameters and calibration maps in one table
    # family while preserving their distinct row semantics.
    atomic_parquet(calibrators, FOLDER / "kou_calibrators.parquet")
    metrics, reliability, prediction_manifest = generate_validation_outputs(
        folds, distributions, calibrators
    )
    REPORT_FILE.write_text(
        make_report(
            parameters,
            calibrators,
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
