#!/usr/bin/env python3
"""Run the pre-registered local-volatility Kou sensitivity on validation only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_benchmarks import atomic_parquet, weighted_ece
from run_empirical import (
    ASSET_SYMBOLS,
    CUTOFF_TEXT,
    FOLD_CALIBRATION_BLOCKS,
    FOLD_FILE,
    HORIZONS,
    cutoff_timestamp,
    unique_contract_calendar,
)
from run_kou import (
    FOLDER,
    MINIMUM_HISTORY_MINUTES,
    SIMULATION_PATHS,
    SIMULATION_STEP_MINUTES,
    TAIL_PSEUDOCOUNT,
    file_sha256,
    fit_fold_calibrators,
    generate_validation_outputs,
    load_calibration_predictions,
    load_returns,
    simulate_extrema,
    simulation_seed,
)


OUTPUT_DIR = FOLDER / "kou_local_predictions"
DISTRIBUTION_DIR = FOLDER / "kou_local_distributions"
PARAMETERS_FILE = FOLDER / "kou_local_parameters.parquet"
CALIBRATORS_FILE = FOLDER / "kou_local_calibrators.parquet"
DISTRIBUTION_MANIFEST_FILE = FOLDER / "kou_local_distribution_manifest.parquet"
METRICS_FILE = FOLDER / "kou_local_contract_metrics.parquet"
RELIABILITY_FILE = FOLDER / "kou_local_reliability.parquet"
PREDICTION_MANIFEST_FILE = FOLDER / "kou_local_prediction_manifest.parquet"
REPORT_FILE = FOLDER / "kou_local_report.md"

LOCAL_WINDOW = 60
LOCAL_Z_THRESHOLD = 4.0
MODEL_KEYS = ("kou_local_raw", "kou_local_platt")
EXPERIMENTS = {
    "kou_local_raw": {
        "experiment_id": "E05B-KOU-LZ60-RAW-V1",
        "calibration_method": "raw",
    },
    "kou_local_platt": {
        "experiment_id": "E05B-KOU-LZ60-PLATT-V1",
        "calibration_method": "platt",
    },
}


def local_scale(returns: np.ndarray) -> np.ndarray:
    """Prior-60-return sample volatility, excluding the current return."""

    return (
        pd.Series(np.asarray(returns, dtype=float))
        .rolling(LOCAL_WINDOW, min_periods=LOCAL_WINDOW)
        .std(ddof=1)
        .shift(1)
        .to_numpy()
    )


def fit_local_kou_parameters(
    available_time: np.ndarray,
    returns: np.ndarray,
    cutoff_epoch: int,
) -> dict[str, float | int]:
    end = np.searchsorted(available_time, cutoff_epoch, side="right")
    history = np.asarray(returns[:end], dtype=float)
    history_times = np.asarray(available_time[:end], dtype=np.int64)
    if len(history) < MINIMUM_HISTORY_MINUTES:
        raise ValueError("insufficient return history for local-volatility Kou fit")

    scale = local_scale(history)
    eligible = np.isfinite(history) & np.isfinite(scale) & (scale > 0)
    eligible_returns = history[eligible]
    eligible_scale = scale[eligible]
    if len(eligible_returns) < MINIMUM_HISTORY_MINUTES - LOCAL_WINDOW:
        raise ValueError("insufficient eligible history for local-volatility Kou fit")

    local_z = eligible_returns / eligible_scale
    jump_mask = np.abs(local_z) >= LOCAL_Z_THRESHOLD
    core = eligible_returns[~jump_mask]
    jumps = eligible_returns[jump_mask]
    positive = jumps[jumps > 0]
    negative_magnitude = -jumps[jumps < 0]
    if len(core) < 100 or np.std(core, ddof=1) <= 0:
        raise ValueError("invalid diffusion core for local-volatility Kou fit")
    if len(positive) == 0 or len(negative_magnitude) == 0:
        raise ValueError("both jump directions are required for local-volatility Kou fit")

    eligible_times = history_times[eligible]
    return {
        "history_minutes": int(len(history)),
        "eligible_minutes": int(len(eligible_returns)),
        "excluded_scale_minutes": int(len(history) - len(eligible_returns)),
        "latest_return_available": int(eligible_times[-1]),
        "local_window_minutes": LOCAL_WINDOW,
        "jump_threshold_z": LOCAL_Z_THRESHOLD,
        "jump_count": int(len(jumps)),
        "positive_jump_count": int(len(positive)),
        "negative_jump_count": int(len(negative_magnitude)),
        "drift_per_minute": float(np.mean(core)),
        "sigma_per_sqrt_minute": float(np.std(core, ddof=1)),
        "jump_intensity_per_minute": float(len(jumps) / len(eligible_returns)),
        "up_jump_share": float(len(positive) / len(jumps)),
        "up_eta": float(1 / np.mean(positive)),
        "down_eta": float(1 / np.mean(negative_magnitude)),
    }


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
            fitted = fit_local_kou_parameters(
                *returns_by_asset[asset], cutoff_epoch
            )
            if fitted["latest_return_available"] > cutoff_epoch:
                raise ValueError("local-volatility Kou parameter cutoff violation")
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
                f"fitted local Kou: {cutoff_id} {asset} "
                f"jumps={fitted['jump_count']:,} "
                f"rate={fitted['jump_intensity_per_minute']:.4%}",
                flush=True,
            )

    parameters = pd.DataFrame(parameter_rows)
    distribution_manifest = pd.DataFrame(distribution_rows)
    atomic_parquet(parameters, PARAMETERS_FILE)
    atomic_parquet(distribution_manifest, DISTRIBUTION_MANIFEST_FILE)
    return all_distributions, parameters, distribution_manifest


def _metric_tables(
    metrics: pd.DataFrame, reliability: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return by_fold, overall.merge(overall_ece, on="model_key")


def make_report(
    parameters: pd.DataFrame,
    calibrators: pd.DataFrame,
    distribution_manifest: pd.DataFrame,
    local_metrics: pd.DataFrame,
    local_reliability: pd.DataFrame,
    prediction_manifest: pd.DataFrame,
) -> str:
    metric_files = [
        "benchmark_contract_metrics.parquet",
        "empirical_contract_metrics.parquet",
        "kou_contract_metrics.parquet",
    ]
    reliability_files = [
        "benchmark_reliability.parquet",
        "empirical_reliability.parquet",
        "kou_reliability.parquet",
    ]
    metrics = pd.concat(
        [pd.read_parquet(FOLDER / name) for name in metric_files] + [local_metrics],
        ignore_index=True,
    )
    reliability = pd.concat(
        [pd.read_parquet(FOLDER / name) for name in reliability_files]
        + [local_reliability],
        ignore_index=True,
    )
    by_fold, overall = _metric_tables(metrics, reliability)
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
    }
    original_parameters = pd.read_parquet(FOLDER / "kou_parameters.parquet")
    original_rate = original_parameters.groupby("asset")[
        "jump_intensity_per_minute"
    ].median()
    local_rate = parameters.groupby("asset")["jump_intensity_per_minute"].median()

    lines = [
        "# Local-volatility Kou sensitivity and family closure",
        "",
        "This is the pre-registered post-result sensitivity. It uses validation",
        "roles only; evaluation and execution data were not read.",
        "",
        "## Jump-frequency diagnostic",
        "",
        "| Asset | Original 4-MAD | Local-z60 | Local implied interval |",
        "|---|---:|---:|---:|",
    ]
    for asset in ASSET_SYMBOLS:
        interval = 1 / local_rate[asset]
        lines.append(
            f"| {asset} | {original_rate[asset]:.4%} | {local_rate[asset]:.4%} | "
            f"1 per {interval:.1f} min |"
        )

    lines.extend(
        [
            "",
            "## Local-z60 fitted parameters",
            "",
            "| Asset | Fits | Median eligible minutes | Median jumps | Median sigma/min | Median lambda/min |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    parameter_summary = parameters.groupby("asset").agg(
        fits=("cutoff_id", "size"),
        eligible=("eligible_minutes", "median"),
        jumps=("jump_count", "median"),
        sigma=("sigma_per_sqrt_minute", "median"),
        intensity=("jump_intensity_per_minute", "median"),
    )
    for asset in ASSET_SYMBOLS:
        row = parameter_summary.loc[asset]
        lines.append(
            f"| {asset} | {int(row.fits)} | {int(row.eligible):,} | "
            f"{int(row.jumps):,} | {row.sigma:.8f} | {row.intensity:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Platt parameters",
            "",
            "| Fold | Training-only OOF blocks | Contracts | Rows | Slope | Intercept |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in calibrators.itertuples(index=False):
        lines.append(
            f"| {row.fold_id} | {row.calibration_blocks} | "
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
        for model_key in MODEL_KEYS:
            row = by_fold[
                (by_fold["fold_id"] == fold_id)
                & (by_fold["model_key"] == model_key)
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
        row = overall[overall["model_key"] == model_key].iloc[0]
        lines.append(
            f"| {labels[model_key]} | {int(row.fold_contract_appearances):,} | "
            f"{row.brier:.6f} | {row.log_loss:.6f} | {row.ece_10:.6f} |"
        )

    indexed = overall.set_index("model_key")
    raw_wins_brier = int(
        (
            by_fold[by_fold.model_key == "kou_local_platt"].set_index("fold_id").brier
            < by_fold[by_fold.model_key == "kou_local_raw"].set_index("fold_id").brier
        ).sum()
    )
    raw_wins_log = int(
        (
            by_fold[by_fold.model_key == "kou_local_platt"].set_index("fold_id").log_loss
            < by_fold[by_fold.model_key == "kou_local_raw"].set_index("fold_id").log_loss
        ).sum()
    )
    family_winner = min(
        MODEL_KEYS,
        key=lambda key: (indexed.loc[key, "brier"], indexed.loc[key, "log_loss"]),
    )
    tournament_winner = min(
        labels,
        key=lambda key: (indexed.loc[key, "brier"], indexed.loc[key, "log_loss"]),
    )
    lines.extend(
        [
            "",
            "## Frozen decision and Kou closure",
            "",
            f"- Within local-z60, **{labels[family_winner]}** wins by the registered Brier-then-log-loss rule.",
            f"- Local-z60 + Platt beats local-z60 raw on Brier in {raw_wins_brier}/4 folds and log loss in {raw_wins_log}/4 folds.",
            f"- Across implemented probability models, **{labels[tournament_winner]}** remains the leader.",
            f"- Local-z60 + Platt versus original Kou + Platt: Brier {indexed.loc['kou_local_platt', 'brier']:.6f} versus {indexed.loc['kou_platt', 'brier']:.6f}; log loss {indexed.loc['kou_local_platt', 'log_loss']:.6f} versus {indexed.loc['kou_platt', 'log_loss']:.6f}.",
            "- The Kou family is now closed for the current Phase-3 tournament. No further threshold, rolling-window, seed, or parameter variants will be added.",
            "",
            "## Artifact coverage",
            "",
            f"- Prediction partitions: {len(prediction_manifest):,}.",
            f"- Prediction rows: {int(prediction_manifest['rows'].sum()):,}.",
            f"- Parameter fits: {len(parameters):,}.",
            f"- Distribution audit rows: {len(distribution_manifest):,}.",
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
    calibrators = fit_fold_calibrators(
        calibration_blocks, calibration_tag="kou_local_platt_v1"
    )
    del calibration_blocks
    atomic_parquet(calibrators, CALIBRATORS_FILE)
    metrics, reliability, prediction_manifest = generate_validation_outputs(
        folds,
        distributions,
        calibrators,
        experiments=EXPERIMENTS,
        model_keys=MODEL_KEYS,
        output_dir=OUTPUT_DIR,
        metrics_file=METRICS_FILE,
        reliability_file=RELIABILITY_FILE,
        prediction_manifest_file=PREDICTION_MANIFEST_FILE,
        model_id="kou_local_z60_v1",
        feature_set_id="spot_distance_time_kou_local_v1",
        calibration_tag="kou_local_platt_v1",
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
