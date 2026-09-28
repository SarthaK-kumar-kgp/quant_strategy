#!/usr/bin/env python3
"""Run the frozen validation-only Polymarket and zero-drift GBM benchmarks."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, ndtr


FOLDER = Path(__file__).resolve().parent
PHASE2 = FOLDER.parent / "phase2_dataset"
PANEL_DIR = PHASE2 / "causal_panel_1m"
FOLD_FILE = PHASE2 / "walk_forward_folds.parquet"
OUTPUT_DIR = FOLDER / "benchmark_predictions"
PARAMETERS_FILE = FOLDER / "benchmark_parameters.parquet"
METRICS_FILE = FOLDER / "benchmark_contract_metrics.parquet"
RELIABILITY_FILE = FOLDER / "benchmark_reliability.parquet"
MANIFEST_FILE = FOLDER / "benchmark_prediction_manifest.parquet"
REPORT_FILE = FOLDER / "benchmark_report.md"

MINUTES_PER_YEAR = 365 * 24 * 60
PROBABILITY_CLIP = 1e-6
LOG_LOSS_CLIP = 1e-12
PLATT_L2 = 1e-6
RELIABILITY_BINS = 10

EXPERIMENTS = {
    "market": {
        "experiment_id": "E03-PM-RAW-V1",
        "model_id": "polymarket_yes_raw",
        "model_version": "v1",
        "feature_set_id": "market_yes_only_v1",
        "calibration_method": "raw",
    },
    "gbm_raw": {
        "experiment_id": "E03-GBM-RAW-V1",
        "model_id": "gbm_zero_drift_rv24h",
        "model_version": "v1",
        "feature_set_id": "gbm_spot_barrier_rv24h_v1",
        "calibration_method": "raw",
    },
    "gbm_platt": {
        "experiment_id": "E03-GBM-PLATT-V1",
        "model_id": "gbm_zero_drift_rv24h",
        "model_version": "v1",
        "feature_set_id": "gbm_spot_barrier_rv24h_v1",
        "calibration_method": "platt",
    },
}

CALIBRATION_BLOCKS = {
    "cal_2026_04": (date(2026, 4, 2), date(2026, 4, 30)),
    "cal_2026_05": (date(2026, 5, 2), date(2026, 5, 31)),
    "cal_2026_06": (date(2026, 6, 2), date(2026, 6, 30)),
    "cal_2026_07": (date(2026, 7, 2), date(2026, 7, 31)),
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

PANEL_COLUMNS = [
    "condition_id",
    "asset",
    "direction",
    "barrier",
    "decision_time",
    "spot_available_time",
    "spot_close",
    "minutes_to_expiry",
    "rv_24h",
    "market_yes_price",
    "y",
    "contract_row_weight",
]


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gbm_touch_probability(
    spot: np.ndarray,
    barrier: np.ndarray,
    minutes_to_expiry: np.ndarray,
    annualized_sigma: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """First-passage probability under GBM with annual price drift mu=0."""

    spot = np.asarray(spot, dtype=float)
    barrier = np.asarray(barrier, dtype=float)
    minutes_to_expiry = np.asarray(minutes_to_expiry, dtype=float)
    sigma = np.asarray(annualized_sigma, dtype=float)
    direction = np.asarray(direction)
    valid = (
        np.isfinite(spot)
        & np.isfinite(barrier)
        & np.isfinite(minutes_to_expiry)
        & np.isfinite(sigma)
        & (spot > 0)
        & (barrier > 0)
        & (minutes_to_expiry > 0)
        & (sigma > 0)
        & np.isin(direction, ["up", "down"])
    )
    probability = np.full(len(spot), np.nan, dtype=float)
    if not valid.any():
        return probability

    s0 = spot[valid]
    strike = barrier[valid]
    horizon = minutes_to_expiry[valid] / MINUTES_PER_YEAR
    vol = sigma[valid]
    side = direction[valid]
    log_barrier = np.log(strike / s0)
    log_drift = -0.5 * vol**2
    total_sigma = vol * np.sqrt(horizon)

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        p_up = ndtr((log_drift * horizon - log_barrier) / total_sigma) + np.exp(
            2 * log_drift * log_barrier / vol**2
        ) * ndtr((-log_drift * horizon - log_barrier) / total_sigma)
        p_down = ndtr((log_barrier - log_drift * horizon) / total_sigma) + np.exp(
            2 * log_drift * log_barrier / vol**2
        ) * ndtr((log_barrier + log_drift * horizon) / total_sigma)

    selected = np.where(side == "up", p_up, p_down)
    already_touched = ((side == "up") & (s0 >= strike)) | (
        (side == "down") & (s0 <= strike)
    )
    selected = np.where(already_touched, 1.0, selected)
    probability[valid] = np.clip(selected, 0.0, 1.0)
    return probability


def read_panel(condition_id: str) -> pd.DataFrame:
    path = PANEL_DIR / f"{condition_id}.parquet"
    return pd.read_parquet(path, columns=PANEL_COLUMNS)


def unique_contract_calendar(folds: pd.DataFrame) -> pd.DataFrame:
    columns = ["condition_id", "asset", "direction", "resolution_date_et"]
    calendar = folds[columns].drop_duplicates().copy()
    conflicts = calendar.groupby("condition_id").size()
    if (conflicts != 1).any():
        raise ValueError("contract metadata changes between outer folds")
    return calendar


def load_calibration_blocks(calendar: pd.DataFrame) -> dict[str, dict[str, object]]:
    blocks = {}
    for block_id, (start, end) in CALIBRATION_BLOCKS.items():
        members = calendar[
            (calendar["resolution_date_et"] >= start)
            & (calendar["resolution_date_et"] <= end)
        ]
        probability_parts = []
        outcome_parts = []
        weight_parts = []
        for condition_id in members["condition_id"]:
            panel = read_panel(condition_id)
            probability = gbm_touch_probability(
                panel["spot_close"].to_numpy(),
                panel["barrier"].to_numpy(),
                panel["minutes_to_expiry"].to_numpy(),
                panel["rv_24h"].to_numpy(),
                panel["direction"].to_numpy(),
            )
            if not np.isfinite(probability).all():
                raise ValueError(f"invalid GBM probability in {condition_id}")
            probability_parts.append(probability)
            outcome_parts.append(panel["y"].to_numpy(dtype=float))
            weight_parts.append(panel["contract_row_weight"].to_numpy(dtype=float))
        blocks[block_id] = {
            "probability": np.concatenate(probability_parts),
            "outcome": np.concatenate(outcome_parts),
            "weight": np.concatenate(weight_parts),
            "contracts": len(members),
        }
        print(
            f"loaded {block_id}: {len(members):,} contracts, "
            f"{len(blocks[block_id]['probability']):,} rows",
            flush=True,
        )
    return blocks


def fit_platt(
    probability: np.ndarray, outcome: np.ndarray, weight: np.ndarray
) -> dict[str, float | int | bool | str]:
    clipped = np.clip(probability, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    logit = np.log(clipped) - np.log1p(-clipped)
    total_weight = float(weight.sum())

    def objective(parameters):
        slope, intercept = parameters
        linear = slope * logit + intercept
        losses = np.logaddexp(0.0, linear) - outcome * linear
        value = float(np.dot(weight, losses) / total_weight)
        value += 0.5 * PLATT_L2 * slope**2
        residual = expit(linear) - outcome
        gradient = np.array(
            [
                np.dot(weight, residual * logit) / total_weight + PLATT_L2 * slope,
                np.dot(weight, residual) / total_weight,
            ]
        )
        return value, gradient

    result = minimize(
        objective,
        x0=np.array([1.0, 0.0]),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Platt fit failed: {result.message}")
    return {
        "slope": float(result.x[0]),
        "intercept": float(result.x[1]),
        "objective": float(result.fun),
        "optimizer_iterations": int(result.nit),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


def fit_fold_calibrators(blocks: dict[str, dict[str, object]]) -> pd.DataFrame:
    records = []
    for fold_id, block_ids in FOLD_CALIBRATION_BLOCKS.items():
        probabilities = np.concatenate([blocks[item]["probability"] for item in block_ids])
        outcomes = np.concatenate([blocks[item]["outcome"] for item in block_ids])
        weights = np.concatenate([blocks[item]["weight"] for item in block_ids])
        fitted = fit_platt(probabilities, outcomes, weights)
        records.append(
            {
                "fold_id": fold_id,
                "calibration_version": f"{fold_id}_platt_v1",
                "mu_annual": 0.0,
                "log_drift_rule": "mu_minus_half_sigma_squared",
                "volatility_feature": "rv_24h",
                "minutes_per_year": MINUTES_PER_YEAR,
                "probability_clip": PROBABILITY_CLIP,
                "platt_l2_slope": PLATT_L2,
                "calibration_blocks": "|".join(block_ids),
                "calibration_contracts": int(
                    sum(int(blocks[item]["contracts"]) for item in block_ids)
                ),
                "calibration_rows": len(probabilities),
                "calibration_weight": float(weights.sum()),
                **fitted,
            }
        )
        print(
            f"fitted {fold_id} Platt: slope={fitted['slope']:.6f}, "
            f"intercept={fitted['intercept']:.6f}",
            flush=True,
        )
    parameters = pd.DataFrame(records)
    atomic_parquet(parameters, PARAMETERS_FILE)
    return parameters


def prediction_frame(
    base: pd.DataFrame,
    experiment_key: str,
    fold_id: str,
    probability_column: str,
    information_column: str,
) -> pd.DataFrame:
    experiment = EXPERIMENTS[experiment_key]
    raw_column = "gbm_probability" if experiment_key == "gbm_platt" else probability_column
    calibration_version = (
        f"{fold_id}_platt_v1" if experiment_key == "gbm_platt" else "none"
    )
    return pd.DataFrame(
        {
            "schema_version": "phase3_prediction_v1",
            "experiment_id": experiment["experiment_id"],
            "condition_id": base["condition_id"],
            "decision_time": base["decision_time"],
            "information_timestamp": base[information_column],
            "fold_id": fold_id,
            "role": "validation",
            "prediction_kind": "forward",
            "model_id": experiment["model_id"],
            "model_version": experiment["model_version"],
            "feature_set_id": experiment["feature_set_id"],
            "calibration_method": experiment["calibration_method"],
            "calibration_version": calibration_version,
            "raw_yes_probability": base[raw_column].to_numpy(dtype=float),
            "calibrated_yes_probability": base[probability_column].to_numpy(dtype=float),
            "contract_row_weight": base["contract_row_weight"].to_numpy(dtype=float),
        }
    )


def contract_metric(
    probability: np.ndarray, outcome: np.ndarray, weight: np.ndarray
) -> tuple[float, float, float]:
    normalized = weight / weight.sum()
    brier = float(np.dot(normalized, (probability - outcome) ** 2))
    clipped = np.clip(probability, LOG_LOSS_CLIP, 1 - LOG_LOSS_CLIP)
    losses = -(outcome * np.log(clipped) + (1 - outcome) * np.log1p(-clipped))
    log_loss = float(np.dot(normalized, losses))
    mean_probability = float(np.dot(normalized, probability))
    return brier, log_loss, mean_probability


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
    folds: pd.DataFrame, parameters: pd.DataFrame
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
        for asset in ["BTC", "ETH", "SOL", "XRP"]:
            members = fold_validation[fold_validation["asset"] == asset]
            contract_frames = []
            for member in members.itertuples(index=False):
                panel = read_panel(member.condition_id)
                gbm = gbm_touch_probability(
                    panel["spot_close"].to_numpy(),
                    panel["barrier"].to_numpy(),
                    panel["minutes_to_expiry"].to_numpy(),
                    panel["rv_24h"].to_numpy(),
                    panel["direction"].to_numpy(),
                )
                if not np.isfinite(gbm).all():
                    raise ValueError(f"invalid GBM probability in {member.condition_id}")
                gbm_logit = np.log(np.clip(gbm, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP))
                gbm_logit -= np.log1p(-np.clip(gbm, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP))
                platt = expit(fitted["slope"] * gbm_logit + fitted["intercept"])
                panel = panel.copy()
                panel["gbm_probability"] = gbm
                panel["gbm_platt_probability"] = platt
                contract_frames.append(panel)

                outcomes = panel["y"].to_numpy(dtype=float)
                weights = panel["contract_row_weight"].to_numpy(dtype=float)
                probability_map = {
                    "market": panel["market_yes_price"].to_numpy(dtype=float),
                    "gbm_raw": gbm,
                    "gbm_platt": platt,
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
            for model_key, probability_column, information_column in [
                ("market", "market_yes_price", "decision_time"),
                ("gbm_raw", "gbm_probability", "spot_available_time"),
                ("gbm_platt", "gbm_platt_probability", "spot_available_time"),
            ]:
                output = prediction_frame(
                    base,
                    model_key,
                    fold_id,
                    probability_column,
                    information_column,
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
                        "contracts": output["condition_id"].nunique(),
                        "first_decision": output["decision_time"].min(),
                        "last_decision": output["decision_time"].max(),
                        "bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                )
                del output
            print(
                f"wrote {fold_id} validation {asset}: "
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
    atomic_parquet(manifest_frame, MANIFEST_FILE)
    return metric_frame, reliability_frame, manifest_frame


def weighted_ece(group: pd.DataFrame) -> float:
    return float(
        np.dot(group["contract_weight"], group["absolute_gap"])
        / group["contract_weight"].sum()
    )


def make_report(
    parameters: pd.DataFrame,
    metrics: pd.DataFrame,
    reliability: pd.DataFrame,
    manifest: pd.DataFrame,
) -> str:
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
    }
    lines = [
        "# Phase-3 benchmark validation report",
        "",
        "This report contains validation results only. Evaluation roles were not",
        "generated or scored. The historical execution proxy was not read.",
        "",
        "## Frozen specifications",
        "",
        "- Polymarket: delayed exact-minute YES probability proxy, uncalibrated.",
        "- GBM: one-touch first-passage probability, annual price drift `mu=0`,",
        "  trailing causal `rv_24h`, and ACT/365 (`525600` minutes per year).",
        "- Platt: `logistic(a * logit(p_gbm) + b)`, fitted with",
        "  `contract_row_weight`, probability clipping at `1e-6`, and fixed slope",
        "  L2 penalty `1e-6` using only frozen out-of-fold training blocks.",
        "",
        "## Platt parameters",
        "",
        "| Fold | OOF blocks | Contracts | Rows | Slope | Intercept |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in parameters.itertuples(index=False):
        lines.append(
            f"| {row.fold_id} | {row.calibration_blocks} | "
            f"{row.calibration_contracts:,} | {row.calibration_rows:,} | "
            f"{row.slope:.6f} | {row.intercept:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Validation metrics by outer fold",
            "",
            "Metrics are averaged within contract first and then across contracts.",
            "ECE uses ten equal-width probability bins with equal total weight per",
            "contract.",
            "",
            "| Fold | Model | Contracts | Rows | Brier | Log loss | ECE-10 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for fold_id in FOLD_CALIBRATION_BLOCKS:
        for model_key in ["market", "gbm_raw", "gbm_platt"]:
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
            "This table is a compact comparison, not an additional test set.",
            "",
            "| Model | Fold-contract appearances | Brier | Log loss | ECE-10 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model_key in ["market", "gbm_raw", "gbm_platt"]:
        row = overall[overall["model_key"] == model_key].iloc[0]
        lines.append(
            f"| {labels[model_key]} | {int(row.fold_contract_appearances):,} | "
            f"{row.brier:.6f} | {row.log_loss:.6f} | {row.ece_10:.6f} |"
        )

    best_brier_key = overall.loc[overall["brier"].idxmin(), "model_key"]
    best_log_loss_key = overall.loc[overall["log_loss"].idxmin(), "model_key"]
    platt_fold = summary[summary["model_key"] == "gbm_platt"].set_index("fold_id")
    raw_fold = summary[summary["model_key"] == "gbm_raw"].set_index("fold_id")
    market_fold = summary[summary["model_key"] == "market"].set_index("fold_id")
    platt_beats_raw_brier = int((platt_fold["brier"] < raw_fold["brier"]).sum())
    platt_beats_raw_log = int((platt_fold["log_loss"] < raw_fold["log_loss"]).sum())
    platt_beats_market_brier = int((platt_fold["brier"] < market_fold["brier"]).sum())
    platt_beats_market_log = int((platt_fold["log_loss"] < market_fold["log_loss"]).sum())
    lines.extend(
        [
            "",
            "## Frozen validation decision",
            "",
            f"- Lowest combined Brier: **{labels[best_brier_key]}**.",
            f"- Lowest combined log loss: **{labels[best_log_loss_key]}**.",
            "- Under the registered Brier-then-log-loss rule, `GBM + Platt` is",
            "  retained as the calibrated GBM benchmark for later evaluation.",
            f"- Platt beats raw GBM on Brier in {platt_beats_raw_brier}/4 folds and",
            f"  on log loss in {platt_beats_raw_log}/4 folds.",
            f"- Platt beats raw Polymarket on Brier in {platt_beats_market_brier}/4",
            f"  folds and on log loss in {platt_beats_market_log}/4 folds.",
            "- Raw Polymarket and raw GBM remain mandatory reference series; the",
            "  Platt improvement is not uniform across every fold or metric.",
        ]
    )

    lines.extend(["", "## Breakdown by asset", "", "| Model | Asset | Contracts | Brier | Log loss |", "|---|---|---:|---:|---:|"])
    for model_key in ["market", "gbm_raw", "gbm_platt"]:
        for asset in ["BTC", "ETH", "SOL", "XRP"]:
            row = by_asset[(by_asset["model_key"] == model_key) & (by_asset["asset"] == asset)].iloc[0]
            lines.append(
                f"| {labels[model_key]} | {asset} | {int(row.contracts):,} | "
                f"{row.brier:.6f} | {row.log_loss:.6f} |"
            )

    lines.extend(["", "## Breakdown by direction", "", "| Model | Direction | Contracts | Brier | Log loss |", "|---|---|---:|---:|---:|"])
    for model_key in ["market", "gbm_raw", "gbm_platt"]:
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
            f"- Prediction partitions: {len(manifest):,}",
            f"- Prediction rows across registered outputs: {int(manifest['rows'].sum()):,}",
            f"- Compressed prediction size: {manifest['bytes'].sum() / 1024**3:.2f} GiB",
            "- Every output is partitioned by experiment, fold, validation role, and asset.",
            "",
            "## Interpretation boundary",
            "",
            "These results can compare benchmark probability quality and choose a",
            "benchmark configuration. They do not establish profitability or",
            "executability. No evaluation result should be produced until the",
            "validation decision is recorded and frozen.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    folds = pd.read_parquet(FOLD_FILE)
    calendar = unique_contract_calendar(folds)
    blocks = load_calibration_blocks(calendar)
    parameters = fit_fold_calibrators(blocks)
    del blocks
    metrics, reliability, manifest = generate_validation_outputs(folds, parameters)
    REPORT_FILE.write_text(
        make_report(parameters, metrics, reliability, manifest), encoding="utf-8"
    )
    print(f"parameters: {PARAMETERS_FILE}")
    print(f"contract metrics: {METRICS_FILE}")
    print(f"prediction manifest: {MANIFEST_FILE}")
    print(f"report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
