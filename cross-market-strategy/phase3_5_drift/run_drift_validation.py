#!/usr/bin/env python3
"""Generate and compare the six frozen Phase-3.5 drift validation trials."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit


FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parent
PHASE3 = PROJECT / "phase3_models"
if str(PHASE3) not in sys.path:
    sys.path.insert(0, str(PHASE3))
if str(FOLDER) not in sys.path:
    sys.path.insert(0, str(FOLDER))

from fit_drift_models import (  # noqa: E402
    BASE_PANEL_COLUMNS,
    CALIBRATORS_FILE,
    PARAMETERS_FILE,
    drift_map,
    gbm_touch_probability_drift,
    models_from_parameters,
    predict_har_drift,
)
from run_benchmarks import (  # noqa: E402
    PROBABILITY_CLIP,
    RELIABILITY_BINS,
    atomic_parquet,
    contract_metric,
    file_sha256,
    weighted_ece,
)
from run_dvol_ablation import (  # noqa: E402
    ASSETS as DVOL_ASSETS,
    DVOL_FEATURE_NAMES,
    augment_with_dvol,
    load_dvol,
)
from run_empirical import (  # noqa: E402
    ASSET_SYMBOLS,
    FOLD_CALIBRATION_BLOCKS,
    FOLD_DISTRIBUTION_CUTOFF,
    FOLD_FILE,
    PANEL_DIR,
)
from run_har import FEATURE_NAMES, prepare_asset_data  # noqa: E402


OUTPUT_DIR = FOLDER / "drift_predictions"
METRICS_FILE = FOLDER / "drift_contract_metrics.parquet"
RELIABILITY_FILE = FOLDER / "drift_reliability.parquet"
MANIFEST_FILE = FOLDER / "drift_prediction_manifest.parquet"
COMPARISON_FILE = FOLDER / "drift_comparison_metrics.parquet"
REPORT_FILE = FOLDER / "drift_validation_report.md"

DVOL_PANEL_COLUMNS = [
    "dvol_available_time",
    "dvol_age_minutes",
    "dvol_sigma",
    "dvol_available",
]
PANEL_COLUMNS = list(dict.fromkeys(BASE_PANEL_COLUMNS + DVOL_PANEL_COLUMNS))

EXPERIMENTS = {
    "gbm_drift_raw": {
        "experiment_id": "E11D-GBM-DRIFT-RAW-V1",
        "model_id": "gbm_causal_drift_rv24h_v1",
        "feature_set_id": "gbm_spot_barrier_rv24h_drift365_v1",
        "calibration_method": "raw",
        "branch": "gbm_drift",
        "assets": tuple(ASSET_SYMBOLS),
    },
    "gbm_drift_platt": {
        "experiment_id": "E11D-GBM-DRIFT-PLATT-V1",
        "model_id": "gbm_causal_drift_rv24h_v1",
        "feature_set_id": "gbm_spot_barrier_rv24h_drift365_v1",
        "calibration_method": "platt",
        "branch": "gbm_drift",
        "assets": tuple(ASSET_SYMBOLS),
    },
    "har_drift_raw": {
        "experiment_id": "E11D-HAR-DRIFT-RAW-V1",
        "model_id": "har_range_causal_drift_v1",
        "feature_set_id": "har_d1_w7_m30_pk24_gk24_drift365_v1",
        "calibration_method": "raw",
        "branch": "har_drift",
        "assets": tuple(ASSET_SYMBOLS),
    },
    "har_drift_platt": {
        "experiment_id": "E11D-HAR-DRIFT-PLATT-V1",
        "model_id": "har_range_causal_drift_v1",
        "feature_set_id": "har_d1_w7_m30_pk24_gk24_drift365_v1",
        "calibration_method": "platt",
        "branch": "har_drift",
        "assets": tuple(ASSET_SYMBOLS),
    },
    "har_dvol_drift_raw": {
        "experiment_id": "E11D-HAR-DVOL-DRIFT-RAW-V1",
        "model_id": "har_range_btceth_dvol_causal_drift_v1",
        "feature_set_id": "har_d1_w7_m30_pk24_gk24_dvol_drift365_v1",
        "calibration_method": "raw",
        "branch": "har_dvol_drift",
        "assets": DVOL_ASSETS,
    },
    "har_dvol_drift_platt": {
        "experiment_id": "E11D-HAR-DVOL-DRIFT-PLATT-V1",
        "model_id": "har_range_btceth_dvol_causal_drift_v1",
        "feature_set_id": "har_d1_w7_m30_pk24_gk24_dvol_drift365_v1",
        "calibration_method": "platt",
        "branch": "har_dvol_drift",
        "assets": DVOL_ASSETS,
    },
}

PAIRS = {
    "gbm_drift_raw": ("benchmark_contract_metrics.parquet", "gbm_raw"),
    "gbm_drift_platt": ("benchmark_contract_metrics.parquet", "gbm_platt"),
    "har_drift_raw": ("har_contract_metrics.parquet", "har_raw"),
    "har_drift_platt": ("har_contract_metrics.parquet", "har_platt"),
    "har_dvol_drift_raw": ("dvol_contract_metrics.parquet", "btceth_dvol_raw"),
    "har_dvol_drift_platt": (
        "dvol_contract_metrics.parquet",
        "btceth_dvol_platt",
    ),
}

LABELS = {
    "gbm_drift_raw": "GBM drift raw",
    "gbm_drift_platt": "GBM drift + Platt",
    "har_drift_raw": "HAR drift raw",
    "har_drift_platt": "HAR drift + Platt",
    "har_dvol_drift_raw": "BTC/ETH HAR+DVOL drift raw",
    "har_dvol_drift_platt": "BTC/ETH HAR+DVOL drift + Platt",
}

CONTROL_LABELS = {
    "gbm_raw": "GBM zero-drift raw",
    "gbm_platt": "GBM zero-drift + Platt",
    "har_raw": "HAR zero-drift raw",
    "har_platt": "HAR zero-drift + Platt",
    "btceth_dvol_raw": "BTC/ETH HAR+DVOL zero-drift raw",
    "btceth_dvol_platt": "BTC/ETH HAR+DVOL zero-drift + Platt",
}


def apply_platt(probability: np.ndarray, calibrator: dict[str, object]) -> np.ndarray:
    clipped = np.clip(probability, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    logit = np.log(clipped) - np.log1p(-clipped)
    return expit(float(calibrator["slope"]) * logit + float(calibrator["intercept"]))


def update_reliability(
    accumulator: dict[tuple[str, str, int], np.ndarray],
    fold_id: str,
    model_key: str,
    probability: np.ndarray,
    outcome: np.ndarray,
    weight: np.ndarray,
) -> None:
    bins = np.minimum(
        (probability * RELIABILITY_BINS).astype(int), RELIABILITY_BINS - 1
    )
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
    calibration_version: str,
) -> pd.DataFrame:
    experiment = EXPERIMENTS[model_key]
    if experiment["branch"] == "har_dvol_drift":
        information_timestamp = pd.concat(
            [base["spot_available_time"], base["dvol_available_time"]], axis=1
        ).max(axis=1)
    else:
        information_timestamp = base["spot_available_time"]
    return pd.DataFrame(
        {
            "schema_version": "phase3_prediction_v1",
            "experiment_id": experiment["experiment_id"],
            "condition_id": base["condition_id"],
            "decision_time": base["decision_time"],
            "information_timestamp": information_timestamp,
            "fold_id": fold_id,
            "role": "validation",
            "prediction_kind": "forward",
            "model_id": experiment["model_id"],
            "model_version": "v1",
            "feature_set_id": experiment["feature_set_id"],
            "calibration_method": experiment["calibration_method"],
            "calibration_version": calibration_version,
            "raw_yes_probability": base[raw_column].to_numpy(dtype=float),
            "calibrated_yes_probability": base[probability_column].to_numpy(
                dtype=float
            ),
            "contract_row_weight": base["contract_row_weight"].to_numpy(
                dtype=float
            ),
        }
    )


def load_frozen_inputs():
    drift_parameters = pd.read_parquet(PARAMETERS_FILE)
    calibrators = pd.read_parquet(CALIBRATORS_FILE)
    if len(drift_parameters) != 20 or len(calibrators) != 12:
        raise ValueError("frozen drift fitting artifacts have unexpected row counts")
    if not calibrators["optimizer_success"].all():
        raise ValueError("one or more frozen Platt fits did not converge")
    drifts = drift_map(drift_parameters)
    calibrator_map = calibrators.set_index(["branch", "fold_id"]).to_dict("index")

    base_assets = {asset: prepare_asset_data(asset) for asset in ASSET_SYMBOLS}
    har_parameters = pd.read_parquet(PHASE3 / "har_parameters.parquet")
    har_models = models_from_parameters(har_parameters, FEATURE_NAMES)

    dvol_assets = {}
    for asset in DVOL_ASSETS:
        source_time, sigma = load_dvol(asset)
        dvol_assets[asset], _ = augment_with_dvol(
            base_assets[asset], source_time, sigma
        )
    dvol_parameters = pd.read_parquet(PHASE3 / "dvol_parameters.parquet")
    dvol_parameters = dvol_parameters[
        dvol_parameters["variant"] == "btceth_dvol"
    ]
    dvol_models = models_from_parameters(dvol_parameters, DVOL_FEATURE_NAMES)
    return drifts, calibrator_map, base_assets, har_models, dvol_assets, dvol_models


def generate_validation_outputs(
    folds: pd.DataFrame,
    drifts: dict[tuple[str, str], float],
    calibrators: dict[tuple[str, str], dict[str, object]],
    base_assets: dict[str, object],
    har_models: dict[str, dict[int, dict[str, object]]],
    dvol_assets: dict[str, object],
    dvol_models: dict[str, dict[int, dict[str, object]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    reliability = defaultdict(lambda: np.zeros(4, dtype=float))

    for fold_id in FOLD_CALIBRATION_BLOCKS:
        validation = folds[
            (folds["fold_id"] == fold_id) & (folds["role"] == "validation")
        ]
        cutoff_id = FOLD_DISTRIBUTION_CUTOFF[fold_id]
        for asset in ASSET_SYMBOLS:
            members = validation[validation["asset"] == asset]
            parts = []
            for member in members.itertuples(index=False):
                panel = pd.read_parquet(
                    PANEL_DIR / f"{member.condition_id}.parquet",
                    columns=PANEL_COLUMNS,
                )
                annual_drift = drifts[(cutoff_id, asset)]
                gbm_raw = gbm_touch_probability_drift(
                    panel["spot_close"].to_numpy(dtype=float),
                    panel["barrier"].to_numpy(dtype=float),
                    panel["minutes_to_expiry"].to_numpy(dtype=float),
                    panel["rv_24h"].to_numpy(dtype=float),
                    panel["direction"].to_numpy(),
                    annual_drift,
                )
                har_raw = predict_har_drift(
                    panel,
                    base_assets[asset],
                    har_models[cutoff_id],
                    annual_drift,
                )
                raw_probability = {
                    "gbm_drift": gbm_raw,
                    "har_drift": har_raw,
                }
                if asset in DVOL_ASSETS:
                    raw_probability["har_dvol_drift"] = predict_har_drift(
                        panel,
                        dvol_assets[asset],
                        dvol_models[cutoff_id],
                        annual_drift,
                        verify_dvol_features=True,
                    )
                if not all(np.isfinite(value).all() for value in raw_probability.values()):
                    raise ValueError(f"invalid drift probability in {member.condition_id}")

                panel = panel.copy()
                for branch, raw in raw_probability.items():
                    fitted = calibrators[(branch, fold_id)]
                    panel[f"{branch}_raw"] = raw
                    panel[f"{branch}_platt"] = apply_platt(raw, fitted)
                parts.append(panel)

                outcome = panel["y"].to_numpy(dtype=float)
                weight = panel["contract_row_weight"].to_numpy(dtype=float)
                for model_key, experiment in EXPERIMENTS.items():
                    if asset not in experiment["assets"]:
                        continue
                    branch = experiment["branch"]
                    suffix = experiment["calibration_method"]
                    probability = panel[f"{branch}_{suffix}"].to_numpy(dtype=float)
                    brier, log_loss, mean_probability = contract_metric(
                        probability, outcome, weight
                    )
                    metrics.append(
                        {
                            "fold_id": fold_id,
                            "role": "validation",
                            "model_key": model_key,
                            "experiment_id": experiment["experiment_id"],
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
            for model_key, experiment in EXPERIMENTS.items():
                if asset not in experiment["assets"]:
                    continue
                branch = experiment["branch"]
                method = experiment["calibration_method"]
                raw_column = f"{branch}_raw"
                probability_column = raw_column if method == "raw" else f"{branch}_platt"
                calibration_version = (
                    "none"
                    if method == "raw"
                    else str(calibrators[(branch, fold_id)]["calibration_version"])
                )
                output = prediction_frame(
                    base,
                    model_key,
                    fold_id,
                    raw_column,
                    probability_column,
                    calibration_version,
                )
                path = (
                    OUTPUT_DIR
                    / experiment["experiment_id"]
                    / fold_id
                    / "validation"
                    / f"{asset}.parquet"
                )
                atomic_parquet(output, path)
                manifests.append(
                    {
                        "experiment_id": experiment["experiment_id"],
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
            print(
                f"wrote drift {fold_id} validation {asset}: "
                f"{len(base):,} rows, {len(members):,} contracts",
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
    atomic_parquet(manifest_frame, MANIFEST_FILE)
    return metric_frame, reliability_frame, manifest_frame


def build_matched_comparison(drift_metrics: pd.DataFrame) -> pd.DataFrame:
    key_columns = [
        "fold_id",
        "condition_id",
        "asset",
        "direction",
        "n_rows",
        "outcome",
    ]
    rows = []
    for drift_key, (control_file, control_key) in PAIRS.items():
        drift = drift_metrics[drift_metrics["model_key"] == drift_key].copy()
        control_all = pd.read_parquet(PHASE3 / control_file)
        control = control_all[control_all["model_key"] == control_key].copy()
        merged = drift.merge(
            control,
            on=key_columns,
            how="outer",
            suffixes=("_drift", "_control"),
            indicator=True,
            validate="one_to_one",
        )
        if not merged["_merge"].eq("both").all() or len(merged) != len(drift):
            raise ValueError(f"unmatched validation rows for {drift_key}")
        selected = merged[
            key_columns
            + [
                "brier_drift",
                "brier_control",
                "log_loss_drift",
                "log_loss_control",
                "mean_probability_drift",
                "mean_probability_control",
            ]
        ].copy()
        selected.insert(0, "drift_model_key", drift_key)
        selected.insert(1, "control_model_key", control_key)
        selected.insert(2, "experiment_id", EXPERIMENTS[drift_key]["experiment_id"])
        selected["brier_delta_drift_minus_control"] = (
            selected["brier_drift"] - selected["brier_control"]
        )
        selected["log_loss_delta_drift_minus_control"] = (
            selected["log_loss_drift"] - selected["log_loss_control"]
        )
        rows.append(selected)
    comparison = pd.concat(rows, ignore_index=True)
    atomic_parquet(comparison, COMPARISON_FILE)
    return comparison


def select_winner(brier_delta: float, log_loss_delta: float) -> str:
    if not np.isclose(brier_delta, 0.0, rtol=0, atol=1e-15):
        return "drift" if brier_delta < 0 else "zero_drift"
    return "drift" if log_loss_delta < 0 else "zero_drift"


def ece_table(reliability: pd.DataFrame) -> pd.DataFrame:
    return (
        reliability.groupby("model_key")
        .apply(weighted_ece, include_groups=False)
        .rename("ece_10")
        .reset_index()
    )


def make_report(
    drift_metrics: pd.DataFrame,
    drift_reliability: pd.DataFrame,
    comparison: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[str, dict[str, str]]:
    overall = comparison.groupby(
        ["drift_model_key", "control_model_key"], as_index=False
    ).agg(
        fold_contract_appearances=("condition_id", "size"),
        drift_brier=("brier_drift", "mean"),
        control_brier=("brier_control", "mean"),
        brier_delta=("brier_delta_drift_minus_control", "mean"),
        drift_log_loss=("log_loss_drift", "mean"),
        control_log_loss=("log_loss_control", "mean"),
        log_loss_delta=("log_loss_delta_drift_minus_control", "mean"),
    )
    by_fold = comparison.groupby(
        ["drift_model_key", "control_model_key", "fold_id"], as_index=False
    ).agg(
        contracts=("condition_id", "size"),
        drift_brier=("brier_drift", "mean"),
        control_brier=("brier_control", "mean"),
        brier_delta=("brier_delta_drift_minus_control", "mean"),
        drift_log_loss=("log_loss_drift", "mean"),
        control_log_loss=("log_loss_control", "mean"),
        log_loss_delta=("log_loss_delta_drift_minus_control", "mean"),
    )
    by_asset = comparison.groupby(
        ["drift_model_key", "asset"], as_index=False
    ).agg(
        contracts=("condition_id", "size"),
        brier_delta=("brier_delta_drift_minus_control", "mean"),
        log_loss_delta=("log_loss_delta_drift_minus_control", "mean"),
    )
    by_direction = comparison.groupby(
        ["drift_model_key", "direction"], as_index=False
    ).agg(
        contracts=("condition_id", "size"),
        brier_delta=("brier_delta_drift_minus_control", "mean"),
        log_loss_delta=("log_loss_delta_drift_minus_control", "mean"),
    )
    drift_ece = ece_table(drift_reliability).set_index("model_key")["ece_10"]
    control_reliability = pd.concat(
        [
            pd.read_parquet(PHASE3 / "benchmark_reliability.parquet"),
            pd.read_parquet(PHASE3 / "har_reliability.parquet"),
            pd.read_parquet(PHASE3 / "dvol_reliability.parquet"),
        ],
        ignore_index=True,
    )
    control_ece = ece_table(control_reliability).set_index("model_key")["ece_10"]

    decisions = {}
    lines = [
        "# Phase 3.5 matched drift validation report",
        "",
        "This is the predeclared outer-validation comparison. Each estimated-drift "
        "trial is matched to its frozen zero-drift control on identical contracts "
        "and decision rows. Evaluation roles and the Phase-4 execution proxy were "
        "not opened.",
        "",
        "Negative deltas favour estimated drift. Selection uses combined "
        "contract-weighted Brier score first and log loss only as the tie-breaker.",
        "",
        "## Combined matched results",
        "",
        "| Drift trial | Control | Appearances | Control Brier | Drift Brier | Delta | Control log loss | Drift log loss | Delta | Control ECE-10 | Drift ECE-10 | Decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in overall.itertuples(index=False):
        winner = select_winner(row.brier_delta, row.log_loss_delta)
        decisions[row.drift_model_key] = winner
        lines.append(
            f"| {LABELS[row.drift_model_key]} | {CONTROL_LABELS[row.control_model_key]} | "
            f"{row.fold_contract_appearances:,} | {row.control_brier:.6f} | "
            f"{row.drift_brier:.6f} | {row.brier_delta:+.6f} | "
            f"{row.control_log_loss:.6f} | {row.drift_log_loss:.6f} | "
            f"{row.log_loss_delta:+.6f} | {control_ece[row.control_model_key]:.6f} | "
            f"{drift_ece[row.drift_model_key]:.6f} | "
            f"**{winner.replace('_', ' ')}** |"
        )

    lines.extend(
        [
            "",
            "## Fold stability",
            "",
            "| Drift trial | Fold | Contracts | Brier delta | Log-loss delta | Fold winner |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in by_fold.itertuples(index=False):
        winner = select_winner(row.brier_delta, row.log_loss_delta)
        lines.append(
            f"| {LABELS[row.drift_model_key]} | {row.fold_id} | {row.contracts:,} | "
            f"{row.brier_delta:+.6f} | {row.log_loss_delta:+.6f} | "
            f"{winner.replace('_', ' ')} |"
        )

    lines.extend(
        [
            "",
            "## Asset stability",
            "",
            "| Drift trial | Asset | Contracts | Brier delta | Log-loss delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in by_asset.itertuples(index=False):
        lines.append(
            f"| {LABELS[row.drift_model_key]} | {row.asset} | {row.contracts:,} | "
            f"{row.brier_delta:+.6f} | {row.log_loss_delta:+.6f} |"
        )

    lines.extend(
        [
            "",
            "## Direction stability",
            "",
            "| Drift trial | Direction | Contracts | Brier delta | Log-loss delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in by_direction.itertuples(index=False):
        lines.append(
            f"| {LABELS[row.drift_model_key]} | {row.direction} | "
            f"{row.contracts:,} | {row.brier_delta:+.6f} | "
            f"{row.log_loss_delta:+.6f} |"
        )

    lines.extend(["", "## Frozen decisions", ""])
    for key in EXPERIMENTS:
        row = overall[overall["drift_model_key"] == key].iloc[0]
        winner = decisions[key]
        fold_rows = by_fold[by_fold["drift_model_key"] == key]
        brier_fold_wins = int((fold_rows["brier_delta"] < 0).sum())
        log_fold_wins = int((fold_rows["log_loss_delta"] < 0).sum())
        lines.append(
            f"- `{EXPERIMENTS[key]['experiment_id']}`: **{winner.replace('_', ' ')}**; "
            f"combined Brier delta {row.brier_delta:+.6f}, log-loss delta "
            f"{row.log_loss_delta:+.6f}; drift wins {brier_fold_wins}/4 folds on "
            f"Brier and {log_fold_wins}/4 on log loss."
        )
    lines.extend(
        [
            "",
            "These decisions select probability specifications, not trading "
            "strategies. They establish neither profitability nor execution "
            "quality.",
            "",
            "## Artifact audit",
            "",
            f"- Drift contract-metric rows: {len(drift_metrics):,}.",
            f"- Matched comparison rows: {len(comparison):,}.",
            f"- Prediction partitions: {len(manifest):,}.",
            f"- Prediction rows across the six trials: {int(manifest['rows'].sum()):,}.",
            f"- Compressed prediction size: {manifest['bytes'].sum() / 1024**3:.2f} GiB.",
            "- Every comparison passed an exact one-to-one match on fold, contract, "
            "asset, direction, row count, and outcome.",
            "- Evaluation roles and `execution_proxy_60s/` remain unopened.",
            "",
        ]
    )
    return "\n".join(lines), decisions


def main() -> int:
    folds = pd.read_parquet(FOLD_FILE)
    frozen = load_frozen_inputs()
    metrics, reliability, manifest = generate_validation_outputs(folds, *frozen)
    comparison = build_matched_comparison(metrics)
    report, decisions = make_report(metrics, reliability, comparison, manifest)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print("frozen decisions:", flush=True)
    for key, decision in decisions.items():
        print(f"  {EXPERIMENTS[key]['experiment_id']}: {decision}", flush=True)
    print(f"metrics: {METRICS_FILE}", flush=True)
    print(f"comparison: {COMPARISON_FILE}", flush=True)
    print(f"manifest: {MANIFEST_FILE}", flush=True)
    print(f"report: {REPORT_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
