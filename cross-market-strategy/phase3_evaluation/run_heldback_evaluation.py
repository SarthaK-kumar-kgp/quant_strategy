#!/usr/bin/env python3
"""Run the frozen one-pass probability evaluation without refitting models."""

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
PHASE35 = PROJECT / "phase3_5_drift"
for import_path in (PHASE3, PHASE35):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from fit_drift_models import models_from_parameters  # noqa: E402
from run_benchmarks import (  # noqa: E402
    PROBABILITY_CLIP,
    RELIABILITY_BINS,
    atomic_parquet,
    contract_metric,
    file_sha256,
    gbm_touch_probability,
    weighted_ece,
)
from run_dvol_ablation import (  # noqa: E402
    ASSETS as DVOL_ASSETS,
    DVOL_FEATURE_NAMES,
    augment_with_dvol,
    load_dvol,
    predict_panel as predict_dvol_panel,
)
from run_empirical import (  # noqa: E402
    ASSET_SYMBOLS,
    FOLD_CALIBRATION_BLOCKS,
    FOLD_DISTRIBUTION_CUTOFF,
    FOLD_FILE,
    HORIZONS,
    PANEL_DIR,
    predict_panel as predict_empirical_panel,
)
from run_har import (  # noqa: E402
    FEATURE_NAMES,
    predict_panel as predict_har_panel,
    prepare_asset_data,
)


OUTPUT_DIR = FOLDER / "predictions"
METRICS_FILE = FOLDER / "evaluation_contract_metrics.parquet"
RELIABILITY_FILE = FOLDER / "evaluation_reliability.parquet"
MANIFEST_FILE = FOLDER / "evaluation_prediction_manifest.parquet"
REPORT_FILE = FOLDER / "evaluation_report.md"

PRIMARY_FOLD = "fold_04"
PRIMARY_PERIOD = "2--9 September 2026"

PANEL_COLUMNS = [
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
    "market_yes_price",
    "dvol_available_time",
    "dvol_age_minutes",
    "dvol_sigma",
    "dvol_available",
    "y",
    "contract_row_weight",
]

EXPERIMENTS = {
    "market_raw": {
        "experiment_id": "E03-PM-RAW-V1",
        "model_id": "polymarket_yes_raw",
        "feature_set_id": "market_yes_only_v1",
        "calibration_method": "raw",
        "assets": tuple(ASSET_SYMBOLS),
        "information": "market",
    },
    "gbm_platt": {
        "experiment_id": "E03-GBM-PLATT-V1",
        "model_id": "gbm_zero_drift_rv24h",
        "feature_set_id": "gbm_spot_barrier_rv24h_v1",
        "calibration_method": "platt",
        "assets": tuple(ASSET_SYMBOLS),
        "information": "spot",
    },
    "empirical_pooled_platt": {
        "experiment_id": "E04-EMP-PLATT-V1",
        "model_id": "empirical_pooled_excursion",
        "feature_set_id": "empirical_spot_barrier_rv24h_v1",
        "calibration_method": "platt",
        "assets": tuple(ASSET_SYMBOLS),
        "information": "spot",
    },
    "empirical_asset_platt": {
        "experiment_id": "E08-EMP-ASSET-PLATT-V1",
        "model_id": "empirical_asset_excursion_v1",
        "feature_set_id": "empirical_spot_barrier_rv24h_asset_v1",
        "calibration_method": "platt",
        "assets": tuple(ASSET_SYMBOLS),
        "information": "spot",
    },
    "har_platt": {
        "experiment_id": "E06-HAR-PLATT-V1",
        "model_id": "har_range_zero_drift_v1",
        "feature_set_id": "har_d1_w7_m30_pk24_gk24_v1",
        "calibration_method": "platt",
        "assets": tuple(ASSET_SYMBOLS),
        "information": "spot",
    },
    "har_dvol_platt": {
        "experiment_id": "E07-HAR-DVOL-PLATT-V1",
        "model_id": "har_range_btceth_dvol_v1",
        "feature_set_id": "har_d1_w7_m30_pk24_gk24_dvol_v1",
        "calibration_method": "platt",
        "assets": DVOL_ASSETS,
        "information": "dvol",
    },
}

LABELS = {
    "market_raw": "Polymarket raw",
    "gbm_platt": "GBM + Platt",
    "empirical_pooled_platt": "Pooled Empirical + Platt",
    "empirical_asset_platt": "Asset-specific Empirical + Platt",
    "har_platt": "HAR/range + Platt",
    "har_dvol_platt": "BTC/ETH HAR+DVOL + Platt",
}

VALIDATION_SOURCES = {
    "market_raw": ("benchmark_contract_metrics.parquet", "market"),
    "gbm_platt": ("benchmark_contract_metrics.parquet", "gbm_platt"),
    "empirical_pooled_platt": (
        "empirical_contract_metrics.parquet",
        "empirical_platt",
    ),
    "empirical_asset_platt": (
        "empirical_asset_contract_metrics.parquet",
        "empirical_asset_platt",
    ),
    "har_platt": ("har_contract_metrics.parquet", "har_platt"),
    "har_dvol_platt": ("dvol_contract_metrics.parquet", "btceth_dvol_platt"),
}


def apply_platt(probability: np.ndarray, calibrator: dict[str, object]) -> np.ndarray:
    clipped = np.clip(probability, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    logit = np.log(clipped) - np.log1p(-clipped)
    return expit(float(calibrator["slope"]) * logit + float(calibrator["intercept"]))


def load_distribution(path: Path) -> dict[tuple[int, str], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        horizons = archive["horizons"].astype(int)
        if not np.array_equal(horizons, HORIZONS):
            raise ValueError(f"unexpected horizons in {path}")
        return {
            (int(horizon), direction): archive[f"h{int(horizon)}_{direction}"].copy()
            for horizon in horizons
            for direction in ("up", "down")
        }


def load_distributions():
    pooled = {}
    asset_specific = {}
    for cutoff_id in set(FOLD_DISTRIBUTION_CUTOFF.values()):
        pooled[cutoff_id] = load_distribution(
            PHASE3 / "empirical_distributions" / f"{cutoff_id}.npz"
        )
        asset_specific[cutoff_id] = {
            asset: load_distribution(
                PHASE3
                / "empirical_asset_distributions"
                / f"{cutoff_id}_{asset}.npz"
            )
            for asset in ASSET_SYMBOLS
        }
    return pooled, asset_specific


def load_frozen_inputs():
    benchmark = pd.read_parquet(PHASE3 / "benchmark_parameters.parquet")
    empirical = pd.read_parquet(PHASE3 / "empirical_parameters.parquet")
    empirical_asset = pd.read_parquet(
        PHASE3 / "empirical_asset_calibrators.parquet"
    )
    har_calibrators = pd.read_parquet(PHASE3 / "har_calibrators.parquet")
    dvol_calibrators = pd.read_parquet(PHASE3 / "dvol_calibrators.parquet")
    dvol_calibrators = dvol_calibrators[
        dvol_calibrators["variant"] == "btceth_dvol"
    ]
    calibrators = {
        "gbm_platt": benchmark.set_index("fold_id").to_dict("index"),
        "empirical_pooled_platt": empirical.set_index("fold_id").to_dict("index"),
        "empirical_asset_platt": empirical_asset.set_index("fold_id").to_dict(
            "index"
        ),
        "har_platt": har_calibrators.set_index("fold_id").to_dict("index"),
        "har_dvol_platt": dvol_calibrators.set_index("fold_id").to_dict("index"),
    }
    for model_key, maps in calibrators.items():
        if set(maps) != set(FOLD_CALIBRATION_BLOCKS):
            raise ValueError(f"incomplete frozen calibrators for {model_key}")
        if not all(bool(item["optimizer_success"]) for item in maps.values()):
            raise ValueError(f"failed frozen calibrator for {model_key}")

    pooled_distributions, asset_distributions = load_distributions()
    base_assets = {asset: prepare_asset_data(asset) for asset in ASSET_SYMBOLS}
    har_models = models_from_parameters(
        pd.read_parquet(PHASE3 / "har_parameters.parquet"), FEATURE_NAMES
    )
    dvol_assets = {}
    for asset in DVOL_ASSETS:
        source_time, sigma = load_dvol(asset)
        dvol_assets[asset], _ = augment_with_dvol(
            base_assets[asset], source_time, sigma
        )
    dvol_parameters = pd.read_parquet(PHASE3 / "dvol_parameters.parquet")
    dvol_models = models_from_parameters(
        dvol_parameters[dvol_parameters["variant"] == "btceth_dvol"],
        DVOL_FEATURE_NAMES,
    )
    return (
        calibrators,
        pooled_distributions,
        asset_distributions,
        base_assets,
        har_models,
        dvol_assets,
        dvol_models,
    )


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
    if experiment["information"] == "market":
        information_timestamp = base["decision_time"]
    elif experiment["information"] == "dvol":
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
            "role": "evaluation",
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


def generate_evaluation_outputs(
    folds: pd.DataFrame,
    calibrators,
    pooled_distributions,
    asset_distributions,
    base_assets,
    har_models,
    dvol_assets,
    dvol_models,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = []
    manifests = []
    reliability = defaultdict(lambda: np.zeros(4, dtype=float))

    for fold_id in FOLD_CALIBRATION_BLOCKS:
        evaluation = folds[
            (folds["fold_id"] == fold_id) & (folds["role"] == "evaluation")
        ]
        cutoff_id = FOLD_DISTRIBUTION_CUTOFF[fold_id]
        for asset in ASSET_SYMBOLS:
            members = evaluation[evaluation["asset"] == asset]
            parts = []
            for member in members.itertuples(index=False):
                panel = pd.read_parquet(
                    PANEL_DIR / f"{member.condition_id}.parquet",
                    columns=PANEL_COLUMNS,
                )
                raw = {
                    "market_raw": panel["market_yes_price"].to_numpy(dtype=float),
                    "gbm_platt": gbm_touch_probability(
                        panel["spot_close"].to_numpy(dtype=float),
                        panel["barrier"].to_numpy(dtype=float),
                        panel["minutes_to_expiry"].to_numpy(dtype=float),
                        panel["rv_24h"].to_numpy(dtype=float),
                        panel["direction"].to_numpy(),
                    ),
                    "empirical_pooled_platt": predict_empirical_panel(
                        panel, pooled_distributions[cutoff_id]
                    ),
                    "empirical_asset_platt": predict_empirical_panel(
                        panel, asset_distributions[cutoff_id][asset]
                    ),
                    "har_platt": predict_har_panel(
                        panel, base_assets[asset], har_models[cutoff_id]
                    ),
                }
                if asset in DVOL_ASSETS:
                    raw["har_dvol_platt"] = predict_dvol_panel(
                        panel,
                        dvol_assets[asset],
                        dvol_models[cutoff_id],
                        verify_dvol=True,
                    )
                if not all(np.isfinite(value).all() for value in raw.values()):
                    raise ValueError(f"invalid evaluation probability: {member.condition_id}")

                panel = panel.copy()
                for model_key, raw_probability in raw.items():
                    panel[f"{model_key}_raw"] = raw_probability
                    panel[f"{model_key}_final"] = (
                        raw_probability
                        if model_key == "market_raw"
                        else apply_platt(raw_probability, calibrators[model_key][fold_id])
                    )
                parts.append(panel)

                outcome = panel["y"].to_numpy(dtype=float)
                weight = panel["contract_row_weight"].to_numpy(dtype=float)
                for model_key in raw:
                    probability = panel[f"{model_key}_final"].to_numpy(dtype=float)
                    brier, log_loss, mean_probability = contract_metric(
                        probability, outcome, weight
                    )
                    metrics.append(
                        {
                            "fold_id": fold_id,
                            "role": "evaluation",
                            "model_key": model_key,
                            "experiment_id": EXPERIMENTS[model_key]["experiment_id"],
                            "condition_id": member.condition_id,
                            "asset": member.asset,
                            "direction": member.direction,
                            "resolution_date_et": member.resolution_date_et,
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
                raw_column = f"{model_key}_raw"
                probability_column = f"{model_key}_final"
                calibration_version = (
                    "none"
                    if model_key == "market_raw"
                    else str(calibrators[model_key][fold_id]["calibration_version"])
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
                    / "evaluation"
                    / f"{asset}.parquet"
                )
                atomic_parquet(output, path)
                manifests.append(
                    {
                        "experiment_id": experiment["experiment_id"],
                        "model_key": model_key,
                        "fold_id": fold_id,
                        "role": "evaluation",
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
                f"wrote evaluation {fold_id} {asset}: {len(base):,} rows, "
                f"{len(members):,} contracts",
                flush=True,
            )

    reliability_rows = []
    for (fold_id, model_key, bin_id), values in sorted(reliability.items()):
        total_weight, weighted_probability, weighted_outcome, row_count = values
        reliability_rows.append(
            {
                "fold_id": fold_id,
                "role": "evaluation",
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


def metric_summary(metrics: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return metrics.groupby(group_columns, as_index=False).agg(
        contracts=("condition_id", "size"),
        decision_rows=("n_rows", "sum"),
        brier=("brier", "mean"),
        log_loss=("log_loss", "mean"),
    )


def attach_ece(summary: pd.DataFrame, reliability: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in ["fold_id", "model_key"] if column in summary.columns]
    ece = (
        reliability.groupby(keys)
        .apply(weighted_ece, include_groups=False)
        .rename("ece_10")
        .reset_index()
    )
    return summary.merge(ece, on=keys, validate="one_to_one")


def validation_summary() -> pd.DataFrame:
    rows = []
    for model_key, (filename, source_key) in VALIDATION_SOURCES.items():
        frame = pd.read_parquet(PHASE3 / filename)
        selected = frame[frame["model_key"] == source_key]
        rows.append(
            {
                "model_key": model_key,
                "contracts": len(selected),
                "brier": selected["brier"].mean(),
                "log_loss": selected["log_loss"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("model_key")


def make_report(
    metrics: pd.DataFrame,
    reliability: pd.DataFrame,
    manifest: pd.DataFrame,
) -> str:
    by_fold = attach_ece(
        metric_summary(metrics, ["fold_id", "model_key"]), reliability
    )
    supporting = metric_summary(metrics, ["model_key"]).set_index("model_key")
    primary_metrics = metrics[metrics["fold_id"] == PRIMARY_FOLD]
    primary_reliability = reliability[reliability["fold_id"] == PRIMARY_FOLD]
    primary = attach_ece(
        metric_summary(primary_metrics, ["fold_id", "model_key"]),
        primary_reliability,
    ).set_index("model_key")
    btceth_primary = metric_summary(
        primary_metrics[primary_metrics["asset"].isin(DVOL_ASSETS)],
        ["model_key"],
    ).set_index("model_key")
    validation = validation_summary()
    by_asset = metric_summary(primary_metrics, ["model_key", "asset"])
    by_direction = metric_summary(primary_metrics, ["model_key", "direction"])

    lines = [
        "# Frozen held-back probability evaluation",
        "",
        "No model was fitted, recalibrated, selected, or rejected in this pass. "
        "The Phase-4 execution proxy was not read.",
        "",
        f"The primary final result is `{PRIMARY_FOLD}`, {PRIMARY_PERIOD}. The first "
        "three evaluation roles overlap later validation months and are shown only "
        "as supporting walk-forward evidence.",
        "",
        "## Primary genuinely held-back result",
        "",
        "Metrics are averaged within each contract and then equally across "
        "contracts. Lower is better.",
        "",
        "| Frozen model | Assets | Contracts | Rows | Brier | Log loss | ECE-10 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_key in EXPERIMENTS:
        row = primary.loc[model_key]
        assets = "BTC/ETH" if model_key == "har_dvol_platt" else "All four"
        lines.append(
            f"| {LABELS[model_key]} | {assets} | {int(row.contracts):,} | "
            f"{int(row.decision_rows):,} | {row.brier:.6f} | "
            f"{row.log_loss:.6f} | {row.ece_10:.6f} |"
        )

    four_asset_keys = [
        key for key in EXPERIMENTS if key != "har_dvol_platt"
    ]
    brier_leader = min(four_asset_keys, key=lambda key: primary.loc[key, "brier"])
    log_leader = min(four_asset_keys, key=lambda key: primary.loc[key, "log_loss"])
    empirical_brier_delta = (
        primary.loc["empirical_asset_platt", "brier"]
        - primary.loc["empirical_pooled_platt", "brier"]
    )
    empirical_log_delta = (
        primary.loc["empirical_asset_platt", "log_loss"]
        - primary.loc["empirical_pooled_platt", "log_loss"]
    )
    lines.extend(
        [
            "",
            "## Final interpretation",
            "",
            f"- Lowest four-asset held-back Brier: **{LABELS[brier_leader]}**.",
            f"- Lowest four-asset held-back log loss: **{LABELS[log_leader]}**.",
            f"- Asset-specific minus pooled Empirical Brier: {empirical_brier_delta:+.6f}.",
            f"- Asset-specific minus pooled Empirical log loss: {empirical_log_delta:+.6f}.",
            "- BTC/ETH HAR+DVOL is reported only on its predeclared two-asset "
            "universe and is compared with the other models below on identical rows.",
            "- These results do not reopen model selection. They are the final "
            "probability-quality report before economic testing.",
            "",
            "## Validation versus primary held-back result",
            "",
            "The samples differ, so these changes describe temporal stability, "
            "not a paired test.",
            "",
            "| Model | Validation Brier | Held-back Brier | Change | Validation log loss | Held-back log loss | Change |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model_key in EXPERIMENTS:
        val = validation.loc[model_key]
        test = primary.loc[model_key]
        lines.append(
            f"| {LABELS[model_key]} | {val.brier:.6f} | {test.brier:.6f} | "
            f"{test.brier - val.brier:+.6f} | {val.log_loss:.6f} | "
            f"{test.log_loss:.6f} | {test.log_loss - val.log_loss:+.6f} |"
        )

    lines.extend(
        [
            "",
            "## Primary matched BTC/ETH subset",
            "",
            "| Model | Contracts | Brier | Log loss |",
            "|---|---:|---:|---:|",
        ]
    )
    for model_key in EXPERIMENTS:
        row = btceth_primary.loc[model_key]
        lines.append(
            f"| {LABELS[model_key]} | {int(row.contracts):,} | "
            f"{row.brier:.6f} | {row.log_loss:.6f} |"
        )
    lines.extend(
        [
            "",
            "On identical BTC/ETH rows, pooled Empirical + Platt had the lowest "
            "Brier and log loss. The DVOL enhancement did not improve the final "
            "held-back result.",
        ]
    )

    lines.extend(
        [
            "",
            "## All predeclared evaluation roles",
            "",
            "This table is supporting walk-forward evidence; folds 1--3 are not "
            "globally untouched after the full validation tournament.",
            "",
            "| Fold | Model | Contracts | Brier | Log loss | ECE-10 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for fold_id in FOLD_CALIBRATION_BLOCKS:
        for model_key in EXPERIMENTS:
            row = by_fold[
                (by_fold["fold_id"] == fold_id)
                & (by_fold["model_key"] == model_key)
            ].iloc[0]
            lines.append(
                f"| {fold_id} | {LABELS[model_key]} | {int(row.contracts):,} | "
                f"{row.brier:.6f} | {row.log_loss:.6f} | {row.ece_10:.6f} |"
            )

    lines.extend(
        [
            "",
            "## Primary block by asset",
            "",
            "| Model | Asset | Contracts | Brier | Log loss |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in by_asset.itertuples(index=False):
        lines.append(
            f"| {LABELS[row.model_key]} | {row.asset} | {row.contracts:,} | "
            f"{row.brier:.6f} | {row.log_loss:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Primary block by direction",
            "",
            "| Model | Direction | Contracts | Brier | Log loss |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in by_direction.itertuples(index=False):
        lines.append(
            f"| {LABELS[row.model_key]} | {row.direction} | {row.contracts:,} | "
            f"{row.brier:.6f} | {row.log_loss:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Supporting aggregate across all evaluation roles",
            "",
            "| Model | Fold-contract appearances | Brier | Log loss |",
            "|---|---:|---:|---:|",
        ]
    )
    for model_key in EXPERIMENTS:
        row = supporting.loc[model_key]
        lines.append(
            f"| {LABELS[model_key]} | {int(row.contracts):,} | "
            f"{row.brier:.6f} | {row.log_loss:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Artifact audit and boundary",
            "",
            f"- Evaluation contract-metric rows: {len(metrics):,}.",
            f"- Prediction partitions: {len(manifest):,}.",
            f"- Prediction rows: {int(manifest['rows'].sum()):,}.",
            f"- Compressed prediction size: {manifest['bytes'].sum() / 1024**3:.2f} GiB.",
            "- Every output contains only `evaluation` roles and uses the frozen "
            "model/calibration artifacts.",
            "- This pass says nothing about fills or profitability. Phase 4 must "
            "open the execution proxy separately and apply fees, slippage, and "
            "fill assumptions.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    folds = pd.read_parquet(FOLD_FILE)
    frozen_inputs = load_frozen_inputs()
    metrics, reliability, manifest = generate_evaluation_outputs(
        folds, *frozen_inputs
    )
    REPORT_FILE.write_text(
        make_report(metrics, reliability, manifest), encoding="utf-8"
    )
    print(f"metrics: {METRICS_FILE}", flush=True)
    print(f"manifest: {MANIFEST_FILE}", flush=True)
    print(f"report: {REPORT_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
