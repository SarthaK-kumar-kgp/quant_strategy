#!/usr/bin/env python3
"""Run the frozen pooled and asset-specific empirical isotonic comparison."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from run_benchmarks import (
    PROBABILITY_CLIP,
    RELIABILITY_BINS,
    atomic_parquet,
    contract_metric,
    file_sha256,
)
from run_empirical import (
    ASSET_SYMBOLS,
    CALIBRATION_BLOCKS,
    FOLD_CALIBRATION_BLOCKS,
    FOLD_DISTRIBUTION_CUTOFF,
    FOLD_FILE,
    HORIZONS,
    predict_panel,
    read_panel,
    unique_contract_calendar,
)
from run_har import metric_tables


FOLDER = Path(__file__).resolve().parent
OUTPUT_DIR = FOLDER / "empirical_isotonic_predictions"
CALIBRATORS_FILE = FOLDER / "empirical_isotonic_calibrators.parquet"
METRICS_FILE = FOLDER / "empirical_isotonic_contract_metrics.parquet"
RELIABILITY_FILE = FOLDER / "empirical_isotonic_reliability.parquet"
MANIFEST_FILE = FOLDER / "empirical_isotonic_prediction_manifest.parquet"
REPORT_FILE = FOLDER / "empirical_isotonic_report.md"

STRUCTURES = {
    "pooled": {
        "model_key": "empirical_pooled_isotonic",
        "experiment_id": "E09-EMP-POOL-ISO-V1",
        "model_id": "empirical_pooled_excursion",
        "feature_set_id": "empirical_spot_barrier_rv24h_v1",
        "distribution_dir": FOLDER / "empirical_distributions",
        "distribution_manifest": FOLDER / "empirical_distribution_manifest.parquet",
    },
    "asset_specific": {
        "model_key": "empirical_asset_isotonic",
        "experiment_id": "E09-EMP-ASSET-ISO-V1",
        "model_id": "empirical_asset_excursion_v1",
        "feature_set_id": "empirical_spot_barrier_rv24h_asset_v1",
        "distribution_dir": FOLDER / "empirical_asset_distributions",
        "distribution_manifest": FOLDER
        / "empirical_asset_distribution_manifest.parquet",
    },
}


def fit_weighted_isotonic(
    probability: np.ndarray, outcome: np.ndarray, weight: np.ndarray
) -> IsotonicRegression:
    probability = np.asarray(probability, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    weight = np.asarray(weight, dtype=float)
    if not (len(probability) == len(outcome) == len(weight)) or not len(probability):
        raise ValueError("isotonic inputs must be non-empty and have equal length")
    if not (
        np.isfinite(probability).all()
        and np.isfinite(outcome).all()
        and np.isfinite(weight).all()
    ):
        raise ValueError("isotonic inputs must be finite")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("raw probabilities must lie in [0,1]")
    if ((outcome < 0) | (outcome > 1)).any() or (weight <= 0).any():
        raise ValueError("outcomes must lie in [0,1] and weights must be positive")
    fitted = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
        y_min=PROBABILITY_CLIP,
        y_max=1 - PROBABILITY_CLIP,
    )
    fitted.fit(probability, outcome, sample_weight=weight)
    return fitted


def apply_isotonic(
    fitted: IsotonicRegression, probability: np.ndarray
) -> np.ndarray:
    calibrated = fitted.predict(np.asarray(probability, dtype=float))
    return np.clip(calibrated, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)


def _verify_distribution_files(manifest_path: Path) -> None:
    manifest = pd.read_parquet(manifest_path)
    paths = manifest[["distribution_path", "distribution_sha256"]].drop_duplicates()
    for row in paths.itertuples(index=False):
        path = FOLDER / row.distribution_path
        if not path.exists() or file_sha256(path) != row.distribution_sha256:
            raise ValueError(f"distribution checksum mismatch: {path}")


def _load_distribution(path: Path) -> dict[tuple[int, str], np.ndarray]:
    with np.load(path) as archive:
        return {
            (int(horizon), direction): archive[f"h{int(horizon)}_{direction}"].copy()
            for horizon in HORIZONS
            for direction in ["up", "down"]
        }


def load_distributions() -> tuple[
    dict[str, dict[tuple[int, str], np.ndarray]],
    dict[str, dict[str, dict[tuple[int, str], np.ndarray]]],
]:
    for specification in STRUCTURES.values():
        _verify_distribution_files(specification["distribution_manifest"])
    pooled = {
        cutoff_id: _load_distribution(
            STRUCTURES["pooled"]["distribution_dir"] / f"{cutoff_id}.npz"
        )
        for cutoff_id in set(FOLD_DISTRIBUTION_CUTOFF.values())
        | {item["cutoff_id"] for item in CALIBRATION_BLOCKS.values()}
    }
    asset_specific = {
        cutoff_id: {
            asset: _load_distribution(
                STRUCTURES["asset_specific"]["distribution_dir"]
                / f"{cutoff_id}_{asset}.npz"
            )
            for asset in ASSET_SYMBOLS
        }
        for cutoff_id in pooled
    }
    return pooled, asset_specific


def load_calibration_predictions(
    calendar: pd.DataFrame,
    pooled: dict[str, dict[tuple[int, str], np.ndarray]],
    asset_specific: dict[
        str, dict[str, dict[tuple[int, str], np.ndarray]]
    ],
) -> dict[str, dict[str, object]]:
    blocks = {}
    for block_id, specification in CALIBRATION_BLOCKS.items():
        members = calendar[
            (calendar.resolution_date_et >= specification["start"])
            & (calendar.resolution_date_et <= specification["end"])
        ]
        probability_parts = {structure: [] for structure in STRUCTURES}
        outcome_parts = []
        weight_parts = []
        cutoff_id = specification["cutoff_id"]
        for member in members.itertuples(index=False):
            panel = read_panel(member.condition_id)
            pooled_probability = predict_panel(panel, pooled[cutoff_id])
            asset_probability = predict_panel(
                panel, asset_specific[cutoff_id][member.asset]
            )
            if not (
                np.isfinite(pooled_probability).all()
                and np.isfinite(asset_probability).all()
            ):
                raise ValueError(f"invalid empirical probability in {member.condition_id}")
            probability_parts["pooled"].append(pooled_probability)
            probability_parts["asset_specific"].append(asset_probability)
            outcome_parts.append(panel.y.to_numpy(dtype=float))
            weight_parts.append(panel.contract_row_weight.to_numpy(dtype=float))
        blocks[block_id] = {
            "probability": {
                key: np.concatenate(parts) for key, parts in probability_parts.items()
            },
            "outcome": np.concatenate(outcome_parts),
            "weight": np.concatenate(weight_parts),
            "contracts": len(members),
            "cutoff_id": cutoff_id,
        }
        print(
            f"generated isotonic OOF {block_id}: {len(members):,} contracts, "
            f"{len(blocks[block_id]['outcome']):,} rows",
            flush=True,
        )
    return blocks


def fit_fold_calibrators(
    blocks: dict[str, dict[str, object]],
) -> tuple[dict[tuple[str, str], IsotonicRegression], pd.DataFrame]:
    calibrators = {}
    parameter_rows = []
    for structure in STRUCTURES:
        for fold_id, block_ids in FOLD_CALIBRATION_BLOCKS.items():
            probability = np.concatenate(
                [blocks[item]["probability"][structure] for item in block_ids]
            )
            outcome = np.concatenate([blocks[item]["outcome"] for item in block_ids])
            weight = np.concatenate([blocks[item]["weight"] for item in block_ids])
            fitted = fit_weighted_isotonic(probability, outcome, weight)
            calibrated = apply_isotonic(fitted, probability)
            oof_brier = float(np.dot(weight, (calibrated - outcome) ** 2) / weight.sum())
            calibrators[(structure, fold_id)] = fitted
            common = {
                "structure": structure,
                "fold_id": fold_id,
                "model_distribution_cutoff": FOLD_DISTRIBUTION_CUTOFF[fold_id],
                "calibration_version": f"{fold_id}_empirical_{structure}_isotonic_v1",
                "calibration_blocks": "|".join(block_ids),
                "calibration_distribution_cutoffs": "|".join(
                    str(blocks[item]["cutoff_id"]) for item in block_ids
                ),
                "calibration_contracts": int(
                    sum(int(blocks[item]["contracts"]) for item in block_ids)
                ),
                "calibration_rows": len(probability),
                "calibration_weight": float(weight.sum()),
                "out_of_bounds": "clip",
                "interpolation": "linear",
                "output_minimum": PROBABILITY_CLIP,
                "output_maximum": 1 - PROBABILITY_CLIP,
                "oof_brier": oof_brier,
                "threshold_count": len(fitted.X_thresholds_),
            }
            for threshold_index, (raw_threshold, mapped_probability) in enumerate(
                zip(fitted.X_thresholds_, fitted.y_thresholds_)
            ):
                parameter_rows.append(
                    {
                        **common,
                        "threshold_index": threshold_index,
                        "raw_threshold": float(raw_threshold),
                        "mapped_probability": float(mapped_probability),
                    }
                )
            print(
                f"fitted {structure} {fold_id} isotonic: "
                f"{len(fitted.X_thresholds_):,} thresholds, OOF Brier {oof_brier:.6f}",
                flush=True,
            )
    parameters = pd.DataFrame(parameter_rows)
    atomic_parquet(parameters, CALIBRATORS_FILE)
    return calibrators, parameters


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
    base: pd.DataFrame, structure: str, fold_id: str
) -> pd.DataFrame:
    specification = STRUCTURES[structure]
    return pd.DataFrame(
        {
            "schema_version": "phase3_prediction_v1",
            "experiment_id": specification["experiment_id"],
            "condition_id": base.condition_id,
            "decision_time": base.decision_time,
            "information_timestamp": base.spot_available_time,
            "fold_id": fold_id,
            "role": "validation",
            "prediction_kind": "forward",
            "model_id": specification["model_id"],
            "model_version": "v1",
            "feature_set_id": specification["feature_set_id"],
            "calibration_method": "isotonic",
            "calibration_version": f"{fold_id}_empirical_{structure}_isotonic_v1",
            "raw_yes_probability": base[f"{structure}_raw"].to_numpy(dtype=float),
            "calibrated_yes_probability": base[
                f"{structure}_isotonic"
            ].to_numpy(dtype=float),
            "contract_row_weight": base.contract_row_weight.to_numpy(dtype=float),
        }
    )


def generate_validation_outputs(
    folds: pd.DataFrame,
    pooled: dict[str, dict[tuple[int, str], np.ndarray]],
    asset_specific: dict[
        str, dict[str, dict[tuple[int, str], np.ndarray]]
    ],
    calibrators: dict[tuple[str, str], IsotonicRegression],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = []
    manifests = []
    reliability = defaultdict(lambda: np.zeros(4, dtype=float))
    for fold_id in FOLD_CALIBRATION_BLOCKS:
        validation = folds[(folds.fold_id == fold_id) & (folds.role == "validation")]
        cutoff_id = FOLD_DISTRIBUTION_CUTOFF[fold_id]
        for asset in ASSET_SYMBOLS:
            members = validation[validation.asset == asset]
            parts = []
            for member in members.itertuples(index=False):
                panel = read_panel(member.condition_id).copy()
                raw_probabilities = {
                    "pooled": predict_panel(panel, pooled[cutoff_id]),
                    "asset_specific": predict_panel(
                        panel, asset_specific[cutoff_id][asset]
                    ),
                }
                outcome = panel.y.to_numpy(dtype=float)
                weight = panel.contract_row_weight.to_numpy(dtype=float)
                for structure, raw in raw_probabilities.items():
                    calibrated = apply_isotonic(
                        calibrators[(structure, fold_id)], raw
                    )
                    panel[f"{structure}_raw"] = raw
                    panel[f"{structure}_isotonic"] = calibrated
                    model_key = STRUCTURES[structure]["model_key"]
                    brier, log_loss, mean_probability = contract_metric(
                        calibrated, outcome, weight
                    )
                    metrics.append(
                        {
                            "fold_id": fold_id,
                            "role": "validation",
                            "model_key": model_key,
                            "experiment_id": STRUCTURES[structure]["experiment_id"],
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
                        calibrated,
                        outcome,
                        weight,
                    )
                parts.append(panel)
            base = pd.concat(parts, ignore_index=True)
            for structure in STRUCTURES:
                output = prediction_frame(base, structure, fold_id)
                path = (
                    OUTPUT_DIR
                    / STRUCTURES[structure]["experiment_id"]
                    / fold_id
                    / "validation"
                    / f"{asset}.parquet"
                )
                atomic_parquet(output, path)
                manifests.append(
                    {
                        "experiment_id": STRUCTURES[structure]["experiment_id"],
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
                f"wrote isotonic {fold_id} {asset}: {len(base):,} rows, "
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
                "experiment_id": next(
                    specification["experiment_id"]
                    for specification in STRUCTURES.values()
                    if specification["model_key"] == model_key
                ),
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


def make_report(
    parameters: pd.DataFrame,
    isotonic_metrics: pd.DataFrame,
    isotonic_reliability: pd.DataFrame,
    manifest: pd.DataFrame,
) -> str:
    pooled_metrics = pd.read_parquet(FOLDER / "empirical_contract_metrics.parquet")
    pooled_reliability = pd.read_parquet(FOLDER / "empirical_reliability.parquet")
    asset_metrics = pd.read_parquet(FOLDER / "empirical_asset_contract_metrics.parquet")
    asset_reliability = pd.read_parquet(FOLDER / "empirical_asset_reliability.parquet")
    metrics = pd.concat(
        [pooled_metrics, asset_metrics, isotonic_metrics], ignore_index=True
    )
    reliability = pd.concat(
        [pooled_reliability, asset_reliability, isotonic_reliability],
        ignore_index=True,
    )
    by_fold, overall, by_asset, by_direction = metric_tables(metrics, reliability)
    labels = {
        "empirical_raw": "Pooled raw",
        "empirical_platt": "Pooled + Platt",
        "empirical_pooled_isotonic": "Pooled + isotonic",
        "empirical_asset_raw": "Asset-specific raw",
        "empirical_asset_platt": "Asset-specific + Platt",
        "empirical_asset_isotonic": "Asset-specific + isotonic",
    }
    order = list(labels)
    indexed = overall.set_index("model_key")
    winner = min(
        order,
        key=lambda key: (indexed.loc[key, "brier"], indexed.loc[key, "log_loss"]),
    )
    lines = [
        "# Phase-3 empirical isotonic-calibration comparison",
        "",
        "Weighted isotonic maps were fitted only from the permitted time-ordered",
        "training OOF blocks. Evaluation and Phase-4 execution data were not read.",
        "",
        "## Fitted-map summary",
        "",
        "| Structure | Fold | OOF contracts | OOF rows | Thresholds | OOF Brier |",
        "|---|---|---:|---:|---:|---:|",
    ]
    parameter_summary = parameters.groupby(
        [
            "structure",
            "fold_id",
            "calibration_contracts",
            "calibration_rows",
            "threshold_count",
            "oof_brier",
        ],
        as_index=False,
    ).size()
    for row in parameter_summary.itertuples(index=False):
        lines.append(
            f"| {row.structure} | {row.fold_id} | "
            f"{row.calibration_contracts:,} | {row.calibration_rows:,} | "
            f"{row.threshold_count:,} | {row.oof_brier:.6f} |"
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
    lines.extend(["", "## Frozen calibration decision", ""])
    lines.append(f"- Registered six-way winner: **{labels[winner]}**.")
    for structure, platt_key, iso_key in [
        ("Pooled", "empirical_platt", "empirical_pooled_isotonic"),
        ("Asset-specific", "empirical_asset_platt", "empirical_asset_isotonic"),
    ]:
        platt_fold = by_fold[by_fold.model_key == platt_key].set_index("fold_id")
        iso_fold = by_fold[by_fold.model_key == iso_key].set_index("fold_id")
        lines.extend(
            [
                f"- {structure} isotonic minus Platt Brier: "
                f"{indexed.loc[iso_key, 'brier'] - indexed.loc[platt_key, 'brier']:+.6f}.",
                f"- {structure} isotonic minus Platt log loss: "
                f"{indexed.loc[iso_key, 'log_loss'] - indexed.loc[platt_key, 'log_loss']:+.6f}.",
                f"- {structure} isotonic beats Platt in "
                f"{int((iso_fold.brier < platt_fold.brier).sum())}/4 Brier folds and "
                f"{int((iso_fold.log_loss < platt_fold.log_loss).sum())}/4 log-loss folds.",
            ]
        )
    lines.extend(
        [
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
            f"- Fitted threshold rows: {len(parameters):,}.",
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
    pooled, asset_specific = load_distributions()
    blocks = load_calibration_predictions(calendar, pooled, asset_specific)
    calibrators, parameters = fit_fold_calibrators(blocks)
    del blocks
    metrics, reliability, manifest = generate_validation_outputs(
        folds, pooled, asset_specific, calibrators
    )
    REPORT_FILE.write_text(
        make_report(parameters, metrics, reliability, manifest), encoding="utf-8"
    )
    print(f"calibrators: {CALIBRATORS_FILE}")
    print(f"metrics: {METRICS_FILE}")
    print(f"prediction manifest: {MANIFEST_FILE}")
    print(f"report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
