#!/usr/bin/env python3
"""Validate Phase-3 prediction outputs without reading Phase-4 execution data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCHEMA_VERSION = "phase3_prediction_v1"
REQUIRED_COLUMNS = [
    "schema_version",
    "experiment_id",
    "condition_id",
    "decision_time",
    "information_timestamp",
    "fold_id",
    "role",
    "prediction_kind",
    "model_id",
    "model_version",
    "feature_set_id",
    "calibration_method",
    "calibration_version",
    "raw_yes_probability",
    "calibrated_yes_probability",
    "contract_row_weight",
]
STRING_COLUMNS = [
    "schema_version",
    "experiment_id",
    "condition_id",
    "fold_id",
    "role",
    "prediction_kind",
    "model_id",
    "model_version",
    "feature_set_id",
    "calibration_method",
    "calibration_version",
]
CONSTANT_COLUMNS = [
    "schema_version",
    "experiment_id",
    "model_id",
    "model_version",
    "feature_set_id",
    "calibration_method",
]
ALLOWED_ROLES = {"train", "validation", "evaluation"}
ALLOWED_PREDICTION_KINDS = {"out_of_fold", "forward"}
ALLOWED_CALIBRATION_METHODS = {"raw", "platt", "isotonic"}


class PredictionValidationError(ValueError):
    """Raised when a prediction dataset violates the frozen Phase-3 contract."""


def _fail(message: str) -> None:
    raise PredictionValidationError(message)


def _read_table(path: Path) -> pd.DataFrame:
    resolved = path.resolve()
    if "execution_proxy_60s" in resolved.parts:
        _fail("prediction input cannot be read from execution_proxy_60s")
    if path.is_dir() or path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    _fail(f"unsupported prediction format: {path}")


def _require_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        _fail(f"{label} is missing required columns: {', '.join(missing)}")


def _normalise_utc(frame: pd.DataFrame, column: str, label: str) -> None:
    converted = pd.to_datetime(frame[column], errors="coerce", utc=True)
    if converted.isna().any():
        _fail(f"{label}.{column} contains missing or invalid timestamps")
    frame[column] = converted


def _validate_strings(predictions: pd.DataFrame) -> None:
    for column in STRING_COLUMNS:
        if predictions[column].isna().any():
            _fail(f"{column} contains missing values")
        non_strings = ~predictions[column].map(lambda value: isinstance(value, str))
        if non_strings.any():
            _fail(f"{column} must contain strings only")
        if predictions[column].str.strip().eq("").any():
            _fail(f"{column} contains blank values")


def _validate_constants(predictions: pd.DataFrame) -> None:
    for column in CONSTANT_COLUMNS:
        if predictions[column].nunique(dropna=False) != 1:
            _fail(f"{column} must be constant within one prediction dataset")


def _validate_probabilities(predictions: pd.DataFrame) -> None:
    for column in ["raw_yes_probability", "calibrated_yes_probability"]:
        values = pd.to_numeric(predictions[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            _fail(f"{column} contains missing or non-finite values")
        if ((values < 0.0) | (values > 1.0)).any():
            _fail(f"{column} contains values outside [0,1]")
        predictions[column] = values.astype(float)

    weights = pd.to_numeric(predictions["contract_row_weight"], errors="coerce")
    if weights.isna().any() or not np.isfinite(weights).all() or (weights <= 0).any():
        _fail("contract_row_weight must be finite and strictly positive")
    predictions["contract_row_weight"] = weights.astype(float)

    method = predictions["calibration_method"].iloc[0]
    if method == "raw":
        if not predictions["calibration_version"].eq("none").all():
            _fail("raw calibration requires calibration_version=none")
        if not np.allclose(
            predictions["raw_yes_probability"],
            predictions["calibrated_yes_probability"],
            rtol=0.0,
            atol=1e-15,
        ):
            _fail("raw rows require calibrated_yes_probability to equal raw")
    elif predictions["calibration_version"].eq("none").any():
        _fail("platt and isotonic calibration require a fitted calibration version")

    maps_per_fold = predictions.groupby("fold_id")["calibration_version"].nunique()
    if (maps_per_fold != 1).any():
        _fail("calibration_version must be constant within each fold")


def _validate_role_semantics(predictions: pd.DataFrame) -> None:
    roles = set(predictions["role"].unique())
    if not roles <= ALLOWED_ROLES:
        _fail(f"invalid roles: {sorted(roles - ALLOWED_ROLES)}")

    kinds = set(predictions["prediction_kind"].unique())
    if not kinds <= ALLOWED_PREDICTION_KINDS:
        _fail(f"invalid prediction kinds: {sorted(kinds - ALLOWED_PREDICTION_KINDS)}")

    methods = set(predictions["calibration_method"].unique())
    if not methods <= ALLOWED_CALIBRATION_METHODS:
        _fail(f"invalid calibration methods: {sorted(methods - ALLOWED_CALIBRATION_METHODS)}")

    train_wrong = (predictions["role"] == "train") & (
        predictions["prediction_kind"] != "out_of_fold"
    )
    forward_wrong = predictions["role"].isin({"validation", "evaluation"}) & (
        predictions["prediction_kind"] != "forward"
    )
    if train_wrong.any():
        _fail("all training predictions must use prediction_kind=out_of_fold")
    if forward_wrong.any():
        _fail("validation and evaluation predictions must use prediction_kind=forward")


def _validate_fold_membership(
    predictions: pd.DataFrame, fold_manifest: pd.DataFrame
) -> None:
    manifest_columns = [
        "fold_id",
        "condition_id",
        "role",
        "first_decision",
        "last_decision",
    ]
    _require_columns(fold_manifest, manifest_columns, "fold manifest")
    manifest = fold_manifest[manifest_columns].copy()
    if manifest.duplicated(["fold_id", "condition_id"]).any():
        _fail("fold manifest has duplicate fold_id/condition_id rows")
    _normalise_utc(manifest, "first_decision", "fold manifest")
    _normalise_utc(manifest, "last_decision", "fold manifest")
    manifest = manifest.rename(columns={"role": "manifest_role"})

    joined = predictions.merge(
        manifest,
        on=["fold_id", "condition_id"],
        how="left",
        validate="many_to_one",
    )
    if joined["manifest_role"].isna().any():
        _fail("one or more fold_id/condition_id pairs are absent from the fold manifest")
    if (joined["role"] != joined["manifest_role"]).any():
        _fail("one or more prediction roles disagree with the frozen fold manifest")
    outside = (joined["decision_time"] < joined["first_decision"]) | (
        joined["decision_time"] > joined["last_decision"]
    )
    if outside.any():
        _fail("one or more decision times fall outside the contract's panel bounds")


def _validate_panel_rows(predictions: pd.DataFrame, panel_dir: Path) -> None:
    if "execution_proxy_60s" in panel_dir.resolve().parts:
        _fail("panel checks cannot use execution_proxy_60s")
    if not panel_dir.is_dir():
        _fail(f"causal panel directory does not exist: {panel_dir}")

    for condition_id, predicted in predictions.groupby("condition_id", sort=False):
        panel_path = panel_dir / f"{condition_id}.parquet"
        if not panel_path.is_file():
            _fail(f"missing causal-panel file for condition_id={condition_id}")
        panel = pd.read_parquet(
            panel_path, columns=["decision_time", "contract_row_weight"]
        )
        _normalise_utc(panel, "decision_time", f"panel {condition_id}")
        if panel["decision_time"].duplicated().any():
            _fail(f"causal panel has duplicate decision times: {condition_id}")
        weights = panel.set_index("decision_time")["contract_row_weight"]
        expected = predicted["decision_time"].map(weights)
        if expected.isna().any():
            _fail(f"prediction contains a non-panel decision time: {condition_id}")
        if not np.allclose(
            predicted["contract_row_weight"], expected, rtol=1e-10, atol=1e-15
        ):
            _fail(f"contract_row_weight disagrees with causal panel: {condition_id}")


def validate_dataframe(
    predictions: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    panel_dir: Path | None = None,
) -> dict[str, int]:
    """Validate predictions and return a compact successful-check summary."""

    if predictions.empty:
        _fail("prediction dataset is empty")
    predictions = predictions.copy()
    _require_columns(predictions, REQUIRED_COLUMNS, "prediction dataset")
    _validate_strings(predictions)
    _validate_constants(predictions)

    if predictions["schema_version"].iloc[0] != SCHEMA_VERSION:
        _fail(f"schema_version must be {SCHEMA_VERSION}")

    _normalise_utc(predictions, "decision_time", "prediction dataset")
    _normalise_utc(predictions, "information_timestamp", "prediction dataset")
    if (predictions["information_timestamp"] > predictions["decision_time"]).any():
        _fail("information_timestamp exceeds decision_time")

    duplicate_key = ["experiment_id", "fold_id", "condition_id", "decision_time"]
    if predictions.duplicated(duplicate_key).any():
        _fail(f"prediction dataset has duplicate natural keys: {duplicate_key}")

    _validate_role_semantics(predictions)
    _validate_probabilities(predictions)
    _validate_fold_membership(predictions, fold_manifest)
    if panel_dir is not None:
        _validate_panel_rows(predictions, panel_dir)

    return {
        "rows": len(predictions),
        "contracts": predictions["condition_id"].nunique(),
        "folds": predictions["fold_id"].nunique(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=folder.parent / "phase2_dataset" / "walk_forward_folds.parquet",
    )
    parser.add_argument(
        "--panel-dir",
        type=Path,
        default=folder.parent / "phase2_dataset" / "causal_panel_1m",
    )
    parser.add_argument(
        "--skip-panel-check",
        action="store_true",
        help="check schema and fold bounds only; do not verify exact panel rows",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        predictions = _read_table(args.predictions)
        fold_manifest = pd.read_parquet(args.fold_manifest)
        summary = validate_dataframe(
            predictions,
            fold_manifest,
            None if args.skip_panel_check else args.panel_dir,
        )
    except (PredictionValidationError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1

    print(
        "VALID: "
        f"{summary['rows']:,} rows, "
        f"{summary['contracts']:,} contracts, "
        f"{summary['folds']:,} folds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
