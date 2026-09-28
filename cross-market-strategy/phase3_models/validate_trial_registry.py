#!/usr/bin/env python3
"""Validate the append-only Phase-3 trial registry."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROTOCOL_VERSION = "phase3_protocol_v1"
REQUIRED_COLUMNS = [
    "experiment_id",
    "registered_at_utc",
    "protocol_version",
    "status",
    "model_family",
    "model_id",
    "model_version",
    "parameter_spec",
    "feature_set_id",
    "pooling_spec",
    "asset_universe",
    "dvol_usage",
    "calibration_method",
    "calibration_spec",
    "ensemble_spec",
    "fold_ids",
    "selection_rule",
    "random_seed",
    "code_commit",
    "data_manifest_id",
    "predictions_path",
    "parameters_path",
    "calibration_path",
    "report_path",
    "decision",
    "decision_reason",
    "notes",
]
SPECIFICATION_COLUMNS = [
    "experiment_id",
    "registered_at_utc",
    "protocol_version",
    "status",
    "model_family",
    "model_id",
    "model_version",
    "parameter_spec",
    "feature_set_id",
    "pooling_spec",
    "asset_universe",
    "dvol_usage",
    "calibration_method",
    "calibration_spec",
    "ensemble_spec",
    "fold_ids",
    "selection_rule",
    "random_seed",
    "code_commit",
    "data_manifest_id",
]
ALLOWED_STATUSES = {"registered", "running", "completed", "failed", "rejected"}
ALLOWED_CALIBRATIONS = {"raw", "platt", "isotonic"}
ALLOWED_DECISIONS = {"", "retain", "reject", "failed"}
ALLOWED_FOLDS = {"fold_01", "fold_02", "fold_03", "fold_04"}


class TrialRegistryValidationError(ValueError):
    """Raised when the experiment registry violates its frozen schema."""


def _fail(message: str) -> None:
    raise TrialRegistryValidationError(message)


def validate_registry(registry: pd.DataFrame) -> dict[str, int]:
    missing = [column for column in REQUIRED_COLUMNS if column not in registry.columns]
    extra = [column for column in registry.columns if column not in REQUIRED_COLUMNS]
    if missing:
        _fail(f"missing columns: {', '.join(missing)}")
    if extra:
        _fail(f"unexpected columns: {', '.join(extra)}")
    if registry.empty:
        return {"trials": 0, "completed": 0}

    registry = registry.fillna("").copy()
    for column in SPECIFICATION_COLUMNS:
        if registry[column].astype(str).str.strip().eq("").any():
            _fail(f"{column} contains blank values")
    if registry["experiment_id"].duplicated().any():
        _fail("experiment_id values must be unique")
    if not registry["protocol_version"].eq(PROTOCOL_VERSION).all():
        _fail(f"protocol_version must be {PROTOCOL_VERSION}")

    bad_status = set(registry["status"]) - ALLOWED_STATUSES
    if bad_status:
        _fail(f"invalid statuses: {sorted(bad_status)}")
    bad_calibration = set(registry["calibration_method"]) - ALLOWED_CALIBRATIONS
    if bad_calibration:
        _fail(f"invalid calibration methods: {sorted(bad_calibration)}")
    bad_decision = set(registry["decision"]) - ALLOWED_DECISIONS
    if bad_decision:
        _fail(f"invalid decisions: {sorted(bad_decision)}")

    timestamps = pd.to_datetime(registry["registered_at_utc"], errors="coerce", utc=True)
    if timestamps.isna().any():
        _fail("registered_at_utc contains invalid timestamps")

    for value in registry["fold_ids"]:
        folds = {item for item in str(value).split("|") if item}
        if not folds or not folds <= ALLOWED_FOLDS:
            _fail(f"invalid fold_ids value: {value}")

    raw_wrong = (registry["calibration_method"] == "raw") & (
        registry["calibration_spec"] != "none"
    )
    fitted_wrong = registry["calibration_method"].isin({"platt", "isotonic"}) & (
        registry["calibration_spec"] == "none"
    )
    if raw_wrong.any() or fitted_wrong.any():
        _fail("calibration_method and calibration_spec are inconsistent")

    finished = registry["status"].isin({"completed", "rejected"})
    for column in ["predictions_path", "report_path", "decision", "decision_reason"]:
        if registry.loc[finished, column].astype(str).str.strip().eq("").any():
            _fail(f"finished trials require {column}")
    failed = registry["status"] == "failed"
    if (registry.loc[failed, "decision"] != "failed").any():
        _fail("failed trials require decision=failed")
    if registry.loc[failed, "decision_reason"].astype(str).str.strip().eq("").any():
        _fail("failed trials require decision_reason")

    return {
        "trials": len(registry),
        "completed": int((registry["status"] == "completed").sum()),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "registry",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "trial_registry.csv",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        registry = pd.read_csv(args.registry, dtype=str, keep_default_na=False)
        summary = validate_registry(registry)
    except (TrialRegistryValidationError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(
        f"VALID: {summary['trials']:,} registered trials, "
        f"{summary['completed']:,} completed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
