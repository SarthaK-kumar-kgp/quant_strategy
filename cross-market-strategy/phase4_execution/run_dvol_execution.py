#!/usr/bin/env python3
"""Run BTC/ETH HAR+DVOL + Platt under the frozen Phase-4 protocol."""

from __future__ import annotations

import run_har_execution as engine


def configure() -> None:
    experiment_id = "E07-HAR-DVOL-PLATT-V1"
    engine.MODEL_EXPERIMENT_ID = experiment_id
    engine.MODEL_LABEL = "BTC/ETH HAR+DVOL + Platt"
    engine.PROTOCOL_FILE_NAME = "DVOL_PROTOCOL.md"
    engine.MODEL_ASSETS = ("BTC", "ETH")
    engine.VALIDATION_DIR = engine.PHASE3 / "dvol_predictions" / experiment_id
    engine.EVALUATION_DIR = (
        engine.PHASE3_EVALUATION / "predictions" / experiment_id
    )
    engine.LEDGER_FILE = engine.FOLDER / "dvol_attempt_ledger.parquet"
    engine.SUMMARY_FILE = engine.FOLDER / "dvol_execution_summary.parquet"
    engine.BREAKDOWN_FILE = engine.FOLDER / "dvol_full_cost_breakdown.parquet"
    engine.REPORT_FILE = engine.FOLDER / "dvol_execution_report.md"


def main() -> None:
    configure()
    engine.main()


if __name__ == "__main__":
    main()
