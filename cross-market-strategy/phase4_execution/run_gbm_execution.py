#!/usr/bin/env python3
"""Run zero-drift GBM + Platt under the frozen Phase-4 protocol."""

from __future__ import annotations

import run_har_execution as engine


def configure() -> None:
    experiment_id = "E03-GBM-PLATT-V1"
    engine.MODEL_EXPERIMENT_ID = experiment_id
    engine.MODEL_LABEL = "GBM + Platt"
    engine.PROTOCOL_FILE_NAME = "GBM_PROTOCOL.md"
    engine.VALIDATION_DIR = engine.PHASE3 / "benchmark_predictions" / experiment_id
    engine.EVALUATION_DIR = (
        engine.PHASE3_EVALUATION / "predictions" / experiment_id
    )
    engine.LEDGER_FILE = engine.FOLDER / "gbm_attempt_ledger.parquet"
    engine.SUMMARY_FILE = engine.FOLDER / "gbm_execution_summary.parquet"
    engine.BREAKDOWN_FILE = engine.FOLDER / "gbm_full_cost_breakdown.parquet"
    engine.REPORT_FILE = engine.FOLDER / "gbm_execution_report.md"


def main() -> None:
    configure()
    engine.main()


if __name__ == "__main__":
    main()
