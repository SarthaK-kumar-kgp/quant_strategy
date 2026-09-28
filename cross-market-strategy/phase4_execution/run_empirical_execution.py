#!/usr/bin/env python3
"""Run pooled Empirical + Platt under the frozen Phase-4 execution protocol."""

from __future__ import annotations

import run_har_execution as engine


def configure() -> None:
    experiment_id = "E04-EMP-PLATT-V1"
    engine.MODEL_EXPERIMENT_ID = experiment_id
    engine.MODEL_LABEL = "Pooled Empirical + Platt"
    engine.PROTOCOL_FILE_NAME = "EMPIRICAL_PROTOCOL.md"
    engine.VALIDATION_DIR = (
        engine.PHASE3 / "empirical_predictions" / experiment_id
    )
    engine.EVALUATION_DIR = (
        engine.PHASE3_EVALUATION / "predictions" / experiment_id
    )
    engine.LEDGER_FILE = engine.FOLDER / "empirical_attempt_ledger.parquet"
    engine.SUMMARY_FILE = engine.FOLDER / "empirical_execution_summary.parquet"
    engine.BREAKDOWN_FILE = engine.FOLDER / "empirical_full_cost_breakdown.parquet"
    engine.REPORT_FILE = engine.FOLDER / "empirical_execution_report.md"


def main() -> None:
    configure()
    engine.main()


if __name__ == "__main__":
    main()
