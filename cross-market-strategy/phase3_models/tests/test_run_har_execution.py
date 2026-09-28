from pathlib import Path
import sys

import numpy as np
import pandas as pd


PHASE4 = Path(__file__).resolve().parents[2] / "phase4_execution"
if str(PHASE4) not in sys.path:
    sys.path.insert(0, str(PHASE4))

import run_har_execution as execution


def example_rows() -> pd.DataFrame:
    times = pd.to_datetime(
        ["2026-05-01T00:00:00Z", "2026-05-01T00:01:00Z"]
    )
    return pd.DataFrame(
        {
            "experiment_id": execution.MODEL_EXPERIMENT_ID,
            "condition_id": ["c1", "c1"],
            "decision_time": times,
            "fold_id": "fold_01",
            "role": "validation",
            "calibrated_yes_probability": [0.70, 0.80],
            "yes_signal_price_available": True,
            "yes_signal_price": [0.60, 0.50],
            "no_signal_price_available": True,
            "no_signal_price": [0.40, 0.50],
            "yes_proxy_available_60s": [True, True],
            "yes_proxy_available_at": times + pd.Timedelta(seconds=60),
            "yes_proxy_price": [0.61, 0.51],
            "yes_proxy_delay_seconds": 60.0,
            "no_proxy_available_60s": [True, True],
            "no_proxy_available_at": times + pd.Timedelta(seconds=60),
            "no_proxy_price": [0.39, 0.49],
            "no_proxy_delay_seconds": 60.0,
            "proxy_is_transaction": False,
            "proxy_is_executable_quote": False,
            "asset": "BTC",
            "direction": "up",
            "barrier": 100.0,
            "barrier_group_id": "g1",
            "official_window_end": pd.Timestamp("2026-05-02T00:00:00Z"),
            "y": True,
        }
    )


def test_first_qualifying_attempt_is_not_retried() -> None:
    attempts = execution.build_attempts(example_rows(), 0.05)
    assert len(attempts) == 1
    assert attempts.iloc[0]["decision_time"] == pd.Timestamp(
        "2026-05-01T00:00:00Z"
    )
    assert attempts.iloc[0]["chosen_side"] == "YES"


def test_full_cost_formula_and_hold_to_resolution_pnl() -> None:
    attempts = execution.build_attempts(example_rows(), 0.05)
    row = attempts.iloc[0]
    expected_price = 0.61 * 1.01
    expected_fee = 0.07 * expected_price * (1.0 - expected_price)
    expected_all_in = expected_price + expected_fee
    assert np.isclose(row["fee_per_share"], expected_fee)
    assert np.isclose(row["all_in_price"], expected_all_in)
    assert bool(row["full_fill"])
    assert np.isclose(row["full_pnl_100"], 100.0 / expected_all_in - 100.0)


def test_missing_proxy_is_recorded_without_retry() -> None:
    rows = example_rows()
    rows.loc[0, "yes_proxy_available_60s"] = False
    rows.loc[0, "yes_proxy_price"] = np.nan
    attempts = execution.build_attempts(rows, 0.05)
    assert len(attempts) == 1
    assert attempts.iloc[0]["full_status"] == "no_proxy_sample"
    assert not bool(attempts.iloc[0]["full_fill"])
