from pathlib import Path
import sys

import numpy as np
import pandas as pd


PHASE5 = Path(__file__).resolve().parents[2] / "phase5_portfolio"
if str(PHASE5) not in sys.path:
    sys.path.insert(0, str(PHASE5))

import run_pooled_empirical_portfolio as portfolio


class FakeMarks:
    def __init__(self, later_prices=None):
        self.later_prices = later_prices or {}

    def latest(self, condition_id, side, timestamp, fallback_time, fallback_price):
        if timestamp > fallback_time and condition_id in self.later_prices:
            return self.later_prices[condition_id]
        return fallback_price


def signals(rows):
    defaults = {
        "fold_id": "fold_01",
        "asset": "BTC",
        "direction": "up",
        "barrier_group_id": "g1",
        "chosen_side": "YES",
        "decision_time": pd.Timestamp("2026-05-01T00:00:00Z"),
        "proxy_available_at": pd.Timestamp("2026-05-01T00:00:01Z"),
        "official_window_end": pd.Timestamp("2026-05-02T00:00:00Z"),
        "net_edge": 0.20,
        "proxy_price": 0.50,
        "all_in_price": 0.50,
        "token_won": True,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def run(frame, *, bankroll=1_000.0, fill=1.0, variant="uncontrolled_fixed"):
    return portfolio.simulate_portfolio(
        frame,
        FakeMarks(),
        period="test",
        threshold=0.10,
        starting_bankroll=bankroll,
        fill_fraction=fill,
        variant=variant,
    )[0]


def test_cash_is_locked_and_released_at_resolution():
    frame = signals(
        [
            {"condition_id": "c1", "barrier_group_id": "g1"},
            {
                "condition_id": "c2",
                "barrier_group_id": "g2",
                "proxy_available_at": pd.Timestamp("2026-05-01T00:01:01Z"),
                "decision_time": pd.Timestamp("2026-05-01T00:01:00Z"),
            },
            {
                "condition_id": "c3",
                "barrier_group_id": "g3",
                "proxy_available_at": pd.Timestamp("2026-05-02T00:00:00Z"),
                "decision_time": pd.Timestamp("2026-05-01T23:59:59Z"),
                "official_window_end": pd.Timestamp("2026-05-03T00:00:00Z"),
            },
        ]
    )
    trades = run(frame, bankroll=100.0)
    status = trades.set_index("condition_id")["status"].to_dict()
    assert status == {"c1": "filled", "c2": "skipped", "c3": "filled"}
    assert trades.set_index("condition_id").loc["c2", "skip_reason"] == "insufficient_cash"


def test_strongest_only_selects_highest_edge_in_group():
    frame = signals(
        [
            {"condition_id": "weak", "net_edge": 0.15},
            {"condition_id": "strong", "net_edge": 0.30},
        ]
    )
    trades = run(frame, variant="strongest_only_fixed")
    assert trades.set_index("condition_id").loc["strong", "status"] == "filled"
    assert trades.set_index("condition_id").loc["weak", "skip_reason"] == "weaker_same_time_group_signal"


def test_controlled_multiple_enforces_group_cost_cap():
    frame = signals(
        [
            {"condition_id": "c1", "net_edge": 0.30},
            {"condition_id": "c2", "net_edge": 0.20},
        ]
    )
    trades = run(frame, bankroll=1_000.0, variant="controlled_multiple_fixed")
    assert list(trades["status"]) == ["filled", "skipped"]
    assert trades.iloc[1]["skip_reason"] == "group_exposure_limit"


def test_taper_reduces_next_budget_after_marked_drawdown():
    frame = signals(
        [
            {"condition_id": "c1", "barrier_group_id": "g1", "proxy_price": 1.0, "all_in_price": 1.0},
            {
                "condition_id": "c2",
                "barrier_group_id": "g2",
                "proxy_available_at": pd.Timestamp("2026-05-01T00:01:01Z"),
                "decision_time": pd.Timestamp("2026-05-01T00:01:00Z"),
            },
        ]
    )
    trades, _ = portfolio.simulate_portfolio(
        frame,
        FakeMarks({"c1": 0.0}),
        period="test",
        threshold=0.10,
        starting_bankroll=1_000.0,
        fill_fraction=1.0,
        variant="uncontrolled_taper",
    )
    second = trades.set_index("condition_id").loc["c2"]
    assert np.isclose(second["pre_trade_drawdown"], 0.10)
    assert np.isclose(second["filled_budget"], 80.0)


def test_mark_cache_keeps_threshold_specific_fallbacks_separate():
    marks = portfolio.MarkStore()
    marks.histories[("c1", "YES")] = (np.array([], dtype=np.int64), np.array([]))
    timestamp = pd.Timestamp("2026-05-01T01:00:00Z")
    first = marks.latest(
        "c1", "YES", timestamp, pd.Timestamp("2026-05-01T00:00:00Z"), 0.20
    )
    second = marks.latest(
        "c1", "YES", timestamp, pd.Timestamp("2026-05-01T00:30:00Z"), 0.80
    )
    assert first == 0.20
    assert second == 0.80
