#!/usr/bin/env python3
"""Run the frozen Phase-5 pooled Empirical + Platt portfolio experiment."""

from __future__ import annotations

from collections import defaultdict
import heapq
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parent
PHASE4 = PROJECT / "phase4_execution"
EXECUTION_DIR = PROJECT / "phase2_dataset" / "execution_proxy_60s"

MODEL_EXPERIMENT_ID = "E04-EMP-PLATT-V1"
MODEL_LABEL = "Pooled Empirical + Platt"
PROTOCOL_VERSION = "phase5_portfolio_v1"

SOURCE_LEDGER = PHASE4 / "empirical_attempt_ledger.parquet"
TRADE_LEDGER_FILE = FOLDER / "pooled_empirical_portfolio_trades.parquet"
EQUITY_LEDGER_FILE = FOLDER / "pooled_empirical_daily_equity.parquet"
SUMMARY_FILE = FOLDER / "pooled_empirical_portfolio_summary.parquet"
BREAKDOWN_FILE = FOLDER / "pooled_empirical_portfolio_breakdown.parquet"
REPORT_FILE = FOLDER / "pooled_empirical_portfolio_report.md"

THRESHOLDS = (0.05, 0.10, 0.15)
BANKROLLS = (1_000.0, 2_000.0, 5_000.0, 10_000.0)
FILL_FRACTIONS = (0.10, 0.25, 0.50, 1.00)
MAX_INTENDED_BUDGET = 100.0
TOTAL_EXPOSURE_FRACTION = 0.50
ASSET_EXPOSURE_FRACTION = 0.25
GROUP_EXPOSURE_FRACTION = 0.10
EPSILON = 1e-9

VARIANTS = {
    "uncontrolled_fixed": {"taper": False, "strongest": False, "controlled": False},
    "uncontrolled_taper": {"taper": True, "strongest": False, "controlled": False},
    "strongest_only_fixed": {"taper": False, "strongest": True, "controlled": False},
    "strongest_only_taper": {"taper": True, "strongest": True, "controlled": False},
    "controlled_multiple_fixed": {"taper": False, "strongest": False, "controlled": True},
    "controlled_multiple_taper": {"taper": True, "strongest": False, "controlled": True},
}

IDENTITY_COLUMNS = [
    "model",
    "experiment_id",
    "period",
    "threshold",
    "starting_bankroll",
    "fill_fraction",
    "variant",
]


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


class MarkStore:
    """Lazy causal same-token price histories used only for proxy marking."""

    def __init__(self, execution_dir: Path = EXECUTION_DIR) -> None:
        self.execution_dir = execution_dir
        self.histories: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self.lookup_cache: dict[tuple[str, str, int, int, float], float] = {}

    def _load_condition(self, condition_id: str) -> None:
        path = self.execution_dir / f"{condition_id}.parquet"
        frame = pd.read_parquet(
            path,
            columns=[
                "decision_time",
                "yes_signal_price_available",
                "yes_signal_price",
                "no_signal_price_available",
                "no_signal_price",
            ],
        ).sort_values("decision_time", kind="mergesort")
        times = frame["decision_time"].astype("int64").to_numpy()
        for side, prefix in (("YES", "yes"), ("NO", "no")):
            prices = frame[f"{prefix}_signal_price"].astype(float).to_numpy()
            available = frame[f"{prefix}_signal_price_available"].fillna(False).to_numpy(bool)
            valid = available & np.isfinite(prices) & (prices >= 0.0) & (prices <= 1.0)
            self.histories[(condition_id, side)] = (times[valid], prices[valid])

    def latest(
        self,
        condition_id: str,
        side: str,
        timestamp: pd.Timestamp,
        fallback_time: pd.Timestamp,
        fallback_price: float,
    ) -> float:
        timestamp_ns = int(timestamp.value)
        cache_key = (
            condition_id,
            side,
            timestamp_ns,
            int(fallback_time.value),
            float(fallback_price),
        )
        if cache_key in self.lookup_cache:
            return self.lookup_cache[cache_key]
        key = (condition_id, side)
        if key not in self.histories:
            self._load_condition(condition_id)
        times, prices = self.histories[key]
        index = int(np.searchsorted(times, timestamp_ns, side="right") - 1)
        if index >= 0 and times[index] > int(fallback_time.value):
            price = float(prices[index])
        else:
            price = float(fallback_price)
        self.lookup_cache[cache_key] = price
        return price


def marked_equity(
    cash: float,
    positions: dict[int, dict[str, Any]],
    timestamp: pd.Timestamp,
    marks: MarkStore,
) -> tuple[float, float]:
    marked_value = 0.0
    for position in positions.values():
        price = marks.latest(
            position["condition_id"],
            position["chosen_side"],
            timestamp,
            position["entry_time"],
            position["entry_proxy_price"],
        )
        marked_value += position["shares"] * price
    return cash + marked_value, marked_value


def drawdown_taper(equity: float, running_peak: float) -> tuple[float, float]:
    drawdown = max(0.0, (running_peak - equity) / running_peak) if running_peak > 0 else 0.0
    return drawdown, max(0.0, 1.0 - drawdown / 0.50)


def simulate_portfolio(
    signals: pd.DataFrame,
    marks: MarkStore,
    *,
    period: str,
    threshold: float,
    starting_bankroll: float,
    fill_fraction: float,
    variant: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate one chronological bankroll path."""

    rules = VARIANTS[variant]
    ordered = signals.copy()
    ordered["entry_time"] = pd.to_datetime(ordered["proxy_available_at"], utc=True)
    ordered["resolution_time"] = pd.to_datetime(ordered["official_window_end"], utc=True)
    ordered = ordered.sort_values(
        ["entry_time", "net_edge", "condition_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    if not ordered.empty and (ordered["entry_time"] >= ordered["resolution_time"]).any():
        raise ValueError("portfolio signal has entry at or after resolution")
    if ordered["condition_id"].duplicated().any():
        raise ValueError("portfolio input has more than one signal per contract")

    identity = {
        "model": MODEL_LABEL,
        "experiment_id": MODEL_EXPERIMENT_ID,
        "period": period,
        "threshold": float(threshold),
        "starting_bankroll": float(starting_bankroll),
        "fill_fraction": float(fill_fraction),
        "variant": variant,
    }
    cash = float(starting_bankroll)
    running_peak = float(starting_bankroll)
    positions: dict[int, dict[str, Any]] = {}
    resolution_heap: list[tuple[int, int]] = []
    used_groups: set[str] = set()
    sequence = 0
    records: list[dict[str, Any]] = []
    max_open_positions = 0
    max_locked_cost = 0.0
    max_asset_exposure = 0.0
    max_group_exposure = 0.0
    max_event_drawdown = 0.0

    def exposure() -> tuple[float, dict[str, float], dict[str, float]]:
        total = sum(float(position["filled_budget"]) for position in positions.values())
        by_asset: dict[str, float] = defaultdict(float)
        by_group: dict[str, float] = defaultdict(float)
        for position in positions.values():
            by_asset[position["asset"]] += float(position["filled_budget"])
            by_group[position["barrier_group_id"]] += float(position["filled_budget"])
        return total, by_asset, by_group

    def update_exposure_peaks() -> None:
        nonlocal max_open_positions, max_locked_cost, max_asset_exposure, max_group_exposure
        total, by_asset, by_group = exposure()
        max_open_positions = max(max_open_positions, len(positions))
        max_locked_cost = max(max_locked_cost, total)
        max_asset_exposure = max(max_asset_exposure, max(by_asset.values(), default=0.0))
        max_group_exposure = max(max_group_exposure, max(by_group.values(), default=0.0))

    def settle_through(timestamp: pd.Timestamp) -> None:
        nonlocal cash, running_peak, max_event_drawdown
        settled = False
        while resolution_heap and resolution_heap[0][0] <= int(timestamp.value):
            _, position_id = heapq.heappop(resolution_heap)
            position = positions.pop(position_id)
            cash += float(position["settlement_proceeds"])
            settled = True
        if settled:
            equity, _ = marked_equity(cash, positions, timestamp, marks)
            running_peak = max(running_peak, equity)
            drawdown, _ = drawdown_taper(equity, running_peak)
            max_event_drawdown = max(max_event_drawdown, drawdown)

    for entry_time, batch in ordered.groupby("entry_time", sort=True):
        entry_time = pd.Timestamp(entry_time)
        settle_through(entry_time)
        batch_seen_groups: set[str] = set()
        for row in batch.itertuples(index=False):
            group_id = str(row.barrier_group_id)
            base_record = {
                **identity,
                "fold_id": row.fold_id,
                "condition_id": row.condition_id,
                "asset": row.asset,
                "direction": row.direction,
                "barrier_group_id": group_id,
                "chosen_side": row.chosen_side,
                "decision_time": row.decision_time,
                "entry_time": entry_time,
                "resolution_time": row.resolution_time,
                "net_edge": float(row.net_edge),
                "proxy_price": float(row.proxy_price),
                "all_in_price": float(row.all_in_price),
                "token_won": bool(row.token_won),
            }

            if rules["strongest"] and group_id in batch_seen_groups:
                records.append({**base_record, "status": "skipped", "skip_reason": "weaker_same_time_group_signal"})
                continue
            if rules["strongest"] and group_id in used_groups:
                records.append({**base_record, "status": "skipped", "skip_reason": "group_already_used"})
                continue
            if rules["strongest"]:
                batch_seen_groups.add(group_id)

            equity, marked_value = marked_equity(cash, positions, entry_time, marks)
            running_peak = max(running_peak, equity)
            drawdown, taper = drawdown_taper(equity, running_peak)
            max_event_drawdown = max(max_event_drawdown, drawdown)
            if not rules["taper"]:
                taper = 1.0
            intended_budget = MAX_INTENDED_BUDGET * taper
            filled_budget = intended_budget * fill_fraction
            diagnostic = {
                "pre_trade_cash": cash,
                "pre_trade_marked_open_value": marked_value,
                "pre_trade_equity": equity,
                "pre_trade_drawdown": drawdown,
                "taper_multiplier": taper,
                "intended_budget": intended_budget,
                "filled_budget": filled_budget,
            }
            if filled_budget <= EPSILON:
                records.append(
                    {**base_record, **diagnostic, "status": "skipped", "skip_reason": "taper_zero"}
                )
                continue
            if cash + EPSILON < filled_budget:
                records.append(
                    {**base_record, **diagnostic, "status": "skipped", "skip_reason": "insufficient_cash"}
                )
                continue

            total, by_asset, by_group = exposure()
            if rules["controlled"]:
                if total + filled_budget > starting_bankroll * TOTAL_EXPOSURE_FRACTION + EPSILON:
                    reason = "total_exposure_limit"
                elif by_asset.get(row.asset, 0.0) + filled_budget > starting_bankroll * ASSET_EXPOSURE_FRACTION + EPSILON:
                    reason = "asset_exposure_limit"
                elif by_group.get(group_id, 0.0) + filled_budget > starting_bankroll * GROUP_EXPOSURE_FRACTION + EPSILON:
                    reason = "group_exposure_limit"
                else:
                    reason = ""
                if reason:
                    records.append(
                        {**base_record, **diagnostic, "status": "skipped", "skip_reason": reason}
                    )
                    continue

            shares = filled_budget / float(row.all_in_price)
            settlement_proceeds = shares if bool(row.token_won) else 0.0
            pnl = settlement_proceeds - filled_budget
            cash -= filled_budget
            sequence += 1
            position = {
                "position_id": sequence,
                "condition_id": row.condition_id,
                "asset": row.asset,
                "barrier_group_id": group_id,
                "chosen_side": row.chosen_side,
                "entry_time": entry_time,
                "resolution_time": row.resolution_time,
                "entry_proxy_price": float(row.proxy_price),
                "filled_budget": filled_budget,
                "shares": shares,
                "settlement_proceeds": settlement_proceeds,
            }
            positions[sequence] = position
            heapq.heappush(resolution_heap, (int(row.resolution_time.value), sequence))
            if rules["strongest"]:
                used_groups.add(group_id)
            update_exposure_peaks()
            post_equity, _ = marked_equity(cash, positions, entry_time, marks)
            running_peak = max(running_peak, post_equity)
            post_drawdown, _ = drawdown_taper(post_equity, running_peak)
            max_event_drawdown = max(max_event_drawdown, post_drawdown)
            records.append(
                {
                    **base_record,
                    **diagnostic,
                    "status": "filled",
                    "skip_reason": "",
                    "shares": shares,
                    "settlement_proceeds": settlement_proceeds,
                    "trade_pnl": pnl,
                    "post_trade_cash": cash,
                    "post_trade_equity": post_equity,
                }
            )

    if positions:
        final_time = pd.Timestamp(max(position["resolution_time"] for position in positions.values()))
        settle_through(final_time)
    if positions or resolution_heap:
        raise AssertionError("open positions remain after final settlement")

    trades = pd.DataFrame.from_records(records)
    for column in (
        "pre_trade_cash",
        "pre_trade_marked_open_value",
        "pre_trade_equity",
        "pre_trade_drawdown",
        "taper_multiplier",
        "intended_budget",
        "filled_budget",
        "shares",
        "settlement_proceeds",
        "trade_pnl",
        "post_trade_cash",
        "post_trade_equity",
    ):
        if column not in trades:
            trades[column] = np.nan
    diagnostics = {
        "final_cash": cash,
        "max_open_positions": max_open_positions,
        "max_locked_cost": max_locked_cost,
        "max_asset_exposure": max_asset_exposure,
        "max_group_exposure": max_group_exposure,
        "max_event_drawdown": max_event_drawdown,
    }
    return trades, diagnostics


def build_daily_equity(
    trades: pd.DataFrame,
    marks: MarkStore,
    starting_bankroll: float,
) -> pd.DataFrame:
    filled = trades[trades["status"].eq("filled")].copy()
    identity = {column: trades.iloc[0][column] for column in IDENTITY_COLUMNS}
    if filled.empty:
        timestamp = pd.Timestamp("1970-01-01", tz="UTC")
        return pd.DataFrame(
            [{**identity, "timestamp": timestamp, "cash": starting_bankroll, "marked_open_value": 0.0,
              "locked_cost": 0.0, "open_positions": 0, "equity": starting_bankroll, "drawdown": 0.0}]
        )

    first_entry = filled["entry_time"].min()
    last_resolution = filled["resolution_time"].max()
    start_day = first_entry.floor("D")
    end_day = last_resolution.floor("D")
    close_times = pd.date_range(start_day, end_day, freq="D") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    timestamps = [start_day - pd.Timedelta(nanoseconds=1), *close_times]
    records: list[dict[str, Any]] = []
    running_peak = starting_bankroll
    for timestamp in timestamps:
        entered = filled[filled["entry_time"] <= timestamp]
        resolved = entered[entered["resolution_time"] <= timestamp]
        open_rows = entered[entered["resolution_time"] > timestamp]
        cash = (
            starting_bankroll
            - float(entered["filled_budget"].sum())
            + float(resolved["settlement_proceeds"].sum())
        )
        marked_value = 0.0
        locked_cost = float(open_rows["filled_budget"].sum())
        for row in open_rows.itertuples(index=False):
            price = marks.latest(
                row.condition_id,
                row.chosen_side,
                timestamp,
                row.entry_time,
                row.proxy_price,
            )
            marked_value += float(row.shares) * price
        equity = cash + marked_value
        running_peak = max(running_peak, equity)
        drawdown, _ = drawdown_taper(equity, running_peak)
        records.append(
            {
                **identity,
                "timestamp": timestamp,
                "cash": cash,
                "marked_open_value": marked_value,
                "locked_cost": locked_cost,
                "open_positions": int(len(open_rows)),
                "equity": equity,
                "drawdown": drawdown,
            }
        )
    return pd.DataFrame.from_records(records)


def return_statistics(returns: pd.Series, periods_per_year: float) -> tuple[float, float, float, float]:
    values = returns.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if not len(values):
        return np.nan, np.nan, np.nan, np.nan
    standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    sharpe = float(np.mean(values) / standard_deviation * np.sqrt(periods_per_year)) if standard_deviation > 0 else np.nan
    downside = float(np.sqrt(np.mean(np.minimum(values, 0.0) ** 2)))
    sortino = float(np.mean(values) / downside * np.sqrt(periods_per_year)) if downside > 0 else np.nan
    volatility = standard_deviation * np.sqrt(periods_per_year) if np.isfinite(standard_deviation) else np.nan
    downside_deviation = downside * np.sqrt(periods_per_year)
    return sharpe, sortino, volatility, downside_deviation


def maximum_drawdown_duration(drawdown: pd.Series) -> int:
    longest = 0
    current = 0
    for value in drawdown.to_numpy(float):
        if value > EPSILON:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def summarize_run(
    trades: pd.DataFrame,
    daily: pd.DataFrame,
    diagnostics: dict[str, Any],
    phase4_attempts: int,
    full_cost_eligible: int,
) -> dict[str, Any]:
    identity = {column: trades.iloc[0][column] for column in IDENTITY_COLUMNS}
    filled = trades[trades["status"].eq("filled")]
    pnl = float(filled["trade_pnl"].sum())
    deployed = float(filled["filled_budget"].sum())
    final_equity = float(daily.iloc[-1]["equity"])
    daily_returns = daily.set_index("timestamp")["equity"].pct_change().dropna()
    weekly_equity = daily.set_index("timestamp")["equity"].resample("W-SUN").last()
    weekly_returns = weekly_equity.pct_change().dropna()
    daily_sharpe, daily_sortino, annual_volatility, annual_downside = return_statistics(daily_returns, 365.0)
    weekly_sharpe, weekly_sortino, _, _ = return_statistics(weekly_returns, 52.0)
    max_drawdown = float(daily["drawdown"].max())
    elapsed_days = max(
        (daily["timestamp"].max() - daily["timestamp"].min()).total_seconds() / 86400.0,
        1.0,
    )
    total_return = final_equity / float(identity["starting_bankroll"]) - 1.0
    annualized_return = (1.0 + total_return) ** (365.0 / elapsed_days) - 1.0 if total_return > -1.0 else -1.0
    winning = filled[filled["trade_pnl"] > 0]
    losing = filled[filled["trade_pnl"] < 0]
    gross_profit = float(winning["trade_pnl"].sum())
    gross_loss = float(-losing["trade_pnl"].sum())
    average_win = float(winning["trade_pnl"].mean()) if len(winning) else np.nan
    average_loss = float(-losing["trade_pnl"].mean()) if len(losing) else np.nan
    q05 = float(daily_returns.quantile(0.05)) if len(daily_returns) else np.nan
    tail = daily_returns[daily_returns <= q05] if np.isfinite(q05) else pd.Series(dtype=float)
    skip_counts = trades.loc[trades["status"].eq("skipped"), "skip_reason"].value_counts()
    return {
        **identity,
        "phase4_attempts": int(phase4_attempts),
        "full_cost_eligible": int(full_cost_eligible),
        "portfolio_fills": int(len(filled)),
        "portfolio_skips": int((trades["status"] == "skipped").sum()),
        "wins": int((filled["trade_pnl"] > 0).sum()),
        "losses": int((filled["trade_pnl"] < 0).sum()),
        "starting_equity": float(identity["starting_bankroll"]),
        "final_equity": final_equity,
        "net_pnl": pnl,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "deployed_capital": deployed,
        "return_on_deployed": pnl / deployed if deployed else np.nan,
        "win_rate": float((filled["trade_pnl"] > 0).mean()) if len(filled) else np.nan,
        "average_win": average_win,
        "average_loss": average_loss,
        "payoff_ratio": average_win / average_loss if average_loss > 0 else np.nan,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.nan,
        "daily_sharpe": daily_sharpe,
        "daily_sortino": daily_sortino,
        "weekly_sharpe": weekly_sharpe,
        "weekly_sortino": weekly_sortino,
        "maximum_drawdown": max_drawdown,
        "maximum_drawdown_dollars": max_drawdown * float(daily.loc[daily["drawdown"].idxmax(), "equity"]) / max(1.0 - max_drawdown, EPSILON) if max_drawdown else 0.0,
        "drawdown_duration_days": maximum_drawdown_duration(daily["drawdown"]),
        "calmar_ratio": annualized_return / max_drawdown if max_drawdown > 0 else np.nan,
        "daily_var_95": max(0.0, -q05) if np.isfinite(q05) else np.nan,
        "daily_cvar_95": max(0.0, -float(tail.mean())) if len(tail) else np.nan,
        "annualized_volatility": annual_volatility,
        "annualized_downside_deviation": annual_downside,
        "average_capital_utilization": float((daily["locked_cost"] / float(identity["starting_bankroll"])).mean()),
        "peak_open_positions": int(diagnostics["max_open_positions"]),
        "peak_locked_cost": float(diagnostics["max_locked_cost"]),
        "peak_asset_exposure": float(diagnostics["max_asset_exposure"]),
        "peak_group_exposure": float(diagnostics["max_group_exposure"]),
        "maximum_event_drawdown": float(diagnostics["max_event_drawdown"]),
        "skipped_insufficient_cash": int(skip_counts.get("insufficient_cash", 0)),
        "skipped_taper_zero": int(skip_counts.get("taper_zero", 0)),
        "skipped_correlation": int(skip_counts.get("group_already_used", 0) + skip_counts.get("weaker_same_time_group_signal", 0)),
        "skipped_total_limit": int(skip_counts.get("total_exposure_limit", 0)),
        "skipped_asset_limit": int(skip_counts.get("asset_exposure_limit", 0)),
        "skipped_group_limit": int(skip_counts.get("group_exposure_limit", 0)),
        "daily_observations": int(len(daily_returns)),
        "weekly_observations": int(len(weekly_returns)),
    }


def build_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    filled = trades[trades["status"].eq("filled")].copy()
    filled["entry_month"] = filled["entry_time"].dt.strftime("%Y-%m")
    records: list[dict[str, Any]] = []
    for dimension, source in (("asset", "asset"), ("side", "chosen_side"), ("month", "entry_month")):
        columns = [*IDENTITY_COLUMNS, source]
        for key, group in filled.groupby(columns, observed=True, sort=True):
            values = dict(zip(columns, key))
            bucket = values.pop(source)
            pnl = float(group["trade_pnl"].sum())
            deployed = float(group["filled_budget"].sum())
            records.append(
                {
                    **values,
                    "dimension": dimension,
                    "bucket": str(bucket),
                    "fills": int(len(group)),
                    "wins": int((group["trade_pnl"] > 0).sum()),
                    "deployed_capital": deployed,
                    "net_pnl": pnl,
                    "return_on_deployed": pnl / deployed if deployed else np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def markdown_summary(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| Period | Threshold | Variant | Bankroll | Fills | Skips | P&L | Return | Max DD | Sharpe | Sortino | Peak locked |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        sharpe = "—" if pd.isna(row.daily_sharpe) else f"{row.daily_sharpe:.2f}"
        sortino = "—" if pd.isna(row.daily_sortino) else f"{row.daily_sortino:.2f}"
        lines.append(
            f"| {row.period} | {row.threshold:.0%} | {row.variant} | ${row.starting_bankroll:,.0f} "
            f"| {row.portfolio_fills:,} | {row.portfolio_skips:,} | ${row.net_pnl:,.2f} "
            f"| {row.total_return:.2%} | {row.maximum_drawdown:.2%} | {sharpe} | {sortino} "
            f"| ${row.peak_locked_cost:,.2f} |"
        )
    return lines


def write_report(summary: pd.DataFrame) -> None:
    central = summary[summary["fill_fraction"].eq(0.25)].sort_values(
        ["period", "threshold", "variant", "starting_bankroll"]
    )
    validation = central[central["period"].eq("validation_may_aug")]
    heldback = central[central["period"].eq("heldback_sep")]
    lines = [
        f"# {MODEL_LABEL} Phase 5 portfolio report",
        "",
        "Generated under the frozen `PHASE5_PROTOCOL.md`. The central table uses",
        "the 25% deterministic fill assumption. These are proxy-marked historical",
        "simulations, not verified executable portfolios.",
        "",
        "## Validation: May--August 2026",
        "",
        *markdown_summary(validation),
        "",
        "## Held-back check: 2--9 September 2026",
        "",
        *markdown_summary(heldback),
        "",
        "## Interpretation boundary",
        "",
        "Cash is genuinely locked until official resolution, but entries and marks",
        "still use historical price samples rather than executable asks, bids, and",
        "depth. September contains too few daily/weekly observations for stable risk",
        "ratios. No final model or threshold is selected until GBM and HAR are run",
        "under the same protocol.",
    ]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs(trades: pd.DataFrame, daily: pd.DataFrame, summary: pd.DataFrame) -> None:
    filled = trades[trades["status"].eq("filled")]
    if (filled["filled_budget"] <= 0).any():
        raise AssertionError("non-positive filled budget")
    if (filled["post_trade_cash"] < -EPSILON).any():
        raise AssertionError("portfolio spent more cash than available")
    if not np.allclose(
        summary["final_equity"] - summary["starting_equity"],
        summary["net_pnl"],
        atol=1e-7,
    ):
        raise AssertionError("final equity and realized P&L disagree")
    if (daily["cash"] < -EPSILON).any():
        raise AssertionError("daily cash is negative")
    if summary["portfolio_fills"].gt(summary["full_cost_eligible"]).any():
        raise AssertionError("portfolio fills exceed Phase 4 eligible signals")


def main() -> None:
    source = pd.read_parquet(SOURCE_LEDGER)
    if source["experiment_id"].nunique() != 1 or source["experiment_id"].iloc[0] != MODEL_EXPERIMENT_ID:
        raise ValueError("wrong Phase 4 model ledger")
    source = source[source["threshold"].isin(THRESHOLDS)].copy()
    partitions = {
        "validation_may_aug": source[source["role"].eq("validation")],
        "heldback_sep": source[source["role"].eq("evaluation") & source["fold_id"].eq("fold_04")],
    }
    marks = MarkStore()
    all_trades: list[pd.DataFrame] = []
    all_daily: list[pd.DataFrame] = []
    all_summaries: list[dict[str, Any]] = []

    for period, partition in partitions.items():
        for threshold in THRESHOLDS:
            threshold_rows = partition[np.isclose(partition["threshold"], threshold)]
            eligible = threshold_rows[threshold_rows["full_fill"]].copy()
            if eligible.empty:
                raise ValueError(f"no eligible signals for {period} at {threshold}")
            for bankroll in BANKROLLS:
                for fill_fraction in FILL_FRACTIONS:
                    for variant in VARIANTS:
                        trades, diagnostics = simulate_portfolio(
                            eligible,
                            marks,
                            period=period,
                            threshold=threshold,
                            starting_bankroll=bankroll,
                            fill_fraction=fill_fraction,
                            variant=variant,
                        )
                        daily = build_daily_equity(trades, marks, bankroll)
                        summary = summarize_run(
                            trades,
                            daily,
                            diagnostics,
                            phase4_attempts=len(threshold_rows),
                            full_cost_eligible=len(eligible),
                        )
                        all_trades.append(trades)
                        all_daily.append(daily)
                        all_summaries.append(summary)
            print(
                f"completed {period} threshold={threshold:.0%}: "
                f"{len(eligible):,} eligible signals",
                flush=True,
            )

    trades = pd.concat(all_trades, ignore_index=True)
    daily = pd.concat(all_daily, ignore_index=True)
    summary = pd.DataFrame.from_records(all_summaries).sort_values(
        ["period", "threshold", "starting_bankroll", "fill_fraction", "variant"],
        ignore_index=True,
    )
    breakdown = build_breakdown(trades)
    validate_outputs(trades, daily, summary)
    atomic_parquet(trades, TRADE_LEDGER_FILE)
    atomic_parquet(daily, EQUITY_LEDGER_FILE)
    atomic_parquet(summary, SUMMARY_FILE)
    atomic_parquet(breakdown, BREAKDOWN_FILE)
    write_report(summary)
    print(f"wrote {TRADE_LEDGER_FILE}: {len(trades):,} rows", flush=True)
    print(f"wrote {EQUITY_LEDGER_FILE}: {len(daily):,} rows", flush=True)
    print(f"wrote {SUMMARY_FILE}: {len(summary):,} rows", flush=True)
    print(f"wrote {BREAKDOWN_FILE}: {len(breakdown):,} rows", flush=True)
    print(f"wrote {REPORT_FILE}", flush=True)


if __name__ == "__main__":
    main()
