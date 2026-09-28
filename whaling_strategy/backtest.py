"""
Backtester for Strategy 1 — Smart Money Follow.

Requires collect_data.py to have been run first.

Usage:
    python backtest.py
    python backtest.py --start=2024-01-01 --warmup-months=3 --top-n=5
    python backtest.py --results=my_results.csv

Timeline:
    start_date  → anchor_date       : warmup window  (score wallets, no signals)
    anchor_date → end of data       : live simulation (signals + registry updates)
"""

import argparse
import bisect
import math
from collections import defaultdict
from typing import Optional
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config import EDGE_Z_THRESHOLD, MIN_RESOLVED_TRADES, MIN_WHALE_TRADE_USD

# ── Paths (must match collect_data.py) ────────────────────────────────────────
WALLET_TRADES_DIR = "data/wallets"
MARKETS_FILE      = "data/markets.parquet"
PRICES_DIR        = "data/prices"
RESULTS_FILE      = "backtest_results.parquet"

# ── Backtest knobs ─────────────────────────────────────────────────────────────
ROLLING_WINDOW   = 30     # last N resolved trades used for drop/promote decisions
RESCORE_DAYS     = 7      # re-score registry every N simulated days
POSITION_PCT     = 0.10   # our position = 10% of whale's trade
MAX_POSITION_USD = 5_000  # hard cap per position


class Backtester:

    def __init__(
        self,
        start_date:     datetime,
        warmup_months:  int   = 3,
        top_n:          int   = 5,
        min_trade_usd:  float = MIN_WHALE_TRADE_USD,
        z_threshold:    float = EDGE_Z_THRESHOLD,
        min_trades:     int   = MIN_RESOLVED_TRADES,
        rolling_window: int   = ROLLING_WINDOW,
    ):
        self.start_date     = start_date
        self.anchor_date    = start_date + timedelta(days=30 * warmup_months)
        self.top_n          = top_n
        self.min_trade_usd  = min_trade_usd
        self.z_threshold    = z_threshold
        self.min_trades     = min_trades
        self.rolling_window = rolling_window

        # Raw data
        self._wallet_trades: dict[str, list[dict]] = {}  # address → sorted trades
        self._markets:       dict[str, dict]        = {}  # condition_id → metadata
        self._prices:        dict[str, dict]        = {}  # condition_id → {timestamps, prices}

        # Simulation state
        self._registry:        dict[str, dict]  = {}  # address → score
        self._open_positions:  dict[tuple, dict] = {}  # (whale,cid,outcome) → entry info
        self._closed_trades:   list[dict]        = []

    # ── Data loading ───────────────────────────────────────────────────────────

    def load(self):
        self._load_wallet_trades()
        self._load_markets()
        self._load_prices()
        print(
            f"[Backtest] Loaded: {len(self._wallet_trades)} wallets  |  "
            f"{len(self._markets)} markets  |  "
            f"{len(self._prices)} price series"
        )

    def _load_wallet_trades(self):
        for path in Path(WALLET_TRADES_DIR).glob("*.parquet"):
            df = pd.read_parquet(path)
            if df.empty:
                continue
            # Parquet preserves types — just ensure nullable columns are clean
            df["timestamp"]  = pd.to_numeric(df["timestamp"],  errors="coerce").fillna(0)
            df["amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce").fillna(0)
            df["price"]      = pd.to_numeric(df["price"],      errors="coerce").fillna(0.5)
            trades  = df.to_dict("records")
            for t in trades:
                t["tags"] = _parse_tags(t.get("tags"))
            address = trades[0].get("wallet") or path.stem
            self._wallet_trades[address] = sorted(trades, key=lambda t: t["timestamp"])

    def _load_markets(self):
        path = Path(MARKETS_FILE)
        if not path.exists():
            print(f"[Backtest] WARNING: {MARKETS_FILE} not found — resolution P&L unavailable")
            return
        for row in pd.read_parquet(path).to_dict("records"):
            cid = row.get("condition_id", "")
            if not cid:
                continue
            row["tags"]     = _parse_tags(row.get("tags"))
            row["resolved"] = bool(row.get("resolved", False))
            end_str = str(row.get("end_date") or "")
            try:
                end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                row["end_date_ts"] = end_dt.timestamp()
            except Exception:
                row["end_date_ts"] = float("inf")
            self._markets[cid] = row

    def _load_prices(self):
        for path in Path(PRICES_DIR).glob("*.parquet"):
            cid = path.stem
            df  = pd.read_parquet(path).sort_values("timestamp")
            self._prices[cid] = {
                "timestamps": df["timestamp"].astype(int).tolist(),
                "prices":     df["price"].astype(float).tolist(),
            }

    # ── Wallet scoring ─────────────────────────────────────────────────────────

    def _score(self, address: str, cutoff_ts: float, rolling: bool = False) -> dict:
        """
        Score wallet using only resolved BUYs before cutoff_ts.
        rolling=True → use only the last self.rolling_window resolved trades.
        """
        trades = self._wallet_trades.get(address, [])
        resolved_buys = [
            t for t in trades
            if t.get("type", "").upper() == "BUY"
            and t["resolved"]
            and t["winner"] is not None
            and t["timestamp"] <= cutoff_ts
        ]
        if rolling and len(resolved_buys) > self.rolling_window:
            resolved_buys = resolved_buys[-self.rolling_window:]

        n = len(resolved_buys)
        if n == 0:
            return {"address": address, "z_score": 0.0, "trade_count": 0,
                    "qualified": False, "clusters": {}}

        prices  = [t["price"] for t in resolved_buys]
        won     = [1.0 if t["winner"] else 0.0 for t in resolved_buys]
        edge    = sum(w - p for w, p in zip(won, prices)) / n
        se      = math.sqrt(sum(p * (1 - p) for p in prices) / n / n) or 1.0
        z       = edge / se
        clusters = self._cluster_scores(resolved_buys)

        return {
            "address":     address,
            "edge":        round(edge, 4),
            "z_score":     round(z, 4),
            "trade_count": n,
            "qualified":   n >= self.min_trades and z >= self.z_threshold,
            "clusters":    clusters,
        }

    def _cluster_scores(self, resolved_buys: list[dict]) -> dict:
        by_cluster: dict[str, list] = defaultdict(list)
        for t in resolved_buys:
            for tag in (t.get("tags") or ["Unknown"]):
                by_cluster[tag].append(t)

        result = {}
        for cluster, ctrades in by_cluster.items():
            n      = len(ctrades)
            prices = [t["price"] for t in ctrades]
            won    = [1.0 if t["winner"] else 0.0 for t in ctrades]
            edge   = sum(w - p for w, p in zip(won, prices)) / n
            se     = math.sqrt(sum(p * (1 - p) for p in prices) / n / n) or 1.0
            z      = edge / se
            result[cluster] = {
                "edge": round(edge, 4), "z_score": round(z, 4),
                "trade_count": n,
                "qualified": n >= self.min_trades and z >= self.z_threshold,
            }
        return result

    def _qualified_clusters(self, score: dict) -> list[str]:
        return [c for c, info in score.get("clusters", {}).items() if info.get("qualified")]

    # ── Registry management ────────────────────────────────────────────────────

    def _seed_registry(self, anchor_ts: float):
        """
        Score all wallets using only warmup-window data.
        Take top-N qualified by z-score as the starting registry.
        """
        candidates = []
        for address in self._wallet_trades:
            s = self._score(address, anchor_ts, rolling=False)
            if s["qualified"]:
                candidates.append(s)

        candidates.sort(key=lambda s: s["z_score"], reverse=True)

        for s in candidates[: self.top_n]:
            self._registry[s["address"]] = s
            print(
                f"  [Seed] {s['address'][:12]}...  "
                f"z={s['z_score']:.2f}  trades={s['trade_count']}"
            )

        if not self._registry:
            print("  [Seed] WARNING: no wallets qualified at anchor date. "
                  "Try a wider wallet pool or shorter warmup.")
        else:
            print(f"  [Seed] Registry seeded with {len(self._registry)} wallet(s)\n")

    def _update_registry(self, current_ts: float):
        """
        Re-score tracked wallets (rolling window).
        Drop those whose rolling z-score fell below threshold.
        Promote any new wallet in the pool that now qualifies.
        """
        # Drop failing
        for address in list(self._registry):
            s = self._score(address, current_ts, rolling=True)
            if not s["qualified"]:
                print(f"  [Drop] {address[:12]}...  z={s['z_score']:.2f} — below threshold")
                del self._registry[address]
            else:
                self._registry[address] = s

        # Promote new qualifiers
        for address in self._wallet_trades:
            if address in self._registry:
                continue
            s = self._score(address, current_ts, rolling=True)
            if s["qualified"]:
                self._registry[address] = s
                print(
                    f"  [Add]  {address[:12]}...  "
                    f"z={s['z_score']:.2f}  trades={s['trade_count']}"
                )

    # ── Price lookup ───────────────────────────────────────────────────────────

    def _price_at(self, condition_id: str, ts: float) -> Optional[float]:
        series = self._prices.get(condition_id)
        if not series or not series["timestamps"]:
            return None
        idx = bisect.bisect_left(series["timestamps"], int(ts))
        if idx == 0:
            return series["prices"][0]
        if idx >= len(series["timestamps"]):
            return series["prices"][-1]
        before = series["prices"][idx - 1]
        after  = series["prices"][idx]
        diff_before = abs(series["timestamps"][idx - 1] - ts)
        diff_after  = abs(series["timestamps"][idx]     - ts)
        return after if diff_after < diff_before else before

    # ── Entry / exit logic ─────────────────────────────────────────────────────

    def _try_entry(self, trade: dict) -> Optional[dict]:
        address = trade.get("wallet", "")
        if address not in self._registry:
            return None
        if trade.get("type", "").upper() != "BUY":
            return None
        if trade["amount_usd"] < self.min_trade_usd:
            return None

        score         = self._registry[address]
        qual_clusters = self._qualified_clusters(score)
        matching      = [t for t in (trade.get("tags") or []) if t in qual_clusters]
        if not matching:
            return None

        cid         = trade.get("condition_id", "")
        entry_price = self._price_at(cid, trade["timestamp"]) or trade["price"]
        if not entry_price:
            return None
        position    = round(min(trade["amount_usd"] * POSITION_PCT, MAX_POSITION_USD), 2)

        return {
            "whale":        address,
            "condition_id": cid,
            "outcome":      trade.get("outcome", ""),
            "cluster":      matching[0],
            "entry_price":  entry_price,
            "entry_ts":     trade["timestamp"],
            "position_usd": position,
            "z_score":      score["z_score"],
        }

    def _try_exit_by_sell(self, trade: dict) -> Optional[tuple]:
        """Return position key if this SELL from a tracked whale closes a position."""
        if trade.get("type", "").upper() != "SELL":
            return None
        address = trade.get("wallet", "")
        if address not in self._registry:
            return None
        key = (address, trade.get("condition_id", ""), trade.get("outcome", ""))
        return key if key in self._open_positions else None

    def _open_position(self, entry: dict):
        key = (entry["whale"], entry["condition_id"], entry["outcome"])
        if key in self._open_positions:
            return  # already tracking
        self._open_positions[key] = entry
        print(
            f"  [ENTRY] {entry['whale'][:10]}...  "
            f"{entry['outcome']:4s}  @ {entry['entry_price']:.3f}  "
            f"cluster={entry['cluster']}  ${entry['position_usd']:,.0f}"
        )

    def _close_position(self, key: tuple, exit_price: float, exit_ts: float, reason: str):
        entry = self._open_positions.pop(key, None)
        if not entry:
            return
        pnl_usd = round(entry["position_usd"] * (exit_price / entry["entry_price"] - 1), 2)
        pnl_pct = round((exit_price / entry["entry_price"] - 1) * 100, 2)
        self._closed_trades.append({
            "whale":        entry["whale"],
            "condition_id": entry["condition_id"],
            "outcome":      entry["outcome"],
            "cluster":      entry["cluster"],
            "entry_ts":     _fmt_ts(entry["entry_ts"]),
            "exit_ts":      _fmt_ts(exit_ts),
            "entry_price":  entry["entry_price"],
            "exit_price":   exit_price,
            "position_usd": entry["position_usd"],
            "pnl_usd":      pnl_usd,
            "pnl_pct":      pnl_pct,
            "reason":       reason,
            "z_score":      entry["z_score"],
        })
        flag = "+" if pnl_usd >= 0 else ""
        print(
            f"  [{reason.upper()[:6]}] {entry['whale'][:10]}...  "
            f"entry={entry['entry_price']:.3f}  exit={exit_price:.3f}  "
            f"PnL ${flag}{pnl_usd:,.2f}  ({flag}{pnl_pct:.1f}%)"
        )

    def _resolve_open_positions(self, current_ts: float):
        """Close positions in markets whose end_date has passed."""
        for key in list(self._open_positions):
            _, cid, outcome = key
            market          = self._markets.get(cid, {})
            if not market.get("resolved"):
                continue
            if current_ts < market.get("end_date_ts", float("inf")):
                continue
            winning = str(market.get("winning_outcome") or "").lower()
            did_win = outcome.lower() == winning
            self._close_position(key, 1.0 if did_win else 0.0, current_ts, "resolution")

    # ── Main replay ────────────────────────────────────────────────────────────

    def run(self):
        self.load()

        anchor_ts = self.anchor_date.timestamp()
        all_ts    = [
            t["timestamp"]
            for trades in self._wallet_trades.values()
            for t in trades
            if t["timestamp"] > 0
        ]
        if not all_ts:
            print("[Backtest] No trades loaded — run collect_data.py first.")
            return
        end_ts = max(all_ts)

        print(f"\n[Backtest] Warmup    : {self.start_date.date()} → {self.anchor_date.date()}")
        print(f"[Backtest] Simulation: {self.anchor_date.date()} → "
              f"{datetime.fromtimestamp(end_ts, tz=timezone.utc).date()}")
        print(f"[Backtest] Registry  : top {self.top_n} at anchor "
              f"(z >= {self.z_threshold}, N >= {self.min_trades})\n")

        # Build chronological event stream from anchor date onwards
        events = sorted(
            (t for trades in self._wallet_trades.values() for t in trades
             if t["timestamp"] >= anchor_ts),
            key=lambda t: t["timestamp"],
        )

        print(f"[Backtest] Anchor day — seeding registry from warmup window:")
        self._seed_registry(anchor_ts)

        last_rescore_ts  = anchor_ts
        rescore_interval = RESCORE_DAYS * 86_400

        for trade in events:
            ts = trade["timestamp"]

            # Periodic re-score
            if ts - last_rescore_ts >= rescore_interval:
                date = datetime.fromtimestamp(ts, tz=timezone.utc).date()
                print(f"\n[Rescore — {date}]  registry size: {len(self._registry)}")
                self._update_registry(ts)
                last_rescore_ts = ts

            # Resolve positions where market has closed
            self._resolve_open_positions(ts)

            # Exit: whale is selling something we hold
            exit_key = self._try_exit_by_sell(trade)
            if exit_key:
                exit_price = (
                    self._price_at(exit_key[1], ts)
                    or trade["price"]
                )
                self._close_position(exit_key, exit_price, ts, reason="whale_exit")
                continue

            # Entry: qualified whale BUY in a matching cluster
            entry = self._try_entry(trade)
            if entry:
                self._open_position(entry)

        # Force-close anything still open at last available price
        if self._open_positions:
            print(f"\n[Backtest] Force-closing {len(self._open_positions)} remaining positions...")
            for key in list(self._open_positions):
                _, cid, _ = key
                series     = self._prices.get(cid)
                last_price = series["prices"][-1] if series else None
                if last_price is not None:
                    self._close_position(key, last_price, end_ts, "end_of_data")
                else:
                    self._open_positions.pop(key, None)

        self._print_summary()
        self._save_results()

    # ── Output ─────────────────────────────────────────────────────────────────

    def _print_summary(self):
        n = len(self._closed_trades)
        if not n:
            print("\n[Backtest] No completed trades in this period.")
            return

        total_pnl = sum(t["pnl_usd"] for t in self._closed_trades)
        wins      = sum(1 for t in self._closed_trades if t["pnl_usd"] > 0)
        by_reason: dict[str, list] = defaultdict(list)
        by_cluster: dict[str, list] = defaultdict(list)
        for t in self._closed_trades:
            by_reason[t["reason"]].append(t["pnl_usd"])
            by_cluster[t["cluster"]].append(t["pnl_usd"])

        print(f"\n{'='*60}")
        print(f"  Total trades  : {n}")
        print(f"  Win rate      : {wins/n*100:.1f}%  ({wins}/{n})")
        print(f"  Total PnL     : ${total_pnl:+,.2f}")
        print(f"  By exit type:")
        for reason, pnls in sorted(by_reason.items()):
            print(f"    {reason:<14}  {len(pnls):3d} trades  ${sum(pnls):+,.2f}")
        print(f"  By cluster:")
        for cluster, pnls in sorted(by_cluster.items(), key=lambda x: sum(x[1]), reverse=True):
            print(f"    {cluster:<14}  {len(pnls):3d} trades  ${sum(pnls):+,.2f}")
        print(f"{'='*60}\n")

    def _save_results(self, path: str = RESULTS_FILE):
        if not self._closed_trades:
            return
        pd.DataFrame(self._closed_trades).to_parquet(path, index=False)
        print(f"[Backtest] Results → {path}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_tags(raw) -> list[str]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return [raw] if raw else []
    return []


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="Strategy 1 Backtester")
    p.add_argument("--start",         default="2024-01-01",
                   help="Start of warmup window (ISO date, default 2024-01-01)")
    p.add_argument("--warmup-months", type=int, default=3,
                   help="Months of warmup before signals begin (default 3)")
    p.add_argument("--top-n",         type=int, default=5,
                   help="Initial registry size at anchor day (default 5)")
    p.add_argument("--results",       default=RESULTS_FILE,
                   help=f"Output CSV path (default {RESULTS_FILE})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    bt   = Backtester(
        start_date    = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc),
        warmup_months = args.warmup_months,
        top_n         = args.top_n,
    )
    bt.run()
