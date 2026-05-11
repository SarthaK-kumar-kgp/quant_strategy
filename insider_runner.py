"""
Live + backtest runner for Strategy 2 (Insider Capture) — threshold-based.

Consumes the calibrated knobs produced by analyze_insider_data.py.

Two modes share the SAME signal/PnL logic so live and backtest results
are directly comparable:

    live      — polls Polymarket every N seconds for markets nearing close,
                emits a signal when a trade passes all knob filters, opens
                a paper position, marks-to-market at resolution.
    backtest  — replays a historical parquet (data/insider_raw.parquet)
                through the same logic, producing a comparable signal log.

Usage:
    python insider_runner.py backtest
    python insider_runner.py backtest --knobs=knobs.json
    python insider_runner.py live      --knobs=knobs.json
    python insider_runner.py live      --paper-size=500 --poll-secs=120

Output: signals_<mode>.parquet  — one row per emitted signal, with eventual
PnL filled in once each market resolves. Feed this back into
analyze_insider_data.py to verify live results match backtest expectations.
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from config import GAMMA_API, DATA_API
from data_fetcher import PolymarketFetcher


# ══════════════════════════════════════════════════════════════════════════════
#  KNOBS — defaults come straight from analyze_insider_data.py recommendations.
#  Override at runtime via --knobs=<path-to-json> with the same field names.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Knobs:
    # Filter knobs (calibrated by analyze_insider_data.py)
    min_usd:               float = 100_000   # min trade size to consider
    max_hours_before_close: float = 24       # only trades within this many hours of close
    max_wallet_age_days:   Optional[float] = None   # None = any age
    side_filter:           Optional[str]   = None   # "BUY" / "SELL" / None
    min_wallet_history_usd: Optional[float] = None  # require some on-chain history

    # Position sizing (paper trading)
    follow_size_usd:       float = 100       # how much to "stake" per signal
    confirmation_required: bool  = False     # if True, need a 2nd large same-side trade

    # Live polling cadence
    poll_secs:             int   = 60        # seconds between market scans
    market_scan_horizon_h: int   = 168       # only watch markets closing in next N hours

    @classmethod
    def from_json(cls, path: Path) -> "Knobs":
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def to_json(self, path: Path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))


# ══════════════════════════════════════════════════════════════════════════════
#  Pure helpers (shared by live + backtest)
# ══════════════════════════════════════════════════════════════════════════════

def passes_filters(
    trade_usd:          float,
    hours_before_close: float,
    wallet_age_days:    Optional[float],
    side:               str,
    knobs:              Knobs,
) -> bool:
    """A single trade either fires a signal or it doesn't — same logic both modes."""
    if trade_usd < knobs.min_usd:
        return False
    if hours_before_close > knobs.max_hours_before_close:
        return False
    if hours_before_close < 0:                       # post-resolution trade — skip
        return False
    if knobs.max_wallet_age_days is not None:
        if wallet_age_days is None or wallet_age_days > knobs.max_wallet_age_days:
            return False
    if knobs.side_filter is not None:
        if (side or "").upper() != knobs.side_filter.upper():
            return False
    return True


def compute_pnl(side: str, entry_price: float, resolution_outcome: str) -> tuple[float, float]:
    """
    Returns (roi_pct, payoff_per_dollar). Trades are on the YES token.
       BUY  YES at p:   payoff = $1 if YES wins, $0 if NO  → roi = (1-p)/p or -100%
       SELL YES at p:   payoff = $1 if NO  wins, $0 if YES → roi = p/(1-p) or -100%
    """
    p = max(min(entry_price, 0.9999), 0.0001)
    is_buy = (side or "").upper() == "BUY"
    yes    = (resolution_outcome or "").upper() == "YES"

    if is_buy:
        roi = (1 - p) / p if yes else -1.0
    else:
        roi = p / (1 - p) if not yes else -1.0
    return roi, 1.0 + roi


# ══════════════════════════════════════════════════════════════════════════════
#  Signal log (shared schema between live + backtest)
# ══════════════════════════════════════════════════════════════════════════════

SIGNAL_COLS = [
    "mode", "signal_emitted_ts",
    "condition_id", "question",
    "wallet", "side", "trade_price", "trade_usd", "trade_ts",
    "hours_before_close", "wallet_age_days_at_trade",
    "follow_size_usd", "entry_price",
    "resolution_ts", "resolution_outcome",
    "roi_pct", "pnl_usd", "status",
]


def _empty_signal_log() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in SIGNAL_COLS})


def _append_signal(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    return pd.concat([df, pd.DataFrame([row])[SIGNAL_COLS]], ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST MODE — replay parquet through the same passes_filters() logic
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(knobs: Knobs, in_path: Path, out_path: Path):
    if not in_path.exists():
        print(f"[backtest] No file at {in_path}. Run collect_insider_data.py first.")
        return

    df = pd.read_parquet(in_path)
    print(f"[backtest] Loaded {len(df):,} historical trades from {in_path}")
    print(f"[backtest] Knobs: {asdict(knobs)}\n")

    log = _empty_signal_log()
    skipped_reasons = {"size": 0, "hours": 0, "age": 0, "side": 0}

    for _, t in df.iterrows():
        # Per-row reason tracking so the user can see WHY a trade got rejected
        size_ok  = t["trade_usd"] >= knobs.min_usd
        hours_ok = 0 <= t["hours_before_close"] <= knobs.max_hours_before_close
        age      = t.get("wallet_age_days_at_trade")
        age_ok   = (knobs.max_wallet_age_days is None) or (
            pd.notna(age) and age <= knobs.max_wallet_age_days
        )
        side_ok  = (knobs.side_filter is None) or (
            str(t.get("side", "")).upper() == knobs.side_filter.upper()
        )

        if not size_ok:    skipped_reasons["size"]  += 1; continue
        if not hours_ok:   skipped_reasons["hours"] += 1; continue
        if not age_ok:     skipped_reasons["age"]   += 1; continue
        if not side_ok:    skipped_reasons["side"]  += 1; continue

        roi, _ = compute_pnl(
            side               = str(t.get("side", "")),
            entry_price        = float(t["trade_price"]),
            resolution_outcome = str(t.get("resolution_outcome", "")),
        )
        pnl_usd = roi * knobs.follow_size_usd

        log = _append_signal(log, {
            "mode":                       "backtest",
            "signal_emitted_ts":          int(t["trade_ts"]),
            "condition_id":               t.get("condition_id"),
            "question":                   t.get("question"),
            "wallet":                     t.get("wallet"),
            "side":                       t.get("side"),
            "trade_price":                float(t["trade_price"]),
            "trade_usd":                  float(t["trade_usd"]),
            "trade_ts":                   int(t["trade_ts"]),
            "hours_before_close":         float(t["hours_before_close"]),
            "wallet_age_days_at_trade":   age if pd.notna(age) else None,
            "follow_size_usd":            knobs.follow_size_usd,
            "entry_price":                float(t["trade_price"]),
            "resolution_ts":              int(t["resolution_ts"]),
            "resolution_outcome":         t.get("resolution_outcome"),
            "roi_pct":                    round(roi * 100, 2),
            "pnl_usd":                    round(pnl_usd, 2),
            "status":                     "resolved",
        })

    log.to_parquet(out_path, index=False)

    # ── Summary ──
    n = len(log)
    print(f"[backtest] Signals fired: {n:,}")
    print(f"[backtest] Skipped — size:{skipped_reasons['size']:,}  "
          f"hours:{skipped_reasons['hours']:,}  age:{skipped_reasons['age']:,}  "
          f"side:{skipped_reasons['side']:,}")
    if n:
        wins   = (log["roi_pct"] > 0).sum()
        total_pnl = log["pnl_usd"].sum()
        total_stake = n * knobs.follow_size_usd
        print(f"[backtest] Hit rate     : {wins/n:.1%} ({wins}/{n})")
        print(f"[backtest] Total PnL    : ${total_pnl:+,.2f} on ${total_stake:,.0f} staked")
        print(f"[backtest] Mean ROI     : {log['roi_pct'].mean():+.2f}%")
        print(f"[backtest] Median ROI   : {log['roi_pct'].median():+.2f}%")
    print(f"[backtest] Signals saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE MODE — poll Polymarket; emit signals; mark-to-market at resolution
# ══════════════════════════════════════════════════════════════════════════════

class LiveRunner:
    """Polls active markets nearing close, fires signals on qualifying trades,
    holds open paper positions until each market resolves, then writes PnL."""

    def __init__(self, knobs: Knobs, out_path: Path):
        self.knobs       = knobs
        self.out_path    = out_path
        self.log         = (
            pd.read_parquet(out_path) if out_path.exists() else _empty_signal_log()
        )
        self.seen_trades: set[str] = set()
        # Resume open positions from previous run
        if not self.log.empty:
            self.seen_trades = set(self.log["trade_ts"].astype(str).tolist())

    async def run(self):
        async with PolymarketFetcher() as fetcher:
            print(f"[live] Knobs: {asdict(self.knobs)}")
            print(f"[live] Polling every {self.knobs.poll_secs}s — "
                  f"horizon {self.knobs.market_scan_horizon_h}h\n")
            while True:
                try:
                    await self._scan_for_signals(fetcher)
                    await self._resolve_open_positions(fetcher)
                    self._save()
                except Exception as e:
                    print(f"[live] error in cycle: {e}")
                await asyncio.sleep(self.knobs.poll_secs)

    async def _scan_for_signals(self, fetcher: PolymarketFetcher):
        """Pull markets closing in next horizon, check their recent trades."""
        markets = await self._markets_closing_soon(fetcher)
        if not markets:
            return
        sem = asyncio.Semaphore(4)
        await asyncio.gather(*[
            self._check_market_for_signals(fetcher, m, sem) for m in markets
        ])

    async def _markets_closing_soon(self, fetcher: PolymarketFetcher) -> list[dict]:
        """Active Sports + Politics markets resolving within the scan horizon."""
        now      = datetime.now(timezone.utc)
        cutoff   = now + timedelta(hours=self.knobs.market_scan_horizon_h)
        results: list = []
        # tag_id 1 = Sports, 2 = Politics
        for tag_id in (1, 2):
            offset = 0
            while True:
                try:
                    batch = await fetcher._get(f"{GAMMA_API}/markets", params={
                        "closed": "false", "tag_id": str(tag_id),
                        "limit": 500, "offset": offset,
                    })
                except Exception:
                    break
                rows = batch if isinstance(batch, list) else batch.get("data", [])
                if not rows:
                    break
                for m in rows:
                    end = m.get("endDateIso") or m.get("endDate") or ""
                    try:
                        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
                    except Exception:
                        continue
                    if now <= end_dt <= cutoff:
                        results.append(m)
                if len(rows) < 500:
                    break
                offset += 500
        return results

    async def _check_market_for_signals(
        self, fetcher: PolymarketFetcher, market: dict, sem: asyncio.Semaphore
    ):
        end = market.get("endDateIso") or market.get("endDate") or ""
        try:
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except Exception:
            return
        resolution_ts = int(end_dt.timestamp())

        yes_tok = _extract_yes_token(market)
        if not yes_tok:
            return

        async with sem:
            try:
                trades = await fetcher._get(f"{DATA_API}/trades", params={
                    "market": yes_tok, "limit": 200,
                })
            except Exception:
                return

        if not isinstance(trades, list):
            return

        cid = market.get("conditionId") or market.get("condition_id", "")
        for t in trades:
            ts = int(float(t.get("timestamp") or 0) or 0)
            if not ts:
                continue
            tid = f"{cid}:{ts}:{t.get('proxyWallet','')}"
            if tid in self.seen_trades:
                continue
            self.seen_trades.add(tid)

            hours_before = (resolution_ts - ts) / 3600
            usd          = float(t.get("usdcSize", 0) or 0)
            side         = str(t.get("side", "")).upper()
            wallet       = t.get("proxyWallet", "")

            wallet_age = None
            if self.knobs.max_wallet_age_days is not None and wallet:
                wallet_age = await self._wallet_age_days(fetcher, wallet)

            if not passes_filters(usd, hours_before, wallet_age, side, self.knobs):
                continue

            self._emit_signal(t, market, resolution_ts, hours_before, wallet_age)

    async def _wallet_age_days(self, fetcher, wallet: str) -> Optional[float]:
        try:
            page = await fetcher._get(f"{DATA_API}/activity", params={
                "user": wallet, "limit": 500, "offset": 0,
            })
        except Exception:
            return None
        rows = page if isinstance(page, list) else page.get("data", [])
        if not rows:
            return 0.0
        timestamps = [float(t.get("timestamp", 0) or 0) for t in rows if t.get("timestamp")]
        if not timestamps:
            return None
        first = min(timestamps)
        return (time.time() - first) / 86400

    def _emit_signal(self, trade: dict, market: dict,
                     resolution_ts: int, hours_before: float,
                     wallet_age: Optional[float]):
        price = float(trade.get("price", 0) or 0)
        side  = str(trade.get("side", "")).upper()
        cid   = market.get("conditionId", "")
        row = {
            "mode":                       "live",
            "signal_emitted_ts":          int(time.time()),
            "condition_id":               cid,
            "question":                   market.get("question", ""),
            "wallet":                     trade.get("proxyWallet", ""),
            "side":                       side,
            "trade_price":                price,
            "trade_usd":                  float(trade.get("usdcSize", 0) or 0),
            "trade_ts":                   int(float(trade.get("timestamp") or 0)),
            "hours_before_close":         round(hours_before, 2),
            "wallet_age_days_at_trade":   wallet_age,
            "follow_size_usd":            self.knobs.follow_size_usd,
            "entry_price":                price,
            "resolution_ts":              resolution_ts,
            "resolution_outcome":         None,
            "roi_pct":                    None,
            "pnl_usd":                    None,
            "status":                     "open",
        }
        self.log = _append_signal(self.log, row)
        print(f"[SIGNAL] {row['question'][:50]}  "
              f"{side} @ {price:.3f}  ${row['trade_usd']:,.0f}  "
              f"{hours_before:.1f}h to close  "
              f"wallet={row['wallet'][:10]}...")

    async def _resolve_open_positions(self, fetcher: PolymarketFetcher):
        """For each open paper position, check if the market has resolved.
        If yes, fill in resolution_outcome + PnL."""
        if self.log.empty:
            return
        open_mask = self.log["status"] == "open"
        if not open_mask.any():
            return
        open_cids = self.log.loc[open_mask, "condition_id"].unique()
        for cid in open_cids:
            try:
                m = await fetcher._get(f"{GAMMA_API}/markets/{cid}")
            except Exception:
                continue
            if not isinstance(m, dict) or not m.get("closed"):
                continue
            outcome = (m.get("winner") or m.get("winningOutcome") or "").upper()
            if not outcome:
                continue
            mask = open_mask & (self.log["condition_id"] == cid)
            for idx in self.log.index[mask]:
                roi, _ = compute_pnl(
                    side               = str(self.log.at[idx, "side"]),
                    entry_price        = float(self.log.at[idx, "entry_price"]),
                    resolution_outcome = outcome,
                )
                self.log.at[idx, "resolution_outcome"] = outcome
                self.log.at[idx, "roi_pct"]            = round(roi * 100, 2)
                self.log.at[idx, "pnl_usd"]            = round(roi * self.knobs.follow_size_usd, 2)
                self.log.at[idx, "status"]             = "resolved"
                print(f"[CLOSE] {self.log.at[idx, 'question'][:50]}  "
                      f"outcome={outcome}  roi={roi:+.1%}  "
                      f"pnl=${self.log.at[idx, 'pnl_usd']:+,.2f}")

    def _save(self):
        if not self.log.empty:
            self.log.to_parquet(self.out_path, index=False)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_yes_token(market: dict) -> Optional[str]:
    raw = market.get("clobTokenIds")
    if raw:
        try:
            ids = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(ids, list) and ids and ids[0]:
                return str(ids[0])
        except Exception:
            pass
    for t in market.get("tokens") or []:
        if isinstance(t, dict) and (t.get("outcome") or "").upper() == "YES":
            return t.get("token_id") or t.get("tokenId")
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _load_knobs(args) -> Knobs:
    if args.knobs:
        knobs = Knobs.from_json(Path(args.knobs))
        print(f"[knobs] loaded from {args.knobs}")
    else:
        knobs = Knobs()
    # CLI overrides
    if args.min_usd        is not None: knobs.min_usd               = args.min_usd
    if args.max_hours      is not None: knobs.max_hours_before_close = args.max_hours
    if args.max_age        is not None: knobs.max_wallet_age_days   = args.max_age
    if args.side           is not None: knobs.side_filter           = args.side
    if args.paper_size     is not None: knobs.follow_size_usd       = args.paper_size
    if args.poll_secs      is not None: knobs.poll_secs             = args.poll_secs
    return knobs


def main():
    p = argparse.ArgumentParser(description="Strategy 2 live + backtest runner")
    p.add_argument("mode", choices=["live", "backtest"])
    p.add_argument("--knobs",       help="Path to knobs JSON (overrides defaults)")
    p.add_argument("--in",  dest="in_path",
                   default="data/insider_raw.parquet",
                   help="(backtest) Input parquet from collect_insider_data.py")
    p.add_argument("--out", dest="out_path", default=None,
                   help="Output signal parquet (default: signals_<mode>.parquet)")

    # CLI knob overrides
    p.add_argument("--min_usd",     type=float)
    p.add_argument("--max_hours",   type=float)
    p.add_argument("--max_age",     type=float)
    p.add_argument("--side",        choices=["BUY", "SELL"])
    p.add_argument("--paper_size",  type=float)
    p.add_argument("--poll_secs",   type=int)

    args  = p.parse_args()
    knobs = _load_knobs(args)
    out   = Path(args.out_path or f"signals_{args.mode}.parquet")

    if args.mode == "backtest":
        run_backtest(knobs, Path(args.in_path), out)
    else:
        asyncio.run(LiveRunner(knobs, out).run())


if __name__ == "__main__":
    main()
