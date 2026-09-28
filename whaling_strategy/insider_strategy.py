"""
Insider / Anomalous Timing Detection — Strategy 2

Detects trades that look like insider knowledge:
  - Abnormally large vs. market baseline  (z_size)
  - Placed close to resolution            (w_prox)
  - Aggressive one-sided order flow       (z_ofi)
  - From a fresh / single-purpose wallet  (novelty)

Entry points:
    python insider_strategy.py                      # WebSocket (default)
    python insider_strategy.py --poll               # fast-poll fallback
    python insider_strategy.py --require-same-wallet  # looser confirmation mode
"""

import argparse
import asyncio
import csv
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import websockets

from config import (
    BASELINE_LOOKBACK_DAYS,
    CLOB_API,
    CONFIRMATION_MIN_SIZE_RATIO,
    CONFIRMATION_WINDOW_S,
    INSIDER_SIGNALS_CSV,
    IS_MIN_THRESHOLD,
    MAX_CONCURRENT_REQUESTS,
    NOVELTY_FUNDING_DAYS,
    NOVELTY_WALLET_AGE_DAYS,
    OFI_WINDOW_MINUTES,
    SIZE_ZSCORE_TRIGGER,
    WS_POLL_FALLBACK_INTERVAL_S,
    WS_URL,
)
from data_fetcher import PolymarketFetcher, extract_wallet_address


# ── Market Baseline ────────────────────────────────────────────────────────────

class MarketBaseline:
    """
    Rolling 7-day statistics for a single market.
    Tracks trade size distribution and order-flow imbalance (OFI).
    Updated live as new trades arrive; also warmed from historical data on startup.
    """

    def __init__(self, condition_id: str):
        self.condition_id = condition_id
        # Size distribution — store raw values, recompute stats incrementally
        self._sizes: deque = deque(maxlen=10_000)
        self._size_mean = 0.0
        self._size_std  = 1.0
        # OFI per rolling 1-hour window
        self._ofi_history: deque = deque(maxlen=7 * 24)  # 7 days × 24 h
        self._ofi_mean    = 0.0
        self._ofi_std     = 1.0
        # Current (open) OFI window accumulators
        self._window_start  = time.time()
        self._window_buy    = 0.0
        self._window_sell   = 0.0

    def ingest(self, amount_usd: float, side: str, ts: float = None):
        ts = ts or time.time()

        self._sizes.append(amount_usd)
        self._update_size_stats()

        if ts - self._window_start >= OFI_WINDOW_MINUTES * 60:
            self._flush_ofi_window()
            self._window_start = ts
            self._window_buy   = 0.0
            self._window_sell  = 0.0

        if side.upper() == "BUY":
            self._window_buy  += amount_usd
        else:
            self._window_sell += amount_usd

    def _flush_ofi_window(self):
        total = self._window_buy + self._window_sell
        if total > 0:
            ofi = (self._window_buy - self._window_sell) / total
            self._ofi_history.append(ofi)
            self._update_ofi_stats()

    def _update_size_stats(self):
        n = len(self._sizes)
        if n < 2:
            return
        vals = list(self._sizes)
        mean = sum(vals) / n
        std  = math.sqrt(sum((x - mean) ** 2 for x in vals) / n) or 1.0
        self._size_mean, self._size_std = mean, std

    def _update_ofi_stats(self):
        n = len(self._ofi_history)
        if n < 2:
            return
        vals = list(self._ofi_history)
        mean = sum(vals) / n
        std  = math.sqrt(sum((x - mean) ** 2 for x in vals) / n) or 1.0
        self._ofi_mean, self._ofi_std = mean, std

    def size_zscore(self, amount_usd: float) -> float:
        return (amount_usd - self._size_mean) / self._size_std if self._size_std else 0.0

    def current_ofi_zscore(self) -> float:
        total = self._window_buy + self._window_sell
        if total == 0 or self._ofi_std == 0:
            return 0.0
        ofi = (self._window_buy - self._window_sell) / total
        return (ofi - self._ofi_mean) / self._ofi_std

    def is_warm(self) -> bool:
        return len(self._sizes) >= 10


# ── Wallet Novelty ─────────────────────────────────────────────────────────────

class WalletNoveltyChecker:
    """
    Scores 0–3 based on how "fresh" a wallet looks.
    Proxies blockchain novelty using Polymarket activity history.
    Results are cached so each wallet is only fetched once per session.
    """

    def __init__(self):
        self._cache: dict[str, int] = {}

    async def score(self, address: str, fetcher: PolymarketFetcher) -> int:
        if address not in self._cache:
            self._cache[address] = await self._compute(address, fetcher)
        return self._cache[address]

    async def _compute(self, address: str, fetcher: PolymarketFetcher) -> int:
        try:
            trades = await fetcher.fetch_wallet_trades(address, resolved_only=False)
        except Exception:
            return 0

        if not trades:
            return 3  # never traded before on Polymarket — maximally novel

        timestamps = [float(t.get("timestamp", 0) or 0) for t in trades]
        first_ts   = min(ts for ts in timestamps if ts > 0) if any(timestamps) else 0

        now = datetime.now(timezone.utc)
        if first_ts:
            first_seen = datetime.fromtimestamp(first_ts, tz=timezone.utc)
            age_days   = (now - first_seen).days
        else:
            age_days   = 9999

        is_new              = age_days < NOVELTY_WALLET_AGE_DAYS
        recently_funded     = age_days < NOVELTY_FUNDING_DAYS
        unique_categories   = {tag for t in trades for tag in (t.get("tags") or [])}
        is_single_purpose   = len(unique_categories) <= 2

        return int(is_new) + int(recently_funded) + int(is_single_purpose)


# ── Insider Scorer ─────────────────────────────────────────────────────────────

class InsiderScorer:
    """Pure function — computes the composite IS score from the four components."""

    @staticmethod
    def compute(
        z_size: float,
        hours_to_resolution: float,
        z_ofi: float,
        novelty: int,
    ) -> float:
        w_prox = math.exp(-hours_to_resolution / 24)
        return round(z_size * w_prox * (1 + 0.3 * z_ofi) * (1 + 0.2 * novelty), 4)


# ── Pending Signal & Confirmation Tracker ──────────────────────────────────────

@dataclass
class PendingSignal:
    original_trade:      dict
    is_score:            float
    condition_id:        str
    side:                str
    amount_usd:          float
    wallet:              str
    created_at:          float = field(default_factory=time.time)


class ConfirmationTracker:
    """
    Holds trades that have crossed IS_MIN but haven't been confirmed yet.

    Two confirmation modes (configurable at runtime):
      require_different_wallet=True  — confirms only if a DIFFERENT wallet
                                       buys the same side (rules out order splitting)
      require_different_wallet=False — any same-direction trade confirms
                                       (faster but looser)

    In both modes the confirming trade must be >= min_size_ratio × original size.
    """

    def __init__(
        self,
        require_different_wallet: bool = True,
        min_size_ratio: float          = CONFIRMATION_MIN_SIZE_RATIO,
    ):
        self.require_different_wallet = require_different_wallet
        self.min_size_ratio           = min_size_ratio
        self._pending: list[PendingSignal] = []

    def add(self, trade: dict, is_score: float):
        ps = PendingSignal(
            original_trade = trade,
            is_score       = is_score,
            condition_id   = trade.get("condition_id", ""),
            side           = trade.get("type", "").upper(),
            amount_usd     = trade.get("amount_usd", 0) or 0,
            wallet         = extract_wallet_address(trade) or "",
        )
        self._pending.append(ps)
        print(
            f"[Insider] Pending  IS={is_score:.2f} | "
            f"{trade.get('title', 'unknown market')[:45]} | "
            f"${ps.amount_usd:,.0f} — waiting for confirmation "
            f"({'diff wallet' if self.require_different_wallet else 'any wallet'}, "
            f"min {self.min_size_ratio:.0%} size)"
        )

    def check(self, new_trade: dict) -> Optional[PendingSignal]:
        """Returns the pending signal if new_trade confirms it, else None."""
        cid       = new_trade.get("condition_id", "")
        side      = new_trade.get("type", "").upper()
        amount    = new_trade.get("amount_usd", 0) or 0
        wallet    = extract_wallet_address(new_trade) or ""

        for ps in self._pending:
            if ps.condition_id != cid:
                continue
            if ps.side != side:
                continue
            if amount < ps.amount_usd * self.min_size_ratio:
                continue
            if self.require_different_wallet and wallet == ps.wallet:
                continue
            return ps
        return None

    def expire(self):
        cutoff       = time.time() - CONFIRMATION_WINDOW_S
        before       = len(self._pending)
        self._pending = [ps for ps in self._pending if ps.created_at >= cutoff]
        expired      = before - len(self._pending)
        if expired:
            print(f"[Insider] {expired} pending signal(s) expired without confirmation.")


# ── Main Strategy ──────────────────────────────────────────────────────────────

class InsiderStrategy:
    """
    Full pipeline:
      1. Warm baselines from 7 days of historical CLOB data (startup)
      2. Listen to live trades via WebSocket (or fast poll as fallback)
      3. Score each trade with the IS formula
      4. Wait for confirmation within CONFIRMATION_WINDOW_S
      5. Emit confirmed signals to insider_signals.csv
    """

    def __init__(
        self,
        require_different_wallet: bool = True,
        min_size_ratio: float          = CONFIRMATION_MIN_SIZE_RATIO,
        is_min_threshold: float        = IS_MIN_THRESHOLD,
    ):
        self._is_min      = is_min_threshold
        self._baselines:  dict[str, MarketBaseline] = {}
        self._markets:    dict[str, dict]           = {}   # condition_id → market info
        self._novelty     = WalletNoveltyChecker()
        self._confirmer   = ConfirmationTracker(require_different_wallet, min_size_ratio)
        self._emitted:    set[str]                  = set()
        self._csv         = Path(INSIDER_SIGNALS_CSV)
        self._init_csv()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _init_csv(self):
        if not self._csv.exists():
            with open(self._csv, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timestamp", "market_id", "market_title", "cluster",
                    "trade_wallet", "trade_size_usd", "entry_price", "outcome",
                    "z_size", "z_ofi", "w_prox", "novelty", "is_score",
                    "hours_to_resolution",
                    "confirmed_by_wallet", "confirm_size_usd",
                    "confirmation_mode",
                ])

    def _baseline(self, condition_id: str) -> MarketBaseline:
        if condition_id not in self._baselines:
            self._baselines[condition_id] = MarketBaseline(condition_id)
        return self._baselines[condition_id]

    def _hours_to_resolution(self, condition_id: str) -> float:
        market   = self._markets.get(condition_id, {})
        end_date = (
            market.get("end_date_iso")
            or market.get("endDate")
            or market.get("game_start_time")
        )
        if not end_date:
            return 24 * 30  # unknown → treat as far away (low w_prox weight)
        try:
            end_dt = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
            hours  = max(0.0, (end_dt - datetime.now(timezone.utc)).total_seconds() / 3600)
            return hours
        except Exception:
            return 24 * 30

    # ── Baseline warmup ────────────────────────────────────────────────────────

    async def warm_baselines(self, fetcher: PolymarketFetcher):
        print("[InsiderBaseline] Fetching active markets...")
        markets = await fetcher.fetch_active_markets()
        print(f"[InsiderBaseline] Warming baselines for {len(markets)} markets "
              f"(last {BASELINE_LOOKBACK_DAYS} days)...")

        cutoff = datetime.now(timezone.utc) - timedelta(days=BASELINE_LOOKBACK_DAYS)

        async def _warm(market: dict):
            cid = (
                market.get("conditionId")
                or market.get("condition_id")
                or market.get("id", "")
            )
            if not cid:
                return
            self._markets[cid] = market
            baseline = self._baseline(cid)
            try:
                raw = await fetcher.fetch_clob_market_trades(cid)
                for t in raw:
                    ts = float(t.get("match_time") or t.get("timestamp") or 0)
                    if ts and datetime.fromtimestamp(ts, tz=timezone.utc) < cutoff:
                        continue
                    price  = float(t.get("price", 0) or 0)
                    size   = float(t.get("size", 0) or 0)
                    side   = t.get("side", "BUY")
                    baseline.ingest(price * size, side, ts)
            except Exception:
                pass

        sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        async def _warm_limited(m):
            async with sem:
                await _warm(m)

        await asyncio.gather(*[_warm_limited(m) for m in markets])
        warm = sum(1 for b in self._baselines.values() if b.is_warm())
        print(f"[InsiderBaseline] Done — {warm}/{len(markets)} markets have warm baselines.")

    # ── Trade processing ───────────────────────────────────────────────────────

    async def _handle(self, trade: dict, fetcher: PolymarketFetcher):
        cid    = trade.get("condition_id", "")
        side   = trade.get("type", "").upper()
        amount = trade.get("amount_usd", 0) or 0
        wallet = extract_wallet_address(trade) or ""

        baseline = self._baseline(cid)
        baseline.ingest(amount, side)

        # Always check if this trade confirms something already pending
        confirmed = self._confirmer.check(trade)
        if confirmed:
            self._confirmer.expire()
            await self._emit(confirmed, confirming_trade=trade, fetcher=fetcher)
            return

        # Only score BUY trades as potential insider entries
        if side != "BUY" or not baseline.is_warm():
            return

        z_size = baseline.size_zscore(amount)
        if z_size < SIZE_ZSCORE_TRIGGER:
            return  # fast path: skip cheap computation for normal-sized trades

        z_ofi   = baseline.current_ofi_zscore()
        hours   = self._hours_to_resolution(cid)
        novelty = await self._novelty.score(wallet, fetcher)

        is_score = InsiderScorer.compute(z_size, hours, z_ofi, novelty)
        if is_score >= self._is_min:
            self._confirmer.add(trade, is_score)

    # ── Signal emission ────────────────────────────────────────────────────────

    async def _emit(
        self,
        pending: PendingSignal,
        confirming_trade: dict,
        fetcher: PolymarketFetcher,
    ):
        key = f"{pending.condition_id}:{pending.wallet}:{pending.created_at:.0f}"
        if key in self._emitted:
            return
        self._emitted.add(key)

        trade    = pending.original_trade
        baseline = self._baseline(pending.condition_id)
        hours    = self._hours_to_resolution(pending.condition_id)
        novelty  = await self._novelty.score(pending.wallet, fetcher)

        z_size   = baseline.size_zscore(pending.amount_usd)
        z_ofi    = baseline.current_ofi_zscore()
        w_prox   = math.exp(-hours / 24)

        row = {
            "timestamp":            datetime.utcnow().isoformat(),
            "market_id":            pending.condition_id,
            "market_title":         trade.get("title", ""),
            "cluster":              (trade.get("tags") or ["Unknown"])[0],
            "trade_wallet":         pending.wallet,
            "trade_size_usd":       round(pending.amount_usd, 2),
            "entry_price":          float(trade.get("price", 0) or 0),
            "outcome":              trade.get("outcome", ""),
            "z_size":               round(z_size, 3),
            "z_ofi":                round(z_ofi, 3),
            "w_prox":               round(w_prox, 4),
            "novelty":              novelty,
            "is_score":             pending.is_score,
            "hours_to_resolution":  round(hours, 2),
            "confirmed_by_wallet":  extract_wallet_address(confirming_trade) or "",
            "confirm_size_usd":     round(confirming_trade.get("amount_usd", 0) or 0, 2),
            "confirmation_mode":    (
                "diff_wallet" if self._confirmer.require_different_wallet else "any_wallet"
            ),
        }

        with open(self._csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=row.keys()).writerow(row)

        print(
            f"\n{'='*60}\n"
            f"[INSIDER SIGNAL]  IS={pending.is_score:.2f}\n"
            f"  Market   : {trade.get('title', '')[:55]}\n"
            f"  Wallet   : {pending.wallet[:12]}...\n"
            f"  Size     : ${pending.amount_usd:,.0f}  @ {trade.get('price', 0):.3f}  → {trade.get('outcome', '')}\n"
            f"  Scores   : z_size={z_size:.1f}  z_ofi={z_ofi:.2f}  w_prox={w_prox:.3f}  novelty={novelty}\n"
            f"  Time     : {hours:.1f}h to resolution\n"
            f"  Confirm  : {extract_wallet_address(confirming_trade) or 'n/a'} "
            f"  ${row['confirm_size_usd']:,.0f}\n"
            f"{'='*60}\n"
        )

    # ── WebSocket runner ───────────────────────────────────────────────────────

    async def run_websocket(self, fetcher: PolymarketFetcher):
        """
        Subscribe to active market token IDs via Polymarket CLOB WebSocket.
        Auto-reconnects on disconnect.
        """
        asset_ids = _collect_asset_ids(self._markets)

        if not asset_ids:
            print("[InsiderWS] No asset IDs available — switching to fast poll.")
            await self._run_poll(fetcher)
            return

        print(f"[InsiderWS] Connecting — subscribing to {len(asset_ids)} asset IDs...")
        seen: set[str] = set()

        while True:
            try:
                async with websockets.connect(
                    WS_URL,
                    ping_interval=30,
                    ping_timeout=10,
                ) as ws:
                    await ws.send(json.dumps({"assets_ids": asset_ids, "type": "market"}))
                    print("[InsiderWS] Connected and subscribed.")

                    async for raw in ws:
                        events = json.loads(raw)
                        if not isinstance(events, list):
                            events = [events]

                        for event in events:
                            if event.get("event_type") != "trade":
                                continue
                            trade = _normalise_ws_event(event)
                            tid   = trade.get("id", "")
                            if tid in seen:
                                continue
                            seen.add(tid)
                            trade = (await fetcher._enrich([trade]))[0]
                            await self._handle(trade, fetcher)

            except (websockets.ConnectionClosed, OSError, Exception) as e:
                print(f"[InsiderWS] Disconnected ({e!r}) — reconnecting in 5s...")
                await asyncio.sleep(5)

    # ── Polling fallback ───────────────────────────────────────────────────────

    async def _run_poll(self, fetcher: PolymarketFetcher):
        print(f"[InsiderPoll] Polling every {WS_POLL_FALLBACK_INTERVAL_S}s...")
        seen: set[str] = set()

        while True:
            try:
                trades = await fetcher.fetch_recent_trades(min_size_usd=500)
                for trade in trades:
                    tid = trade.get("id", "")
                    if tid in seen:
                        continue
                    seen.add(tid)
                    await self._handle(trade, fetcher)
                self._confirmer.expire()
            except Exception as e:
                print(f"[InsiderPoll] Error: {e}")
            await asyncio.sleep(WS_POLL_FALLBACK_INTERVAL_S)

    # ── Entry point ────────────────────────────────────────────────────────────

    async def run(self, fetcher: PolymarketFetcher, use_websocket: bool = True):
        await self.warm_baselines(fetcher)
        if use_websocket:
            await self.run_websocket(fetcher)
        else:
            await self._run_poll(fetcher)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalise_ws_event(event: dict) -> dict:
    price = float(event.get("price", 0) or 0)
    size  = float(event.get("size", 0) or 0)
    return {
        "id":           event.get("id", ""),
        "condition_id": event.get("market", ""),
        "wallet":       event.get("maker_address") or event.get("owner", ""),
        "type":         event.get("side", "BUY").upper(),
        "outcome":      event.get("outcome", ""),
        "price":        price,
        "size":         size,
        "amount_usd":   round(price * size, 4),
        "timestamp":    str(event.get("timestamp", "")),
        "title":        "",
        "tags":         [],
        "resolved":     False,
        "winner":       None,
    }


def _collect_asset_ids(markets: dict[str, dict]) -> list[str]:
    """
    Extract outcome token IDs from market objects.
    Gamma API usually stores them under 'tokens': [{'token_id': ..., 'outcome': ...}]
    or 'clobTokenIds': [...].
    """
    ids = []
    for market in markets.values():
        tokens = market.get("tokens") or []
        if isinstance(tokens, list):
            for tok in tokens:
                if isinstance(tok, dict):
                    tid = tok.get("token_id") or tok.get("tokenId")
                else:
                    tid = tok
                if tid:
                    ids.append(str(tid))
        clob_ids = market.get("clobTokenIds") or []
        ids.extend(str(i) for i in clob_ids if i)
    return list(set(ids))


# ── CLI ────────────────────────────────────────────────────────────────────────

async def _main(use_websocket: bool, require_different_wallet: bool):
    strategy = InsiderStrategy(
        require_different_wallet = require_different_wallet,
        min_size_ratio           = CONFIRMATION_MIN_SIZE_RATIO,
        is_min_threshold         = IS_MIN_THRESHOLD,
    )
    async with PolymarketFetcher() as fetcher:
        await strategy.run(fetcher, use_websocket=use_websocket)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insider/Anomalous Timing — Strategy 2")
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Use fast polling instead of WebSocket",
    )
    parser.add_argument(
        "--require-same-wallet",
        action="store_true",
        help="Allow the same wallet to confirm its own signal (looser, faster)",
    )
    args = parser.parse_args()

    asyncio.run(_main(
        use_websocket            = not args.poll,
        require_different_wallet = not args.require_same_wallet,
    ))
