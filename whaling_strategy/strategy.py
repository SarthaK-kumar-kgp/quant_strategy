"""
Smart Money Follow — Strategy 1

Entry point:
    python strategy.py                  # full run (leaderboard seed + live loops)
    python strategy.py --smoke          # seed top 5, score them, then exit
    python strategy.py --backfill       # backfill top N leaders to CSV and exit
    python strategy.py --backfill --n=20  # backfill top 20

Change how many leaders to track:
    Edit NUM_LEADERBOARD_LEADERS below, or pass --n=<int> to --backfill.
"""

import argparse
import asyncio
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import (
    MIN_WHALE_TRADE_USD,
    ORGANIC_MIN_TRADE_USD,
    ORGANIC_POLL_INTERVAL_S,
    SIGNAL_POLL_INTERVAL_S,
    SIGNALS_CSV,
)
from data_fetcher import PolymarketFetcher, extract_wallet_address
from wallet_scorer import WalletScorer
from whale_registry import WhaleRegistry

# ── Change this to track more / fewer leaderboard leaders ─────────────────────
NUM_LEADERBOARD_LEADERS = 5


# ── Organic Discovery ──────────────────────────────────────────────────────────

class OrganicDiscovery:
    """
    Watches live trades from all wallets.
    When an unknown wallet accumulates enough resolved trades and passes the
    scoring threshold, it is automatically promoted into the WhaleRegistry.
    Runs as a background asyncio task — separate from signal detection.
    """

    def __init__(self, registry: WhaleRegistry):
        self._registry   = registry
        self._candidates: dict[str, list[dict]] = defaultdict(list)
        self._seen_ids:   set[str]              = set()

    def observe(self, trade: dict):
        tid = trade.get("id", "")
        if tid in self._seen_ids:
            return
        self._seen_ids.add(tid)

        address = extract_wallet_address(trade)
        if not address or self._registry.is_tracked(address):
            return

        self._candidates[address].append(trade)
        self._maybe_promote(address)

    def _maybe_promote(self, address: str):
        trades = self._candidates[address]
        resolved_buys = [
            t for t in trades
            if t.get("type", "").upper() == "BUY"
            and t.get("resolved")
            and t.get("winner") is not None
        ]
        if len(resolved_buys) < 50:
            return

        score = WalletScorer.score(address, trades)
        if score["qualified"]:
            self._registry.add(address, score, source="organic")
            del self._candidates[address]

    async def run(self, fetcher: PolymarketFetcher):
        print("[OrganicDiscovery] Started — watching all trades >= "
              f"${ORGANIC_MIN_TRADE_USD:,}")
        while True:
            try:
                trades = await fetcher.fetch_recent_trades(
                    min_size_usd=ORGANIC_MIN_TRADE_USD
                )
                for trade in trades:
                    self.observe(trade)
            except Exception as e:
                print(f"[OrganicDiscovery] Error: {e}")
            await asyncio.sleep(ORGANIC_POLL_INTERVAL_S)


# ── Signal Generator ───────────────────────────────────────────────────────────

# ── Exit Tracker ───────────────────────────────────────────────────────────────

_EXIT_POLL_MIN_USD = 1_000   # lower threshold so partial exits aren't missed

class ExitTracker:
    """
    Watches for SELL trades from whales whose BUYs we copied.
    When the originating whale posts a SELL on the same market + outcome,
    an exit signal is printed so you know to close your position.

    Usage:
        - SignalGenerator calls exit_tracker.track(signal) after each entry signal
        - ExitTracker.run() polls independently and fires exits when found
    """

    def __init__(self, registry: WhaleRegistry):
        self._registry = registry
        # (whale_address, condition_id, outcome) → original entry signal
        self._open: dict[tuple, dict] = {}

    def track(self, signal: dict):
        """Register an open position to watch for an exit."""
        key = (signal["whale_address"], signal["market_id"], signal["outcome"])
        self._open[key] = signal

    def _check(self, trade: dict) -> Optional[dict]:
        """Return the pending signal if this trade is a confirming SELL, else None."""
        if trade.get("type", "").upper() != "SELL":
            return None
        address = extract_wallet_address(trade)
        if not address or not self._registry.is_tracked(address):
            return None
        key = (address, trade.get("condition_id", ""), trade.get("outcome", ""))
        return self._open.pop(key, None)

    async def run(self, fetcher: PolymarketFetcher):
        print(f"[ExitTracker] Started — polling for whale exits "
              f"(min ${_EXIT_POLL_MIN_USD:,})")
        while True:
            try:
                trades = await fetcher.fetch_recent_trades(
                    min_size_usd=_EXIT_POLL_MIN_USD
                )
                for trade in trades:
                    signal = self._check(trade)
                    if signal:
                        self._emit_exit(signal, trade)
            except Exception as e:
                print(f"[ExitTracker] Error: {e}")
            await asyncio.sleep(SIGNAL_POLL_INTERVAL_S)

    @staticmethod
    def _emit_exit(signal: dict, exit_trade: dict):
        exit_price  = float(exit_trade.get("price", 0) or 0)
        entry_price = signal.get("entry_price", 0) or 0
        pnl_pct     = (exit_price / entry_price - 1) * 100 if entry_price else 0
        flag        = "+" if pnl_pct >= 0 else ""
        print(
            f"\n{'='*55}\n"
            f"[EXIT SIGNAL]\n"
            f"  Whale  : {signal['whale_address'][:12]}...\n"
            f"  Market : {signal.get('market_title', '')[:50]}\n"
            f"  Outcome: {signal.get('outcome', '')}\n"
            f"  Entry  : {entry_price:.3f}  →  Exit: {exit_price:.3f}\n"
            f"  Est PnL: {flag}{pnl_pct:.1f}%\n"
            f"{'='*55}\n"
        )


# ── Signal Generator ────────────────────────────────────────────────────────────

class SignalGenerator:
    """
    Polls for new trades from tracked whales.
    When a tracked whale with qualified edge in a cluster places a trade >= $5k
    in a matching market, a signal row is written to signals.csv.
    """

    _COLUMNS = [
        "timestamp",
        "whale_address",
        "market_id",
        "market_title",
        "cluster",
        "trade_size_usd",
        "entry_price",
        "outcome",
        "whale_z_score",
        "whale_edge_in_cluster",
        "whale_pnl",
        "recommended_size_usd",   # = min(10% of whale trade, capital limits)
    ]

    def __init__(self, registry: WhaleRegistry, exit_tracker: Optional["ExitTracker"] = None):
        self._registry     = registry
        self._exit_tracker = exit_tracker
        self._seen_ids: set[str] = set()
        self._csv      = Path(SIGNALS_CSV)
        self._init_csv()

    def _init_csv(self):
        if not self._csv.exists():
            with open(self._csv, "w", newline="") as f:
                csv.writer(f).writerow(self._COLUMNS)

    def _build_signal(self, trade: dict) -> Optional[dict]:
        tid = trade.get("id", "")
        if tid in self._seen_ids:
            return None
        self._seen_ids.add(tid)

        address = extract_wallet_address(trade)
        if not address or not self._registry.is_tracked(address):
            return None
        if trade.get("type", "").upper() != "BUY":
            return None

        amount_usd = trade.get("amount_usd", 0) or 0
        if amount_usd < MIN_WHALE_TRADE_USD:
            return None

        qualified_clusters = self._registry.qualified_clusters(address)
        matching = [tag for tag in (trade.get("tags") or []) if tag in qualified_clusters]
        if not matching:
            return None

        cluster      = matching[0]
        wallet_info  = self._registry.get(address)
        cluster_info = wallet_info.get("clusters", {}).get(cluster, {})

        return {
            "timestamp":             datetime.utcnow().isoformat(),
            "whale_address":         address,
            "market_id":             trade.get("condition_id", ""),
            "market_title":          trade.get("title", ""),
            "cluster":               cluster,
            "trade_size_usd":        round(amount_usd, 2),
            "entry_price":           float(trade.get("price", 0) or 0),
            "outcome":               trade.get("outcome", ""),
            "whale_z_score":         wallet_info.get("z_score", 0),
            "whale_edge_in_cluster": cluster_info.get("edge", 0),
            "whale_pnl":             wallet_info.get("pnl", 0),
            "recommended_size_usd":  round(min(amount_usd * 0.1, 5_000), 2),
        }

    def _emit(self, signal: dict):
        with open(self._csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._COLUMNS).writerow(signal)
        print(
            f"[Signal] {signal['whale_address'][:10]}...  "
            f"| {signal['market_title'][:45]}  "
            f"| ${signal['trade_size_usd']:,.0f}  "
            f"| cluster={signal['cluster']}  "
            f"| recommend=${signal['recommended_size_usd']:,.0f}"
        )
        if self._exit_tracker:
            self._exit_tracker.track(signal)

    async def run(self, fetcher: PolymarketFetcher):
        print(f"[SignalGenerator] Started — watching tracked wallets "
              f"(min trade ${MIN_WHALE_TRADE_USD:,})")
        while True:
            try:
                trades = await fetcher.fetch_recent_trades(
                    min_size_usd=MIN_WHALE_TRADE_USD
                )
                for trade in trades:
                    signal = self._build_signal(trade)
                    if signal:
                        self._emit(signal)
            except Exception as e:
                print(f"[SignalGenerator] Error: {e}")
            await asyncio.sleep(SIGNAL_POLL_INTERVAL_S)


# ── Leaderboard seeding ────────────────────────────────────────────────────────

async def seed_from_leaderboard(
    fetcher: PolymarketFetcher,
    registry: WhaleRegistry,
    n: int = NUM_LEADERBOARD_LEADERS,
):
    """
    Fetch top-N wallets from the leaderboard, score each one,
    and add them to the registry regardless of whether they pass
    the qualification threshold (we track them all as known smart money).
    """
    print(f"[Seed] Fetching top {n} wallets from leaderboard...")
    top = await fetcher.fetch_leaderboard(n)

    async def _process(wallet: dict):
        address = extract_wallet_address(wallet)
        if not address or registry.is_tracked(address):
            return
        print(f"[Seed] Scoring {address[:10]}...")
        trades = await fetcher.fetch_wallet_trades(address, resolved_only=True)
        if not trades:
            print(f"[Seed] No resolved trades found for {address[:10]}...")
            return
        score = WalletScorer.score(address, trades)
        registry.add(address, score, source="leaderboard")

    await asyncio.gather(*[_process(w) for w in top])
    print(f"[Seed] Done. {registry.summary()}")


# ── Main orchestration ─────────────────────────────────────────────────────────

async def run_strategy(n_leaders: int = NUM_LEADERBOARD_LEADERS):
    """Full run: seed leaderboard, then run discovery + signals + exits concurrently."""
    registry = WhaleRegistry()

    async with PolymarketFetcher() as fetcher:
        await seed_from_leaderboard(fetcher, registry, n=n_leaders)

        exit_tracker = ExitTracker(registry)
        discovery    = OrganicDiscovery(registry)
        signal_gen   = SignalGenerator(registry, exit_tracker=exit_tracker)

        await asyncio.gather(
            discovery.run(fetcher),
            signal_gen.run(fetcher),
            exit_tracker.run(fetcher),
        )


async def run_smoke_test(n_leaders: int = 5):
    """Seed top N from leaderboard, score them, print results, then exit."""
    registry = WhaleRegistry()
    async with PolymarketFetcher() as fetcher:
        await seed_from_leaderboard(fetcher, registry, n=n_leaders)

    print("\n── Registry snapshot ─────────────────────────────────")
    for w in registry.get_all():
        qual_clusters = [c for c, info in w.get("clusters", {}).items() if info.get("qualified")]
        print(
            f"  {w['address'][:12]}...  "
            f"z={w['z_score']:.2f}  pnl=${w['pnl']:,.0f}  "
            f"trades={w['trade_count']}  "
            f"qualified={w['qualified']}  "
            f"clusters={qual_clusters or 'none'}"
        )
    print("──────────────────────────────────────────────────────\n")


async def run_backfill(n: int):
    """Backfill all trades for the top-N leaderboard wallets to CSV."""
    async with PolymarketFetcher() as fetcher:
        await fetcher.backfill(mode="leaderboard", n=n)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Smart Money Follow — Strategy 1")
    parser.add_argument("--smoke",    action="store_true",
                        help="Seed leaderboard, score wallets, print summary, exit")
    parser.add_argument("--backfill", action="store_true",
                        help="Download all trades for top-N leaders to CSV and exit")
    parser.add_argument("--n",        type=int, default=NUM_LEADERBOARD_LEADERS,
                        help=f"Number of leaderboard leaders to use (default {NUM_LEADERBOARD_LEADERS})")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.backfill:
        asyncio.run(run_backfill(n=args.n))
    elif args.smoke:
        asyncio.run(run_smoke_test(n_leaders=args.n))
    else:
        asyncio.run(run_strategy(n_leaders=args.n))
