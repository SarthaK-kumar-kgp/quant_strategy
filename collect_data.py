"""
Data collection for backtesting.

Usage:
    python collect_data.py --wallets=100

Fetches top N wallets from today's all-time leaderboard, then collects
ALL their trades from 2024-06-01 (or their first trade, whichever is later)
up to today. Gets hourly price history for every market they traded.
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import pandas as pd

from data_fetcher import PolymarketFetcher, extract_wallet_address, _infer_tags_from_slug

# ── Constants ──────────────────────────────────────────────────────────────────
COLLECT_CATEGORIES = {"CRYPTO", "POLITICS"}
WALLET_TRADES_DIR  = "data/wallets"
MARKETS_FILE       = "data/markets.parquet"
PRICES_DIR         = "data/prices"
DATA_API           = "https://data-api.polymarket.com"
HARD_START         = datetime(2024, 6, 1, tzinfo=timezone.utc)   # never go before this


# ── Step 1: leaderboard ────────────────────────────────────────────────────────

def fetch_top_wallets(n: int) -> list[str]:
    print(f"[Leaderboard] Fetching top {n} wallets (all-time, today)...")
    seen, addrs = set(), []
    for cat in ["OVERALL", "CRYPTO", "POLITICS"]:
        try:
            r = requests.get(
                f"{DATA_API}/v1/leaderboard",
                params={"orderBy": "PNL", "timePeriod": "ALL",
                        "category": cat, "limit": n},
                timeout=15,
            )
            if r.status_code == 200:
                for w in r.json():
                    addr = w.get("proxyWallet") or w.get("address", "")
                    if addr and addr not in seen:
                        seen.add(addr)
                        addrs.append(addr)
                        if len(addrs) >= n:
                            break
        except Exception as e:
            print(f"[Leaderboard] {cat} error: {e}")
        if len(addrs) >= n:
            break

    print(f"[Leaderboard] Got {len(addrs)} wallets")
    return addrs[:n]


# ── Orchestration ──────────────────────────────────────────────────────────────

async def collect(n_wallets: int):
    today = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    print(f"\n{'='*60}")
    print(f"  Top N wallets   : {n_wallets}")
    print(f"  Hard start date : {HARD_START.date()}")
    print(f"  End date        : {today.date()} (today)")
    print(f"{'='*60}\n")

    Path(WALLET_TRADES_DIR).mkdir(parents=True, exist_ok=True)
    Path(PRICES_DIR).mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    addresses = fetch_top_wallets(n_wallets)

    async with PolymarketFetcher() as fetcher:

        # Step 2 — fetch all trades, determine per-wallet date window
        condition_ids: set  = set()
        yes_token_map: dict = {}
        no_token_map:  dict = {}

        print(f"\n[Collect] Fetching trades for {len(addresses)} wallets...\n")
        await asyncio.gather(*[
            _fetch_and_save_wallet(
                fetcher, addr, today,
                condition_ids, yes_token_map, no_token_map,
            )
            for addr in addresses
        ])

        token_map = {**no_token_map, **yes_token_map}  # YES wins on conflict

        print(f"\n[Collect] Markets found     : {len(condition_ids)}")
        print(f"[Collect] Markets with token : {len(token_map)}")

        # Step 3 — market metadata
        print(f"\n[Collect] Fetching metadata for {len(condition_ids)} markets...")
        await _fetch_and_save_markets(fetcher, condition_ids, yes_token_map)

        # Step 4 — price history (full window: HARD_START → today)
        print(f"\n[Collect] Fetching price history for {len(token_map)} markets...")
        results = await asyncio.gather(*[
            _fetch_prices_by_token(fetcher, cid, tok, HARD_START, today)
            for cid, tok in token_map.items()
        ])
        saved = sum(1 for r in results if r)
        print(f"\n[Collect] Price history: {saved}/{len(token_map)} saved")

    print(f"\n{'='*60}")
    print(f"[Collect] Done.")
    print(f"  Wallets : {WALLET_TRADES_DIR}/")
    print(f"  Markets : {MARKETS_FILE}")
    print(f"  Prices  : {PRICES_DIR}/")
    print(f"{'='*60}\n")


# ── Per-wallet ─────────────────────────────────────────────────────────────────

async def _fetch_and_save_wallet(
    fetcher, address, today,
    condition_ids, yes_token_map, no_token_map,
):
    try:
        trades = await fetcher.fetch_wallet_trades(address, resolved_only=False)
    except Exception as e:
        print(f"[Wallet] {address[:12]}... error: {e}")
        return

    if not trades:
        print(f"[Wallet] {address[:12]}  0 trades, skipping")
        return

    # Find the wallet's earliest trade to determine its actual start
    timestamps = [float(t.get("timestamp") or 0) for t in trades if t.get("timestamp")]
    if timestamps:
        earliest = datetime.fromtimestamp(min(timestamps), tz=timezone.utc)
        window_start = max(earliest, HARD_START)
    else:
        window_start = HARD_START

    # Filter to window_start → today
    filtered = [t for t in trades if _in_date_range(t, window_start, today)]

    print(
        f"[Wallet] {address[:12]}  "
        f"total={len(trades)}  "
        f"earliest={earliest.date() if timestamps else 'unknown'}  "
        f"window={window_start.date()}→{today.date()}  "
        f"kept={len(filtered)}"
    )

    if not filtered:
        return

    path = Path(WALLET_TRADES_DIR) / f"{address.lower()[:20]}.parquet"
    _write_parquet(path, filtered)

    for t in filtered:
        cid     = t.get("condition_id", "")
        asset   = t.get("asset", "")
        outcome = t.get("outcome", "").upper()
        if not cid or not asset:
            continue
        condition_ids.add(cid)
        if outcome == "YES" and cid not in yes_token_map:
            yes_token_map[cid] = asset
        elif outcome in ("NO", "SELL") and cid not in no_token_map:
            no_token_map[cid] = asset


# ── Market metadata ────────────────────────────────────────────────────────────

async def _fetch_and_save_markets(fetcher, condition_ids, yes_token_map):
    markets, failed = [], 0
    for cid in condition_ids:
        try:
            m = await fetcher.fetch_market(cid)
            if m:
                markets.append(m)
            else:
                failed += 1
        except Exception:
            failed += 1

    rows = []
    for m in markets:
        tokens    = m.get("tokens") or []
        yes_token = next(
            (t.get("token_id") or t.get("tokenId")
             for t in tokens
             if isinstance(t, dict) and t.get("outcome", "").upper() == "YES"),
            None,
        )
        cid = m.get("conditionId") or m.get("condition_id", "")
        rows.append({
            "condition_id":    cid,
            "question":        m.get("question", ""),
            "tags":            json.dumps(_extract_tags(m)),
            "end_date":        m.get("endDateIso") or m.get("endDate", ""),
            "resolved":        bool(m.get("resolved", False)),
            "winning_outcome": m.get("winner") or m.get("winningOutcome", "") or "",
            "yes_token_id":    yes_token or yes_token_map.get(cid, ""),
        })

    path = Path(MARKETS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(path, rows)
    print(f"[Collect] {len(rows)} markets saved ({failed} not in Gamma API)")


# ── Price history ──────────────────────────────────────────────────────────────

async def _fetch_prices_by_token(fetcher, cid, token_id, start, end):
    try:
        prices = await fetcher.fetch_prices_history(
            token_id=token_id,
            start_ts=int(start.timestamp()),
            end_ts=int(end.timestamp()),
        )
    except Exception as e:
        print(f"[Price] {cid[:12]} error: {e}")
        return False

    if not prices:
        return False

    ts_min = datetime.fromtimestamp(prices[0]["timestamp"],  tz=timezone.utc).date()
    ts_max = datetime.fromtimestamp(prices[-1]["timestamp"], tz=timezone.utc).date()
    _write_parquet(Path(PRICES_DIR) / f"{cid}.parquet", prices)
    print(f"[Price] {cid[:12]}  {len(prices)} pts  {ts_min}→{ts_max}")
    return True


# ── Helpers ────────────────────────────────────────────────────────────────────

def _in_date_range(trade, start, end):
    ts = float(trade.get("timestamp") or 0)
    if not ts:
        return True
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return start <= dt <= end

def _extract_tags(market):
    raw = market.get("tags") or market.get("categories") or []
    return [t["label"] if isinstance(t, dict) else t for t in raw]

def _write_parquet(path, rows):
    if not rows:
        return
    pd.DataFrame(rows).to_parquet(path, index=False)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wallets", type=int, default=100,
                   help="Number of top wallets to fetch (default 100)")
    args = p.parse_args()
    asyncio.run(collect(args.wallets))