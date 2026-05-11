"""
Insider-trading calibration data collector (Strategy 2).

Fetches all resolved Sports/Politics markets from the last N months,
then for each market pulls trades >= MIN_USD placed within HOURS hours
of resolution. Also fetches each unique wallet's age (first trade ts).

Output: data/insider_raw.parquet  — one row per qualifying trade.

Usage:
    python collect_insider_data.py
    python collect_insider_data.py --months=6 --min_usd=50000 --hours=48
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from config import CLOB_API, GAMMA_API, DATA_API
from data_fetcher import PolymarketFetcher

OUTPUT_FILE   = Path("data/insider_raw.parquet")
MARKETS_CACHE = Path("data/insider_markets_cache.json")
# Numeric tag IDs verified via /tags/<id> — slug filters are silently ignored by Gamma.
CATEGORY_TAG_IDS = {"Sports": 1, "Politics": 2}


# ── Orchestration ──────────────────────────────────────────────────────────────

async def collect(months: int, min_usd: float, hours: int, refresh_markets: bool = False):
    today  = datetime.now(tz=timezone.utc)
    cutoff = today - timedelta(days=months * 30)

    print(f"\n{'='*60}")
    print(f"  Categories          : Sports + Politics")
    print(f"  Markets resolved    : {cutoff.date()} → {today.date()}")
    print(f"  Min trade size      : ${min_usd:,.0f}")
    print(f"  Window before close : {hours}h")
    print(f"{'='*60}\n")

    Path("data").mkdir(exist_ok=True)

    async with PolymarketFetcher() as fetcher:

        # Step 1 — resolved Sports/Politics markets (cached to disk)
        if MARKETS_CACHE.exists() and not refresh_markets:
            markets = json.loads(MARKETS_CACHE.read_text())
            print(f"[1/4] Loaded {len(markets)} markets from cache "
                  f"({MARKETS_CACHE}) — pass --refresh_markets to re-fetch\n")
        else:
            print("[1/4] Fetching resolved markets (Sports + Politics)...")
            markets = await _fetch_resolved_markets(fetcher, cutoff, today)
            MARKETS_CACHE.write_text(json.dumps(markets))
            print(f"      {len(markets)} markets — cached to {MARKETS_CACHE}\n")

        if not markets:
            print("No matching markets. Check Gamma API categories or date range.")
            return

        # Pre-filter markets: any market with total volume < min_usd cannot
        # contain a single trade >= min_usd, so skip it entirely.
        before_n = len(markets)
        markets = [
            m for m in markets
            if float(m.get("volumeNum") or m.get("volume") or 0) >= min_usd
        ]
        print(f"      Pre-filter: {before_n:,} markets → {len(markets):,} "
              f"with volume >= ${min_usd:,.0f}\n")

        if not markets:
            print("No markets pass the volume pre-filter. Lower --min_usd to widen.")
            return

        # Step 2 — qualifying trades per market (concurrent, bounded)
        print(f"[2/4] Pulling trades >= ${min_usd:,.0f} in last {hours}h before close...")
        sem      = asyncio.Semaphore(4)   # lowered from 8 to avoid 429s
        total    = len(markets)
        progress = {"done": 0, "rows": 0, "capped": 0}

        async def _wrapped(m):
            rows, capped = await _fetch_market_qualifying_trades(
                fetcher, m, min_usd, hours, sem
            )
            progress["done"]   += 1
            progress["rows"]   += len(rows)
            progress["capped"] += int(capped)
            if progress["done"] % 25 == 0 or progress["done"] == total:
                print(
                    f"      [{progress['done']:>5}/{total}]  "
                    f"rows={progress['rows']:>5}  "
                    f"capped={progress['capped']:>4}",
                    end="\r",
                )
            return rows, capped

        results = await asyncio.gather(*[_wrapped(m) for m in markets])
        print()  # newline after the \r progress line

        all_rows = [row for batch, _capped in results for row in batch]
        capped_n = sum(1 for _batch, capped in results if capped)
        print(f"      {len(all_rows)} qualifying trades found")
        print(f"      {capped_n:,} / {total:,} markets hit the 3,500-trade cap "
              f"({capped_n / max(total,1):.1%}) — may have missed older trades in those")

        if _DIAG_COUNTS:
            print(f"\n      Diagnostics — error buckets seen during step 2:")
            for bucket, cnt in sorted(_DIAG_COUNTS.items(), key=lambda x: -x[1]):
                print(f"        {bucket:<24} {cnt:>6,}   sample: {_DIAG_SAMPLES[bucket]}")
        print()

        if not all_rows:
            print("No qualifying trades. Try lowering --min_usd or increasing --hours.")
            return

        # Step 3 — wallet ages (unique wallets only)
        unique_wallets = list({r["wallet"] for r in all_rows if r["wallet"]})
        print(f"[3/4] Fetching age for {len(unique_wallets)} unique wallets...")
        age_results = await asyncio.gather(*[
            _fetch_wallet_first_ts(fetcher, w) for w in unique_wallets
        ], return_exceptions=True)
        wallet_first_ts = {
            w: (r if not isinstance(r, Exception) else None)
            for w, r in zip(unique_wallets, age_results)
        }

        # Step 4 — attach wallet age + save
        print("[4/4] Saving...")
        for row in all_rows:
            fts = wallet_first_ts.get(row["wallet"])
            row["wallet_first_trade_ts"]    = fts
            row["wallet_age_days_at_trade"] = (
                round((row["trade_ts"] - fts) / 86400, 1)
                if fts and row["trade_ts"]
                else None
            )

        df = pd.DataFrame(all_rows)
        df.to_parquet(OUTPUT_FILE, index=False)

        print(f"\n{'='*60}")
        print(f"  Rows saved   : {len(df):,}")
        print(f"  Markets      : {df['condition_id'].nunique():,}")
        print(f"  Wallets      : {df['wallet'].nunique():,}")
        print(f"  Output       : {OUTPUT_FILE}")
        print(f"{'='*60}\n")


# ── Step 1: Resolved markets ───────────────────────────────────────────────────

async def _fetch_resolved_markets(
    fetcher, after: datetime, before: datetime, page: int = 500
) -> list[dict]:
    """
    Paginate closed markets for each parent category tag (Sports, Politics)
    and keep only those that resolved within [after, before]. Date filtering
    is client-side because Gamma silently ignores endDateMin/endDateMax.
    """
    after_ts  = int(after.timestamp())
    before_ts = int(before.timestamp())

    all_markets: list = []
    seen_ids:    set  = set()

    for label, tag_id in CATEGORY_TAG_IDS.items():
        offset = 0
        kept_for_tag = 0
        while True:
            try:
                data = await fetcher._get(
                    f"{GAMMA_API}/markets",
                    params={
                        "closed": "true",
                        "tag_id": str(tag_id),
                        "limit":  page,
                        "offset": offset,
                    },
                )
            except Exception as e:
                print(f"\n      [!] {label}: stopped at offset={offset} ({e})")
                break

            batch = data if isinstance(data, list) else data.get("data", [])
            if not batch:
                break

            for m in batch:
                ts = _parse_resolution_ts(m)
                if ts is None or ts < after_ts or ts > before_ts:
                    continue
                mid = m.get("id") or m.get("conditionId") or m.get("condition_id")
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    all_markets.append(m)
                    kept_for_tag += 1

            print(f"      [{label}] offset={offset:>6}  kept: {kept_for_tag}", end="\r")

            if len(batch) < page:
                break
            offset += page

        print()
    return all_markets


# ── Diagnostics ────────────────────────────────────────────────────────────────

# Counts every distinct error/anomaly bucket seen during step 2.
# Printed as a summary table at the end.
_DIAG_COUNTS: dict = {}
_DIAG_SAMPLES: dict = {}

def _diag_count(bucket: str, cid: str, yes_tok: str, offset: int, msg: str):
    _DIAG_COUNTS[bucket] = _DIAG_COUNTS.get(bucket, 0) + 1
    if bucket not in _DIAG_SAMPLES:
        _DIAG_SAMPLES[bucket] = (
            f"cid={cid[:14]} offset={offset} "
            f"yes_tok={(yes_tok or '')[:14]}... msg={msg[:160]}"
        )


# ── Step 2: Qualifying trades per market ───────────────────────────────────────

async def _fetch_market_qualifying_trades(
    fetcher, market: dict, min_usd: float, hours: int, sem: asyncio.Semaphore
) -> tuple[list[dict], bool]:
    """Returns (rows, hit_cap_before_window)."""
    resolution_ts = _parse_resolution_ts(market)
    if not resolution_ts:
        return [], False

    window_open = resolution_ts - hours * 3600

    yes_tok = _extract_yes_token(market)
    if not yes_tok:
        return [], False

    cid      = market.get("conditionId") or market.get("condition_id", "")
    question = market.get("question", "")
    outcome  = (market.get("winner") or market.get("winningOutcome", "") or "").upper()

    rows, offset = [], 0
    hit_cap_before_window = False
    async with sem:
        while True:
            data = None
            for attempt in range(4):
                try:
                    data = await fetcher._get(
                        f"{DATA_API}/trades",
                        params={"market": yes_tok, "limit": 500, "offset": offset},
                    )
                    break
                except Exception as e:
                    msg = str(e)
                    if "429" in msg and attempt < 3:
                        await asyncio.sleep(1.5 * (2 ** attempt))   # 1.5, 3, 6 s
                        continue
                    if "400" in msg:
                        # ONLY treat as a real cap-hit if we're actually past
                        # the 3,500-offset wall. A 400 at offset=0 means the
                        # asset isn't queryable (closed-market token, etc.) and
                        # should not be counted as "hit cap before window".
                        if offset >= 3500:
                            hit_cap_before_window = True
                        else:
                            _diag_count("400_at_offset_0", cid, yes_tok, offset, msg)
                    else:
                        print(f"  [!] {cid[:12]} trades error: {e}")
                    break
            if data is None:
                break

            batch = data if isinstance(data, list) else data.get("data", [])
            if not batch:
                break

            # Data API returns newest-first — stop as soon as we pass window_open
            stop = False
            for t in batch:
                ts = _parse_trade_ts(t)
                if ts is None:
                    continue
                if ts < window_open:
                    stop = True
                    break
                if ts > resolution_ts:
                    continue   # after resolution (skip)

                price  = float(t.get("price", 0) or 0)
                size   = float(t.get("size", 0) or 0)
                amount = float(t.get("usdcSize", 0) or 0) or round(price * size, 2)

                if amount < min_usd:
                    continue

                rows.append({
                    "condition_id":          cid,
                    "question":              question,
                    "resolution_ts":         resolution_ts,
                    "resolution_outcome":    outcome,
                    "wallet":                t.get("proxyWallet") or t.get("maker_address") or t.get("owner", ""),
                    "trade_ts":              ts,
                    "hours_before_close":    round((resolution_ts - ts) / 3600, 2),
                    "side":                  (t.get("side") or t.get("type", "")).upper(),
                    "trade_price":           price,
                    "trade_usd":             amount,
                    "trade_size_tokens":     size,
                    "wallet_first_trade_ts":    None,   # filled in step 3
                    "wallet_age_days_at_trade": None,
                })

            if stop or len(batch) < 500:
                break
            offset += 500

    return rows, hit_cap_before_window


def _extract_yes_token(market: dict) -> Optional[str]:
    """Gamma returns clobTokenIds as a JSON-encoded list [yes_id, no_id]."""
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


def _parse_resolution_ts(market: dict) -> Optional[int]:
    raw = (
        market.get("endDateIso")
        or market.get("endDate")
        or market.get("closeTime")
        or ""
    )
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        return None


def _parse_trade_ts(trade: dict) -> Optional[int]:
    raw = trade.get("match_time") or trade.get("timestamp") or trade.get("created_at")
    if raw is None:
        return None
    try:
        return int(float(raw))
    except Exception:
        return None


# ── Step 3: Wallet age ─────────────────────────────────────────────────────────

async def _fetch_wallet_first_ts(fetcher, address: str) -> Optional[int]:
    """
    Returns the unix timestamp of the wallet's first-ever trade by paginating
    the activity endpoint (newest-first) all the way to the last page.
    Note: the API may cap pagination; the result is a lower bound if capped.
    """
    min_ts, offset = None, 0
    while True:
        try:
            page = await fetcher._get(
                f"{DATA_API}/activity",
                params={"user": address, "limit": 500, "offset": offset},
            )
        except Exception:
            break
        batch = page if isinstance(page, list) else page.get("data", [])
        if not batch:
            break
        for t in batch:
            ts = t.get("timestamp")
            if ts:
                ts = int(float(ts))
                if min_ts is None or ts < min_ts:
                    min_ts = ts
        if len(batch) < 500:
            break
        offset += 500
    return min_ts


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Collect insider calibration data (Strategy 2)")
    p.add_argument("--months",  type=int,   default=6,
                   help="Months of resolved markets to look back (default: 6)")
    p.add_argument("--min_usd", type=float, default=50_000,
                   help="Minimum trade size in USD (default: 50000)")
    p.add_argument("--hours",   type=int,   default=48,
                   help="Hours before market close to search for trades (default: 48)")
    p.add_argument("--refresh_markets", action="store_true",
                   help="Force re-fetch of the markets list (ignores cache)")
    args = p.parse_args()
    asyncio.run(collect(args.months, args.min_usd, args.hours, args.refresh_markets))
