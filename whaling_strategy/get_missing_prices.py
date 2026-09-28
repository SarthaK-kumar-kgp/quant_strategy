"""
Run from your quant_strategy folder:
    python3 fix_missing_prices.py
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd
from typing import Optional

GAMMA_API   = "https://gamma-api.polymarket.com"
CLOB_API    = "https://clob.polymarket.com"
PRICES_DIR  = Path("data/prices")
WALLETS_DIR = Path("data/wallets")
HARD_START  = datetime(2024, 6, 1, tzinfo=timezone.utc)
CONCURRENCY = 10


async def get_yes_token_from_gamma(session, condition_id: str) -> Optional[str]:
    for param in ["conditionId", "condition_id"]:
        try:
            async with session.get(
                f"{GAMMA_API}/markets",
                params={param: condition_id},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    markets = data if isinstance(data, list) else data.get("markets", [])
                    for m in markets:
                        tokens = m.get("tokens") or m.get("clobTokenIds") or []
                        if isinstance(tokens, list) and len(tokens) >= 1:
                            if isinstance(tokens[0], dict):
                                yes = next(
                                    (t.get("token_id") or t.get("tokenId")
                                     for t in tokens
                                     if t.get("outcome", "").upper() == "YES"),
                                    None,
                                )
                                if yes:
                                    return yes
                            else:
                                return str(tokens[0])
        except Exception as e:
            print(f"  [Gamma] {condition_id[:12]} error: {e}")
    return None


async def fetch_prices(session, token_id: str, start_ts: int, end_ts: int) -> list[dict]:
    try:
        async with session.get(
            f"{CLOB_API}/prices-history",
            params={"market": token_id, "interval": "max", "fidelity": 60},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as r:
            if r.status != 200:
                return []
            data = await r.json()
            history = data if isinstance(data, list) else data.get("history", [])
            points = sorted(
                [{"timestamp": int(p["t"]), "price": float(p["p"])}
                 for p in history if "t" in p and "p" in p],
                key=lambda x: x["timestamp"],
            )
            return [p for p in points if start_ts <= p["timestamp"] <= end_ts]
    except Exception as e:
        print(f"  [CLOB] {token_id[:12]} error: {e}")
        return []


def write_parquet(path: Path, rows: list[dict]):
    if rows:
        pd.DataFrame(rows).to_parquet(path, index=False)


async def main():
    today    = datetime.now(tz=timezone.utc)
    end_ts   = int(today.timestamp())
    start_ts = int(HARD_START.timestamp())

    # ── Step 1: scan wallet files ──────────────────────────────────────────────
    wallet_files = list(WALLETS_DIR.glob("*.parquet"))
    print(f"Wallet files found: {len(wallet_files)}")
    if not wallet_files:
        print(f"ERROR: No wallet files in {WALLETS_DIR.resolve()}")
        print(f"  Make sure you're running from your quant_strategy folder")
        return

    # Peek at columns of first file
    sample = pd.read_parquet(wallet_files[0])
    print(f"Sample wallet columns: {list(sample.columns)}")
    print(f"Sample row: {sample.iloc[0].to_dict() if len(sample) else 'empty'}")

    # ── Step 2: build cid → token map ─────────────────────────────────────────
    print(f"\nScanning {len(wallet_files)} wallet files for condition_ids...")
    cid_to_tokens: dict[str, dict] = {}

    for f in wallet_files:
        df = pd.read_parquet(f)
        cols = set(df.columns)

        # Handle different possible column names
        cid_col     = next((c for c in ["condition_id", "conditionId"] if c in cols), None)
        asset_col   = next((c for c in ["asset", "token_id", "tokenId"] if c in cols), None)
        outcome_col = next((c for c in ["outcome", "side"] if c in cols), None)

        if not cid_col:
            print(f"  WARNING: {f.name} has no condition_id column, cols={list(cols)}")
            continue

        for _, row in df.iterrows():
            cid     = str(row.get(cid_col) or "")
            asset   = str(row.get(asset_col) or "") if asset_col else ""
            outcome = str(row.get(outcome_col) or "").upper() if outcome_col else ""
            if not cid or not asset:
                continue
            if cid not in cid_to_tokens:
                cid_to_tokens[cid] = {"yes": None, "no": None}
            if outcome == "YES":
                cid_to_tokens[cid]["yes"] = asset
            elif outcome in ("NO", "SELL") and not cid_to_tokens[cid]["no"]:
                cid_to_tokens[cid]["no"] = asset

    print(f"Total unique condition_ids: {len(cid_to_tokens)}")

    # ── Step 3: find missing ───────────────────────────────────────────────────
    saved_cids = {f.stem for f in PRICES_DIR.glob("*.parquet")}
    print(f"Already have price files : {len(saved_cids)}")

    missing  = {cid: t for cid, t in cid_to_tokens.items() if cid not in saved_cids}
    no_yes   = {cid: t for cid, t in missing.items() if not t["yes"]}
    has_yes  = {cid: t for cid, t in missing.items() if t["yes"]}

    print(f"Missing price files      : {len(missing)}")
    print(f"  No YES token           : {len(no_yes)}")
    print(f"  Have YES token         : {len(has_yes)}")

    if not missing:
        print("\nNothing to fix — all markets already have price files!")
        return

    sem   = asyncio.Semaphore(CONCURRENCY)
    saved = 0

    async with aiohttp.ClientSession() as session:

        # ── Step 4: Gamma lookup for NO-only markets ───────────────────────────
        print(f"\nLooking up YES tokens from Gamma for {len(no_yes)} markets...")
        recovered_yes: dict[str, str] = {}

        async def lookup_yes(cid: str):
            async with sem:
                yes = await get_yes_token_from_gamma(session, cid)
                if yes:
                    recovered_yes[cid] = yes

        await asyncio.gather(*[lookup_yes(cid) for cid in no_yes])
        print(f"YES tokens recovered: {len(recovered_yes)}/{len(no_yes)}")

        # ── Step 5: fetch prices ───────────────────────────────────────────────
        fetch_queue = {}
        fetch_queue.update({cid: t["yes"] for cid, t in has_yes.items()})
        fetch_queue.update(recovered_yes)

        print(f"\nFetching prices for {len(fetch_queue)} markets...")

        async def fetch_and_save(cid: str, token_id: str):
            nonlocal saved
            async with sem:
                prices = await fetch_prices(session, token_id, start_ts, end_ts)
                if prices:
                    ts_min = datetime.fromtimestamp(prices[0]["timestamp"],  tz=timezone.utc).date()
                    ts_max = datetime.fromtimestamp(prices[-1]["timestamp"], tz=timezone.utc).date()
                    write_parquet(PRICES_DIR / f"{cid}.parquet", prices)
                    saved += 1
                    print(f"  ✓ {cid[:16]}  {len(prices)} pts  {ts_min}→{ts_max}")
                else:
                    print(f"  ✗ {cid[:16]}  no data returned")

        await asyncio.gather(*[
            fetch_and_save(cid, tok) for cid, tok in fetch_queue.items()
        ])

    total = len(list(PRICES_DIR.glob("*.parquet")))
    print(f"\n{'='*60}")
    print(f"New price files saved : {saved}")
    print(f"Total price files now : {total}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())