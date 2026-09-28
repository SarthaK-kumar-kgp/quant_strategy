import asyncio
import base64
import csv
import hashlib
import hmac
import time
from pathlib import Path
from typing import Optional

import aiohttp

from config import (
    CLOB_API, DATA_API, GAMMA_API,
    DATA_DIR, MAX_CONCURRENT_REQUESTS, PAGE_SIZE, REQUEST_TIMEOUT_S,
    CLOB_API_KEY, CLOB_SECRET, CLOB_PASSPHRASE, CLOB_WALLET,
)


class PolymarketFetcher:
    """Async Polymarket API client. Use as an async context manager."""

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._sem     = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self._market_cache: dict[str, dict] = {}

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=MAX_CONCURRENT_REQUESTS,
            keepalive_timeout=30,
        )
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S),
            headers={"Accept": "application/json"},
        )
        return self

    async def __aexit__(self, *_):
        await self._session.close()

    # ── Core HTTP ──────────────────────────────────────────────────────────────

    @staticmethod
    def _clob_headers(method: str, path: str) -> dict:
        """HMAC-signed L2 auth headers for CLOB API requests."""
        ts           = str(int(time.time()))
        msg          = ts + method.upper() + path
        secret_bytes = base64.urlsafe_b64decode(CLOB_SECRET)
        sig          = base64.b64encode(
            hmac.new(secret_bytes, msg.encode(), hashlib.sha256).digest()
        ).decode()
        return {
            "POLY_ADDRESS":   CLOB_WALLET,
            "POLY_API_KEY":   CLOB_API_KEY,
            "POLY_SIGNATURE": sig,
            "POLY_TIMESTAMP": ts,
            "POLY_PASSPHRASE": CLOB_PASSPHRASE,
        }

    async def _get(self, url: str, params: dict = None) -> dict:
        # Attach CLOB auth headers for any CLOB API request
        extra_headers = {}
        if url.startswith(CLOB_API):
            from urllib.parse import urlparse, urlencode
            path = urlparse(url).path
            if params:
                path = path + "?" + urlencode(params)
            extra_headers = self._clob_headers("GET", path)

        async with self._sem:
            async with self._session.get(url, params=params, headers=extra_headers) as r:
                r.raise_for_status()
                return await r.json()

    # ── Leaderboard ────────────────────────────────────────────────────────────

    async def fetch_leaderboard(
        self,
        n:           int = 5,
        time_period: str = "ALL",    # DAY | WEEK | MONTH | ALL
        category:    str = "OVERALL" # OVERALL | POLITICS | CRYPTO | SPORTS …
    ) -> list[dict]:
        """Top N wallets by all-time profit from the Polymarket leaderboard."""
        data = await self._get(
            f"{DATA_API}/v1/leaderboard",
            params={
                "orderBy":    "PNL",
                "timePeriod": time_period,
                "category":   category,
                "limit":      n,
                "offset":     0,
            },
        )
        return data if isinstance(data, list) else data.get("data", [])

    # ── Wallet trades ──────────────────────────────────────────────────────────

    async def fetch_wallet_trades(
        self,
        address:      str,
        resolved_only: bool = True,
        categories:   list = None,   # e.g. ["Crypto", "Politics"]
    ) -> list[dict]:
        """
        All trades for a wallet, optionally filtered by category at the API level.
        When categories are provided, makes one paginated call per category and
        combines results — maximising the number of relevant trades fetched.
        """
        if categories:
            seen, all_trades = set(), []
            for cat in categories:
                batch = await self._fetch_activity_pages(address, category=cat)
                for t in batch:
                    tid = t.get("id", "")
                    if tid not in seen:
                        seen.add(tid)
                        all_trades.append(t)
            trades = all_trades
        else:
            trades = await self._fetch_activity_pages(address)

        # Drop non-trade activity (REDEEM, REWARD, MAKER_REBATE, REFERRAL_REWARD …)
        trades = [t for t in trades if t.get("type", "").upper() in ("BUY", "SELL", "TRADE")]

        trades = await self._enrich(trades)
        return [t for t in trades if t["resolved"]] if resolved_only else trades

    async def _fetch_activity_pages(
        self,
        address:  str,
        category: str = None,
    ) -> list[dict]:
        """Paginate the activity endpoint, stopping gracefully at the API ceiling."""
        trades, offset, _printed_sample = [], 0, False
        while True:
            params = {"user": address, "limit": PAGE_SIZE, "offset": offset}
            if category:
                params["category"] = category
            try:
                page = await self._get(f"{DATA_API}/activity", params=params)
            except Exception:
                break   # API ceiling reached — return what we have
            batch = page if isinstance(page, list) else page.get("data", [])
            if not batch:
                break
            # Print raw keys of very first trade ever seen (one-time debug)
            if not _printed_sample and offset == 0:
                raw = batch[0]
                print(f"[DEBUG_RAW] keys={list(raw.keys())}")
                print(f"[DEBUG_RAW] sample={dict(list(raw.items())[:8])}")
                print(f"[DEBUG_SLUG] slug={raw.get('slug')} | eventSlug={raw.get('eventSlug')} | type={raw.get('type')}")
                _printed_sample = True
            # Normalize camelCase API fields → snake_case before enrichment
            trades.extend(_normalise_data_api_trade(t) for t in batch)
            if len(batch) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
        return trades

    # ── Recent trades (all wallets) ────────────────────────────────────────────

    async def fetch_recent_trades(
        self,
        min_size_usd: float = 1_000,
        limit: int = 200,
    ) -> list[dict]:
        """
        Latest trades across all markets from the CLOB.
        Used by both organic discovery and signal detection.
        """
        data = await self._get(
            f"{CLOB_API}/trades",
            params={"limit": limit},
        )
        raw = data if isinstance(data, list) else data.get("data", [])
        normalised = [_normalise_clob_trade(t) for t in raw]
        filtered   = [t for t in normalised if t["amount_usd"] >= min_size_usd]
        return await self._enrich(filtered)

    # ── Market info ────────────────────────────────────────────────────────────

    async def fetch_market(self, condition_id: str) -> dict:
        if condition_id not in self._market_cache:
            await self._load_market(condition_id)
        return self._market_cache.get(condition_id, {})

    async def _load_market(self, condition_id: str):
        if not condition_id:
            return
        for param_name in ("conditionId", "conditionIds", "condition_id", "condition_ids"):
            try:
                data = await self._get(
                    f"{GAMMA_API}/markets",
                    params={param_name: condition_id},
                )
                markets = data if isinstance(data, list) else data.get("data", [])
                # Verify the returned market actually matches our condition_id
                matched = [
                    m for m in markets
                    if (m.get("conditionId") or m.get("condition_id", "")).lower()
                    == condition_id.lower()
                ]
                if matched:
                    self._market_cache[condition_id] = matched[0]
                    return
            except Exception:
                continue
        self._market_cache[condition_id] = {}

    # ── Trade enrichment (tags + resolution) ───────────────────────────────────

    async def _enrich(self, trades: list[dict]) -> list[dict]:
        cids     = {t["condition_id"] for t in trades if t.get("condition_id")}
        uncached = cids - self._market_cache.keys()

        if uncached:
            await asyncio.gather(*[self._load_market(cid) for cid in uncached])

        for trade in trades:
            market = self._market_cache.get(trade.get("condition_id", ""), {})
            trade["resolved"] = bool(market.get("resolved") or market.get("closed"))
            gamma_tags = _extract_tags(market)
            if gamma_tags:
                trade["tags"] = gamma_tags
            trade["title"]  = trade.get("title") or market.get("question", "")
            trade["winner"] = _did_win(trade, market) if trade["resolved"] else None

        return trades

    # ── Insider strategy helpers ───────────────────────────────────────────────

    async def fetch_active_markets(self, limit: int = 500) -> list[dict]:
        """All currently active (unresolved) markets from Gamma API."""
        all_markets, offset = [], 0
        while True:
            data = await self._get(
                f"{GAMMA_API}/markets",
                params={"active": "true", "closed": "false", "limit": limit, "offset": offset},
            )
            batch = data if isinstance(data, list) else data.get("data", [])
            if not batch:
                break
            all_markets.extend(batch)
            if len(batch) < limit:
                break
            offset += limit
        return all_markets

    async def fetch_clob_market_trades(
        self,
        condition_id: str,
        limit: int = 500,
    ) -> list[dict]:
        """All CLOB trades for one market, fully paginated."""
        all_trades, cursor = [], None
        while True:
            params: dict = {"market": condition_id, "limit": limit}
            if cursor:
                params["next_cursor"] = cursor
            data   = await self._get(f"{CLOB_API}/trades", params=params)
            if isinstance(data, list):
                all_trades.extend(data)
                break
            batch  = data.get("data", [])
            cursor = data.get("next_cursor")
            all_trades.extend(batch)
            # "LTE=" is Polymarket's sentinel for "no more pages"
            if not batch or not cursor or cursor == "LTE=":
                break
        return all_trades

    async def fetch_prices_history(
        self,
        token_id: str,
        start_ts: int,
        end_ts:   int,
    ) -> list[dict]:
        """
        Hourly YES-outcome price history for a token.
        Returns [{timestamp: int, price: float}] sorted by time, filtered to [start_ts, end_ts].

        Uses interval=max (no time params) then filters locally — the CLOB API
        rejects startTs/endTs when interval=max is set (returns 400).

        token_id comes from trade data: the 'asset' field on a YES-outcome trade.
        """
        data = await self._get(
            f"{CLOB_API}/prices-history",
            params={
                "market":   token_id,
                "interval": "max",
                "fidelity": 60,      # hourly — plenty for backtesting
            },
        )
        history = data if isinstance(data, list) else data.get("history", [])
        points = sorted(
            [
                {"timestamp": int(p["t"]), "price": float(p["p"])}
                for p in history
                if "t" in p and "p" in p
            ],
            key=lambda x: x["timestamp"],
        )
        # Filter to requested date window locally
        return [p for p in points if start_ts <= p["timestamp"] <= end_ts]

    # ── Backfill ───────────────────────────────────────────────────────────────

    async def backfill(
        self,
        mode: str,
        wallet: Optional[str]        = None,
        wallets: Optional[list[str]] = None,
        n: int                        = 20,
        output_dir: str               = DATA_DIR,
    ) -> None:
        """
        Fetch and save all trades for one or more wallets to CSV.

        mode='single'      — one wallet   (requires wallet=<address>)
        mode='bulk'        — many wallets (requires wallets=[...])
        mode='leaderboard' — top-N from leaderboard (requires n=<int>)
        """
        Path(output_dir).mkdir(exist_ok=True)

        if mode == "single":
            if not wallet:
                raise ValueError("wallet= is required for mode='single'")
            await self._save_csv(wallet, output_dir)

        elif mode == "bulk":
            if not wallets:
                raise ValueError("wallets= is required for mode='bulk'")
            await asyncio.gather(*[self._save_csv(w, output_dir) for w in wallets])

        elif mode == "leaderboard":
            top   = await self.fetch_leaderboard(n)
            addrs = [extract_wallet_address(w) for w in top]
            addrs = [a for a in addrs if a]
            await asyncio.gather(*[self._save_csv(a, output_dir) for a in addrs])

        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose: single | bulk | leaderboard")

    async def _save_csv(self, address: str, output_dir: str):
        trades = await self.fetch_wallet_trades(address, resolved_only=False)
        if not trades:
            print(f"[backfill] No trades for {address[:10]}...")
            return
        path = Path(output_dir) / f"{address.lower()[:10]}_trades.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys(), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(trades)
        print(f"[backfill] {len(trades):,} trades -> {path}")


# ── Field-name normalisation helpers ───────────────────────────────────────────
# CLOB API uses different field names than the Data API activity endpoint.
# We normalise everything into one internal schema so the rest of the code
# never has to worry about which source a trade came from.

def _normalise_clob_trade(t: dict) -> dict:
    price = float(t.get("price", 0) or 0)
    size  = float(t.get("size", 0) or 0)
    return {
        "id":           t.get("id", ""),
        "condition_id": t.get("market") or t.get("conditionId", ""),
        "wallet":       t.get("maker_address") or t.get("owner", ""),
        "type":         t.get("side", "").upper(),
        "outcome":      t.get("outcome", ""),
        "price":        price,
        "size":         size,
        "amount_usd":   round(price * size, 4),
        "timestamp":    t.get("match_time") or t.get("timestamp", ""),
        "title":        t.get("title", ""),
        "tags":         [],
        "resolved":     False,
        "winner":       None,
    }


_CRYPTO_TOKENS = frozenset({
    "btc", "eth", "bitcoin", "ethereum", "sol", "solana",
    "bnb", "xrp", "doge", "dogecoin", "matic", "avax", "avalanche",
    "link", "chainlink", "usdc", "usdt", "defi", "nft", "blockchain",
    "crypto", "token", "coin", "polygon", "arbitrum", "optimism",
})

_POLITICS_TOKENS = frozenset({
    "election", "president", "presidential", "senate", "congress",
    "democrat", "republican", "gop", "trump", "biden", "harris",
    "kamala", "vote", "ballot", "midterm", "governor", "legislation",
    "political", "politics", "campaign",
})


def _infer_tags_from_slug(slug: str, event_slug: str = "") -> list[str]:
    combined = (slug + " " + event_slug).lower().replace("-", " ").replace("_", " ")
    tokens   = set(combined.split())
    tags = []
    if tokens & _CRYPTO_TOKENS:
        tags.append("Crypto")
    if tokens & _POLITICS_TOKENS:
        tags.append("Politics")
    return tags


def _normalise_data_api_trade(t: dict) -> dict:
    price      = float(t.get("price", 0) or 0)
    size       = float(t.get("size", 0) or 0)
    amount_usd = float(t.get("usdcSize", 0) or 0) or round(price * size, 4)
    return {
        "id":           t.get("id", ""),
        "condition_id": t.get("conditionId") or t.get("condition_id", ""),
        "wallet":       t.get("proxyWallet") or t.get("proxy_wallet", ""),
        "type":         t.get("type", "").upper(),
        "outcome":      t.get("outcome", ""),
        "price":        price,
        "size":         size,
        "amount_usd":   amount_usd,
        "timestamp":    t.get("timestamp", ""),
        "title":        t.get("title", ""),
        "tags":         _infer_tags_from_slug(t.get("slug", ""), t.get("eventSlug", "")),
        "asset":        t.get("asset", ""),   # token_id for the traded outcome
        "resolved":     t.get("resolved", False),
        "winner":       None,
    }


def _extract_tags(market: dict) -> list[str]:
    raw = market.get("tags") or market.get("categories") or []
    return [t["label"] if isinstance(t, dict) else t for t in raw]


def _did_win(trade: dict, market: dict) -> Optional[bool]:
    winner = market.get("winner") or market.get("winning_outcome")
    if not winner:
        return None
    return trade.get("outcome", "").lower() == str(winner).lower()


def extract_wallet_address(obj: dict) -> Optional[str]:
    """Extract wallet address from leaderboard or trade dict regardless of field name."""
    return (
        obj.get("proxyWallet")
        or obj.get("proxy_wallet")
        or obj.get("address")
        or obj.get("wallet")
        or obj.get("maker_address")
        or obj.get("owner")
    )