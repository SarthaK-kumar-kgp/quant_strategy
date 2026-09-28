# Polymarket Quantitative Strategy — Dissertation Project

This repository contains two independent quantitative trading strategies built on
[Polymarket](https://polymarket.com), a decentralised prediction market platform.

Both strategies are developed as part of a dissertation exploring whether systematic
edge can be extracted from prediction market microstructure.

---

## Strategies

| # | Name | File | Status |
|---|------|------|--------|
| 1 | Smart Money Follow | `strategy.py` / `backtest.py` | Data layer needs fix (see `current_problems.md`) |
| 2 | Insider Capture | `insider_runner.py` | Data collection in progress |

See `strategy1.md` and `strategy2.md` for full hypothesis, data requirements,
and workflow for each.

---

## Repository Layout

```
quant_strategy/
│
├── README.md                   ← this file
├── strategy1.md                ← Strategy 1 full documentation
├── strategy2.md                ← Strategy 2 full documentation
├── current_problems.md         ← known data issues and workarounds
│
├── config.py                   ← shared constants and API endpoints
├── data_fetcher.py             ← async HTTP wrapper (Gamma + Data API + CLOB)
├── polymarket_credentials.py   ← CLOB auth credentials (HMAC L2)
│
├── Strategy 1 ─────────────────────────────────────────────────────────
├── collect_data.py             ← fetch leaderboard wallets + trade history
├── strategy.py                 ← live runner
├── backtest.py                 ← historical backtester
├── wallet_scorer.py            ← edge/z-score scorer per wallet
├── whale_registry.py           ← registry of currently-followed wallets
├── wallets.py                  ← leaderboard API helper
│
├── Strategy 2 ─────────────────────────────────────────────────────────
├── collect_insider_data.py     ← fetch resolved markets + qualifying trades
├── analyze_insider_data.py     ← calibration analysis → recommended knobs
├── insider_runner.py           ← live + backtest runner (uses calibrated knobs)
│
└── data/
    ├── markets.parquet         ← Strategy 1: market metadata
    ├── wallets/                ← Strategy 1: per-wallet trade parquets
    ├── prices/                 ← Strategy 1: hourly price history per market
    ├── insider_raw.parquet     ← Strategy 2: qualifying trades for calibration
    └── insider_markets_cache.json  ← Strategy 2: cached market list (step 1)
```

---

## Shared Infrastructure

### APIs used

| API | Base URL | Auth | Purpose |
|-----|----------|------|---------|
| Gamma | `gamma-api.polymarket.com` | None | Market metadata, tags, resolution |
| Data API | `data-api.polymarket.com` | None | Public trade history, activity, leaderboard |
| CLOB | `clob.polymarket.com` | HMAC L2 | Order placement (live trading only) |

### Key known API quirks

- Gamma `tag_slug` filter is silently ignored — use numeric `tag_id` (Sports=1, Politics=2)
- Gamma has a hard offset cap of 100,000 (returns 422 above that)
- Data API `/trades` has a hard offset cap of 3,500 (returns 400 above that)
- Data API time-window filters (`before_ts`, `end_ts`, etc.) are all silently ignored
- Data API `/trades` returns trades newest-first and cannot be reversed

---

## Quickstart

```bash
pip install pandas pyarrow aiohttp asyncio requests websockets

# Strategy 2 (recommended starting point — data pipeline is cleaner)
python collect_insider_data.py --months=2 --min_usd=100000 --hours=24
python analyze_insider_data.py
python insider_runner.py backtest --knobs=knobs.json

# Strategy 1
python collect_data.py --wallets=100
python backtest.py --start=2024-09-01 --warmup-months=2 --top-n=5
```
