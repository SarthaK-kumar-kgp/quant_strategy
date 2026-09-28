# Strategy 1 — Smart Money Follow

## Hypothesis

Polymarket publishes an all-time leaderboard of wallets ranked by profit-and-loss.
The top performers are not lucky — they are systematically better at pricing outcomes
than the market average, and their edge persists across markets and time periods.

If we can identify wallets with statistically significant positive edge (measured as
a z-score on their calibration-adjusted hit rate) and copy their open positions
shortly after they enter, we can capture a fraction of their alpha.

**Core assumption:** Skill in prediction markets is persistent. A wallet that has
demonstrated z ≥ 2.5 edge over 50+ resolved trades is unlikely to be a noise outlier,
and is expected to continue outperforming.

---

## Motivation

Prediction market literature (Tetlock, Mellers et al.) shows that "superforecasters"
— a small minority — consistently outperform base rates. Polymarket's on-chain,
public trade history makes it possible to identify and follow these actors in real time,
something impossible in traditional financial markets where order books are anonymous
or access is restricted.

---

## What It Tries to Capture

- **Persistent forecasting edge** of top leaderboard wallets across Sports and Politics markets
- **Order-flow momentum** — when a skilled wallet takes a large position, the market
  often underreacts initially, creating a short-lived mispricing
- **Information advantage** — top wallets often trade days before public consensus shifts,
  giving a follower a favourable entry price

---

## Data Requirements

### What the strategy needs

| Dataset | Description | File |
|---------|-------------|------|
| Top-N wallet addresses | From Polymarket all-time PnL leaderboard | fetched by `wallets.py` |
| Per-wallet trade history | Every BUY trade per wallet from HARD_START onwards | `data/wallets/<address>.parquet` |
| Market metadata | Question, tags, `end_date`, `resolved`, `winning_outcome` | `data/markets.parquet` |
| Price history | Hourly YES-token price for every market touched | `data/prices/<condition_id>.parquet` |

### Schema — `data/wallets/<address>.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `condition_id` | str | Unique market identifier |
| `wallet` | str | Wallet address |
| `type` | str | Always `'TRADE'` (Data API) |
| `outcome` | str | Which side they bought: `Yes`, `No`, `Up`, `Down` |
| `price` | float | Entry price (0–1) |
| `size` | float | Tokens purchased |
| `amount_usd` | float | Dollar value of the trade |
| `timestamp` | int | Unix timestamp |
| `title` | str | Market question |
| `tags` | list | Category tags |
| `resolved` | bool | Whether market has resolved |
| `winner` | str/None | Winning outcome (None if unresolved) |

### Schema — `data/markets.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `condition_id` | str | Unique market identifier |
| `question` | str | Market question text |
| `tags` | str | JSON-encoded category list |
| `end_date` | str | ISO 8601 resolution date |
| `resolved` | bool | Whether market has resolved |
| `winning_outcome` | str | `'Yes'`/`'No'` or empty if unresolved |
| `yes_token_id` | str | ERC-1155 token ID for the YES outcome |

### Schema — `data/prices/<condition_id>.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | int | Unix timestamp (hourly) |
| `price` | float | YES token mid price at that hour |

---

## How the Strategy Works

### Step 1 — Collect (offline, `collect_data.py`)

1. Hit the Polymarket leaderboard API for top-N wallets by all-time PnL
2. For each wallet, fetch all trades from `HARD_START` (2024-06-01) to today
3. For each market touched, fetch metadata (question, end date, resolution) via Gamma API
4. Fetch hourly price history for the YES token of every market

### Step 2 — Score (warmup window, `backtest.py` / `wallet_scorer.py`)

Each wallet is scored on its **resolved BUY trades** using a calibration-adjusted z-score:

```
edge  = mean(win_i - price_i)        # actual outcome minus implied probability
SE    = std(win_i - price_i) / √N
z     = edge / SE
```

A wallet **qualifies** for the registry if:
- N ≥ 50 resolved BUY trades (sufficient sample size)
- z ≥ 2.5 (edge is statistically significant at ~99% one-tailed confidence)

### Step 3 — Registry (live simulation, `whale_registry.py`)

The top-N qualified wallets form the **whale registry**. The registry is re-evaluated
weekly using a rolling rescore — wallets can enter or exit as their track record updates.

### Step 4 — Signal & Position (`strategy.py`)

When a registry wallet places a new BUY trade on an unresolved market:
- Open a **paper position** copying their side and outcome
- Size the position proportionally (configurable)
- Track the position until market resolution
- Close at the resolution price, record PnL

### Step 5 — Evaluate (`backtest.py`)

The backtester replays the above in historical simulation:
- Warmup window: score wallets, build initial registry
- Simulation window: fire signals as registry wallets trade, mark-to-market at resolution

### Key configurable knobs

| Knob | Default | Meaning |
|------|---------|---------|
| `--top-n` | 5 | How many qualified wallets to follow |
| `--warmup-months` | 3 | Months of history to score wallets before simulation |
| `--start` | 2024-06-01 | Start of warmup window |
| `EDGE_Z_THRESHOLD` | 2.5 | Minimum z-score to qualify |
| `MIN_RESOLVED_TRADES` | 50 | Minimum resolved trade count |

---

## Files

| File | Role |
|------|------|
| `collect_data.py` | Data collection — leaderboard → trades → markets → prices |
| `wallet_scorer.py` | Scoring logic — computes edge, SE, z-score, cluster scores |
| `whale_registry.py` | Maintains the set of currently-followed wallets |
| `wallets.py` | Leaderboard API wrapper |
| `strategy.py` | Live runner (WebSocket + polling) |
| `backtest.py` | Historical backtester |

---

## Output

- `backtest_results.parquet` — one row per closed paper trade with columns:
  `whale`, `condition_id`, `outcome`, `cluster`, `entry_ts`, `exit_ts`,
  `entry_price`, `exit_price`, `position_usd`, `pnl_usd`, `pnl_pct`,
  `reason`, `z_score`
