# Strategy 2 — Insider Capture

## Hypothesis

In prediction markets, trades placed unusually close to market resolution by wallets
with large position sizes represent a disproportionate information signal. A trader
who commits $100k+ to a market in the final 24 hours is not guessing — they either
have inside information, are significantly better-informed than the market, or have
a strong model edge that only becomes confident as resolution approaches.

If these large late-window trades are directionally correct more than chance would
predict (i.e. hit rate > 50% after adjusting for price), then copying them immediately
upon detection generates positive expected value.

**Core assumption:** Large, late-window trades on Sports and Politics markets are
placed by informed actors. The market price at that moment underestimates the
probability of the outcome the actor is buying.

---

## Motivation

Traditional financial markets conceal order flow. Polymarket is entirely on-chain —
every trade, wallet, and timestamp is public. This creates a rare opportunity to
observe what would be "institutional dark pool flow" in equities, but with full
transparency.

The strategy is specifically targeting **Sports and Politics** because:
- These markets have clear, verifiable resolution events (game scores, election results)
- Insider knowledge is plausible (team insiders, political operatives, pollsters)
- Trade sizes on these markets tend to be more bimodal (retail vs. informed) than
  crypto price markets where everyone has equal information access

---

## What It Tries to Capture

- **Informed late-window flow** — large trades placed in the final hours of a market
  by wallets that appear purpose-built (young, single-category)
- **Calibration mispricing** — prices at the time of the signal trade are expected
  to be wrong in the direction of the trade
- **Signal-to-noise edge** — by requiring minimum trade size and maximum time-to-close,
  we filter out uninformed retail noise and focus on statistically anomalous behaviour

---

## Data Requirements

### What the strategy needs

| Dataset | Description | File |
|---------|-------------|------|
| Resolved Sports/Politics markets | Last N months of closed markets from Gamma API | `data/insider_markets_cache.json` |
| Qualifying trades per market | All trades ≥ min_usd within hours_window of resolution | `data/insider_raw.parquet` |
| Wallet first-trade timestamps | To compute wallet age at time of trade | fetched live from Data API |

### Schema — `data/insider_raw.parquet`

One row per qualifying trade (i.e. trade that passed the size + window filter during collection).

| Column | Type | Description |
|--------|------|-------------|
| `condition_id` | str | Unique market identifier |
| `question` | str | Market question text |
| `resolution_ts` | int | Unix timestamp of market close |
| `resolution_outcome` | str | `'YES'` or `'NO'` (from Gamma `winner` field) |
| `wallet` | str | `proxyWallet` of the trader |
| `trade_ts` | int | Unix timestamp of the trade |
| `hours_before_close` | float | How far before resolution the trade was placed |
| `side` | str | `'BUY'` or `'SELL'` (YES token) |
| `trade_price` | float | Price at time of trade (0–1) |
| `trade_usd` | float | Dollar size of the trade |
| `trade_size_tokens` | float | Token quantity |
| `wallet_first_trade_ts` | int/None | Unix timestamp of wallet's first-ever Polymarket trade |
| `wallet_age_days_at_trade` | float/None | Age of wallet in days at time of trade |

### Derived columns (added by `analyze_insider_data.py`)

| Column | Description |
|--------|-------------|
| `picked_winner` | bool — did this trade's direction match the resolved outcome? |
| `roi` | Expected return if copied at trade price and held to resolution |

---

## Pipeline — Step by Step

### Step 1 — Collect resolved markets (`collect_insider_data.py`)

Paginates Gamma API for all **closed** Sports (tag_id=1) and Politics (tag_id=2) markets
from the last N months. Results are cached to `data/insider_markets_cache.json` so
subsequent runs skip this step unless `--refresh_markets` is passed.

**Pre-filter:** any market with total volume < `min_usd` cannot contain a single trade
that large, so it is skipped before hitting the Data API at all.

### Step 2 — Fetch qualifying trades (`collect_insider_data.py`)

For each market that passes the volume pre-filter:
1. Extract the YES token ID from the `clobTokenIds` field
2. Call `data-api.polymarket.com/trades?market=<yes_token>&limit=500&offset=<N>`
3. Walk trades newest-first; stop when timestamp drops below `resolution_ts - hours`
4. Keep only trades where `trade_usd >= min_usd`
5. Retry on 429 (rate limit) with exponential backoff; treat 400 at offset ≥ 3,500
   as the hard pagination cap

### Step 3 — Wallet ages (`collect_insider_data.py`)

For each unique wallet that appears in a qualifying trade, fetch their first-ever
activity timestamp from `data-api.polymarket.com/activity` to compute wallet age.
This is used as a proxy for whether the wallet was purpose-built for this trade.

### Step 4 — Analyze and calibrate (`analyze_insider_data.py`)

Loads `data/insider_raw.parquet` and runs 7 analysis sections:
1. Dataset overview (row counts, naive baseline hit rate)
2. Feature distributions (percentiles of size, hours, wallet age)
3. Univariate hit rates (each feature bucketed separately)
4. Grid search (all combinations of the four knobs; ranked by Wilson LB hit rate)
5. Repeat-offender wallets (wallets that appear multiple times; hit rate + PnL)
6. Sports vs Politics split (heuristic regex categorisation)
7. Recommended starter knobs (strictest config with Wilson LB ≥ 55% and n ≥ 20)

### Step 5 — Run strategy (`insider_runner.py`)

Takes the recommended knobs (from a `knobs.json` file or CLI flags) and runs in one
of two modes:

**Backtest mode:** Replays `data/insider_raw.parquet` through the `passes_filters()`
function. For each row that passes all knob filters, computes the hypothetical ROI
if you had copied the trade at its price and held to resolution.

**Live mode:** Polls Polymarket every N seconds for active Sports/Politics markets
resolving within the scan horizon. For each recent trade on those markets that passes
all knob filters, emits a paper signal. Checks open positions periodically and marks
them to market once the Gamma API shows the market as closed.

---

## Signal Logic

```python
def passes_filters(trade_usd, hours_before_close, wallet_age_days, side, knobs):
    if trade_usd < knobs.min_usd:               return False
    if hours_before_close > knobs.max_hours:    return False
    if hours_before_close < 0:                  return False   # post-resolution
    if knobs.max_wallet_age_days is not None:
        if wallet_age_days > knobs.max_wallet_age_days: return False
    if knobs.side_filter is not None:
        if side != knobs.side_filter:           return False
    return True
```

The **same function** is called in both backtest and live mode so results are
directly comparable.

---

## PnL Model

Trades are on the YES token. ROI is calculated as:

| Scenario | ROI |
|----------|-----|
| BUY YES at price p, outcome = YES | `(1 - p) / p` |
| BUY YES at price p, outcome = NO | `-100%` |
| SELL YES at price p, outcome = NO | `p / (1 - p)` |
| SELL YES at price p, outcome = YES | `-100%` |

Paper position size is fixed at `follow_size_usd` dollars per signal.

---

## Configurable Knobs

All knobs are set at the top of `analyze_insider_data.py` (for calibration) and
consumed by `insider_runner.py` (for execution). Save the recommended values to
`knobs.json` to pass between the two scripts.

| Knob | What it controls |
|------|-----------------|
| `min_usd` | Minimum trade size in USD to fire a signal |
| `max_hours_before_close` | Only trades within this window of resolution |
| `max_wallet_age_days` | Only wallets younger than this (proxy for purpose-built) |
| `side_filter` | `"BUY"`, `"SELL"`, or `None` for either |
| `follow_size_usd` | Paper stake per signal (does not affect hit rate analysis) |
| `confirmation_required` | Require a second qualifying trade before firing signal |

---

## Files

| File | Role |
|------|------|
| `collect_insider_data.py` | Data collection — markets → trades → wallet ages |
| `analyze_insider_data.py` | Calibration — finds optimal knobs from historical data |
| `insider_runner.py` | Execution — backtest and live paper trading using calibrated knobs |

---

## Output

**`signals_backtest.parquet`** / **`signals_live.parquet`** — one row per emitted signal:

| Column | Description |
|--------|-------------|
| `mode` | `'backtest'` or `'live'` |
| `signal_emitted_ts` | When the signal was fired |
| `condition_id` | Market identifier |
| `question` | Market question |
| `wallet` | Wallet that placed the triggering trade |
| `side` | `BUY` or `SELL` |
| `trade_price` | Price at signal time |
| `trade_usd` | Size of the triggering trade |
| `hours_before_close` | Time to resolution when signal fired |
| `wallet_age_days_at_trade` | Wallet age at time of trade |
| `follow_size_usd` | Paper stake amount |
| `entry_price` | Price we "entered" at (= trade_price) |
| `resolution_ts` | Market resolution timestamp |
| `resolution_outcome` | `YES` or `NO` |
| `roi_pct` | Return on stake (%) |
| `pnl_usd` | Dollar PnL on the paper position |
| `status` | `'open'` or `'resolved'` |
