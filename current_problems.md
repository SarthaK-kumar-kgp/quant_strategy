# Current Data Problems

This document tracks the known data collection issues blocking both strategies,
what has been tried, and the current state of each fix.

---

## Strategy 2 — Insider Capture

### Problem 1: Data API hard pagination cap at offset 3,500

**What happens:** `data-api.polymarket.com/trades` returns `HTTP 400 Bad Request`
when `offset >= 3500`. There is no way to paginate past 3,500 trades on any market.

**Impact:** For high-volume markets (2024 US election, Super Bowl, etc.) where
thousands of trades happen in the final 24 hours, we cannot reach the relevant
time window because the most recent 3,500 trades already sit entirely within that
window, consuming the cap before we see any older trades. These markets are silently
skipped or return incomplete data.

**What was tried:**
- Switching from CLOB `/trades` (requires HMAC L2 auth) to public Data API `/trades` ✓
- Testing 18 candidate time-window filter parameters (`before_ts`, `end_ts`, `to_ts`,
  `beforeTs`, `endTs`, `after`, `from`, `startDate`, `endDate`, etc.) — all silently
  ignored by the API. Confirmed via a probe script on a frozen closed market.
- Polygon subgraph as a fallback for high-volume markets — not yet implemented

**Current state:** The `hit_cap_before_window` flag in `collect_insider_data.py`
now only triggers at `offset >= 3500` (real cap), not on earlier 400 errors. A
diagnostic counter (`_DIAG_COUNTS`) tracks how many markets fail at `offset=0`
(asset not queryable) vs at the real cap. Output looks like:

```
Diagnostics — error buckets seen during step 2:
  400_at_offset_0       2,800   sample: cid=0x320045... offset=0 ...
```

**Workaround in place:** Markets that cap before reaching the time window are
flagged and counted but not discarded — the trades we DID retrieve are still
searched for qualifying trades. In practice most qualifying ($100k+) trades
appear at the very end of the window (final few hours), which the Data API
returns in the first few pages (newest-first), so most of the cap-hit markets
are still usable.

---

### Problem 2: 100% cap rate on early markets — 0 qualifying rows

**What happens:** The first ~3,000 markets processed show `capped=100%` and
`rows=0`. Progress looks like:

```
[ 2950/14147]  rows=    0  capped=2950
```

**Root cause (identified):** Two distinct failure modes were being conflated
under a single "capped" label:
1. **True cap** — market had >3,500 trades, API returned 400 at `offset=3500`.
   These are real giant markets. Among their most-recent 3,500 trades, no single
   trade was ≥ `min_usd` ($100k). This is correct behaviour — $100k+ trades are
   rare even on large markets.
2. **Asset not queryable** — API returned 400 at `offset=0`, meaning the YES
   token for that closed market is no longer indexed by the Data API. These
   markets should be silently skipped, not counted as cap-hits.

**Fix applied:** The code now distinguishes 400-at-offset-0 (asset issue, silent
skip) from 400-at-offset-3500 (real cap). The diagnostic counter shows the split.

**Current state:** Data collection is still running. Once complete, the diagnostic
output will show exactly how many markets fell into each bucket. If the asset-not-
queryable rate is very high (>50%) we may need to rebuild the markets list using
a different token source.

---

### Problem 3: Very large market pool slowing collection

**What happens:** With `--months=2 --min_usd=100000`, the Gamma API returns
~100k markets total, of which ~14k pass the volume pre-filter (`volume >= $100k`).
Processing 14k markets at 4 concurrent requests with rate-limit backoffs takes
several hours.

**What was tried:**
- Lowering concurrency from 8 → 4 (reduces 429 rate-limit errors)
- Exponential backoff on 429 (1.5s, 3s, 6s)
- Volume pre-filter to skip markets that cannot possibly contain a `$min_usd` trade
- Progress counter every 25 markets so the run isn't silent

**Current state:** Still running. No data yet in `data/insider_raw.parquet`.

---

## Strategy 1 — Smart Money Follow

Strategy 1 has three separate data layer problems that make the backtester
produce `0` results regardless of threshold settings.

### Problem 1: All market resolutions are missing

**What happens:** `data/markets.parquet` has 559 rows, all with `resolved=False`
and `winning_outcome=''`.

**Root cause:** `collect_data.py` calls `fetcher.fetch_market(cid)` once at
collection time and writes whatever the API returns. Most of the 559 markets were
fetched while they were still open (or are recurring short-window markets that
resolve within minutes). The collector has no refresh step — it never re-checks
markets after they close.

**Impact:** `wallet_scorer.py` cannot compute z-scores because `winner` is always
`None`. The backtester seeds 0 wallets, fires 0 signals, produces 0 results.

**Fix needed:** Add a `refresh_resolutions()` pass that re-calls the Gamma API
for every market in `markets.parquet` where `resolved=False`, and overwrites the
row with the updated `resolved`/`winning_outcome` values. This is a one-time
repair, after which the daily collection only needs to refresh recent markets.

---

### Problem 2: Trade `type` field is always `'TRADE'`, not `'BUY'`/`'SELL'`

**What happens:** Every row in every `data/wallets/<address>.parquet` has
`type='TRADE'`. The scorer filters `type == 'BUY'` and finds nothing.

**Root cause:** The Polymarket Data API returns `type='TRADE'` for all trades.
The actual side (which outcome the wallet bought) is in the `outcome` field:
`'Yes'`, `'No'`, `'Up'`, `'Down'`, etc. The scorer was written against an assumed
schema where buys and sells are labelled, which doesn't exist in the API response.

**Impact:** `_resolved_buys()` in `wallet_scorer.py` returns an empty list for
every wallet. Z-scores are undefined; no wallet qualifies.

**Fix needed:** Rewrite `_resolved_buys()` to select rows based on the `outcome`
field rather than `type`. The semantics need careful handling because markets have
both binary YES/NO outcomes and directional Up/Down outcomes.

---

### Problem 3: Wallet trade history is too short

**What happens:** Some wallets in `data/wallets/` have all their trades on a
single day (`2026-05-09`), suggesting only recent activity was fetched, not the
full historical window back to `HARD_START` (2024-06-01).

**Root cause:** `collect_data.py` calls `fetcher.fetch_wallet_trades()` which
paginates the Data API activity endpoint. High-volume wallets may have thousands
of trades, and the fetcher likely hits a pagination cap or timeout before reaching
the historical window.

**Impact:** Wallets scored on 1 day of data do not have enough resolved trades
to qualify (need N ≥ 50). Even if the scorer's `type` filter were fixed, these
wallets would still not qualify.

**Fix needed:** Implement date-windowed pagination in `fetch_wallet_trades()` so
it explicitly fetches backwards in time until it reaches `HARD_START` or the
wallet's first-ever trade, whichever is later.

---

## Summary Table

| Strategy | Problem | Impact | Status |
|----------|---------|--------|--------|
| S2 | Data API 3,500-trade offset cap | Misses older trades on giant markets | Partially mitigated — cap correctly identified, no time filter exists |
| S2 | 400 at offset=0 misclassified as cap-hit | Inflated cap counter | Fixed — now correctly bucketed |
| S2 | 14k markets takes hours to process | No data yet | Collection in progress |
| S1 | All markets unresolved in parquet | 0 wallet scores | Not fixed — needs refresh pass |
| S1 | `type='TRADE'` instead of `BUY`/`SELL` | 0 resolved buys found | Not fixed — needs scorer rewrite |
| S1 | Wallet trade history too short | Wallets don't qualify by trade count | Not fixed — needs paginator fix |

---

## Recommended Next Steps

1. **Wait for Strategy 2 collection to finish**, then run `analyze_insider_data.py`.
   If the dataset is too small, re-run with `--min_usd=25000 --months=6` to widen.

2. **For Strategy 1**, fix in this order:
   a. Write a `refresh_resolutions.py` script that re-fetches market metadata for
      all 559 markets and updates `resolved`/`winning_outcome`
   b. Patch `wallet_scorer.py` to use the `outcome` field for side determination
   c. Fix `fetch_wallet_trades()` to paginate back to `HARD_START`
   d. Re-run the backtester

3. **If the 400-at-offset-0 rate is very high for Strategy 2**, consider using the
   Polygon subgraph (`api.thegraph.com/subgraphs/name/polymarket/...`) as an
   alternative trade source — it has no offset cap and supports time-range filters.
