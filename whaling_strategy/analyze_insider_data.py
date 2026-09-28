"""
Calibration analysis for Strategy 2 (Insider Trading Capture).

Loads data/insider_raw.parquet (produced by collect_insider_data.py) and
helps you choose the four core knobs:

    1. min_usd            — trade-size threshold
    2. max_hours          — only trades within this many hours of resolution
    3. max_wallet_age     — only wallets younger than this (in days)
    4. side filter        — should we follow BUY only, SELL only, or both?

For each candidate setting it reports:
    • how many signals fire
    • hit rate (% of signals that picked the eventual winner)
    • mean expected ROI per signal (assuming we copy each signal at its trade price)
    • Wilson lower bound on hit rate (so we don't chase noise from tiny samples)

Usage:
    python analyze_insider_data.py
    python analyze_insider_data.py --in=data/insider_raw.parquet
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
#  TUNING KNOBS — edit these to match your collection params and curiosity
# ══════════════════════════════════════════════════════════════════════════════
#
# Collection settings used to produce the parquet (just for reference / logging):
COLLECTION_MIN_USD = 100_000     # collector's --min_usd
COLLECTION_HOURS   = 24          # collector's --hours
COLLECTION_MONTHS  = 2           # collector's --months
#
# Bucket edges for the univariate "hit rate by feature" tables (Section 3).
# Each list defines bucket boundaries — `np.digitize` puts each row into a bin.
# Choose edges that REVEAL variation across your data range.
# Since trades are all >= $100k, bucket *above* $100k. Same idea for hours <=24.
USD_BUCKETS    = [150_000, 250_000, 500_000, 1_000_000, 2_500_000]
HOURS_BUCKETS  = [3, 6, 12, 18, 24]
AGE_BUCKETS    = [7, 30, 90, 180, 365]
#
# Grid-search ranges (Section 4 + Section 7). The analyzer evaluates every
# combination of these. Keep lists small — total combos = product of sizes.
GRID_MIN_USDS  = [100_000, 250_000, 500_000, 1_000_000]
GRID_MAX_HOURS = [3, 6, 12, 24]
GRID_MAX_AGES  = [None, 30, 90, 365]   # None = "any age"
GRID_SIDES     = [None, "BUY", "SELL"] # None = "either side"
#
# Statistical floors — keeps small-sample noise from polluting recommendations.
MIN_N_FOR_GRID  = 5      # ignore knob combos with fewer signals than this
MIN_N_FOR_MEAN  = 15     # require at least this many to rank by mean ROI
MIN_N_FOR_REC   = 20     # required signal count for the "recommended starter knobs"
WILSON_LB_FLOOR = 0.55   # required hit-rate Wilson LB for the recommendation
#
# Repeat-offender wallet table (Section 5)
MIN_SIGNALS_PER_WALLET = 3
#
# ══════════════════════════════════════════════════════════════════════════════


# ── Helpers ────────────────────────────────────────────────────────────────────

def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    """Lower 95% CI on a binomial proportion. Useful for ranking by hit rate
    without being fooled by 1/1 = 100% buckets."""
    if n == 0:
        return 0.0
    phat = successes / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (center - margin) / denom


def bucket_label(edges, idx):
    if idx == 0:
        return f"<{edges[0]}"
    if idx == len(edges):
        return f">={edges[-1]}"
    return f"{edges[idx-1]}–{edges[idx]}"


def print_section(title: str):
    print(f"\n{'='*78}\n  {title}\n{'='*78}")


# ── Derived columns ────────────────────────────────────────────────────────────

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns we need everywhere."""
    df = df.copy()

    # All trades in the parquet are on the YES token (per the collector).
    # So:  BUY  YES wins  if resolution_outcome == YES
    #      SELL YES wins  if resolution_outcome == NO
    outcome_yes = df["resolution_outcome"].astype(str).str.upper().eq("YES")
    is_buy      = df["side"].astype(str).str.upper().eq("BUY")
    df["picked_winner"] = (is_buy & outcome_yes) | (~is_buy & ~outcome_yes)

    # ROI per dollar staked, holding the trade to resolution.
    #   BUY  YES at price p:  payoff = 1 if YES else 0  →  ROI = (1-p)/p if YES, -1 if NO
    #   SELL YES at price p:  shorting YES → payoff if NO wins
    #                         (we approximate: ROI = (p)/(1-p) if NO wins, -1 if YES wins)
    p = df["trade_price"].clip(lower=1e-4, upper=1 - 1e-4)
    buy_roi  = np.where(outcome_yes,  (1 - p) / p, -1.0)
    sell_roi = np.where(~outcome_yes, p / (1 - p), -1.0)
    df["roi"] = np.where(is_buy, buy_roi, sell_roi)

    df["trade_dt"]      = pd.to_datetime(df["trade_ts"], unit="s", utc=True)
    df["resolution_dt"] = pd.to_datetime(df["resolution_ts"], unit="s", utc=True)

    return df


# ── Sections ───────────────────────────────────────────────────────────────────

def overview(df: pd.DataFrame):
    print_section("1. Dataset overview")
    print(f"  Collection params used   : "
          f"min_usd=${COLLECTION_MIN_USD:,}, hours={COLLECTION_HOURS}, "
          f"months={COLLECTION_MONTHS}")
    print(f"  Total qualifying trades  : {len(df):,}")
    print(f"  Unique markets           : {df['condition_id'].nunique():,}")
    print(f"  Unique wallets           : {df['wallet'].nunique():,}")
    print(f"  Date range (trade_ts)    : "
          f"{df['trade_dt'].min().date()} → {df['trade_dt'].max().date()}")
    print(f"  Total USD across signals : ${df['trade_usd'].sum():,.0f}")
    print(f"  BUY / SELL split         : "
          f"{(df['side'].str.upper() == 'BUY').mean():.1%} BUY  /  "
          f"{(df['side'].str.upper() == 'SELL').mean():.1%} SELL")
    print(f"  Outcome split            : "
          f"{(df['resolution_outcome'].str.upper() == 'YES').mean():.1%} YES  /  "
          f"{(df['resolution_outcome'].str.upper() == 'NO').mean():.1%} NO")
    print(f"  Naive hit rate (no flt)  : {df['picked_winner'].mean():.1%}")
    age = df['wallet_age_days_at_trade'].dropna()
    print(f"  Wallet age coverage      : {len(age)/len(df):.1%} of rows have age")
    if len(age):
        print(f"     median age            : {age.median():.0f} days")
        print(f"     fraction <30d old     : {(age < 30).mean():.1%}")


def distributions(df: pd.DataFrame):
    print_section("2. Feature distributions")
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    for col in ["trade_usd", "hours_before_close", "wallet_age_days_at_trade"]:
        s = df[col].dropna()
        if not len(s):
            continue
        q = s.quantile(quantiles)
        print(f"\n  {col}  (n={len(s):,})")
        for p, v in zip(quantiles, q):
            print(f"     p{int(p*100):>3} = {v:>12,.2f}")


def _hit_table(df: pd.DataFrame, edges, col: str, label: str):
    """Bucket `col` by `edges`, report hit rate, ROI, and Wilson LB."""
    df = df.dropna(subset=[col])
    if df.empty:
        print(f"\n  (no rows have {col} populated — skipping)")
        return
    idx = np.digitize(df[col].values, edges)
    rows = []
    for i in range(len(edges) + 1):
        mask = idx == i
        n = int(mask.sum())
        if n == 0:
            continue
        wins = int(df.loc[mask, "picked_winner"].sum())
        rows.append({
            "bucket":  bucket_label(edges, i),
            "n":       n,
            "hit_rate": wins / n,
            "wilson_lb": wilson_lower_bound(wins, n),
            "mean_roi": df.loc[mask, "roi"].mean(),
            "med_roi":  df.loc[mask, "roi"].median(),
        })
    out = pd.DataFrame(rows)
    print(f"\n  Hit rate by {label}:")
    print(out.to_string(index=False, formatters={
        "hit_rate":  "{:.1%}".format,
        "wilson_lb": "{:.1%}".format,
        "mean_roi":  "{:+.1%}".format,
        "med_roi":   "{:+.1%}".format,
    }))


def feature_hit_rates(df: pd.DataFrame):
    print_section("3. Hit rate by single feature (univariate)")

    _hit_table(df, USD_BUCKETS,   "trade_usd",                "trade size (USD)")
    _hit_table(df, HOURS_BUCKETS, "hours_before_close",       "hours before close")
    _hit_table(df, AGE_BUCKETS,   "wallet_age_days_at_trade", "wallet age (days)")

    # Side
    print("\n  Hit rate by side:")
    for side in sorted(df["side"].dropna().unique()):
        sub = df[df["side"] == side]
        wins = int(sub["picked_winner"].sum())
        n = len(sub)
        print(f"     {side:<6} n={n:>6}  hit={wins/n:.1%}  "
              f"wilson_lb={wilson_lower_bound(wins, n):.1%}  "
              f"mean_roi={sub['roi'].mean():+.1%}")


def grid_search(df: pd.DataFrame):
    """Joint sweep over the four knobs. Reports the top configurations by
    Wilson LB on hit rate (so we trust them) AND by total ROI captured."""
    print_section("4. Grid search — find the best knob combination")

    results = []
    for min_usd in GRID_MIN_USDS:
        for mh in GRID_MAX_HOURS:
            for ma in GRID_MAX_AGES:
                for sd in GRID_SIDES:
                    sub = df[
                        (df["trade_usd"] >= min_usd) &
                        (df["hours_before_close"] <= mh)
                    ]
                    if ma is not None:
                        sub = sub[sub["wallet_age_days_at_trade"].fillna(9999) <= ma]
                    if sd is not None:
                        sub = sub[sub["side"].str.upper() == sd]
                    n = len(sub)
                    if n < MIN_N_FOR_GRID:
                        continue
                    wins = int(sub["picked_winner"].sum())
                    results.append({
                        "min_usd":   min_usd,
                        "max_hours": mh,
                        "max_age":   ma if ma is not None else "any",
                        "side":      sd if sd is not None else "any",
                        "n":         n,
                        "hit_rate":  wins / n,
                        "wilson_lb": wilson_lower_bound(wins, n),
                        "mean_roi":  sub["roi"].mean(),
                        "tot_roi$":  (sub["roi"] * sub["trade_usd"]).sum(),
                    })

    if not results:
        print("  No configurations had >=10 signals — collect more data first.")
        return

    res = pd.DataFrame(results)

    fmts = {
        "hit_rate":  "{:.1%}".format,
        "wilson_lb": "{:.1%}".format,
        "mean_roi":  "{:+.1%}".format,
        "tot_roi$":  "${:+,.0f}".format,
    }

    print("\n  Top 15 configs by Wilson lower bound on hit rate "
          "(trustworthy edge):")
    print(res.sort_values("wilson_lb", ascending=False)
              .head(15).to_string(index=False, formatters=fmts))

    print("\n  Top 15 configs by total ROI captured "
          "(absolute dollars made if we'd copied each signal):")
    print(res.sort_values("tot_roi$", ascending=False)
              .head(15).to_string(index=False, formatters=fmts))

    print(f"\n  Top 15 configs by mean ROI per signal "
          f"(min n>={MIN_N_FOR_MEAN} to filter noise):")
    print(res[res["n"] >= MIN_N_FOR_MEAN]
            .sort_values("mean_roi", ascending=False)
            .head(15).to_string(index=False, formatters=fmts))


def repeat_offender_wallets(df: pd.DataFrame):
    """Wallets that appear in many signals — these are the "regulars" and
    the most interesting ones for a live strategy that follows specific addresses."""
    print_section("5. Top wallets by signal volume + accuracy")

    g = df.groupby("wallet").agg(
        n_signals    = ("picked_winner", "size"),
        wins         = ("picked_winner", "sum"),
        markets      = ("condition_id", "nunique"),
        usd_total    = ("trade_usd", "sum"),
        avg_age_days = ("wallet_age_days_at_trade", "mean"),
        mean_roi     = ("roi", "mean"),
    )
    g["hit_rate"]  = g["wins"] / g["n_signals"]
    g["wilson_lb"] = [
        wilson_lower_bound(int(w), int(n))
        for w, n in zip(g["wins"], g["n_signals"])
    ]
    g["pnl_usd"]   = (df.assign(_pnl=df["roi"] * df["trade_usd"])
                       .groupby("wallet")["_pnl"].sum())

    fmts = {
        "hit_rate":  "{:.1%}".format,
        "wilson_lb": "{:.1%}".format,
        "mean_roi":  "{:+.1%}".format,
        "usd_total": "${:,.0f}".format,
        "pnl_usd":   "${:+,.0f}".format,
    }

    repeat = g[g["n_signals"] >= MIN_SIGNALS_PER_WALLET]
    if repeat.empty:
        print(f"  No wallets with >={MIN_SIGNALS_PER_WALLET} signals yet — need more data.")
        return

    print(f"\n  {len(repeat):,} wallets have >={MIN_SIGNALS_PER_WALLET} qualifying signals")
    print("\n  Top 20 by Wilson LB on hit rate (most consistently right):")
    print(repeat.sort_values("wilson_lb", ascending=False)
                .head(20).to_string(formatters=fmts))

    print("\n  Top 20 by total realized PnL (biggest dollar winners):")
    print(repeat.sort_values("pnl_usd", ascending=False)
                .head(20).to_string(formatters=fmts))


def category_breakdown(df: pd.DataFrame):
    """If we tagged Sports vs Politics in collection (we don't yet) we'd
    split here. For now use a heuristic on the question text."""
    print_section("6. Sports vs Politics breakdown (heuristic)")

    sports_kw   = r"NFL|NBA|MLB|NHL|UFC|tennis|soccer|cricket|F1|formula|champions league|super bowl|world cup|cup|league"
    politics_kw = r"election|president|senate|congress|vote|primary|caucus|nominee|governor|prime minister"
    is_sports   = df["question"].str.contains(sports_kw,   case=False, regex=True, na=False)
    is_politics = df["question"].str.contains(politics_kw, case=False, regex=True, na=False)
    df = df.assign(
        category = np.select([is_sports, is_politics],
                              ["Sports", "Politics"],
                              default="Other")
    )
    print()
    print(df.groupby("category").agg(
        n        = ("picked_winner", "size"),
        hit_rate = ("picked_winner", "mean"),
        mean_roi = ("roi", "mean"),
        usd      = ("trade_usd", "sum"),
    ).to_string(formatters={
        "hit_rate": "{:.1%}".format,
        "mean_roi": "{:+.1%}".format,
        "usd":      "${:,.0f}".format,
    }))


def recommendation(df: pd.DataFrame):
    """Pick a single 'starter' configuration to run live."""
    print_section("7. Suggested live-strategy starter knobs")

    # We pick the strictest knob combination that still satisfies the
    # statistical floors defined at the top of the file.
    candidates = []
    for min_usd in GRID_MIN_USDS:
        for mh in GRID_MAX_HOURS:
            for ma in GRID_MAX_AGES:
                sub = df[
                    (df["trade_usd"] >= min_usd) &
                    (df["hours_before_close"] <= mh)
                ]
                if ma is not None:
                    sub = sub[sub["wallet_age_days_at_trade"].fillna(9999) <= ma]
                n = len(sub)
                if n < MIN_N_FOR_REC:
                    continue
                wins = int(sub["picked_winner"].sum())
                wlb  = wilson_lower_bound(wins, n)
                if wlb < WILSON_LB_FLOOR:
                    continue
                candidates.append({
                    "min_usd":   min_usd,
                    "max_hours": mh,
                    "max_age":   ma if ma is not None else "any",
                    "n":         n,
                    "hit_rate":  wins / n,
                    "wilson_lb": wlb,
                    "mean_roi":  sub["roi"].mean(),
                })

    if not candidates:
        print(f"\n  No knob combo passed hit_rate Wilson LB >= {WILSON_LB_FLOOR:.0%} "
              f"with n >= {MIN_N_FOR_REC}.")
        print("  Either the strategy doesn't have edge in this dataset, OR")
        print("  the dataset is still too small. Re-run after more data lands.")
        return

    best = pd.DataFrame(candidates).sort_values(
        ["wilson_lb", "n"], ascending=[False, False]
    ).head(10)

    print(f"\n  Configurations that clear hit_rate Wilson LB >= {WILSON_LB_FLOOR:.0%} "
          f"& n >= {MIN_N_FOR_REC}:")
    print(best.to_string(index=False, formatters={
        "hit_rate":  "{:.1%}".format,
        "wilson_lb": "{:.1%}".format,
        "mean_roi":  "{:+.1%}".format,
    }))
    print("\n  → Recommended starting knobs are the top row above.")
    print("    These are the *strictest* edge with statistical support; loosening")
    print("    will boost signal count but lower the hit-rate floor.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(in_path: Path):
    if not in_path.exists():
        print(f"No file at {in_path}. Run collect_insider_data.py first.")
        return

    df = pd.read_parquet(in_path)
    if df.empty:
        print(f"{in_path} has 0 rows.")
        return

    df = enrich(df)

    overview(df)
    distributions(df)
    feature_hit_rates(df)
    grid_search(df)
    repeat_offender_wallets(df)
    category_breakdown(df)
    recommendation(df)

    print(f"\n{'='*78}\n  Done. Re-run after each new collection to refresh knobs.\n{'='*78}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Calibration analysis for Strategy 2")
    p.add_argument("--in", dest="in_path", default="data/insider_raw.parquet",
                   help="Path to parquet produced by collect_insider_data.py")
    args = p.parse_args()
    main(Path(args.in_path))
