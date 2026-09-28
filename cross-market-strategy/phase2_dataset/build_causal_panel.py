from pathlib import Path

import numpy as np
import pandas as pd


folder = Path(__file__).resolve().parent
project = folder.parent.parent
registry_file = folder.parent / "phase1_registry" / "eligible_market_registry.parquet"
price_folder = project / "data" / "pm_prices_clob_1m"
no_price_folder = project / "data" / "pm_prices_clob_1m_no"
trade_bucket_folder = project / "data" / "pm_prices"
spot_folder = project / "data" / "spot"
vol_folder = project / "data" / "vol"

panel_folder = folder / "causal_panel_1m"
proxy_folder = folder / "execution_proxy_60s"
manifest_file = folder / "panel_manifest.parquet"
excluded_file = folder / "panel_exclusions.parquet"
feature_coverage_file = folder / "feature_coverage.csv"
tape_audit_file = folder / "transaction_tape_audit.parquet"
report_file = folder / "leakage_and_coverage_report.md"
execution_report_file = folder / "execution_proxy_limitations.md"

minutes_per_year = 365 * 24 * 60
market_delay_seconds = 60
dvol_delay_seconds = 3600
dvol_max_age_seconds = 7200

assets = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}

feature_scopes = {
    "spot_close": "all_assets",
    "spot_high": "all_assets",
    "spot_low": "all_assets",
    "ret_1m": "all_assets",
    "ret_5m": "all_assets",
    "ret_15m": "all_assets",
    "ret_1h": "all_assets",
    "ret_6h": "all_assets",
    "ret_24h": "all_assets",
    "rv_1h": "all_assets",
    "rv_6h": "all_assets",
    "rv_24h": "all_assets",
    "rv_7d": "all_assets",
    "parkinson_1h": "all_assets",
    "parkinson_6h": "all_assets",
    "parkinson_24h": "all_assets",
    "garman_klass_1h": "all_assets",
    "garman_klass_6h": "all_assets",
    "garman_klass_24h": "all_assets",
    "jump_flag": "all_assets",
    "jump_zscore": "all_assets",
    "jump_count_up_1h": "all_assets",
    "jump_count_down_1h": "all_assets",
    "jump_count_6h": "all_assets",
    "jump_abs_return_6h": "all_assets",
    "volatility_ratio_1h_24h": "all_assets",
    "volume_1h": "all_assets",
    "volume_ratio_1h_24h": "all_assets",
    "volatility_regime": "all_assets",
    "return_regime": "all_assets",
    "log_distance_to_barrier": "all_assets",
    "distance_to_barrier_pct": "all_assets",
    "distance_in_expiry_sigma": "all_assets",
    "market_yes_price": "all_assets",
    "market_yes_change": "all_assets_lagged_optional",
    "dvol_sigma": "btc_eth_only",
}


def write_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def rolling_volatility(returns, window):
    variance = returns.pow(2).rolling(window, min_periods=window).mean()
    return np.sqrt(variance * minutes_per_year)


def rolling_range_volatility(values, window):
    variance = values.rolling(window, min_periods=window).mean()
    return np.sqrt(variance.clip(lower=0) * minutes_per_year)


def load_spot(asset):
    path = spot_folder / f"{assets[asset]}_1m.parquet"
    frame = pd.read_parquet(path).sort_values("ts").reset_index(drop=True)
    log_close = np.log(frame["close"])
    returns = log_close.diff()

    frame["ret_1m"] = returns
    for minutes, name in [(5, "ret_5m"), (15, "ret_15m"), (60, "ret_1h"),
                          (360, "ret_6h"), (1440, "ret_24h")]:
        frame[name] = log_close.diff(minutes)

    for minutes, name in [(60, "rv_1h"), (360, "rv_6h"),
                          (1440, "rv_24h"), (10080, "rv_7d")]:
        frame[name] = rolling_volatility(returns, minutes)

    log_range = np.log(frame["high"] / frame["low"])
    parkinson = log_range.pow(2) / (4 * np.log(2))
    log_open_close = np.log(frame["close"] / frame["open"])
    garman_klass = 0.5 * log_range.pow(2) - (2 * np.log(2) - 1) * log_open_close.pow(2)

    for minutes, suffix in [(60, "1h"), (360, "6h"), (1440, "24h")]:
        frame[f"parkinson_{suffix}"] = rolling_range_volatility(parkinson, minutes)
        frame[f"garman_klass_{suffix}"] = rolling_range_volatility(garman_klass, minutes)

    prior_sigma = returns.rolling(60, min_periods=60).std().shift(1)
    jump_zscore = returns / prior_sigma.replace(0, np.nan)
    jump_flag = jump_zscore.abs() >= 4
    jump_up = (jump_flag & (returns > 0)).astype(int)
    jump_down = (jump_flag & (returns < 0)).astype(int)
    jump_return = returns.abs().where(jump_flag, 0)

    frame["jump_flag"] = jump_flag.astype(int)
    frame["jump_zscore"] = jump_zscore
    frame["jump_count_up_1h"] = jump_up.rolling(60, min_periods=60).sum()
    frame["jump_count_down_1h"] = jump_down.rolling(60, min_periods=60).sum()
    frame["jump_count_6h"] = jump_flag.astype(int).rolling(360, min_periods=360).sum()
    frame["jump_abs_return_6h"] = jump_return.rolling(360, min_periods=360).sum()

    frame["volatility_ratio_1h_24h"] = frame["rv_1h"] / frame["rv_24h"].replace(0, np.nan)
    frame["volume_1h"] = frame["volume"].rolling(60, min_periods=60).sum()
    volume_24h_average = frame["volume_1h"].rolling(1440, min_periods=1440).mean()
    frame["volume_ratio_1h_24h"] = frame["volume_1h"] / volume_24h_average.replace(0, np.nan)

    ratio = frame["volatility_ratio_1h_24h"]
    frame["volatility_regime"] = np.select([ratio < 0.8, ratio > 1.2], [-1, 1], default=0)
    six_hour_scale = frame["rv_24h"] * np.sqrt(360 / minutes_per_year)
    standardized_return = frame["ret_6h"] / six_hour_scale.replace(0, np.nan)
    frame["return_regime"] = np.select(
        [standardized_return < -1, standardized_return > 1], [-1, 1], default=0
    )

    columns = ["ts", "open", "high", "low", "close", "volume"] + [
        name for name in feature_scopes if name in frame.columns
    ]
    return {column: frame[column].to_numpy() for column in columns}


def load_dvol(asset):
    if asset not in ["BTC", "ETH"]:
        return None
    path = vol_folder / f"DVOL_{asset}.parquet"
    frame = pd.read_parquet(path, columns=["ts", "sigma"]).sort_values("ts")
    return {
        "source_ts": frame["ts"].to_numpy(dtype=np.int64),
        "available_ts": frame["ts"].to_numpy(dtype=np.int64) + dvol_delay_seconds,
        "sigma": frame["sigma"].to_numpy(dtype=float),
    }


def first_touch(spot, barrier, direction, start_time, end_time):
    first_minute = int(np.ceil(start_time.timestamp() / 60) * 60)
    end_second = int(end_time.timestamp())
    left = np.searchsorted(spot["ts"], first_minute, side="left")
    right = np.searchsorted(spot["ts"], end_second, side="left")
    values = spot["high"][left:right] if direction == "up" else spot["low"][left:right]
    if direction == "up":
        matches = np.flatnonzero(values >= barrier)
    else:
        matches = np.flatnonzero(values <= barrier)
    return int(spot["ts"][left + matches[0]]) if len(matches) else None


def attach_dvol(panel, decision_seconds, dvol):
    count = len(panel)
    panel["dvol_source_time"] = pd.Series(
        pd.NaT, index=panel.index, dtype="datetime64[ns, UTC]"
    )
    panel["dvol_available_time"] = pd.Series(
        pd.NaT, index=panel.index, dtype="datetime64[ns, UTC]"
    )
    panel["dvol_age_minutes"] = np.nan
    panel["dvol_sigma"] = np.nan
    panel["dvol_available"] = False
    if dvol is None or count == 0:
        return panel

    indices = np.searchsorted(dvol["available_ts"], decision_seconds, side="right") - 1
    valid = indices >= 0
    safe_indices = np.maximum(indices, 0)
    ages = decision_seconds - dvol["available_ts"][safe_indices]
    valid &= ages >= 0
    valid &= ages <= dvol_max_age_seconds
    valid_rows = np.flatnonzero(valid)
    valid_indices = indices[valid]

    panel.loc[valid_rows, "dvol_source_time"] = pd.to_datetime(
        dvol["source_ts"][valid_indices], unit="s", utc=True
    ).to_numpy()
    panel.loc[valid_rows, "dvol_available_time"] = pd.to_datetime(
        dvol["available_ts"][valid_indices], unit="s", utc=True
    ).to_numpy()
    panel.loc[valid_rows, "dvol_age_minutes"] = ages[valid] / 60
    panel.loc[valid_rows, "dvol_sigma"] = dvol["sigma"][valid_indices]
    panel.loc[valid_rows, "dvol_available"] = True
    return panel


def next_price_sample(decision_times, decision_seconds, prices):
    count = len(decision_times)
    next_bucket = pd.Series(pd.NaT, index=range(count), dtype="datetime64[ns, UTC]")
    next_available = pd.Series(pd.NaT, index=range(count), dtype="datetime64[ns, UTC]")
    next_price = np.full(count, np.nan)
    next_delay = np.full(count, np.nan)
    valid = np.zeros(count, dtype=bool)
    if prices.empty:
        return valid, next_bucket, next_available, next_price, next_delay

    available_times = prices["available_time"]
    available_seconds = (available_times.astype("int64") // 10**9).to_numpy(dtype=np.int64)
    next_indices = np.searchsorted(available_seconds, decision_seconds, side="right")
    in_bounds = next_indices < len(prices)
    safe_indices = np.minimum(next_indices, len(prices) - 1)
    delays = available_seconds[safe_indices] - decision_seconds
    valid = in_bounds & (delays > 0) & (delays <= 60)

    rows = np.flatnonzero(valid)
    source_indices = next_indices[valid]
    next_bucket.iloc[rows] = prices["bucket_ts"].iloc[source_indices].to_numpy()
    next_available.iloc[rows] = prices["available_time"].iloc[source_indices].to_numpy()
    next_price[rows] = prices["last_price"].iloc[source_indices].to_numpy(dtype=float)
    next_delay[rows] = delays[valid]
    return valid, next_bucket, next_available, next_price, next_delay


def current_price_sample(decision_times, decision_seconds, prices):
    count = len(decision_times)
    current_bucket = pd.Series(pd.NaT, index=range(count), dtype="datetime64[ns, UTC]")
    current_available = pd.Series(pd.NaT, index=range(count), dtype="datetime64[ns, UTC]")
    current_price = np.full(count, np.nan)
    valid = np.zeros(count, dtype=bool)
    if prices.empty:
        return valid, current_bucket, current_available, current_price

    available_seconds = (
        prices["available_time"].astype("int64") // 10**9
    ).to_numpy(dtype=np.int64)
    indices = np.searchsorted(available_seconds, decision_seconds, side="left")
    in_bounds = indices < len(prices)
    safe_indices = np.minimum(indices, len(prices) - 1)
    valid = in_bounds & (available_seconds[safe_indices] == decision_seconds)
    rows = np.flatnonzero(valid)
    source_indices = indices[valid]
    current_bucket.iloc[rows] = prices["bucket_ts"].iloc[source_indices].to_numpy()
    current_available.iloc[rows] = prices["available_time"].iloc[source_indices].to_numpy()
    current_price[rows] = prices["last_price"].iloc[source_indices].to_numpy(dtype=float)
    return valid, current_bucket, current_available, current_price


def load_proxy_prices(path):
    if not path.exists():
        return pd.DataFrame(columns=["bucket_ts", "last_price", "available_time"])
    prices = pd.read_parquet(path, columns=["bucket_ts", "last_price"])
    prices["bucket_ts"] = pd.to_datetime(prices["bucket_ts"], utc=True)
    prices = prices[prices["last_price"].between(0, 1)].copy()
    prices = prices.sort_values("bucket_ts").drop_duplicates("bucket_ts", keep="last")
    prices["available_time"] = prices["bucket_ts"] + pd.Timedelta(seconds=market_delay_seconds)
    return prices


def make_execution_proxy(row, decision_times, decision_seconds, yes_prices, no_prices):
    yes_signal = current_price_sample(decision_times, decision_seconds, yes_prices)
    no_signal = current_price_sample(decision_times, decision_seconds, no_prices)
    signal_both = yes_signal[0] & no_signal[0]
    signal_sum = np.where(signal_both, yes_signal[3] + no_signal[3], np.nan)
    signal_gap = signal_sum - 1

    yes = next_price_sample(decision_times, decision_seconds, yes_prices)
    no = next_price_sample(decision_times, decision_seconds, no_prices)
    both_available = yes[0] & no[0]
    same_time = both_available & yes[2].eq(no[2]).to_numpy()
    price_sum = np.where(same_time, yes[3] + no[3], np.nan)
    complement_gap = price_sum - 1

    return pd.DataFrame({
        "condition_id": row.condition_id,
        "decision_time": decision_times.to_numpy(),
        "lookup_window_end": (decision_times + pd.Timedelta(seconds=60)).to_numpy(),
        "yes_token_id": str(row.yes_token_id),
        "no_token_id": str(row.no_token_id),
        "yes_signal_price_available": yes_signal[0],
        "yes_signal_bucket_start": yes_signal[1],
        "yes_signal_available_at": yes_signal[2],
        "yes_signal_price": yes_signal[3],
        "no_signal_price_available": no_signal[0],
        "no_signal_bucket_start": no_signal[1],
        "no_signal_available_at": no_signal[2],
        "no_signal_price": no_signal[3],
        "both_signal_prices_available": signal_both,
        "signal_yes_no_price_sum": signal_sum,
        "signal_yes_no_complement_gap": signal_gap,
        "signal_yes_no_absolute_complement_gap": np.abs(signal_gap),
        "yes_proxy_available_60s": yes[0],
        "yes_proxy_bucket_start": yes[1],
        "yes_proxy_available_at": yes[2],
        "yes_proxy_price": yes[3],
        "yes_proxy_delay_seconds": yes[4],
        "no_proxy_available_60s": no[0],
        "no_proxy_bucket_start": no[1],
        "no_proxy_available_at": no[2],
        "no_proxy_price": no[3],
        "no_proxy_delay_seconds": no[4],
        "both_proxy_available_60s": both_available,
        "proxy_samples_same_time": same_time,
        "yes_no_price_sum": price_sum,
        "yes_no_complement_gap": complement_gap,
        "yes_no_absolute_complement_gap": np.abs(complement_gap),
        "proxy_is_transaction": False,
        "proxy_is_executable_quote": False,
    })


def make_panel(row, spot, dvol):
    price_file = price_folder / f"{row.condition_id}.parquet"
    if not price_file.exists():
        return None, None, "missing_clob_price_file", None

    prices = pd.read_parquet(price_file)
    needed = {"bucket_ts", "last_price", "vwap", "n_trades", "volume_usd"}
    if not needed.issubset(prices.columns):
        return None, None, "invalid_clob_price_schema", None

    prices["bucket_ts"] = pd.to_datetime(prices["bucket_ts"], utc=True)
    prices = prices[prices["last_price"].between(0, 1)].copy()
    prices = prices.sort_values("bucket_ts").drop_duplicates("bucket_ts", keep="last")
    if prices.empty:
        return None, None, "no_valid_clob_price_samples", None

    prices["available_time"] = prices["bucket_ts"] + pd.Timedelta(seconds=market_delay_seconds)
    prices["pm_gap_minutes"] = prices["bucket_ts"].diff().dt.total_seconds() / 60
    prices["market_yes_change"] = prices["last_price"].diff()
    decision_seconds_all = (prices["available_time"].astype("int64") // 10**9).to_numpy(dtype=np.int64)

    touch_second = first_touch(
        spot, float(row.barrier), row.direction,
        row.official_window_start, row.official_window_end,
    )
    if int(touch_second is not None) != int(row.touched):
        return None, None, "touch_recheck_disagrees_with_phase1", touch_second

    listing_second = row.listing_start.timestamp()
    window_start_second = row.official_window_start.timestamp()
    window_end_second = row.official_window_end.timestamp()
    usable = decision_seconds_all >= max(listing_second, window_start_second)
    usable &= decision_seconds_all < window_end_second
    if touch_second is not None:
        usable &= decision_seconds_all < touch_second
    selected = prices.loc[usable].copy()
    decision_seconds = decision_seconds_all[usable]
    if selected.empty:
        reason = "touch_before_every_usable_decision" if touch_second is not None else "no_price_before_window_end"
        return None, None, reason, touch_second

    spot_indices = np.searchsorted(spot["ts"], decision_seconds, side="left") - 1
    valid_spot = spot_indices >= 0
    selected = selected.loc[valid_spot].copy()
    decision_seconds = decision_seconds[valid_spot]
    spot_indices = spot_indices[valid_spot]
    if selected.empty:
        return None, None, "no_completed_spot_candle", touch_second

    decision_times = selected["available_time"].reset_index(drop=True)
    spot_close = spot["close"][spot_indices]
    if row.direction == "up":
        log_distance = np.log(float(row.barrier) / spot_close)
        distance_pct = float(row.barrier) / spot_close - 1
    else:
        log_distance = np.log(spot_close / float(row.barrier))
        distance_pct = spot_close / float(row.barrier) - 1

    minutes_to_expiry = (window_end_second - decision_seconds) / 60
    expiry_scale = spot["rv_24h"][spot_indices] * np.sqrt(minutes_to_expiry / minutes_per_year)
    distance_in_sigma = log_distance / np.where(expiry_scale > 0, expiry_scale, np.nan)
    touch_time = pd.to_datetime(touch_second, unit="s", utc=True) if touch_second is not None else pd.NaT

    panel = pd.DataFrame({
        "condition_id": row.condition_id,
        "asset": row.asset,
        "direction": row.direction,
        "barrier": float(row.barrier),
        "barrier_group_id": row.barrier_group_id,
        "barrier_group_size": int(row.barrier_group_size),
        "barrier_rank_easiest_first": int(row.barrier_rank_easiest_first),
        "listing_start": row.listing_start,
        "official_window_start": row.official_window_start,
        "official_window_end": row.official_window_end,
        "pm_bucket_start": selected["bucket_ts"].to_numpy(),
        "decision_time": decision_times.to_numpy(),
        "pm_delay_seconds": market_delay_seconds,
        "pm_gap_minutes": selected["pm_gap_minutes"].to_numpy(dtype=float),
        "minutes_to_expiry": minutes_to_expiry,
        "minutes_since_listing": (decision_seconds - listing_second) / 60,
        "minutes_since_window_start": (decision_seconds - window_start_second) / 60,
        "window_elapsed_fraction": (decision_seconds - window_start_second) / (
            window_end_second - window_start_second
        ),
        "spot_candle_start": pd.to_datetime(spot["ts"][spot_indices], unit="s", utc=True),
        "spot_available_time": pd.to_datetime(spot["ts"][spot_indices] + 60, unit="s", utc=True),
        "spot_close": spot_close,
        "spot_high": spot["high"][spot_indices],
        "spot_low": spot["low"][spot_indices],
        "log_distance_to_barrier": log_distance,
        "distance_to_barrier_pct": distance_pct,
        "distance_in_expiry_sigma": distance_in_sigma,
        "market_yes_price": selected["last_price"].to_numpy(dtype=float),
        "market_yes_change": selected["market_yes_change"].to_numpy(dtype=float),
        "market_no_probability_proxy": 1 - selected["last_price"].to_numpy(dtype=float),
        "pm_source_points": selected["n_trades"].to_numpy(dtype=int),
        "pm_source_volume_usd": selected["volume_usd"].to_numpy(dtype=float),
        "y": int(row.touched),
    })

    spot_feature_names = [
        name for name, scope in feature_scopes.items()
        if scope == "all_assets" and name in spot and name not in panel.columns
    ]
    for name in spot_feature_names:
        panel[name] = spot[name][spot_indices]

    panel["first_touch_time"] = pd.Series(
        [touch_time] * len(panel), dtype="datetime64[ns, UTC]"
    )
    panel = attach_dvol(panel, decision_seconds, dvol)
    panel["contract_row_weight"] = 1 / len(panel)
    no_prices = load_proxy_prices(no_price_folder / f"{row.condition_id}.parquet")
    proxy = make_execution_proxy(row, decision_times, decision_seconds, prices, no_prices)

    assert (panel["decision_time"] - panel["pm_bucket_start"] == pd.Timedelta(seconds=60)).all()
    assert (panel["spot_available_time"] <= panel["decision_time"]).all()
    assert (panel["decision_time"] < panel["official_window_end"]).all()
    if touch_second is not None:
        assert (panel["decision_time"] < touch_time).all()
    assert panel["pm_bucket_start"].is_unique
    assert np.isclose(panel["contract_row_weight"].sum(), 1)
    return panel, proxy, "", touch_second


def coverage_rows(panel, totals):
    for feature, scope in feature_scopes.items():
        if feature not in panel:
            continue
        key = (feature, scope)
        if key not in totals:
            totals[key] = [0, 0]
        if scope == "btc_eth_only" and panel["asset"].iloc[0] not in ["BTC", "ETH"]:
            continue
        totals[key][0] += len(panel)
        totals[key][1] += int(panel[feature].notna().sum())


def audit_transaction_buckets(registry):
    rows = []
    for number, row in enumerate(registry.itertuples(index=False), start=1):
        path = trade_bucket_folder / f"{row.condition_id}.parquet"
        item = {
            "condition_id": row.condition_id,
            "asset": row.asset,
            "file_exists": path.exists(),
            "bucket_rows": 0,
            "first_bucket": pd.NaT,
            "last_bucket": pd.NaT,
            "buckets_with_volume": 0,
            "total_volume_usd": np.nan,
            "exact_event_timestamp_available": False,
            "token_id_available": False,
            "aggressor_side_available": False,
            "individual_trade_price_available": False,
            "usable_for_exact_60s_fill_lookup": False,
        }
        if path.exists():
            data = pd.read_parquet(path)
            item["bucket_rows"] = len(data)
            if len(data):
                times = pd.to_datetime(data["bucket_ts"], utc=True)
                item["first_bucket"] = times.min()
                item["last_bucket"] = times.max()
                item["buckets_with_volume"] = int(data["volume_usd"].notna().sum())
                item["total_volume_usd"] = float(data["volume_usd"].sum(min_count=1))
        rows.append(item)
        if number % 2000 == 0:
            print(f"audited transaction buckets {number:,}/{len(registry):,}", flush=True)
    return pd.DataFrame(rows)


def build_feature_coverage(totals):
    rows = []
    for (feature, scope), (total, non_null) in totals.items():
        rows.append({
            "feature": feature,
            "scope": scope,
            "eligible_rows": total,
            "non_null_rows": non_null,
            "coverage": non_null / total if total else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["scope", "feature"])


def make_report(registry, manifest, exclusions, feature_coverage, tape_audit):
    total_rows = int(manifest["n_rows"].sum())
    no_signal_rows = int(manifest["no_signal_price_rows"].sum())
    both_signal_rows = int(manifest["both_signal_price_rows"].sum())
    yes_proxy_rows = int(manifest["yes_proxy_rows_60s"].sum())
    no_proxy_rows = int(manifest["no_proxy_rows_60s"].sum())
    same_time_rows = int(manifest["yes_no_same_time_rows_60s"].sum())
    reason_counts = exclusions["exclusion_reason"].value_counts()
    asset_contracts = manifest["asset"].value_counts()
    asset_rows = manifest.groupby("asset")["n_rows"].sum().sort_values(ascending=False)

    def bullets(series):
        if series.empty:
            return "- None"
        return "\n".join(f"- `{name}`: {int(value):,}" for name, value in series.items())

    missing = feature_coverage[feature_coverage["coverage"] < 1]
    missing_lines = "\n".join(
        f"- `{row.feature}` ({row.scope}): {row.coverage:.2%}"
        for row in missing.itertuples(index=False)
    ) or "- None"

    return f"""# Phase 2 leakage and coverage report

## Result

- Frozen Phase-1 contracts: **{len(registry):,}**
- Contracts with causal decision rows: **{len(manifest):,}**
- Contracts without usable decision rows: **{len(exclusions):,}**
- Causal decision rows: **{total_rows:,}**
- Rows with an exact-time observed NO price at the decision: **{no_signal_rows:,} ({no_signal_rows / total_rows:.2%})**
- Rows with exact-time observed prices for both sides: **{both_signal_rows:,} ({both_signal_rows / total_rows:.2%})**
- Rows with a subsequent YES price sample within 60 seconds: **{yes_proxy_rows:,} ({yes_proxy_rows / total_rows:.2%})**
- Rows with a subsequent NO price sample within 60 seconds: **{no_proxy_rows:,} ({no_proxy_rows / total_rows:.2%})**
- Rows with same-time YES and NO proxy samples: **{same_time_rows:,} ({same_time_rows / total_rows:.2%})**

## Timing rules

- Every Polymarket sample is delayed by 60 seconds before it becomes usable.
- No missing Polymarket minute is generated or locally forward-filled.
- Spot features use only candles whose one-minute interval has completed.
- Panels begin after listing and inside the official measurement window.
- Panels stop before the Binance candle containing the first barrier touch.
- Final outcome `y` is retained only as an evaluation label.
- DVOL is delayed by one hour and discarded when more than two hours stale.

## Contract exclusions

{bullets(reason_counts)}

## Usable contracts by asset

{bullets(asset_contracts)}

## Decision rows by asset

{bullets(asset_rows)}

## Features below complete coverage in their declared scope

{missing_lines}

## Execution evidence

- Goldsky aggregate files present: **{int(tape_audit['file_exists'].sum()):,}/{len(tape_audit):,} contracts**
- Files with at least one aggregate bucket: **{int((tape_audit['bucket_rows'] > 0).sum()):,}**
- Files usable for exact 60-second fill lookup: **0**

The Goldsky files contain aggregate five-minute buckets, not raw events. The
CLOB files contain YES- and NO-token price-history samples, not trades, order-book
depth, or executable asks. The separate 60-second lookup is therefore a price
proxy and must never be described as a historical fill.
"""


def make_execution_report(registry, manifest, tape_audit):
    total_rows = int(manifest["n_rows"].sum())
    yes_signal_rows = int(manifest["yes_signal_price_rows"].sum())
    no_signal_rows = int(manifest["no_signal_price_rows"].sum())
    both_signal_rows = int(manifest["both_signal_price_rows"].sum())
    signal_gap_rows = int(manifest["signal_abs_gap_rows"].sum())
    signal_mean_gap = manifest["signal_abs_gap_sum"].sum() / signal_gap_rows
    signal_max_gap = manifest["signal_abs_gap_max"].max()
    signal_gaps_over_1_cent = int(manifest["signal_abs_gap_over_1_cent"].sum())
    signal_gaps_over_2_cents = int(manifest["signal_abs_gap_over_2_cents"].sum())
    yes_proxy_rows = int(manifest["yes_proxy_rows_60s"].sum())
    no_proxy_rows = int(manifest["no_proxy_rows_60s"].sum())
    same_time_rows = int(manifest["yes_no_same_time_rows_60s"].sum())
    gap_rows = int(manifest["yes_no_abs_gap_rows"].sum())
    mean_gap = manifest["yes_no_abs_gap_sum"].sum() / gap_rows if gap_rows else np.nan
    max_gap = manifest["yes_no_abs_gap_max"].max() if gap_rows else np.nan
    gaps_over_1_cent = int(manifest["yes_no_abs_gap_over_1_cent"].sum())
    gaps_over_2_cents = int(manifest["yes_no_abs_gap_over_2_cents"].sum())
    return f"""# Execution proxy limitations

The historical files do not contain a usable event-level, token-level execution tape.

## What is available

- One-minute CLOB price-history paths for both YES and NO tokens.
- Aggregate Goldsky fill buckets for {int((tape_audit['bucket_rows'] > 0).sum()):,} of {len(registry):,} eligible contracts.
- Exact-time YES decision prices for {yes_signal_rows:,} rows ({yes_signal_rows / total_rows:.2%}) and separately observed NO decision prices for {no_signal_rows:,} rows ({no_signal_rows / total_rows:.2%}).
- Exact-time prices for both sides on {both_signal_rows:,} rows ({both_signal_rows / total_rows:.2%}); their mean absolute `YES + NO - 1` gap is {signal_mean_gap:.6f}, with a maximum of {signal_max_gap:.6f}.
- Decision-price complement gaps above 1 cent: {signal_gaps_over_1_cent:,} ({signal_gaps_over_1_cent / signal_gap_rows:.4%}); above 2 cents: {signal_gaps_over_2_cents:,} ({signal_gaps_over_2_cents / signal_gap_rows:.4%}).
- A next-YES-price-sample lookup within 60 seconds for {yes_proxy_rows:,} of {total_rows:,} decision rows ({yes_proxy_rows / total_rows:.2%}).
- A next-NO-price-sample lookup within 60 seconds for {no_proxy_rows:,} of {total_rows:,} decision rows ({no_proxy_rows / total_rows:.2%}).
- Same-time YES and NO samples for {same_time_rows:,} rows ({same_time_rows / total_rows:.2%}).
- Mean absolute `YES + NO - 1` gap on same-time samples: {mean_gap:.6f}; maximum: {max_gap:.6f}.
- Complement gaps above 1 cent: {gaps_over_1_cent:,} ({gaps_over_1_cent / gap_rows:.4%}); above 2 cents: {gaps_over_2_cents:,} ({gaps_over_2_cents / gap_rows:.4%}).

## What is not available

- Historical bid/ask quotes or depth.
- Exact raw fill timestamps in the saved files.
- The original token ID for each aggregated Goldsky fill.
- Maker/taker or buyer-initiated aggressor classification.
- Proof that a hypothetical additional order would have filled.

`execution_proxy_60s/` therefore contains **token-specific price-sample
sensitivities**, not transactions, fills, or executable quotes. Phase 4 may use
the separately observed YES and NO paths for proxy-executed stress tests, but
must retain slippage and partial-fill scenarios. True execution testing still
requires order-book data or the later forward test.
"""


def main():
    registry = pd.read_parquet(registry_file).sort_values(
        ["asset", "official_window_start", "barrier_group_id", "barrier"]
    )
    panel_folder.mkdir(parents=True, exist_ok=True)
    proxy_folder.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    exclusion_rows = []
    feature_totals = {}
    completed = 0

    for asset in assets:
        asset_rows = registry[registry["asset"] == asset]
        spot = load_spot(asset)
        dvol = load_dvol(asset)
        print(f"building {asset}: {len(asset_rows):,} contracts", flush=True)

        for row in asset_rows.itertuples(index=False):
            panel, proxy, reason, touch_second = make_panel(row, spot, dvol)
            if panel is None:
                exclusion_rows.append({
                    "condition_id": row.condition_id,
                    "asset": row.asset,
                    "question": row.question,
                    "barrier_group_id": row.barrier_group_id,
                    "official_window_start": row.official_window_start,
                    "official_window_end": row.official_window_end,
                    "first_touch_time": pd.to_datetime(touch_second, unit="s", utc=True)
                    if touch_second is not None else pd.NaT,
                    "exclusion_reason": reason,
                })
            else:
                write_parquet(panel, panel_folder / f"{row.condition_id}.parquet")
                write_parquet(proxy, proxy_folder / f"{row.condition_id}.parquet")
                coverage_rows(panel, feature_totals)
                manifest_rows.append({
                    "condition_id": row.condition_id,
                    "asset": row.asset,
                    "direction": row.direction,
                    "barrier_group_id": row.barrier_group_id,
                    "n_rows": len(panel),
                    "first_decision": panel["decision_time"].min(),
                    "last_decision": panel["decision_time"].max(),
                    "first_touch_time": panel["first_touch_time"].iloc[0],
                    "yes_signal_price_rows": int(proxy["yes_signal_price_available"].sum()),
                    "no_signal_price_rows": int(proxy["no_signal_price_available"].sum()),
                    "both_signal_price_rows": int(proxy["both_signal_prices_available"].sum()),
                    "signal_abs_gap_rows": int(proxy["signal_yes_no_absolute_complement_gap"].notna().sum()),
                    "signal_abs_gap_sum": float(proxy["signal_yes_no_absolute_complement_gap"].sum()),
                    "signal_abs_gap_max": float(proxy["signal_yes_no_absolute_complement_gap"].max()),
                    "signal_abs_gap_over_1_cent": int((proxy["signal_yes_no_absolute_complement_gap"] > 0.01).sum()),
                    "signal_abs_gap_over_2_cents": int((proxy["signal_yes_no_absolute_complement_gap"] > 0.02).sum()),
                    "yes_proxy_rows_60s": int(proxy["yes_proxy_available_60s"].sum()),
                    "no_proxy_rows_60s": int(proxy["no_proxy_available_60s"].sum()),
                    "yes_no_same_time_rows_60s": int(proxy["proxy_samples_same_time"].sum()),
                    "yes_no_abs_gap_rows": int(proxy["yes_no_absolute_complement_gap"].notna().sum()),
                    "yes_no_abs_gap_sum": float(proxy["yes_no_absolute_complement_gap"].sum()),
                    "yes_no_abs_gap_max": float(proxy["yes_no_absolute_complement_gap"].max()),
                    "yes_no_abs_gap_over_1_cent": int((proxy["yes_no_absolute_complement_gap"] > 0.01).sum()),
                    "yes_no_abs_gap_over_2_cents": int((proxy["yes_no_absolute_complement_gap"] > 0.02).sum()),
                    "dvol_rows": int(panel["dvol_available"].sum()),
                    "max_pm_gap_minutes": float(panel["pm_gap_minutes"].max()),
                })
            completed += 1
            if completed % 1000 == 0:
                print(f"processed {completed:,}/{len(registry):,} contracts", flush=True)

        del spot
        del dvol

    manifest = pd.DataFrame(manifest_rows)
    exclusions = pd.DataFrame(exclusion_rows)
    feature_coverage = build_feature_coverage(feature_totals)
    tape_audit = audit_transaction_buckets(registry)

    assert len(manifest) + len(exclusions) == len(registry)
    assert manifest["condition_id"].is_unique
    assert exclusions["condition_id"].is_unique
    assert manifest["no_proxy_rows_60s"].sum() > 0
    required = feature_coverage[feature_coverage["scope"] == "all_assets"]
    assert (required["coverage"] == 1).all()

    write_parquet(manifest, manifest_file)
    write_parquet(exclusions, excluded_file)
    feature_coverage.to_csv(feature_coverage_file, index=False)
    write_parquet(tape_audit, tape_audit_file)
    report_file.write_text(
        make_report(registry, manifest, exclusions, feature_coverage, tape_audit),
        encoding="utf-8",
    )
    execution_report_file.write_text(
        make_execution_report(registry, manifest, tape_audit), encoding="utf-8"
    )

    print(f"contracts: {len(registry):,}")
    print(f"usable contracts: {len(manifest):,}")
    print(f"decision rows: {int(manifest['n_rows'].sum()):,}")
    print(f"excluded contracts: {len(exclusions):,}")
    print(exclusions["exclusion_reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
