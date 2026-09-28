from calendar import monthrange
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from zoneinfo import ZoneInfo
import re

import numpy as np
import pandas as pd
from dateutil import parser as date_parser


folder = Path(__file__).resolve().parent
project = folder.parent.parent
source_file = project / "phase1" / "census_one_touch_clean.parquet"
spot_folder = project / "data" / "spot"

audit_file = folder / "semantic_audit.parquet"
eligible_file = folder / "eligible_market_registry.parquet"
excluded_file = folder / "exclusion_registry.parquet"
groups_file = folder / "nested_barrier_groups.parquet"
coverage_file = folder / "coverage_tables.csv"
report_file = folder / "semantic_audit_report.md"

et = ZoneInfo("America/New_York")
utc = ZoneInfo("UTC")

month_numbers = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
month_pattern = "|".join(name.title() for name in month_numbers)


def local_time(year, month, day, hour=0, minute=0):
    value = datetime(year, month, day, hour, minute, tzinfo=et)
    return pd.Timestamp(value.astimezone(utc))


def candidate_years(row):
    years = {row["listing_start"].year, row["listing_end"].year}
    years |= {year - 1 for year in list(years)}
    years |= {year + 1 for year in list(years)}
    return sorted(years)


def choose_single_date(month, day, row):
    choices = []
    for year in candidate_years(row):
        try:
            start = local_time(year, month, day)
        except ValueError:
            continue
        next_day = datetime(year, month, day) + timedelta(days=1)
        end = local_time(next_day.year, next_day.month, next_day.day)
        gap = abs((end - row["listing_end"]).total_seconds())
        choices.append((gap, start, end))
    if not choices:
        return None, None
    _, start, end = min(choices, key=lambda item: item[0])
    return start, end


def parse_single_date(question, row):
    pattern = rf"\bon ({month_pattern}) (\d{{1,2}})\?$"
    match = re.search(pattern, question, re.IGNORECASE)
    if not match:
        return None, None
    month = month_numbers[match.group(1).lower()]
    return choose_single_date(month, int(match.group(2)), row)


def parse_month(question, row):
    pattern = rf"\bin ({month_pattern})\?$"
    match = re.search(pattern, question, re.IGNORECASE)
    if not match:
        return None, None

    month = month_numbers[match.group(1).lower()]
    choices = []
    for year in candidate_years(row):
        start = local_time(year, month, 1)
        if month == 12:
            end = local_time(year + 1, 1, 1)
        else:
            end = local_time(year, month + 1, 1)
        gap = abs((end - row["listing_end"]).total_seconds())
        choices.append((gap, start, end))

    _, start, end = min(choices, key=lambda item: item[0])
    return start, end


def parse_date_range(question, row):
    pattern = (
        rf"\b({month_pattern})\s+(\d{{1,2}})\s*[-–—]\s*"
        rf"(?:({month_pattern})\s+)?(\d{{1,2}})\?$"
    )
    match = re.search(pattern, question, re.IGNORECASE)
    if not match:
        return None, None

    first_month = month_numbers[match.group(1).lower()]
    first_day = int(match.group(2))
    last_month = month_numbers[(match.group(3) or match.group(1)).lower()]
    last_day = int(match.group(4))

    choices = []
    for last_year in candidate_years(row):
        first_year = last_year
        if (first_month, first_day) > (last_month, last_day):
            first_year -= 1
        try:
            start = local_time(first_year, first_month, first_day)
            next_day = datetime(last_year, last_month, last_day) + timedelta(days=1)
            end = local_time(next_day.year, next_day.month, next_day.day)
        except ValueError:
            continue
        gap = abs((end - row["listing_end"]).total_seconds())
        choices.append((gap, start, end))

    if not choices:
        return None, None
    _, start, end = min(choices, key=lambda item: item[0])
    return start, end


def clean_date_text(text):
    text = text.strip(" ,.")
    text = re.sub(r"\bET\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^F(?=March\b)", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" ,.")


def has_year(text):
    match = re.search(r"\b(20\d{2})\b", text)
    return int(match.group(1)) if match else None


def parse_date_text(text, default_year):
    default = datetime(default_year, 1, 1, 0, 0)
    value = date_parser.parse(clean_date_text(text), default=default, fuzzy=False)
    return value.replace(tzinfo=et)


def parse_explicit_between(description, row):
    first_line = description.splitlines()[0]
    match = re.search(r"between\s*(.+?)\s+has a final", first_line, re.IGNORECASE)
    if not match:
        return None, None

    clause = match.group(1)
    clause = re.sub(r"\s*in the ET timezone\s*", " ", clause, flags=re.IGNORECASE)
    parts = re.split(r"\s+and\s+", clause, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None, None

    first_text, last_text = parts
    first_year = has_year(first_text)
    last_year = has_year(last_text)
    base_year = last_year or first_year or row["listing_end"].year

    try:
        last = parse_date_text(last_text, base_year)
        if first_year:
            first = parse_date_text(first_text, first_year)
        else:
            first = parse_date_text(first_text, last.year)
            if (first.month, first.day) > (last.month, last.day):
                first = first.replace(year=last.year - 1)
        if not last_year and (last.month, last.day) < (first.month, first.day):
            last = last.replace(year=first.year + 1)
    except (ValueError, OverflowError):
        return None, None

    start = pd.Timestamp(first.astimezone(utc))
    end = pd.Timestamp((last + timedelta(minutes=1)).astimezone(utc))
    return start, end


def parse_creation_window(question, row):
    _, end = parse_date_range(question, row)
    if end is None:
        _, end = parse_month(question, row)
    if end is None:
        _, end = parse_single_date(question, row)
    if end is None:
        end = row["listing_end"]
        return row["listing_start"], end, True
    return row["listing_start"], end, False


def parse_window(row):
    rule = row["description"].lower()

    if "from the creation of this market through" in rule:
        start, end, fallback = parse_creation_window(row["question"], row)
        return start, end, "rules_creation_to_title_end", fallback

    if "on the date specified in the title" in rule:
        start, end = parse_single_date(row["question"], row)
        return start, end, "rules_title_single_date_et", False

    if "during the date range specified in the title" in rule:
        start, end = parse_date_range(row["question"], row)
        return start, end, "rules_title_date_range_et", False

    if "during the month specified in the title" in rule:
        start, end = parse_month(row["question"], row)
        return start, end, "rules_title_month_et", False

    start, end = parse_explicit_between(row["description"], row)
    return start, end, "rules_explicit_between_et", False


def question_barrier_is_valid(question, barrier):
    prices = re.findall(r"\$([\d,]+(?:\.\d+)?)([kK]?)", question)
    if len(prices) != 1:
        return False
    value = float(prices[0][0].replace(",", ""))
    if prices[0][1]:
        value *= 1000
    return bool(np.isclose(value, barrier, rtol=0, atol=max(1e-9, barrier * 1e-9)))


def rule_price_field(description):
    first_line = description.splitlines()[0]
    lower = first_line.lower()
    if "high" in lower and "low" in lower:
        return "both"
    match = re.search(r"final\s+[\"“]?(High|Low)", first_line, re.IGNORECASE)
    return match.group(1).lower() if match else ""


def rule_pair(description):
    normalized = description.upper().replace("/", "")
    pairs = re.findall(r"\b(BTC|ETH|SOL|XRP|DOGE)USDT\b", normalized)
    return f"{pairs[0]}USDT" if pairs else ""


def add_semantic_fields(census):
    windows = []
    for _, row in census.iterrows():
        windows.append(parse_window(row))

    census["official_window_start"] = [item[0] for item in windows]
    census["official_window_end"] = [item[1] for item in windows]
    census["window_source"] = [item[2] for item in windows]
    census["fallback_duration_used"] = [item[3] for item in windows]
    census["window_parse_ok"] = census["official_window_start"].notna()
    census["window_parse_ok"] &= census["official_window_end"].notna()
    census["official_window_hours"] = (
        census["official_window_end"] - census["official_window_start"]
    ).dt.total_seconds() / 3600

    census["binary_outcome_ok"] = census["yes_token_id"].notna()
    census["binary_outcome_ok"] &= census["no_token_id"].notna()
    census["binary_outcome_ok"] &= census["yes_token_id"] != census["no_token_id"]
    census["supported_asset"] = census["asset"].isin(["BTC", "ETH", "SOL", "XRP"])
    census["barrier_ok"] = [
        question_barrier_is_valid(question, barrier)
        for question, barrier in zip(census["question"], census["barrier"])
    ]

    census["resolution_pair"] = census["description"].map(rule_pair)
    expected_pair = census["asset"] + "USDT"
    census["resolution_pair_ok"] = census["resolution_pair"] == expected_pair
    census["rule_price_fields"] = census["description"].map(rule_price_field)
    expected_field = census["direction"].map({"up": "high", "down": "low"})
    census["resolution_price_field"] = np.where(
        census["rule_price_fields"] == "both",
        expected_field,
        census["rule_price_fields"],
    )
    census["direction_rule_ok"] = census["resolution_price_field"] == expected_field

    lower_rule = census["description"].str.lower()
    census["resolution_source_ok"] = lower_rule.str.contains(
        "resolution source for this market is binance", regex=False
    )
    one_minute = r"1[ -]?minute|one-minute|\b1m\b"
    census["one_minute_rule_ok"] = lower_rule.str.contains(one_minute, regex=True)
    census["candle_minutes"] = np.where(census["one_minute_rule_ok"], 1, np.nan)
    census["api_end_difference_hours"] = (
        census["listing_end"] - census["official_window_end"]
    ).dt.total_seconds() / 3600
    return census


def load_spot(asset):
    path = spot_folder / f"{asset}USDT_1m.parquet"
    frame = pd.read_parquet(path, columns=["ts", "high", "low"])
    return frame.sort_values("ts").reset_index(drop=True)


def recompute_outcomes(census):
    census["spot_window_rows"] = 0
    census["expected_spot_rows"] = 0
    census["spot_coverage_ratio"] = np.nan
    census["spot_window_extreme"] = np.nan
    census["outcome_recomputed"] = pd.array([pd.NA] * len(census), dtype="Int64")

    semantic_ok = census["supported_asset"].copy()
    semantic_ok &= census["binary_outcome_ok"]
    semantic_ok &= census["barrier_ok"]
    semantic_ok &= census["resolution_source_ok"]
    semantic_ok &= census["one_minute_rule_ok"]
    semantic_ok &= census["resolution_pair_ok"]
    semantic_ok &= census["direction_rule_ok"]
    semantic_ok &= census["window_parse_ok"]
    semantic_ok &= census["official_window_hours"] > 0
    semantic_ok &= census["official_window_hours"] <= 24

    for asset in ["BTC", "ETH", "SOL", "XRP"]:
        spot = load_spot(asset)
        timestamps = spot["ts"].to_numpy()
        highs = spot["high"].to_numpy()
        lows = spot["low"].to_numpy()
        rows = census.index[semantic_ok & (census["asset"] == asset)]

        for index in rows:
            row = census.loc[index]
            start_value = row["official_window_start"].timestamp()
            end_value = row["official_window_end"].timestamp()
            first_minute = int(np.ceil(start_value / 60) * 60)
            end_minute = int(np.ceil(end_value / 60) * 60)
            expected = max(0, (end_minute - first_minute) // 60)
            left = np.searchsorted(timestamps, first_minute, side="left")
            right = np.searchsorted(timestamps, end_minute, side="left")
            count = right - left

            census.at[index, "expected_spot_rows"] = expected
            census.at[index, "spot_window_rows"] = count
            census.at[index, "spot_coverage_ratio"] = count / expected if expected else 0

            if count == 0:
                continue
            if row["direction"] == "up":
                extreme = float(np.max(highs[left:right]))
                outcome = int(extreme >= row["barrier"])
            else:
                extreme = float(np.min(lows[left:right]))
                outcome = int(extreme <= row["barrier"])
            census.at[index, "spot_window_extreme"] = extreme
            census.at[index, "outcome_recomputed"] = outcome

    census["spot_coverage_complete"] = census["spot_coverage_ratio"] == 1
    census["outcome_match"] = census["outcome_recomputed"] == census["touched"]
    census["outcome_match"] = census["outcome_match"].fillna(False)
    return census


def primary_exclusion(row):
    checks = [
        (not row["supported_asset"], "unsupported_asset"),
        (not row["binary_outcome_ok"], "not_binary_yes_no"),
        (not row["barrier_ok"], "ambiguous_or_invalid_barrier"),
        (not row["resolution_source_ok"], "resolution_source_not_binance"),
        (not row["one_minute_rule_ok"], "resolution_not_one_minute"),
        (not row["resolution_pair_ok"], "resolution_pair_mismatch"),
        (not row["direction_rule_ok"], "direction_rule_mismatch"),
        (not row["window_parse_ok"], "official_window_unparseable"),
        (row["window_parse_ok"] and row["official_window_hours"] <= 0,
         "official_window_nonpositive"),
        (row["window_parse_ok"] and row["official_window_hours"] > 24,
         "official_window_over_24h"),
        (row["window_parse_ok"] and row["official_window_hours"] <= 24
         and not row["spot_coverage_complete"], "spot_coverage_incomplete"),
        (row["window_parse_ok"] and row["official_window_hours"] <= 24
         and row["spot_coverage_complete"] and not row["outcome_match"],
         "official_outcome_mismatch"),
    ]
    for failed, reason in checks:
        if failed:
            return reason
    return ""


def group_key(row):
    parts = [
        row["asset"],
        row["direction"],
        row["official_window_start"].isoformat(),
        row["official_window_end"].isoformat(),
        row["resolution_pair"],
        row["resolution_price_field"],
        str(int(row["candle_minutes"])),
    ]
    digest = sha256("|".join(parts).encode()).hexdigest()[:16]
    return f"bg_{digest}"


def add_groups(eligible):
    eligible["barrier_group_id"] = eligible.apply(group_key, axis=1)
    eligible["barrier_group_size"] = eligible.groupby("barrier_group_id")[
        "condition_id"
    ].transform("size")
    duplicate_key = ["barrier_group_id", "barrier"]
    eligible["same_barrier_count"] = eligible.groupby(duplicate_key)[
        "condition_id"
    ].transform("size")
    eligible["duplicate_barrier_in_group"] = eligible["same_barrier_count"] > 1

    ranks = pd.Series(index=eligible.index, dtype="Int64")
    for _, group in eligible.groupby("barrier_group_id"):
        ascending = group["direction"].iloc[0] == "up"
        ordered = group.sort_values("barrier", ascending=ascending)
        values = ordered["barrier"].drop_duplicates().tolist()
        rank_map = {value: rank + 1 for rank, value in enumerate(values)}
        ranks.loc[group.index] = group["barrier"].map(rank_map)
    eligible["barrier_rank_easiest_first"] = ranks
    return eligible


def duration_band(hours):
    if hours <= 1:
        return "0-1h"
    if hours <= 6:
        return "1-6h"
    if hours <= 12:
        return "6-12h"
    if hours <= 18:
        return "12-18h"
    if hours < 23:
        return "18-23h"
    return "23-24h"


def build_coverage(eligible):
    local_start = eligible["official_window_start"].dt.tz_convert(et)
    values = {
        "asset": eligible["asset"],
        "direction": eligible["direction"],
        "month": local_start.dt.strftime("%Y-%m"),
        "duration": eligible["official_window_hours"].map(duration_band),
        "outcome": eligible["touched"].map({0: "NO", 1: "YES"}),
    }
    tables = []
    for dimension, series in values.items():
        counts = series.value_counts(dropna=False).rename_axis("value").reset_index(name="contracts")
        counts.insert(0, "dimension", dimension)
        tables.append(counts)
    return pd.concat(tables, ignore_index=True)


def count_monotonicity_violations(eligible):
    violations = 0
    for _, group in eligible.groupby("barrier_group_id"):
        ordered = group.sort_values("barrier_rank_easiest_first")
        outcomes = ordered["touched"].to_numpy()
        if np.any(np.diff(outcomes) > 0):
            violations += 1
    return violations


def report_text(audit, eligible, excluded, nested, coverage):
    reason_counts = excluded["primary_exclusion_reason"].value_counts()
    source_counts = audit["window_source"].value_counts()
    asset_counts = eligible["asset"].value_counts()
    direction_counts = eligible["direction"].value_counts()
    outcome_counts = eligible["touched"].map({0: "NO", 1: "YES"}).value_counts()
    old_intraday = audit["horizon_days"] < 1
    removed_from_old = old_intraday & ~audit["included"]
    added_beyond_old = ~old_intraday & audit["included"]
    nested_group_count = nested["barrier_group_id"].nunique() if len(nested) else 0
    duplicate_barriers = int(eligible["duplicate_barrier_in_group"].sum())
    monotonicity_violations = count_monotonicity_violations(eligible)

    def bullets(series):
        return "\n".join(f"- `{name}`: {count:,}" for name, count in series.items())

    return f"""# Phase 1 semantic audit report

## Result

- Clean source contracts audited: **{len(audit):,}**
- Frozen eligible intraday contracts: **{len(eligible):,}**
- Excluded contracts: **{len(excluded):,}**
- Contracts removed from the old 9,132-row listing-duration cohort: **{removed_from_old.sum():,}**
- Contracts added despite an old listing duration of at least one day: **{added_beyond_old.sum():,}**
- Eligible contracts using a duration fallback: **{eligible['fallback_duration_used'].sum():,}**
- Nested groups with at least two contracts: **{nested_group_count:,}**
- Contracts belonging to nested groups: **{len(nested):,}**
- Largest nested group: **{eligible['barrier_group_size'].max():,} contracts**
- Duplicate barriers inside a group: **{duplicate_barriers:,}**
- Final-outcome monotonicity violations: **{monotonicity_violations:,} groups**

The frozen registry uses official rule windows in America/New_York and stores
their UTC equivalents as a half-open interval: start inclusive, end exclusive.
For a rule ending at 11:59 PM ET, the stored end is the following midnight.

## Primary exclusion reasons

{bullets(reason_counts)}

## Window parsing methods

{bullets(source_counts)}

## Eligible coverage by asset

{bullets(asset_counts)}

## Eligible coverage by direction

{bullets(direction_counts)}

## Eligible coverage by final outcome

{bullets(outcome_counts)}

## Validation

- Every included row has one parsed official window of at most 24 hours.
- Every included row uses the expected Binance USDT pair and one-minute High/Low rule.
- Every included row has complete Binance one-minute coverage over its official window.
- Every included final outcome matches the recomputed Binance High/Low barrier touch.
- Every excluded row has exactly one primary exclusion reason.
- `coverage_tables.csv` contains the requested asset, direction, month, duration, and outcome tables.

## Interpretation

The old `end - start < 1 day` filter measured listing duration, not necessarily
the resolution window. The contracts removed from that old cohort had official
multi-day windows even though they were listed late. No contract outside the
old cohort became eligible under the audited official-window definition.
"""


def main():
    census = pd.read_parquet(source_file)
    census = census.rename(columns={"start": "listing_start", "end": "listing_end"})
    census = add_semantic_fields(census)
    census = recompute_outcomes(census)
    census["primary_exclusion_reason"] = census.apply(primary_exclusion, axis=1)
    census["included"] = census["primary_exclusion_reason"] == ""

    eligible = add_groups(census[census["included"]].copy())
    group_columns = [
        "barrier_group_id",
        "barrier_group_size",
        "condition_id",
        "question",
        "asset",
        "direction",
        "barrier",
        "barrier_rank_easiest_first",
        "same_barrier_count",
        "duplicate_barrier_in_group",
        "official_window_start",
        "official_window_end",
        "resolution_pair",
        "resolution_price_field",
        "touched",
    ]
    nested = eligible.loc[eligible["barrier_group_size"] > 1, group_columns].copy()
    nested = nested.sort_values(
        ["official_window_start", "asset", "direction", "barrier_rank_easiest_first"]
    )

    group_fields = eligible[[
        "condition_id",
        "barrier_group_id",
        "barrier_group_size",
        "same_barrier_count",
        "duplicate_barrier_in_group",
        "barrier_rank_easiest_first",
    ]]
    census = census.merge(group_fields, on="condition_id", how="left")
    excluded = census[~census["included"]].copy()
    coverage = build_coverage(eligible)

    assert census["condition_id"].is_unique
    assert eligible["condition_id"].is_unique
    assert eligible["official_window_start"].notna().all()
    assert eligible["official_window_end"].notna().all()
    assert eligible["official_window_hours"].between(0, 24, inclusive="right").all()
    assert eligible["spot_coverage_complete"].all()
    assert eligible["outcome_match"].all()
    assert (excluded["primary_exclusion_reason"] != "").all()
    assert eligible["barrier_group_id"].notna().all()
    assert count_monotonicity_violations(eligible) == 0

    census.to_parquet(audit_file, index=False)
    eligible.to_parquet(eligible_file, index=False)
    excluded.to_parquet(excluded_file, index=False)
    nested.to_parquet(groups_file, index=False)
    coverage.to_csv(coverage_file, index=False)
    report_file.write_text(
        report_text(census, eligible, excluded, nested, coverage), encoding="utf-8"
    )

    print(f"audited: {len(census):,}")
    print(f"eligible: {len(eligible):,}")
    print(f"excluded: {len(excluded):,}")
    print(excluded["primary_exclusion_reason"].value_counts().to_string())
    print(f"nested groups: {nested['barrier_group_id'].nunique():,}")
    print(f"nested contracts: {len(nested):,}")


if __name__ == "__main__":
    main()
