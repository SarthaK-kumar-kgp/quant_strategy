from pathlib import Path

import numpy as np
import pandas as pd

from build_causal_panel import (
    load_proxy_prices,
    make_execution_proxy,
    make_execution_report,
    make_report,
    write_parquet,
)


folder = Path(__file__).resolve().parent
project = folder.parent.parent

registry_file = folder.parent / "phase1_registry" / "eligible_market_registry.parquet"
panel_folder = folder / "causal_panel_1m"
yes_price_folder = project / "data" / "pm_prices_clob_1m"
no_price_folder = project / "data" / "pm_prices_clob_1m_no"

manifest_file = folder / "panel_manifest.parquet"
exclusions_file = folder / "panel_exclusions.parquet"
coverage_file = folder / "feature_coverage.csv"
tape_audit_file = folder / "transaction_tape_audit.parquet"

staging_folder = folder / "execution_proxy_60s_staging"
staging_manifest = folder / "panel_manifest_staging.parquet"
staging_audit = folder / "execution_proxy_audit_staging.parquet"
staging_leakage_report = folder / "leakage_and_coverage_report_staging.md"
staging_execution_report = folder / "execution_proxy_limitations_staging.md"


def validate_side(proxy, side):
    available = proxy[f"{side}_proxy_available_60s"]
    rows = proxy.loc[available]
    if rows.empty:
        return
    assert rows[f"{side}_proxy_price"].between(0, 1).all()
    assert (rows[f"{side}_proxy_available_at"] > rows["decision_time"]).all()
    assert (rows[f"{side}_proxy_available_at"] <= rows["lookup_window_end"]).all()
    assert (
        rows[f"{side}_proxy_available_at"] - rows[f"{side}_proxy_bucket_start"]
        == pd.Timedelta(seconds=60)
    ).all()


def validate_signal_price(proxy, side):
    available = proxy[f"{side}_signal_price_available"]
    rows = proxy.loc[available]
    if rows.empty:
        return
    assert rows[f"{side}_signal_price"].between(0, 1).all()
    assert (rows[f"{side}_signal_available_at"] == rows["decision_time"]).all()
    assert (
        rows[f"{side}_signal_available_at"] - rows[f"{side}_signal_bucket_start"]
        == pd.Timedelta(seconds=60)
    ).all()


def audit_row(row, proxy, yes_prices, no_prices):
    gaps = proxy["yes_no_absolute_complement_gap"].dropna()
    signal_gaps = proxy["signal_yes_no_absolute_complement_gap"].dropna()
    return {
        "condition_id": row.condition_id,
        "asset": row.asset,
        "direction": row.direction,
        "decision_rows": len(proxy),
        "yes_source_points": len(yes_prices),
        "no_source_points": len(no_prices),
        "yes_signal_price_rows": int(proxy["yes_signal_price_available"].sum()),
        "no_signal_price_rows": int(proxy["no_signal_price_available"].sum()),
        "both_signal_price_rows": int(proxy["both_signal_prices_available"].sum()),
        "signal_abs_gap_rows": len(signal_gaps),
        "signal_abs_gap_sum": float(signal_gaps.sum()),
        "signal_abs_gap_mean": float(signal_gaps.mean()) if len(signal_gaps) else np.nan,
        "signal_abs_gap_median": float(signal_gaps.median()) if len(signal_gaps) else np.nan,
        "signal_abs_gap_p95": float(signal_gaps.quantile(0.95)) if len(signal_gaps) else np.nan,
        "signal_abs_gap_max": float(signal_gaps.max()) if len(signal_gaps) else np.nan,
        "signal_abs_gap_over_1_cent": int((signal_gaps > 0.01).sum()),
        "signal_abs_gap_over_2_cents": int((signal_gaps > 0.02).sum()),
        "yes_proxy_rows_60s": int(proxy["yes_proxy_available_60s"].sum()),
        "no_proxy_rows_60s": int(proxy["no_proxy_available_60s"].sum()),
        "both_proxy_rows_60s": int(proxy["both_proxy_available_60s"].sum()),
        "yes_no_same_time_rows_60s": int(proxy["proxy_samples_same_time"].sum()),
        "yes_no_abs_gap_rows": len(gaps),
        "yes_no_abs_gap_sum": float(gaps.sum()),
        "yes_no_abs_gap_mean": float(gaps.mean()) if len(gaps) else np.nan,
        "yes_no_abs_gap_median": float(gaps.median()) if len(gaps) else np.nan,
        "yes_no_abs_gap_p95": float(gaps.quantile(0.95)) if len(gaps) else np.nan,
        "yes_no_abs_gap_max": float(gaps.max()) if len(gaps) else np.nan,
        "yes_no_abs_gap_over_1_cent": int((gaps > 0.01).sum()),
        "yes_no_abs_gap_over_2_cents": int((gaps > 0.02).sum()),
    }


def main():
    if staging_folder.exists():
        raise RuntimeError(f"Remove or rename the old staging folder first: {staging_folder}")

    registry = pd.read_parquet(registry_file).set_index("condition_id", drop=False)
    manifest = pd.read_parquet(manifest_file)
    staging_folder.mkdir(parents=True)
    audits = []

    for number, condition_id in enumerate(manifest["condition_id"], start=1):
        row = registry.loc[condition_id]
        panel = pd.read_parquet(
            panel_folder / f"{condition_id}.parquet",
            columns=["decision_time"],
        )
        decision_times = pd.to_datetime(panel["decision_time"], utc=True)
        decision_seconds = (
            decision_times.astype("int64") // 10**9
        ).to_numpy(dtype=np.int64)

        yes_prices = load_proxy_prices(yes_price_folder / f"{condition_id}.parquet")
        no_prices = load_proxy_prices(no_price_folder / f"{condition_id}.parquet")
        if yes_prices.empty or no_prices.empty:
            raise RuntimeError(f"Missing usable YES or NO history for {condition_id}")

        proxy = make_execution_proxy(
            row, decision_times, decision_seconds, yes_prices, no_prices
        )
        assert len(proxy) == len(panel)
        assert proxy["decision_time"].equals(decision_times.reset_index(drop=True))
        assert proxy["yes_token_id"].eq(str(row.yes_token_id)).all()
        assert proxy["no_token_id"].eq(str(row.no_token_id)).all()
        assert not proxy["proxy_is_transaction"].any()
        assert not proxy["proxy_is_executable_quote"].any()
        validate_side(proxy, "yes")
        validate_side(proxy, "no")
        validate_signal_price(proxy, "yes")
        validate_signal_price(proxy, "no")

        write_parquet(proxy, staging_folder / f"{condition_id}.parquet")
        audits.append(audit_row(row, proxy, yes_prices, no_prices))

        if number % 500 == 0 or number == len(manifest):
            print(f"built {number:,}/{len(manifest):,} proxy files", flush=True)

    audit = pd.DataFrame(audits)
    assert len(audit) == len(manifest)
    assert audit["condition_id"].is_unique
    assert audit["decision_rows"].sum() == manifest["n_rows"].sum()
    assert audit["no_proxy_rows_60s"].sum() > 0

    update_columns = [
        "condition_id",
        "yes_signal_price_rows",
        "no_signal_price_rows",
        "both_signal_price_rows",
        "signal_abs_gap_rows",
        "signal_abs_gap_sum",
        "signal_abs_gap_max",
        "signal_abs_gap_over_1_cent",
        "signal_abs_gap_over_2_cents",
        "yes_proxy_rows_60s",
        "no_proxy_rows_60s",
        "yes_no_same_time_rows_60s",
        "yes_no_abs_gap_rows",
        "yes_no_abs_gap_sum",
        "yes_no_abs_gap_max",
        "yes_no_abs_gap_over_1_cent",
        "yes_no_abs_gap_over_2_cents",
    ]
    old_columns = [name for name in update_columns if name != "condition_id"]
    updated_manifest = manifest.drop(columns=old_columns, errors="ignore").merge(
        audit[update_columns], on="condition_id", how="left", validate="one_to_one"
    )

    exclusions = pd.read_parquet(exclusions_file)
    coverage = pd.read_csv(coverage_file)
    tape_audit = pd.read_parquet(tape_audit_file)
    write_parquet(audit, staging_audit)
    write_parquet(updated_manifest, staging_manifest)
    staging_leakage_report.write_text(
        make_report(registry.reset_index(drop=True), updated_manifest, exclusions, coverage, tape_audit),
        encoding="utf-8",
    )
    staging_execution_report.write_text(
        make_execution_report(registry.reset_index(drop=True), updated_manifest, tape_audit),
        encoding="utf-8",
    )

    total_rows = int(audit["decision_rows"].sum())
    no_signal_rows = int(audit["no_signal_price_rows"].sum())
    yes_rows = int(audit["yes_proxy_rows_60s"].sum())
    no_rows = int(audit["no_proxy_rows_60s"].sum())
    same_rows = int(audit["yes_no_same_time_rows_60s"].sum())
    gap_rows = int(audit["yes_no_abs_gap_rows"].sum())
    print(f"decision rows: {total_rows:,}")
    print(f"NO signal-price rows: {no_signal_rows:,} ({no_signal_rows / total_rows:.2%})")
    print(f"YES proxy rows: {yes_rows:,} ({yes_rows / total_rows:.2%})")
    print(f"NO proxy rows: {no_rows:,} ({no_rows / total_rows:.2%})")
    print(f"same-time YES/NO rows: {same_rows:,} ({same_rows / total_rows:.2%})")
    print(f"mean absolute complement gap: {audit['yes_no_abs_gap_sum'].sum() / gap_rows:.6f}")
    print(f"maximum absolute complement gap: {audit['yes_no_abs_gap_max'].max():.6f}")
    print(f"staged proxy -> {staging_folder}")


if __name__ == "__main__":
    main()
