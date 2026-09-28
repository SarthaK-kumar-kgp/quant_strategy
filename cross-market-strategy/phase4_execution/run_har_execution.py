#!/usr/bin/env python3
"""Run the frozen Phase-4 execution proxy for HAR/range + Platt."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parent
PHASE2 = PROJECT / "phase2_dataset"
PHASE3 = PROJECT / "phase3_models"
PHASE3_EVALUATION = PROJECT / "phase3_evaluation"

MODEL_EXPERIMENT_ID = "E06-HAR-PLATT-V1"
MODEL_LABEL = "HAR/range + Platt"
PROTOCOL_FILE_NAME = "PHASE4_PROTOCOL.md"
MODEL_ASSETS = ("BTC", "ETH", "SOL", "XRP")
THRESHOLDS = (0.02, 0.05, 0.10, 0.15)
FILL_FRACTIONS = (0.10, 0.25, 0.50, 1.00)
INTENDED_BUDGET = 100.0
SLIPPAGE_RATE = 0.01
FEE_RATE = 0.07

REGISTRY_FILE = PROJECT / "phase1_registry" / "eligible_market_registry.parquet"
EXECUTION_DIR = PHASE2 / "execution_proxy_60s"
VALIDATION_DIR = PHASE3 / "har_predictions" / MODEL_EXPERIMENT_ID
EVALUATION_DIR = PHASE3_EVALUATION / "predictions" / MODEL_EXPERIMENT_ID

LEDGER_FILE = FOLDER / "har_attempt_ledger.parquet"
SUMMARY_FILE = FOLDER / "har_execution_summary.parquet"
BREAKDOWN_FILE = FOLDER / "har_full_cost_breakdown.parquet"
REPORT_FILE = FOLDER / "har_execution_report.md"

PREDICTION_COLUMNS = [
    "experiment_id",
    "condition_id",
    "decision_time",
    "information_timestamp",
    "fold_id",
    "role",
    "calibrated_yes_probability",
]

EXECUTION_COLUMNS = [
    "condition_id",
    "decision_time",
    "yes_signal_price_available",
    "yes_signal_price",
    "no_signal_price_available",
    "no_signal_price",
    "yes_proxy_available_60s",
    "yes_proxy_available_at",
    "yes_proxy_price",
    "yes_proxy_delay_seconds",
    "no_proxy_available_60s",
    "no_proxy_available_at",
    "no_proxy_price",
    "no_proxy_delay_seconds",
    "proxy_is_transaction",
    "proxy_is_executable_quote",
]

REGISTRY_COLUMNS = [
    "condition_id",
    "asset",
    "direction",
    "barrier",
    "barrier_group_id",
    "official_window_end",
    "outcome_recomputed",
]

STAGES = (
    ("gross_signal", "gross_fill", "gross_pnl_100"),
    ("delayed", "delayed_fill", "delayed_pnl_100"),
    ("slippage", "slippage_fill", "slippage_pnl_100"),
    ("full_cost", "full_fill", "full_pnl_100"),
)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def valid_price(price: pd.Series) -> pd.Series:
    return price.notna() & (price > 0) & (price < 1)


def token_pnl_100(
    token_won: pd.Series,
    all_in_price: pd.Series,
    filled: pd.Series,
) -> pd.Series:
    result = INTENDED_BUDGET * (
        token_won.astype(float) / all_in_price - 1.0
    )
    return result.where(filled, np.nan)


def build_attempts(rows: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Choose one first qualifying attempt per contract for one threshold."""

    frame = rows.copy()
    probability = frame["calibrated_yes_probability"].astype(float)
    yes_ok = frame["yes_signal_price_available"].fillna(False) & valid_price(
        frame["yes_signal_price"]
    )
    no_ok = frame["no_signal_price_available"].fillna(False) & valid_price(
        frame["no_signal_price"]
    )
    yes_edge = (probability - frame["yes_signal_price"]).where(yes_ok, -np.inf)
    no_edge = (
        (1.0 - probability) - frame["no_signal_price"]
    ).where(no_ok, -np.inf)

    frame["chosen_side"] = np.where(
        yes_edge > no_edge,
        "YES",
        np.where(no_edge > yes_edge, "NO", ""),
    )
    frame["gross_edge"] = np.maximum(yes_edge, no_edge)
    candidates = frame[
        (frame["chosen_side"] != "") & (frame["gross_edge"] >= threshold)
    ].copy()
    if candidates.empty:
        return candidates

    candidates = candidates.sort_values(
        ["condition_id", "decision_time"], kind="mergesort"
    ).drop_duplicates("condition_id", keep="first")
    is_yes = candidates["chosen_side"].eq("YES")
    candidates["threshold"] = float(threshold)
    candidates["fair_probability"] = np.where(
        is_yes,
        candidates["calibrated_yes_probability"],
        1.0 - candidates["calibrated_yes_probability"],
    )
    candidates["signal_price"] = np.where(
        is_yes,
        candidates["yes_signal_price"],
        candidates["no_signal_price"],
    )
    candidates["proxy_available"] = np.where(
        is_yes,
        candidates["yes_proxy_available_60s"].fillna(False),
        candidates["no_proxy_available_60s"].fillna(False),
    ).astype(bool)
    candidates["proxy_available_at"] = candidates["yes_proxy_available_at"].where(
        is_yes, candidates["no_proxy_available_at"]
    )
    candidates["proxy_price"] = np.where(
        is_yes,
        candidates["yes_proxy_price"],
        candidates["no_proxy_price"],
    )
    candidates["proxy_delay_seconds"] = np.where(
        is_yes,
        candidates["yes_proxy_delay_seconds"],
        candidates["no_proxy_delay_seconds"],
    )

    candidates["token_won"] = np.where(
        is_yes,
        candidates["y"].astype(bool),
        ~candidates["y"].astype(bool),
    )
    candidates["minutes_to_expiry"] = (
        candidates["official_window_end"] - candidates["decision_time"]
    ).dt.total_seconds() / 60.0

    proxy_valid = candidates["proxy_available"] & valid_price(
        candidates["proxy_price"]
    )
    candidates["delayed_edge"] = (
        candidates["fair_probability"] - candidates["proxy_price"]
    ).where(proxy_valid, np.nan)
    candidates["delayed_fill"] = proxy_valid & (
        candidates["delayed_edge"] >= threshold
    )

    candidates["slippage_price"] = candidates["proxy_price"] * (
        1.0 + SLIPPAGE_RATE
    )
    slippage_valid = candidates["proxy_available"] & valid_price(
        candidates["slippage_price"]
    )
    candidates["slippage_edge"] = (
        candidates["fair_probability"] - candidates["slippage_price"]
    ).where(slippage_valid, np.nan)
    candidates["slippage_fill"] = slippage_valid & (
        candidates["slippage_edge"] >= threshold
    )

    candidates["fee_per_share"] = (
        FEE_RATE
        * candidates["slippage_price"]
        * (1.0 - candidates["slippage_price"])
    ).where(slippage_valid, np.nan)
    candidates["all_in_price"] = (
        candidates["slippage_price"] + candidates["fee_per_share"]
    )
    candidates["net_edge"] = (
        candidates["fair_probability"] - candidates["all_in_price"]
    ).where(slippage_valid, np.nan)
    candidates["full_fill"] = slippage_valid & (
        candidates["net_edge"] >= threshold
    )

    candidates["gross_fill"] = True
    candidates["gross_pnl_100"] = token_pnl_100(
        candidates["token_won"], candidates["signal_price"], candidates["gross_fill"]
    )
    candidates["delayed_pnl_100"] = token_pnl_100(
        candidates["token_won"], candidates["proxy_price"], candidates["delayed_fill"]
    )
    candidates["slippage_pnl_100"] = token_pnl_100(
        candidates["token_won"],
        candidates["slippage_price"],
        candidates["slippage_fill"],
    )
    candidates["full_pnl_100"] = token_pnl_100(
        candidates["token_won"], candidates["all_in_price"], candidates["full_fill"]
    )
    candidates["full_shares_100"] = (
        INTENDED_BUDGET / candidates["all_in_price"]
    ).where(candidates["full_fill"], np.nan)

    candidates["full_status"] = np.select(
        [
            ~candidates["proxy_available"],
            candidates["proxy_available"] & ~valid_price(candidates["proxy_price"]),
            proxy_valid & ~slippage_valid,
            slippage_valid & ~candidates["full_fill"],
            candidates["full_fill"],
        ],
        [
            "no_proxy_sample",
            "invalid_proxy_price",
            "invalid_slippage_price",
            "post_cost_edge_below_threshold",
            "filled",
        ],
        default="invalid_state",
    )
    candidates["proxy_is_transaction"] = candidates[
        "proxy_is_transaction"
    ].fillna(False)
    candidates["proxy_is_executable_quote"] = candidates[
        "proxy_is_executable_quote"
    ].fillna(False)
    return candidates


def load_registry() -> pd.DataFrame:
    registry = pd.read_parquet(REGISTRY_FILE, columns=REGISTRY_COLUMNS)
    if registry["condition_id"].duplicated().any():
        raise ValueError("eligible registry contains duplicate condition IDs")
    registry = registry.rename(columns={"outcome_recomputed": "y"})
    registry["y"] = registry["y"].astype(bool)
    return registry


def load_execution_rows(condition_ids: list[str]) -> pd.DataFrame:
    pieces = []
    for condition_id in condition_ids:
        path = EXECUTION_DIR / f"{condition_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        pieces.append(pd.read_parquet(path, columns=EXECUTION_COLUMNS))
    if not pieces:
        return pd.DataFrame(columns=EXECUTION_COLUMNS)
    execution = pd.concat(pieces, ignore_index=True)
    if execution.duplicated(["condition_id", "decision_time"]).any():
        raise ValueError("execution proxy keys are not unique")
    return execution


def prediction_files(source: Path, role: str) -> list[Path]:
    files = sorted(source.glob(f"fold_*/{role}/*.parquet"))
    expected = {
        (f"fold_{fold:02d}", asset)
        for fold in range(1, 5)
        for asset in MODEL_ASSETS
    }
    observed = {(path.parents[1].name, path.stem) for path in files}
    if observed != expected:
        raise ValueError(
            f"unexpected {role} prediction coverage: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return files


def process_prediction_file(
    path: Path,
    role: str,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    prediction = pd.read_parquet(path, columns=PREDICTION_COLUMNS)
    if prediction.empty:
        raise ValueError(f"empty prediction file: {path}")
    if prediction["experiment_id"].nunique() != 1 or (
        prediction["experiment_id"].iloc[0] != MODEL_EXPERIMENT_ID
    ):
        raise ValueError(f"wrong experiment in {path}")
    if prediction["role"].nunique() != 1 or prediction["role"].iloc[0] != role:
        raise ValueError(f"wrong role in {path}")
    if not prediction["decision_time"].equals(prediction["information_timestamp"]):
        raise ValueError(f"non-causal information timestamp in {path}")
    if prediction.duplicated(["condition_id", "decision_time"]).any():
        raise ValueError(f"duplicate prediction key in {path}")
    if not prediction["calibrated_yes_probability"].between(0, 1).all():
        raise ValueError(f"invalid probability in {path}")

    condition_ids = prediction["condition_id"].drop_duplicates().tolist()
    execution = load_execution_rows(condition_ids)
    merged = prediction.merge(
        execution,
        on=["condition_id", "decision_time"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        missing = int((merged["_merge"] != "both").sum())
        raise ValueError(f"{missing} prediction rows lack execution rows in {path}")
    merged = merged.drop(columns=["_merge", "information_timestamp"])
    merged = merged.merge(
        registry,
        on="condition_id",
        how="left",
        validate="many_to_one",
    )
    if merged["asset"].isna().any():
        raise ValueError(f"prediction conditions missing from registry in {path}")
    if not merged["asset"].eq(path.stem).all():
        raise ValueError(f"asset mismatch in {path}")

    attempts = [build_attempts(merged, threshold) for threshold in THRESHOLDS]
    result = pd.concat(attempts, ignore_index=True)
    print(
        f"processed {path.parents[1].name} {role} {path.stem}: "
        f"{len(prediction):,} rows, {len(result):,} attempts",
        flush=True,
    )
    return result


def build_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    group_specs = [
        (["role", "fold_id", "threshold"], False),
        (["role", "threshold"], True),
    ]
    for group_columns, combined in group_specs:
        for key, group in ledger.groupby(group_columns, observed=True, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            base = dict(zip(group_columns, key))
            if combined:
                base["fold_id"] = "combined"
            for stage, fill_column, pnl_column in STAGES:
                filled = group[fill_column].astype(bool)
                pnl_100 = group.loc[filled, pnl_column]
                for fill_fraction in FILL_FRACTIONS:
                    deployed = float(filled.sum() * INTENDED_BUDGET * fill_fraction)
                    pnl = float(pnl_100.sum() * fill_fraction)
                    records.append(
                        {
                            **base,
                            "cost_stage": stage,
                            "fill_fraction": fill_fraction,
                            "attempts": int(len(group)),
                            "fills": int(filled.sum()),
                            "cancelled_or_unavailable": int((~filled).sum()),
                            "wins": int(group.loc[filled, "token_won"].sum()),
                            "losses": int(filled.sum() - group.loc[filled, "token_won"].sum()),
                            "deployed_capital": deployed,
                            "net_pnl": pnl,
                            "return_on_deployed": pnl / deployed if deployed else np.nan,
                            "win_rate": (
                                float(group.loc[filled, "token_won"].mean())
                                if filled.any()
                                else np.nan
                            ),
                        }
                    )
    summary = pd.DataFrame.from_records(records)
    return summary.sort_values(
        ["role", "fold_id", "threshold", "cost_stage", "fill_fraction"],
        ignore_index=True,
    )


def build_breakdown(ledger: pd.DataFrame) -> pd.DataFrame:
    records = []
    group_columns = ["role", "fold_id", "threshold", "asset", "chosen_side"]
    for key, group in ledger.groupby(group_columns, observed=True, sort=True):
        filled = group["full_fill"].astype(bool)
        pnl_100 = group.loc[filled, "full_pnl_100"]
        deployed = float(filled.sum() * INTENDED_BUDGET * 0.25)
        pnl = float(pnl_100.sum() * 0.25)
        records.append(
            {
                **dict(zip(group_columns, key)),
                "fill_fraction": 0.25,
                "attempts": int(len(group)),
                "fills": int(filled.sum()),
                "wins": int(group.loc[filled, "token_won"].sum()),
                "deployed_capital": deployed,
                "net_pnl": pnl,
                "return_on_deployed": pnl / deployed if deployed else np.nan,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        group_columns, ignore_index=True
    )


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    labels = {
        "threshold": "Threshold",
        "fold_id": "Fold",
        "cost_stage": "Cost stage",
        "attempts": "Attempts",
        "fills": "Fills",
        "net_pnl": "P&L ($)",
        "return_on_deployed": "Return/deployed",
        "win_rate": "Win rate",
    }
    header = [labels.get(column, column) for column in columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column == "threshold":
                values.append(f"{float(value):.0%}")
            elif column in {"return_on_deployed", "win_rate"}:
                values.append("—" if pd.isna(value) else f"{float(value):.2%}")
            elif column == "net_pnl":
                values.append(f"{float(value):,.2f}")
            elif column in {"attempts", "fills"}:
                values.append(f"{int(value):,}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(ledger: pd.DataFrame, summary: pd.DataFrame) -> None:
    central = summary[
        (summary["fill_fraction"] == 0.25)
        & (summary["cost_stage"] == "full_cost")
    ]
    validation = central[
        (central["role"] == "validation") & (central["fold_id"] == "combined")
    ].sort_values("threshold")
    evaluation = central[
        (central["role"] == "evaluation") & (central["fold_id"] == "combined")
    ].sort_values("threshold")
    september = central[
        (central["role"] == "evaluation") & (central["fold_id"] == "fold_04")
    ].sort_values("threshold")

    validation_folds = central[
        (central["role"] == "validation") & (central["fold_id"] != "combined")
    ].sort_values(["threshold", "fold_id"])

    stage = summary[
        (summary["role"] == "validation")
        & (summary["fold_id"] == "combined")
        & (summary["fill_fraction"] == 0.25)
    ].copy()
    stage["stage_order"] = stage["cost_stage"].map(
        {name: order for order, (name, _, _) in enumerate(STAGES)}
    )
    stage = stage.sort_values(["threshold", "stage_order"])

    validation_full = ledger[(ledger["role"] == "validation") & ledger["full_fill"]]
    side_rows = []
    for key, group in validation_full.groupby(
        ["threshold", "chosen_side"], observed=True, sort=True
    ):
        threshold, side = key
        deployed = len(group) * INTENDED_BUDGET * 0.25
        pnl = group["full_pnl_100"].sum() * 0.25
        side_rows.append(
            {
                "threshold": threshold,
                "chosen_side": side,
                "fills": len(group),
                "net_pnl": pnl,
                "return_on_deployed": pnl / deployed,
                "win_rate": group["token_won"].mean(),
            }
        )
    side_summary = pd.DataFrame(side_rows).sort_values(
        ["threshold", "chosen_side"]
    )

    lines = [
        f"# {MODEL_LABEL} execution-proxy report",
        "",
        f"Generated under the frozen `{PROTOCOL_FILE_NAME}`. Results are historical",
        "price-sample sensitivities, not verified fills or an investable portfolio.",
        "",
        "## Full-cost validation result",
        "",
        "The table uses the central 25% fill assumption ($25 per filled trade).",
        "Return on deployed capital is invariant to the deterministic fill fraction.",
        "",
        *markdown_table(
            validation,
            ["threshold", "attempts", "fills", "net_pnl", "return_on_deployed", "win_rate"],
        ),
        "",
        "### Validation stability by fold",
        "",
        *markdown_table(
            validation_folds,
            ["threshold", "fold_id", "fills", "net_pnl", "return_on_deployed"],
        ),
        "",
        "### Validation result by purchased side",
        "",
        *markdown_table(
            side_summary,
            ["threshold", "chosen_side", "fills", "net_pnl", "return_on_deployed", "win_rate"],
        ),
        "",
        "NO purchases generate most of the result. Several YES sub-strategies are",
        "weak or negative, so pooled profitability must not be interpreted as",
        "uniform evidence across sides.",
        "",
        "## Full-cost supporting evaluation result",
        "",
        *markdown_table(
            evaluation,
            ["threshold", "attempts", "fills", "net_pnl", "return_on_deployed", "win_rate"],
        ),
        "",
        "Evaluation folds 1--3 overlap later validation months and are supporting",
        "walk-forward evidence rather than a globally untouched test.",
        "",
        "## Fold-4 September result",
        "",
        *markdown_table(
            september,
            ["threshold", "attempts", "fills", "net_pnl", "return_on_deployed", "win_rate"],
        ),
        "",
        "Fold 4 covers 2--9 September 2026. Its execution proxy was isolated until",
        "this run, but its probability outcomes were already reported in Phase 3.5.",
        "",
        "## Validation cost attrition",
        "",
        "Each row applies that stage's own threshold recheck. Therefore later stages",
        "both add costs and reject attempts; the table is an eligibility-and-cost",
        "funnel, not a matched-trade attribution of fee dollars alone.",
        "",
        *markdown_table(
            stage,
            ["threshold", "cost_stage", "attempts", "fills", "net_pnl", "return_on_deployed"],
        ),
        "",
        "## Status counts",
        "",
    ]
    status = (
        ledger.groupby(["role", "threshold", "full_status"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    lines.extend(
        [
            "| Role | Threshold | Status | Count |",
            "|---|---:|---|---:|",
            *[
                f"| {row.role} | {row.threshold:.0%} | {row.full_status} | {row.count:,} |"
                for row in status.itertuples(index=False)
            ],
            "",
            "## Interpretation boundary",
            "",
            "The historical proxy fields explicitly state that they are neither",
            "transactions nor executable quotes, every filled attempt here uses a",
            "60-second sample, and no depth is available. Large model-market gaps",
            "can therefore reflect stale or non-executable price samples. The positive",
            "numbers are a signal to continue testing, not demonstrated realizable",
            "profit.",
            "",
            "No threshold or model is selected from this report alone. Phase 5 must",
            "apply chronological bankroll, cash-lock, dependence, and drawdown rules",
            "before any historical candidate can be shortlisted for forward paper trading.",
        ]
    )
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    registry = load_registry()
    ledgers = []
    for role, source in (
        ("validation", VALIDATION_DIR),
        ("evaluation", EVALUATION_DIR),
    ):
        for path in prediction_files(source, role):
            ledgers.append(process_prediction_file(path, role, registry))
    ledger = pd.concat(ledgers, ignore_index=True)
    if ledger.empty:
        raise ValueError("HAR execution run produced no attempts")
    if ledger.duplicated(["fold_id", "role", "condition_id", "threshold"]).any():
        raise ValueError("more than one attempt per fold-role-contract-threshold")
    if ledger["proxy_is_transaction"].any() or ledger["proxy_is_executable_quote"].any():
        raise ValueError("execution proxy unexpectedly claims executable evidence")

    output_columns = [
        "experiment_id",
        "fold_id",
        "role",
        "condition_id",
        "asset",
        "direction",
        "barrier",
        "barrier_group_id",
        "official_window_end",
        "decision_time",
        "minutes_to_expiry",
        "threshold",
        "chosen_side",
        "calibrated_yes_probability",
        "fair_probability",
        "signal_price",
        "gross_edge",
        "proxy_available",
        "proxy_available_at",
        "proxy_delay_seconds",
        "proxy_price",
        "delayed_edge",
        "slippage_price",
        "slippage_edge",
        "fee_per_share",
        "all_in_price",
        "net_edge",
        "y",
        "token_won",
        "gross_fill",
        "delayed_fill",
        "slippage_fill",
        "full_fill",
        "full_status",
        "gross_pnl_100",
        "delayed_pnl_100",
        "slippage_pnl_100",
        "full_shares_100",
        "full_pnl_100",
        "proxy_is_transaction",
        "proxy_is_executable_quote",
    ]
    ledger = ledger[output_columns].sort_values(
        ["role", "fold_id", "threshold", "decision_time", "condition_id"],
        ignore_index=True,
    )
    summary = build_summary(ledger)
    breakdown = build_breakdown(ledger)
    atomic_parquet(ledger, LEDGER_FILE)
    atomic_parquet(summary, SUMMARY_FILE)
    atomic_parquet(breakdown, BREAKDOWN_FILE)
    write_report(ledger, summary)
    print(f"wrote {LEDGER_FILE}: {len(ledger):,} attempts", flush=True)
    print(f"wrote {SUMMARY_FILE}: {len(summary):,} rows", flush=True)
    print(f"wrote {BREAKDOWN_FILE}: {len(breakdown):,} rows", flush=True)
    print(f"wrote {REPORT_FILE}", flush=True)


if __name__ == "__main__":
    main()
