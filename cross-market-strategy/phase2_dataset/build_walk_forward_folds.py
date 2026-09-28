from datetime import date
from pathlib import Path

import pandas as pd


folder = Path(__file__).resolve().parent
registry_file = folder.parent / "phase1_registry" / "eligible_market_registry.parquet"
panel_manifest_file = folder / "panel_manifest.parquet"
output_file = folder / "walk_forward_folds.parquet"
report_file = folder / "walk_forward_folds_report.md"

folds = [
    {
        "fold_id": "fold_01",
        "train_end": date(2026, 4, 30),
        "validation_embargo": date(2026, 5, 1),
        "validation_start": date(2026, 5, 2),
        "validation_end": date(2026, 5, 31),
        "evaluation_embargo": date(2026, 6, 1),
        "evaluation_start": date(2026, 6, 2),
        "evaluation_end": date(2026, 6, 30),
    },
    {
        "fold_id": "fold_02",
        "train_end": date(2026, 5, 31),
        "validation_embargo": date(2026, 6, 1),
        "validation_start": date(2026, 6, 2),
        "validation_end": date(2026, 6, 30),
        "evaluation_embargo": date(2026, 7, 1),
        "evaluation_start": date(2026, 7, 2),
        "evaluation_end": date(2026, 7, 31),
    },
    {
        "fold_id": "fold_03",
        "train_end": date(2026, 6, 30),
        "validation_embargo": date(2026, 7, 1),
        "validation_start": date(2026, 7, 2),
        "validation_end": date(2026, 7, 31),
        "evaluation_embargo": date(2026, 8, 1),
        "evaluation_start": date(2026, 8, 2),
        "evaluation_end": date(2026, 8, 31),
    },
    {
        "fold_id": "fold_04",
        "train_end": date(2026, 7, 31),
        "validation_embargo": date(2026, 8, 1),
        "validation_start": date(2026, 8, 2),
        "validation_end": date(2026, 8, 31),
        "evaluation_embargo": date(2026, 9, 1),
        "evaluation_start": date(2026, 9, 2),
        "evaluation_end": date(2026, 9, 9),
    },
]


def role_for_day(day, fold):
    if day <= fold["train_end"]:
        return "train"
    if day == fold["validation_embargo"]:
        return "embargo_before_validation"
    if fold["validation_start"] <= day <= fold["validation_end"]:
        return "validation"
    if day == fold["evaluation_embargo"]:
        return "embargo_before_evaluation"
    if fold["evaluation_start"] <= day <= fold["evaluation_end"]:
        return "evaluation"
    return ""


def make_report(manifest):
    lines = [
        "# Frozen expanding walk-forward folds",
        "",
        "Contracts are assigned by the final included minute of their official",
        "measurement window in America/New_York. Every fold has a one-calendar-day",
        "embargo before validation and evaluation. Whole contracts and whole",
        "nested-barrier groups remain in one role within a fold.",
        "",
    ]
    for fold in folds:
        fold_rows = manifest[manifest["fold_id"] == fold["fold_id"]]
        lines.extend([
            f"## {fold['fold_id']}",
            "",
            f"- Train through: `{fold['train_end']}`",
            f"- Validation: `{fold['validation_start']}` to `{fold['validation_end']}`",
            f"- Evaluation: `{fold['evaluation_start']}` to `{fold['evaluation_end']}`",
            "",
        ])
        counts = fold_rows["role"].value_counts()
        for role in ["train", "embargo_before_validation", "validation",
                     "embargo_before_evaluation", "evaluation"]:
            rows = fold_rows[fold_rows["role"] == role]
            lines.append(
                f"- `{role}`: {int(counts.get(role, 0)):,} contracts, "
                f"{int(rows['n_rows'].sum()):,} decision rows"
            )
        lines.append("")
    lines.extend([
        "## Important interpretation",
        "",
        "These are predeclared rolling research folds, not a newly untouched final",
        "test. Earlier versions of this project already examined parts of the same",
        "historical period. Live forward testing remains necessary after a candidate",
        "is frozen.",
        "",
    ])
    return "\n".join(lines)


def main():
    registry = pd.read_parquet(registry_file)
    panels = pd.read_parquet(panel_manifest_file)
    data = registry.merge(
        panels[["condition_id", "n_rows", "first_decision", "last_decision"]],
        on="condition_id",
        how="inner",
        validate="one_to_one",
    )
    final_minute = data["official_window_end"] - pd.Timedelta(minutes=1)
    data["resolution_date_et"] = final_minute.dt.tz_convert("America/New_York").dt.date

    rows = []
    for fold in folds:
        for row in data.itertuples(index=False):
            role = role_for_day(row.resolution_date_et, fold)
            if not role:
                continue
            rows.append({
                "fold_id": fold["fold_id"],
                "condition_id": row.condition_id,
                "asset": row.asset,
                "direction": row.direction,
                "barrier_group_id": row.barrier_group_id,
                "resolution_date_et": row.resolution_date_et,
                "role": role,
                "n_rows": int(row.n_rows),
                "first_decision": row.first_decision,
                "last_decision": row.last_decision,
                "official_window_start": row.official_window_start,
                "official_window_end": row.official_window_end,
            })

    manifest = pd.DataFrame(rows)
    duplicate_roles = manifest.groupby(
        ["fold_id", "barrier_group_id"]
    )["role"].nunique()
    assert (duplicate_roles == 1).all()
    assert not manifest.duplicated(["fold_id", "condition_id"]).any()
    for fold in folds:
        evaluation = manifest[
            (manifest["fold_id"] == fold["fold_id"])
            & (manifest["role"] == "evaluation")
        ]
        validation = manifest[
            (manifest["fold_id"] == fold["fold_id"])
            & (manifest["role"] == "validation")
        ]
        assert len(validation) > 0
        assert len(evaluation) > 0

    manifest.to_parquet(output_file, index=False, compression="zstd")
    report_file.write_text(make_report(manifest), encoding="utf-8")
    print(f"usable contracts: {len(data):,}")
    print(f"fold membership rows: {len(manifest):,}")
    print(manifest.groupby(["fold_id", "role"]).size().to_string())


if __name__ == "__main__":
    main()
