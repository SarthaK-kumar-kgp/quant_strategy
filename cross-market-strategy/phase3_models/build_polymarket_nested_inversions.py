#!/usr/bin/env python3
"""Build the preserved Polymarket nested-price-inversion research dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_benchmarks import atomic_parquet, file_sha256


FOLDER = Path(__file__).resolve().parent
SOURCE_FILE = FOLDER / "nested_coherence_violations.parquet"
PREDICTION_MANIFEST_FILE = FOLDER / "benchmark_prediction_manifest.parquet"
NESTED_FILE = FOLDER.parent / "phase1_registry" / "nested_barrier_groups.parquet"
OUTPUT_FILE = FOLDER / "polymarket_nested_inversions.parquet"
MANIFEST_FILE = FOLDER / "polymarket_nested_inversions_manifest.parquet"
EXPERIMENT_ID = "E03-PM-RAW-V1"

PREDICTION_COLUMNS = [
    "condition_id",
    "decision_time",
    "information_timestamp",
    "calibrated_yes_probability",
    "contract_row_weight",
]
METADATA_COLUMNS = [
    "condition_id",
    "question",
    "official_window_start",
    "official_window_end",
    "touched",
]


def enrich_cases(
    cases: pd.DataFrame,
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    easier_predictions = predictions[PREDICTION_COLUMNS].rename(
        columns={
            "condition_id": "easier_condition_id",
            "information_timestamp": "easier_information_timestamp",
            "calibrated_yes_probability": "easier_source_yes_price",
            "contract_row_weight": "easier_contract_row_weight",
        }
    )
    harder_predictions = predictions[PREDICTION_COLUMNS].rename(
        columns={
            "condition_id": "harder_condition_id",
            "information_timestamp": "harder_information_timestamp",
            "calibrated_yes_probability": "harder_source_yes_price",
            "contract_row_weight": "harder_contract_row_weight",
        }
    )
    enriched = cases.rename(
        columns={
            "condition_id": "harder_condition_id",
            "easier_probability": "easier_yes_price",
            "calibrated_yes_probability": "harder_yes_price",
            "positive_excess": "gross_yes_price_inversion",
            "easier_rank": "easier_barrier_rank",
            "barrier_rank_easiest_first": "harder_barrier_rank",
        }
    )
    enriched = enriched.merge(
        easier_predictions,
        on=["easier_condition_id", "decision_time"],
        how="left",
        validate="many_to_one",
    ).merge(
        harder_predictions,
        on=["harder_condition_id", "decision_time"],
        how="left",
        validate="many_to_one",
    )
    if enriched[
        ["easier_source_yes_price", "harder_source_yes_price"]
    ].isna().any().any():
        raise ValueError("a stored inversion does not match its source prediction")
    if not (
        np.allclose(
            enriched.easier_yes_price,
            enriched.easier_source_yes_price,
            rtol=0.0,
            atol=1e-15,
        )
        and np.allclose(
            enriched.harder_yes_price,
            enriched.harder_source_yes_price,
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise ValueError("stored inversion prices differ from source predictions")

    easier_metadata = metadata[METADATA_COLUMNS].rename(
        columns={
            "condition_id": "easier_condition_id",
            "question": "easier_question",
            "official_window_start": "easier_window_start",
            "official_window_end": "easier_window_end",
            "touched": "easier_outcome",
        }
    )
    harder_metadata = metadata[METADATA_COLUMNS].rename(
        columns={
            "condition_id": "harder_condition_id",
            "question": "harder_question",
            "official_window_start": "harder_window_start",
            "official_window_end": "harder_window_end",
            "touched": "harder_outcome",
        }
    )
    enriched = enriched.merge(
        easier_metadata, on="easier_condition_id", how="left", validate="many_to_one"
    ).merge(
        harder_metadata, on="harder_condition_id", how="left", validate="many_to_one"
    )
    if not (
        enriched.easier_window_start.eq(enriched.harder_window_start).all()
        and enriched.easier_window_end.eq(enriched.harder_window_end).all()
    ):
        raise ValueError("nested inversion members do not share an official window")
    enriched["source_timestamps_aligned"] = enriched.easier_information_timestamp.eq(
        enriched.harder_information_timestamp
    )
    enriched["easier_source_age_seconds"] = (
        enriched.decision_time - enriched.easier_information_timestamp
    ).dt.total_seconds()
    enriched["harder_source_age_seconds"] = (
        enriched.decision_time - enriched.harder_information_timestamp
    ).dt.total_seconds()
    recomputed = enriched.harder_yes_price - enriched.easier_yes_price
    if not np.allclose(
        recomputed, enriched.gross_yes_price_inversion, rtol=0.0, atol=1e-15
    ):
        raise ValueError("inversion magnitude does not equal harder minus easier YES")
    if (enriched.gross_yes_price_inversion <= 1e-12).any():
        raise ValueError("dataset contains a non-material inversion")
    return enriched.drop(
        columns=["easier_source_yes_price", "harder_source_yes_price"]
    )


def main() -> None:
    cases = pd.read_parquet(SOURCE_FILE)
    if set(cases.experiment_id.unique()) != {EXPERIMENT_ID}:
        raise ValueError("coherence violations contain an unexpected experiment")
    prediction_manifest = pd.read_parquet(PREDICTION_MANIFEST_FILE)
    prediction_manifest = prediction_manifest[
        prediction_manifest.experiment_id == EXPERIMENT_ID
    ]
    metadata = pd.read_parquet(NESTED_FILE, columns=METADATA_COLUMNS)
    if metadata.condition_id.duplicated().any():
        raise ValueError("nested metadata contains duplicate contracts")

    parts = []
    for (fold_id, asset), partition_cases in cases.groupby(
        ["fold_id", "asset"], sort=True
    ):
        manifest_row = prediction_manifest[
            (prediction_manifest.fold_id == fold_id)
            & (prediction_manifest.asset == asset)
        ]
        if len(manifest_row) != 1:
            raise ValueError(f"missing source partition for {fold_id} {asset}")
        manifest_row = manifest_row.iloc[0]
        source_path = FOLDER / manifest_row.path
        if file_sha256(source_path) != manifest_row.sha256:
            raise ValueError(f"source checksum mismatch: {source_path}")
        predictions = pd.read_parquet(source_path, columns=PREDICTION_COLUMNS)
        parts.append(enrich_cases(partition_cases, predictions, metadata))
        print(
            f"preserved {fold_id} {asset}: {len(partition_cases):,} inversions",
            flush=True,
        )
    inversions = pd.concat(parts, ignore_index=True).sort_values(
        [
            "decision_time",
            "barrier_group_id",
            "easier_barrier_rank",
            "harder_barrier_rank",
            "fold_id",
        ],
        ignore_index=True,
    )
    natural_key = [
        "fold_id",
        "barrier_group_id",
        "decision_time",
        "easier_condition_id",
        "harder_condition_id",
    ]
    if inversions.duplicated(natural_key).any():
        raise ValueError("duplicate inversion natural keys")
    atomic_parquet(inversions, OUTPUT_FILE)
    manifest = pd.DataFrame(
        [
            {
                "dataset_version": "polymarket_nested_inversions_v1",
                "source_experiment_id": EXPERIMENT_ID,
                "source_audit": SOURCE_FILE.name,
                "path": OUTPUT_FILE.name,
                "rows": len(inversions),
                "folds": "|".join(sorted(inversions.fold_id.unique())),
                "assets": "|".join(sorted(inversions.asset.unique())),
                "directions": "|".join(sorted(inversions.direction.unique())),
                "first_decision": inversions.decision_time.min(),
                "last_decision": inversions.decision_time.max(),
                "aligned_source_timestamp_rows": int(
                    inversions.source_timestamps_aligned.sum()
                ),
                "minimum_inversion": float(inversions.gross_yes_price_inversion.min()),
                "median_inversion": float(
                    inversions.gross_yes_price_inversion.median()
                ),
                "maximum_inversion": float(inversions.gross_yes_price_inversion.max()),
                "bytes": OUTPUT_FILE.stat().st_size,
                "sha256": file_sha256(OUTPUT_FILE),
            }
        ]
    )
    atomic_parquet(manifest, MANIFEST_FILE)
    print(f"dataset: {OUTPUT_FILE}")
    print(f"manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
