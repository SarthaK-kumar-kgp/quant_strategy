import sys
from pathlib import Path
import unittest

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_polymarket_nested_inversions import enrich_cases  # noqa: E402


class PolymarketNestedInversionTests(unittest.TestCase):
    def test_enrichment_preserves_prices_and_source_times(self):
        decision = pd.Timestamp("2026-05-01T00:00:00Z")
        cases = pd.DataFrame(
            {
                "experiment_id": ["E03-PM-RAW-V1"],
                "fold_id": ["fold_01"],
                "barrier_group_id": ["g1"],
                "decision_time": [decision],
                "asset": ["BTC"],
                "direction": ["up"],
                "easier_condition_id": ["easy"],
                "condition_id": ["hard"],
                "easier_rank": [1],
                "barrier_rank_easiest_first": [2],
                "easier_barrier": [100.0],
                "barrier": [110.0],
                "easier_probability": [0.2],
                "calibrated_yes_probability": [0.3],
                "positive_excess": [0.1],
            }
        )
        predictions = pd.DataFrame(
            {
                "condition_id": ["easy", "hard"],
                "decision_time": [decision, decision],
                "information_timestamp": [decision, decision],
                "calibrated_yes_probability": [0.2, 0.3],
                "contract_row_weight": [0.1, 0.1],
            }
        )
        metadata = pd.DataFrame(
            {
                "condition_id": ["easy", "hard"],
                "question": ["easy?", "hard?"],
                "official_window_start": [decision, decision],
                "official_window_end": [decision + pd.Timedelta(days=1)] * 2,
                "touched": [1, 0],
            }
        )
        enriched = enrich_cases(cases, predictions, metadata)
        self.assertEqual(len(enriched), 1)
        self.assertAlmostEqual(enriched.iloc[0].gross_yes_price_inversion, 0.1)
        self.assertTrue(enriched.iloc[0].source_timestamps_aligned)
        self.assertEqual(enriched.iloc[0].easier_source_age_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
