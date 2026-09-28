import sys
from pathlib import Path
import unittest

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_nested_coherence import audit_partition  # noqa: E402


class NestedCoherenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.metadata = pd.DataFrame(
            {
                "condition_id": ["easy", "middle", "hard"],
                "barrier_group_id": ["g1", "g1", "g1"],
                "barrier_group_size": [3, 3, 3],
                "barrier_rank_easiest_first": [1, 2, 3],
                "asset": ["BTC", "BTC", "BTC"],
                "direction": ["down", "down", "down"],
                "barrier": [100.0, 90.0, 80.0],
            }
        )

    def frame(self, probabilities):
        return pd.DataFrame(
            {
                "condition_id": ["easy", "middle", "hard"],
                "decision_time": pd.to_datetime(
                    ["2026-05-01T00:00:00Z"] * 3, utc=True
                ),
                "fold_id": ["fold_01"] * 3,
                "calibrated_yes_probability": probabilities,
            }
        )

    def test_coherent_order_has_no_violation(self):
        metrics, violations = audit_partition(
            self.frame([0.8, 0.5, 0.2]), self.metadata
        )
        self.assertEqual(metrics["adjacent_pairs"], 2)
        self.assertEqual(metrics["violations"], 0)
        self.assertTrue(violations.empty)

    def test_harder_barrier_with_higher_probability_is_detected(self):
        metrics, violations = audit_partition(
            self.frame([0.8, 0.4, 0.6]), self.metadata
        )
        self.assertEqual(metrics["violations"], 1)
        self.assertAlmostEqual(metrics["maximum_positive_excess"], 0.2)
        self.assertEqual(violations.iloc[0].condition_id, "hard")
        self.assertEqual(violations.iloc[0].easier_condition_id, "middle")

    def test_missing_intermediate_rank_compares_available_neighbors(self):
        predictions = self.frame([0.7, 0.6, 0.8]).iloc[[0, 2]].copy()
        metrics, _ = audit_partition(predictions, self.metadata)
        self.assertEqual(metrics["adjacent_pairs"], 1)
        self.assertEqual(metrics["violations"], 1)

    def test_tolerance_ignores_roundoff(self):
        metrics, _ = audit_partition(
            self.frame([0.5, 0.5 + 5e-13, 0.4]), self.metadata, tolerance=1e-12
        )
        self.assertEqual(metrics["violations"], 0)


if __name__ == "__main__":
    unittest.main()
