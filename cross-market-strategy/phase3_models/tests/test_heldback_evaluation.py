import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "phase3_evaluation"))
sys.path.insert(0, str(PROJECT / "phase3_5_drift"))
sys.path.insert(0, str(PROJECT / "phase3_models"))

from run_heldback_evaluation import (  # noqa: E402
    EXPERIMENTS,
    PRIMARY_FOLD,
    apply_platt,
)


class HeldbackEvaluationTests(unittest.TestCase):
    def test_primary_fold_is_non_overlapping_final_fold(self):
        self.assertEqual(PRIMARY_FOLD, "fold_04")

    def test_frozen_candidate_set_is_exact(self):
        self.assertEqual(
            set(EXPERIMENTS),
            {
                "market_raw",
                "gbm_platt",
                "empirical_pooled_platt",
                "empirical_asset_platt",
                "har_platt",
                "har_dvol_platt",
            },
        )

    def test_platt_map_is_bounded(self):
        probability = np.array([0.0, 0.2, 0.8, 1.0])
        mapped = apply_platt(probability, {"slope": 1.1, "intercept": -0.2})
        self.assertTrue(np.isfinite(mapped).all())
        self.assertTrue(((mapped > 0) & (mapped < 1)).all())


if __name__ == "__main__":
    unittest.main()
