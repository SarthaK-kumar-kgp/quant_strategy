import sys
from pathlib import Path
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_dvol_ablation import (  # noqa: E402
    DVOL_DELAY_SECONDS,
    DVOL_MAX_AGE_SECONDS,
    causal_dvol,
)


class DvolAblationTests(unittest.TestCase):
    def test_source_is_unavailable_before_one_hour_delay(self):
        source = np.array([1000], dtype=np.int64)
        sigma = np.array([0.5])
        selected, available, _ = causal_dvol(
            np.array([1000 + DVOL_DELAY_SECONDS - 1]), source, sigma
        )
        self.assertTrue(np.isnan(selected[0]))
        self.assertEqual(available[0], -1)

    def test_source_becomes_available_at_exact_delay(self):
        source = np.array([1000], dtype=np.int64)
        sigma = np.array([0.5])
        selected, available, age = causal_dvol(
            np.array([1000 + DVOL_DELAY_SECONDS]), source, sigma
        )
        self.assertEqual(selected[0], 0.5)
        self.assertEqual(available[0], 1000 + DVOL_DELAY_SECONDS)
        self.assertEqual(age[0], 0)

    def test_source_is_discarded_after_maximum_age(self):
        source = np.array([1000], dtype=np.int64)
        sigma = np.array([0.5])
        decisions = np.array(
            [
                1000 + DVOL_DELAY_SECONDS + DVOL_MAX_AGE_SECONDS,
                1000 + DVOL_DELAY_SECONDS + DVOL_MAX_AGE_SECONDS + 1,
            ]
        )
        selected, _, _ = causal_dvol(decisions, source, sigma)
        self.assertEqual(selected[0], 0.5)
        self.assertTrue(np.isnan(selected[1]))

    def test_latest_available_observation_is_used(self):
        source = np.array([1000, 4600], dtype=np.int64)
        sigma = np.array([0.5, 0.7])
        selected, available, _ = causal_dvol(
            np.array([4600 + DVOL_DELAY_SECONDS]), source, sigma
        )
        self.assertEqual(selected[0], 0.7)
        self.assertEqual(available[0], 4600 + DVOL_DELAY_SECONDS)


if __name__ == "__main__":
    unittest.main()
