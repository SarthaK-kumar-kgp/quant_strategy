import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "phase3_5_drift"))
sys.path.insert(0, str(PROJECT / "phase3_models"))

from fit_drift_models import (  # noqa: E402
    completed_daily_returns,
    drift_touch_probability_from_variance,
    estimate_drift,
    gbm_touch_probability_drift,
    huber_location,
)
from run_benchmarks import MINUTES_PER_YEAR, gbm_touch_probability  # noqa: E402
from run_har import har_touch_probability  # noqa: E402


class DriftModelTests(unittest.TestCase):
    def test_incomplete_utc_day_is_excluded(self):
        spot = pd.DataFrame(
            {
                "ts": [86340, 86400 + 86340, 2 * 86400],
                "close": [100.0, 101.0, 999.0],
            }
        )
        returns, days, availability = completed_daily_returns(spot, 3 * 86400)
        self.assertTrue(np.allclose(returns, [0.01]))
        self.assertTrue(np.array_equal(days, [1]))
        self.assertTrue(np.array_equal(availability, [2 * 86400]))

    def test_constant_sample_uses_mean_fallback(self):
        location, scale, iterations, fallback = huber_location(np.full(365, 0.001))
        self.assertAlmostEqual(location, 0.001)
        self.assertEqual(scale, 0)
        self.assertEqual(iterations, 0)
        self.assertTrue(fallback)

    def test_noisy_zero_signal_shrinks_to_zero(self):
        returns = np.concatenate([np.tile(np.array([-0.01, 0.01]), 182), [0.0]])
        result = estimate_drift(returns)
        self.assertEqual(result["shrinkage_factor"], 0)
        self.assertEqual(result["final_annual_drift"], 0)

    def test_zero_drift_gbm_matches_frozen_benchmark(self):
        spot = np.array([100.0, 100.0])
        barrier = np.array([105.0, 95.0])
        minutes = np.array([720.0, 720.0])
        sigma = np.array([0.8, 0.8])
        direction = np.array(["up", "down"])
        expected = gbm_touch_probability(spot, barrier, minutes, sigma, direction)
        actual = gbm_touch_probability_drift(
            spot, barrier, minutes, sigma, direction, 0.0
        )
        self.assertTrue(np.allclose(actual, expected, rtol=1e-12, atol=1e-12))

    def test_zero_drift_har_matches_frozen_har(self):
        distance = np.array([0.05, 0.05])
        variance = np.array([0.001, 0.001])
        minutes = np.array([720.0, 720.0])
        direction = np.array(["up", "down"])
        expected = har_touch_probability(distance, variance, direction)
        actual = drift_touch_probability_from_variance(
            distance, variance, minutes, direction, 0.0
        )
        self.assertTrue(np.allclose(actual, expected, rtol=1e-12, atol=1e-12))

    def test_positive_drift_favours_up_barrier(self):
        distance = np.array([0.05, 0.05])
        variance = np.array([0.001, 0.001])
        minutes = np.array([1440.0, 1440.0])
        direction = np.array(["up", "down"])
        zero = drift_touch_probability_from_variance(
            distance, variance, minutes, direction, 0.0
        )
        positive = drift_touch_probability_from_variance(
            distance, variance, minutes, direction, 1.0
        )
        self.assertGreater(positive[0], zero[0])
        self.assertLess(positive[1], zero[1])

    def test_integrated_variance_time_is_in_years(self):
        sigma = 0.7
        minutes = np.array([1440.0])
        variance = np.array([sigma**2 * minutes[0] / MINUTES_PER_YEAR])
        expected = gbm_touch_probability_drift(
            np.array([100.0]),
            np.array([105.0]),
            minutes,
            np.array([sigma]),
            np.array(["up"]),
            0.25,
        )
        actual = drift_touch_probability_from_variance(
            np.array([np.log(1.05)]),
            variance,
            minutes,
            np.array(["up"]),
            0.25,
        )
        self.assertTrue(np.allclose(actual, expected, rtol=1e-12, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
