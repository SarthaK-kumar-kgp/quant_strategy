import sys
from pathlib import Path
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_empirical import (  # noqa: E402
    HORIZONS,
    empirical_touch_probability,
    future_extreme,
    smoothed_survival,
)


class EmpiricalModelTests(unittest.TestCase):
    def setUp(self):
        values = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
        self.distribution = {
            (int(horizon), direction): values
            for horizon in HORIZONS
            for direction in ["up", "down"]
        }

    def test_smoothed_survival_has_no_zero_or_one_tail(self):
        values = np.array([0.5, 1.0, 1.5])
        probabilities = smoothed_survival(values, np.array([-1.0, 3.0]))
        self.assertGreater(probabilities[1], 0.0)
        self.assertLess(probabilities[0], 1.0)

    def test_negative_excursions_count_as_non_exceedances(self):
        values = np.array([-1.0, -0.5, 0.25, 1.0])
        probability = smoothed_survival(values, np.array([0.1]))[0]
        expected = (2 + 0.5) / (4 + 1.0)
        self.assertAlmostEqual(probability, expected)

    def test_future_extreme_excludes_start_and_uses_exact_horizon(self):
        values = np.array([10.0, 11.0, 15.0, 12.0, 9.0])
        maximum = future_extreme(values, horizon=2, kind="max")
        minimum = future_extreme(values, horizon=2, kind="min")
        self.assertEqual(maximum[0], 15.0)
        self.assertEqual(minimum[0], 11.0)

    def test_closer_barrier_has_higher_probability(self):
        probability = empirical_touch_probability(
            distance=np.array([0.01, 0.10]),
            annualized_sigma=np.array([0.60, 0.60]),
            minutes_to_expiry=np.array([60.0, 60.0]),
            direction="up",
            distribution=self.distribution,
        )
        self.assertGreater(probability[0], probability[1])

    def test_probability_is_bounded_for_interpolated_horizon(self):
        probability = empirical_touch_probability(
            distance=np.array([0.03, 0.05]),
            annualized_sigma=np.array([0.50, 0.80]),
            minutes_to_expiry=np.array([45.0, 900.0]),
            direction="down",
            distribution=self.distribution,
        )
        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())

    def test_already_touched_is_one(self):
        probability = empirical_touch_probability(
            distance=np.array([0.0]),
            annualized_sigma=np.array([0.50]),
            minutes_to_expiry=np.array([30.0]),
            direction="up",
            distribution=self.distribution,
        )
        self.assertEqual(probability[0], 1.0)


if __name__ == "__main__":
    unittest.main()
