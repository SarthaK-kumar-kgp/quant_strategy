import sys
from pathlib import Path
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_benchmarks import MINUTES_PER_YEAR, gbm_touch_probability  # noqa: E402
from run_har import (  # noqa: E402
    FEATURE_NAMES,
    fit_ridge_log_har,
    future_average_variance,
    har_touch_probability,
    predict_average_variance,
)


class HarModelTests(unittest.TestCase):
    def test_future_target_excludes_origin_and_uses_exact_horizon(self):
        squared = np.arange(1.0, 8.0)
        result = future_average_variance(squared, 3)
        self.assertAlmostEqual(result[0], np.mean([2.0, 3.0, 4.0]))
        self.assertAlmostEqual(result[3], np.mean([5.0, 6.0, 7.0]))
        self.assertTrue(np.isnan(result[4:]).all())

    def test_ridge_prediction_is_positive_and_bounded(self):
        rng = np.random.default_rng(2)
        features = rng.normal(size=(3000, len(FEATURE_NAMES)))
        target = np.exp(-12 + 0.2 * features[:, 0] + rng.normal(0, 0.1, 3000))
        model = fit_ridge_log_har(features, target)
        prediction = predict_average_variance(features[:20], model)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction > 0).all())
        self.assertTrue((prediction >= model["target_variance_lower"]).all())
        self.assertTrue((prediction <= model["target_variance_upper"]).all())

    def test_integrated_variance_formula_matches_constant_sigma_gbm(self):
        spot = np.array([100.0, 100.0])
        barrier = np.array([105.0, 95.0])
        minutes = np.array([720.0, 720.0])
        sigma = np.array([0.8, 0.8])
        direction = np.array(["up", "down"])
        distance = np.abs(np.log(barrier / spot))
        variance = sigma**2 * minutes / MINUTES_PER_YEAR
        expected = gbm_touch_probability(spot, barrier, minutes, sigma, direction)
        actual = har_touch_probability(distance, variance, direction)
        self.assertTrue(np.allclose(actual, expected, rtol=1e-12, atol=1e-12))

    def test_probability_increases_with_variance(self):
        probability = har_touch_probability(
            np.array([0.05, 0.05]),
            np.array([0.001, 0.01]),
            np.array(["up", "up"]),
        )
        self.assertGreater(probability[1], probability[0])


if __name__ == "__main__":
    unittest.main()
