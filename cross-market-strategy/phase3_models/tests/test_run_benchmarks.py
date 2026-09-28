import sys
from pathlib import Path
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_benchmarks import gbm_touch_probability  # noqa: E402


class GbmBenchmarkTests(unittest.TestCase):
    def test_probability_is_bounded_and_moves_with_distance(self):
        probability = gbm_touch_probability(
            spot=np.array([100.0, 100.0]),
            barrier=np.array([101.0, 110.0]),
            minutes_to_expiry=np.array([1440.0, 1440.0]),
            annualized_sigma=np.array([0.60, 0.60]),
            direction=np.array(["up", "up"]),
        )
        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())
        self.assertGreater(probability[0], probability[1])

    def test_probability_moves_with_time_and_volatility(self):
        time_probability = gbm_touch_probability(
            spot=np.array([100.0, 100.0]),
            barrier=np.array([105.0, 105.0]),
            minutes_to_expiry=np.array([60.0, 1440.0]),
            annualized_sigma=np.array([0.60, 0.60]),
            direction=np.array(["up", "up"]),
        )
        volatility_probability = gbm_touch_probability(
            spot=np.array([100.0, 100.0]),
            barrier=np.array([105.0, 105.0]),
            minutes_to_expiry=np.array([1440.0, 1440.0]),
            annualized_sigma=np.array([0.30, 0.90]),
            direction=np.array(["up", "up"]),
        )
        self.assertGreater(time_probability[1], time_probability[0])
        self.assertGreater(volatility_probability[1], volatility_probability[0])

    def test_up_and_down_are_handled(self):
        probability = gbm_touch_probability(
            spot=np.array([100.0, 100.0]),
            barrier=np.array([105.0, 95.0]),
            minutes_to_expiry=np.array([1440.0, 1440.0]),
            annualized_sigma=np.array([0.60, 0.60]),
            direction=np.array(["up", "down"]),
        )
        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue((probability > 0).all())


if __name__ == "__main__":
    unittest.main()
