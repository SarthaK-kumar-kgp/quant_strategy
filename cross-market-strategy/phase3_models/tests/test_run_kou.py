import sys
from pathlib import Path
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_kou import (  # noqa: E402
    HORIZONS,
    fit_kou_parameters,
    kou_touch_probability,
    simulate_extrema,
    smoothed_probability,
)


class KouModelTests(unittest.TestCase):
    def test_parameter_fit_respects_cutoff(self):
        count = 50_000
        times = np.arange(1, count + 1, dtype=np.int64) * 60
        rng = np.random.default_rng(7)
        returns = rng.normal(0, 0.001, count)
        returns[::100] = 0.02
        returns[50::100] = -0.02
        cutoff = int(times[45_000])
        fitted = fit_kou_parameters(times, returns, cutoff)
        self.assertLessEqual(fitted["latest_return_available"], cutoff)
        self.assertEqual(fitted["history_minutes"], 45_001)
        self.assertGreater(fitted["jump_count"], 0)

    def test_simulation_is_deterministic_and_sorted(self):
        parameters = {
            "sigma_per_sqrt_minute": 0.001,
            "drift_per_minute": 0.0,
            "jump_intensity_per_minute": 0.01,
            "up_jump_share": 0.5,
            "up_eta": 100.0,
            "down_eta": 100.0,
        }
        first = simulate_extrema(parameters, seed=123, path_count=64)
        second = simulate_extrema(parameters, seed=123, path_count=64)
        self.assertTrue(np.array_equal(first[(1440, "up")], second[(1440, "up")]))
        self.assertTrue(np.all(np.diff(first[(1440, "down")]) >= 0))

    def test_smoothed_probability_is_strictly_bounded(self):
        values = np.array([0.1, 0.2, 0.3])
        probability = smoothed_probability(values, np.array([-1.0, 2.0]))
        self.assertTrue((probability > 0).all())
        self.assertTrue((probability < 1).all())

    def test_closer_barrier_has_higher_probability(self):
        values = np.linspace(0.0, 0.2, 128)
        distribution = {
            (int(horizon), direction): values
            for horizon in HORIZONS
            for direction in ["up", "down"]
        }
        probability = kou_touch_probability(
            distance=np.array([0.01, 0.10]),
            minutes_to_expiry=np.array([60.0, 60.0]),
            direction="up",
            distribution=distribution,
        )
        self.assertGreater(probability[0], probability[1])


if __name__ == "__main__":
    unittest.main()
