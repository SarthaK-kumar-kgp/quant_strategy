import sys
from pathlib import Path
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_kou_local import (  # noqa: E402
    LOCAL_WINDOW,
    fit_local_kou_parameters,
    local_scale,
)


class LocalVolatilityKouTests(unittest.TestCase):
    def test_scale_excludes_current_return(self):
        returns = np.ones(LOCAL_WINDOW + 2)
        returns[LOCAL_WINDOW] = 100.0
        scale = local_scale(returns)
        self.assertAlmostEqual(scale[LOCAL_WINDOW], 0.0)
        self.assertGreater(scale[LOCAL_WINDOW + 1], 0.0)

    def test_first_sixty_returns_are_ineligible(self):
        rng = np.random.default_rng(4)
        returns = rng.normal(0, 0.001, 50_000)
        scale = local_scale(returns)
        self.assertTrue(np.isnan(scale[:LOCAL_WINDOW]).all())
        self.assertTrue(np.isfinite(scale[LOCAL_WINDOW:]).all())

    def test_fit_respects_cutoff_and_counts_only_eligible_minutes(self):
        count = 50_000
        times = np.arange(1, count + 1, dtype=np.int64) * 60
        rng = np.random.default_rng(8)
        returns = rng.normal(0, 0.001, count)
        returns[500::1000] = 0.02
        returns[900::1000] = -0.02
        cutoff = int(times[45_000])
        fitted = fit_local_kou_parameters(times, returns, cutoff)
        self.assertLessEqual(fitted["latest_return_available"], cutoff)
        self.assertEqual(fitted["history_minutes"], 45_001)
        self.assertEqual(fitted["eligible_minutes"], 45_001 - LOCAL_WINDOW)
        self.assertGreater(fitted["jump_count"], 0)

    def test_local_standardisation_handles_volatility_regime_change(self):
        rng = np.random.default_rng(10)
        low = rng.normal(0, 0.001, 25_000)
        high = rng.normal(0, 0.004, 25_000)
        returns = np.concatenate([low, high])
        scale = local_scale(returns)
        eligible = np.isfinite(scale) & (scale > 0)
        local_rate = np.mean(np.abs(returns[eligible] / scale[eligible]) >= 4)

        median = np.median(returns)
        global_mad = 1.4826 * np.median(np.abs(returns - median))
        global_rate = np.mean(np.abs(returns - median) > 4 * global_mad)
        self.assertLess(local_rate, global_rate)


if __name__ == "__main__":
    unittest.main()
