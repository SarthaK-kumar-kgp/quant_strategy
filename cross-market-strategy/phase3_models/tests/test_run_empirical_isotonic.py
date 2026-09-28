import sys
from pathlib import Path
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_benchmarks import PROBABILITY_CLIP  # noqa: E402
from run_empirical_isotonic import (  # noqa: E402
    apply_isotonic,
    fit_weighted_isotonic,
)


class EmpiricalIsotonicTests(unittest.TestCase):
    def test_fitted_map_is_monotone_and_bounded(self):
        probability = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        outcome = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
        fitted = fit_weighted_isotonic(probability, outcome, np.ones(5))
        mapped = apply_isotonic(fitted, np.linspace(0.0, 1.0, 101))
        self.assertTrue((np.diff(mapped) >= -1e-15).all())
        self.assertGreaterEqual(mapped.min(), PROBABILITY_CLIP)
        self.assertLessEqual(mapped.max(), 1 - PROBABILITY_CLIP)

    def test_weights_affect_pooled_adjacent_block(self):
        probability = np.array([0.1, 0.2, 0.3])
        outcome = np.array([0.0, 1.0, 0.0])
        weight = np.array([1.0, 3.0, 1.0])
        fitted = fit_weighted_isotonic(probability, outcome, weight)
        mapped = apply_isotonic(fitted, probability)
        self.assertAlmostEqual(mapped[1], 0.75)
        self.assertAlmostEqual(mapped[2], 0.75)

    def test_out_of_range_inputs_use_endpoint_values(self):
        probability = np.array([0.2, 0.5, 0.8])
        outcome = np.array([0.0, 0.5, 1.0])
        fitted = fit_weighted_isotonic(probability, outcome, np.ones(3))
        mapped = apply_isotonic(fitted, np.array([0.0, 0.2, 0.8, 1.0]))
        self.assertAlmostEqual(mapped[0], mapped[1])
        self.assertAlmostEqual(mapped[2], mapped[3])

    def test_invalid_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_weighted_isotonic(
                np.array([0.1, 0.2]),
                np.array([0.0, 1.0]),
                np.array([1.0, 0.0]),
            )


if __name__ == "__main__":
    unittest.main()
