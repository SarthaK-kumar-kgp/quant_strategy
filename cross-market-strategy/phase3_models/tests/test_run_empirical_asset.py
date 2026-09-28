import sys
from pathlib import Path
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_empirical_asset import select_asset_values  # noqa: E402


class EmpiricalAssetPoolingTests(unittest.TestCase):
    def test_selection_uses_only_requested_asset_and_cutoff(self):
        catalog = {
            5: {
                "end": np.array([10, 20, 30, 40]),
                "asset": np.array(["BTC", "ETH", "BTC", "BTC"]),
                "up": np.array([0.4, 9.0, 0.2, 8.0]),
                "down": np.array([0.3, 7.0, 0.1, 6.0]),
            }
        }
        values, ends = select_asset_values(catalog, 5, "BTC", "up", 30)
        self.assertTrue(np.array_equal(values, np.array([0.2, 0.4])))
        self.assertTrue(np.array_equal(ends, np.array([10, 30])))

    def test_direction_is_kept_separate(self):
        catalog = {
            5: {
                "end": np.array([10, 20]),
                "asset": np.array(["SOL", "SOL"]),
                "up": np.array([0.5, 0.6]),
                "down": np.array([0.1, 0.2]),
            }
        }
        up, _ = select_asset_values(catalog, 5, "SOL", "up", 20)
        down, _ = select_asset_values(catalog, 5, "SOL", "down", 20)
        self.assertTrue(np.array_equal(up, np.array([0.5, 0.6])))
        self.assertTrue(np.array_equal(down, np.array([0.1, 0.2])))


if __name__ == "__main__":
    unittest.main()
