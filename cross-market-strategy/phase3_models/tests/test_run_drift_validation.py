import sys
from pathlib import Path
import unittest


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "phase3_5_drift"))
sys.path.insert(0, str(PROJECT / "phase3_models"))

from run_drift_validation import select_winner  # noqa: E402


class DriftValidationTests(unittest.TestCase):
    def test_brier_has_priority(self):
        self.assertEqual(select_winner(-1e-4, 1.0), "drift")
        self.assertEqual(select_winner(1e-4, -1.0), "zero_drift")

    def test_log_loss_breaks_exact_brier_tie(self):
        self.assertEqual(select_winner(0.0, -1e-4), "drift")
        self.assertEqual(select_winner(0.0, 1e-4), "zero_drift")


if __name__ == "__main__":
    unittest.main()
