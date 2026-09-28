import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate_predictions import (  # noqa: E402
    PredictionValidationError,
    validate_dataframe,
)


class PredictionValidatorTests(unittest.TestCase):
    def setUp(self):
        self.times = pd.to_datetime(
            ["2026-05-02T12:00:00Z", "2026-05-02T12:01:00Z"], utc=True
        )
        self.manifest = pd.DataFrame(
            {
                "fold_id": ["fold_01"],
                "condition_id": ["condition_a"],
                "role": ["validation"],
                "first_decision": [self.times[0]],
                "last_decision": [self.times[-1]],
            }
        )
        self.predictions = pd.DataFrame(
            {
                "schema_version": ["phase3_prediction_v1"] * 2,
                "experiment_id": ["E03-gbm-raw-v1"] * 2,
                "condition_id": ["condition_a"] * 2,
                "decision_time": self.times,
                "information_timestamp": self.times,
                "fold_id": ["fold_01"] * 2,
                "role": ["validation"] * 2,
                "prediction_kind": ["forward"] * 2,
                "model_id": ["gbm_driftless"] * 2,
                "model_version": ["v1"] * 2,
                "feature_set_id": ["gbm_core_v1"] * 2,
                "calibration_method": ["raw"] * 2,
                "calibration_version": ["none"] * 2,
                "raw_yes_probability": [0.25, 0.30],
                "calibrated_yes_probability": [0.25, 0.30],
                "contract_row_weight": [0.5, 0.5],
            }
        )

    def test_valid_predictions_match_manifest_and_panel(self):
        with TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)
            pd.DataFrame(
                {
                    "decision_time": self.times,
                    "contract_row_weight": [0.5, 0.5],
                }
            ).to_parquet(panel_dir / "condition_a.parquet", index=False)
            summary = validate_dataframe(
                self.predictions, self.manifest, panel_dir=panel_dir
            )
        self.assertEqual(summary, {"rows": 2, "contracts": 1, "folds": 1})

    def test_future_information_is_rejected(self):
        invalid = self.predictions.copy()
        invalid.loc[0, "information_timestamp"] = (
            invalid.loc[0, "decision_time"] + pd.Timedelta(seconds=1)
        )
        with self.assertRaisesRegex(
            PredictionValidationError, "information_timestamp exceeds"
        ):
            validate_dataframe(invalid, self.manifest)

    def test_fold_role_mismatch_is_rejected(self):
        invalid = self.predictions.copy()
        invalid["role"] = "evaluation"
        with self.assertRaisesRegex(PredictionValidationError, "fold manifest"):
            validate_dataframe(invalid, self.manifest)

    def test_out_of_range_probability_is_rejected(self):
        invalid = self.predictions.copy()
        invalid.loc[1, "raw_yes_probability"] = 1.01
        invalid.loc[1, "calibrated_yes_probability"] = 1.01
        with self.assertRaisesRegex(PredictionValidationError, "outside"):
            validate_dataframe(invalid, self.manifest)

    def test_non_panel_timestamp_is_rejected(self):
        with TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)
            pd.DataFrame(
                {
                    "decision_time": [self.times[0]],
                    "contract_row_weight": [1.0],
                }
            ).to_parquet(panel_dir / "condition_a.parquet", index=False)
            with self.assertRaisesRegex(PredictionValidationError, "non-panel"):
                validate_dataframe(
                    self.predictions, self.manifest, panel_dir=panel_dir
                )


if __name__ == "__main__":
    unittest.main()
