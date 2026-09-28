import sys
from pathlib import Path
import unittest

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate_trial_registry import (  # noqa: E402
    REQUIRED_COLUMNS,
    TrialRegistryValidationError,
    validate_registry,
)


class TrialRegistryValidatorTests(unittest.TestCase):
    def valid_row(self):
        row = {column: "" for column in REQUIRED_COLUMNS}
        row.update(
            {
                "experiment_id": "E03-gbm-raw-v1",
                "registered_at_utc": "2026-09-18T10:00:00Z",
                "protocol_version": "phase3_protocol_v1",
                "status": "registered",
                "model_family": "gbm",
                "model_id": "gbm_driftless",
                "model_version": "v1",
                "parameter_spec": "fixed-driftless-v1",
                "feature_set_id": "gbm_core_v1",
                "pooling_spec": "pooled",
                "asset_universe": "BTC|ETH|SOL|XRP",
                "dvol_usage": "none",
                "calibration_method": "raw",
                "calibration_spec": "none",
                "ensemble_spec": "none",
                "fold_ids": "fold_01|fold_02|fold_03|fold_04",
                "selection_rule": "contract_weighted_brier_v1",
                "random_seed": "none",
                "code_commit": "untracked:test",
                "data_manifest_id": "phase2_folds_v1",
            }
        )
        return row

    def test_empty_header_only_registry_is_valid(self):
        registry = pd.DataFrame(columns=REQUIRED_COLUMNS)
        self.assertEqual(validate_registry(registry), {"trials": 0, "completed": 0})

    def test_registered_trial_is_valid(self):
        registry = pd.DataFrame([self.valid_row()])
        self.assertEqual(validate_registry(registry)["trials"], 1)

    def test_duplicate_experiment_is_rejected(self):
        row = self.valid_row()
        registry = pd.DataFrame([row, row])
        with self.assertRaisesRegex(TrialRegistryValidationError, "unique"):
            validate_registry(registry)


if __name__ == "__main__":
    unittest.main()
