import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.probe_commonvoice_metadata_separability import (
    clean_value,
    evaluate_feature_space,
    normalize_metadata,
    run_probe,
    select_labeled_rows,
    stratified_split,
    verdict,
)


class MetadataSeparabilityProbeTests(unittest.TestCase):
    def test_clean_and_normalize_metadata(self):
        self.assertIsNone(clean_value(None))
        self.assertIsNone(clean_value(""))
        self.assertIsNone(clean_value(float("nan")))
        self.assertEqual(normalize_metadata("gender", "male_masculine"), "male")
        self.assertEqual(normalize_metadata("gender", "female_feminine"), "female")
        self.assertEqual(normalize_metadata("age", "fourties"), "forties")
        self.assertEqual(normalize_metadata("accent", " Canadian English "), "Canadian English")

    def test_select_labeled_rows_filters_sparse_classes(self):
        values = ["male", "male_masculine", "female", "female_feminine", "other"]
        indices, labels, raw_counts = select_labeled_rows(
            values=values,
            field="gender",
            min_class_count=2,
            top_k_classes=0,
            max_rows=0,
            seed=42,
        )

        self.assertEqual(indices, [0, 1, 2, 3])
        self.assertEqual(labels, ["male", "male", "female", "female"])
        self.assertEqual(raw_counts["male"], 2)
        self.assertEqual(raw_counts["female"], 2)
        self.assertEqual(raw_counts["other"], 1)

    def test_stratified_split_keeps_every_class_in_train_and_test(self):
        labels = ["a"] * 10 + ["b"] * 10 + ["c"] * 10
        train, test = stratified_split(labels, test_fraction=0.2, seed=42)

        self.assertEqual(len(test), 6)
        self.assertEqual(len(train), 24)
        self.assertEqual({labels[index] for index in train}, {"a", "b", "c"})
        self.assertEqual({labels[index] for index in test}, {"a", "b", "c"})

    def test_evaluate_feature_space_detects_clear_separation(self):
        torch.manual_seed(42)
        left = torch.randn(30, 4) * 0.05 - 2.0
        right = torch.randn(30, 4) * 0.05 + 2.0
        features = torch.cat([left, right], dim=0)
        labels = ["left"] * 30 + ["right"] * 30

        result = evaluate_feature_space(
            features=features,
            labels=labels,
            test_fraction=0.25,
            seed=42,
            permutations=20,
        )

        self.assertGreaterEqual(result["accuracy"], 0.95)
        self.assertGreaterEqual(result["macro_f1"], 0.95)
        self.assertGreater(result["macro_f1_delta_vs_majority"], 0.40)
        self.assertLessEqual(result["macro_f1_p_value"], 0.10)
        self.assertGreater(result["centroid_separation_ratio"], 1.0)

    def test_run_probe_skips_when_only_one_class_survives(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "artifact.pt"
            torch.save(
                {
                    "data": torch.randn(4, 3),
                    "gender": ["male", "male", None, "male"],
                },
                artifact_path,
            )
            rows = run_probe(
                SimpleNamespace(
                    artifact=str(artifact_path),
                    fields="gender",
                    vae_checkpoint=None,
                    latent_dims=None,
                    max_rows=0,
                    min_class_count=2,
                    top_k_classes=0,
                    test_fraction=0.25,
                    permutations=0,
                    seed=42,
                )
            )

        self.assertEqual(rows[0]["status"], "skipped")
        self.assertEqual(rows[0]["reason"], "fewer_than_two_classes_after_filtering")

    def test_verdict_thresholds_are_conservative(self):
        row = {
            "status": "ok",
            "accuracy_delta_vs_majority": 0.01,
            "macro_f1_delta_vs_majority": 0.01,
            "macro_f1_p_value": 0.5,
        }
        self.assertEqual(verdict(row), "not meaningfully separable")

        row["macro_f1_delta_vs_majority"] = 0.12
        row["macro_f1_p_value"] = 0.05
        self.assertEqual(verdict(row), "moderately separable")


if __name__ == "__main__":
    unittest.main()
