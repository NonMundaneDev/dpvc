import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_historical_metadata_recovery.py"
SPEC = importlib.util.spec_from_file_location("run_historical_metadata_recovery", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class HistoricalMetadataRecoveryTest(unittest.TestCase):
    def test_conditions_preserve_historical_dims_and_polarity(self):
        indexed = {condition.label: condition for condition in MODULE.conditions()}
        self.assertEqual(indexed["male_s1"].control_features, {1: -1.0})
        self.assertEqual(indexed["female_s1"].control_features, {1: 1.0})
        self.assertEqual(indexed["teens_s2"].control_features, {0: -2.0})
        self.assertEqual(indexed["sixties_s2"].control_features, {0: 2.0})

    def test_pair_summary_preserves_direction(self):
        rows = []
        for source, male, female in (("a", 100.0, 130.0), ("b", 110.0, 105.0)):
            for endpoint, value in (("male", male), ("female", female)):
                rows.append(
                    {
                        "model": "m",
                        "source_stem": source,
                        "attribute": "gender",
                        "strength": 1.0,
                        "endpoint": endpoint,
                        "bundle_audio": f"{source}_{endpoint}.wav",
                        "median_f0_hz": str(value),
                        "spectral_centroid_hz": "1000",
                    }
                )
            for endpoint, value in (("teens", 120.0), ("sixties", 100.0)):
                rows.append(
                    {
                        "model": "m",
                        "source_stem": source,
                        "attribute": "age",
                        "strength": 1.0,
                        "endpoint": endpoint,
                        "bundle_audio": f"{source}_{endpoint}.wav",
                        "median_f0_hz": str(value),
                        "spectral_centroid_hz": "1000",
                    }
                )
        doubled = []
        for row in rows:
            doubled.append(row)
            doubled.append({**row, "strength": 2.0, "bundle_audio": "s2_" + row["bundle_audio"]})
        pairs = MODULE.build_pair_rows(doubled)
        summaries = MODULE.summarize_pairs(pairs)
        gender_s1 = next(
            row
            for row in summaries
            if row["attribute"] == "gender" and row["strength"] == 1.0
        )
        self.assertEqual(gender_s1["positive_endpoint_higher_f0"], 1)
        self.assertEqual(gender_s1["median_positive_minus_negative_f0_hz"], "12.5000")


if __name__ == "__main__":
    unittest.main()
