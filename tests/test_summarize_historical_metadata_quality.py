import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_historical_metadata_quality.py"
SPEC = importlib.util.spec_from_file_location("summarize_historical_metadata_quality", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class HistoricalMetadataQualitySummaryTest(unittest.TestCase):
    def test_summarizes_endpoint_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wer = root / "wer.csv"
            mos = root / "mos.csv"
            conditions = [
                "male_s1", "female_s1", "teens_s1", "sixties_s1",
                "male_s2", "female_s2", "teens_s2", "sixties_s2",
            ]
            write_csv(wer, [{"style": condition, "wer": "0.1"} for condition in conditions])
            write_csv(
                mos,
                [{"style": condition, "delta_vs_baseline": "-0.05"} for condition in conditions],
            )
            acoustic = []
            for attribute in ("gender", "age"):
                for strength in (1.0, 2.0):
                    acoustic.append(
                        {
                            "model": "m",
                            "attribute": attribute,
                            "strength": str(strength),
                            "median_positive_minus_negative_f0_hz": "20.0",
                            "positive_endpoint_higher_f0": "2",
                            "pairs": "2",
                        }
                    )
            rows = MODULE.summarize([("m", wer, mos)], acoustic)
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["mean_wer"], "0.1000")
            self.assertEqual(rows[0]["mean_mos_delta"], "-0.0500")


if __name__ == "__main__":
    unittest.main()
