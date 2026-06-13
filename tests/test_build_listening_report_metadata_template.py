import csv
import tempfile
import unittest
from pathlib import Path

from scripts.build_listening_report import write_rating_template


class BuildListeningReportMetadataTemplateTest(unittest.TestCase):
    def test_metadata_control_baseline_rows_are_scoreable(self):
        rows = [
            {"source_stem": "spk1", "style": "baseline", "output_file": "base.wav"},
            {
                "source_stem": "spk1",
                "style": "baseline",
                "gender_control": "female",
                "output_file": "female.wav",
            },
            {
                "source_stem": "spk1",
                "style": "baseline",
                "gender_control": "male",
                "output_file": "male.wav",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ratings.csv"
            write_rating_template(path, rows)
            with path.open(newline="", encoding="utf-8") as handle:
                written = list(csv.DictReader(handle))

        self.assertEqual(len(written), 2)
        self.assertEqual([row["gender_control"] for row in written], ["female", "male"])


if __name__ == "__main__":
    unittest.main()
