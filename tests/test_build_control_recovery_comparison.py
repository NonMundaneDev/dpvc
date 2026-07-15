import unittest

from scripts.build_control_recovery_comparison import (
    aggregate_details,
    build_detail_rows,
)


class BuildControlRecoveryComparisonTests(unittest.TestCase):
    def setUp(self):
        self.legacy_rows = [
            {
                "source_stem": "speaker_a",
                "style": "baseline",
                "source_file": "/tmp/source.wav",
                "output_file": "/tmp/legacy_baseline.wav",
                "style_strength": 0.0,
            },
            {
                "source_stem": "speaker_a",
                "style": "anger",
                "source_file": "/tmp/source.wav",
                "output_file": "/tmp/legacy_anger.wav",
                "style_strength": 5.0,
            },
        ]
        self.current_rows = [
            {
                "source_stem": "speaker_a",
                "style": "baseline",
                "source_file": "/tmp/source.wav",
                "output_file": "/tmp/current_baseline.wav",
                "style_strength": 0.0,
            },
            {
                "source_stem": "speaker_a",
                "style": "anger",
                "source_file": "/tmp/source.wav",
                "output_file": "/tmp/current_anger.wav",
                "style_strength": 5.0,
            },
        ]

    def test_builds_matched_detail_and_aggregate_rows(self):
        legacy_metrics = {
            ("speaker_a", "anger"): {
                "target": "angry",
                "match": "0",
                "wer": "0.2",
                "delta_vs_baseline": "-0.1",
                "novelty_gain_vs_baseline": "0.3",
            }
        }
        current_metrics = {
            ("speaker_a", "anger"): {
                "target": "angry",
                "match": "1",
                "wer": "0.1",
                "delta_vs_baseline": "-0.05",
                "novelty_gain_vs_baseline": "0.4",
            }
        }
        details = build_detail_rows(
            self.legacy_rows,
            self.current_rows,
            legacy_metrics,
            current_metrics,
            "legacy",
            "current",
        )
        self.assertEqual(len(details), 4)
        aggregates = aggregate_details(details, "legacy", "current")
        self.assertEqual(len(aggregates), 1)
        anger = aggregates[0]
        self.assertEqual(anger["style"], "anger")
        self.assertEqual(anger["legacy_recall"], 0.0)
        self.assertEqual(anger["current_recall"], 1.0)
        self.assertAlmostEqual(anger["delta_mean_wer"], -0.1)

    def test_rejects_partially_matched_manifests(self):
        with self.assertRaisesRegex(ValueError, "not fully matched"):
            build_detail_rows(
                self.legacy_rows,
                self.current_rows[:-1],
                {},
                {},
                "legacy",
                "current",
            )


if __name__ == "__main__":
    unittest.main()
