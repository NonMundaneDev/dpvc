import argparse
import tempfile
import unittest
from pathlib import Path

from scripts.build_control_selection_recommendation import (
    classify_generated_gate,
    classify_recommendation,
    classify_source_gate,
    mean,
    to_float,
    write_markdown,
)


class ControlSelectionRecommendationTest(unittest.TestCase):
    def test_source_gate_uses_best_available_signal(self):
        self.assertEqual(classify_source_gate(0.80, None), "source_separable")
        self.assertEqual(classify_source_gate(None, 0.50), "source_supported_quality_sensitive")
        self.assertEqual(classify_source_gate(0.25, 0.10), "source_weak")
        self.assertEqual(classify_source_gate(None, None), "source_unmeasured")

    def test_generated_gate_for_direct_emotion_controls(self):
        self.assertEqual(classify_generated_gate(11, 0.91, 0.12, -0.10, 0.27), "generated_strong")
        self.assertEqual(classify_generated_gate(11, 0.45, 0.39, -0.10, 0.45), "generated_mixed")
        self.assertEqual(classify_generated_gate(11, 0.10, 0.17, -0.05, 0.36), "generated_diagnostic")
        self.assertEqual(classify_generated_gate(11, 0.00, 0.50, -1.00, 0.05), "generated_weak")

    def test_generated_gate_for_nonemotion_controls(self):
        self.assertEqual(classify_generated_gate(0, None, 0.20, -0.05, 0.44), "generated_nonemotion_supported")
        self.assertEqual(classify_generated_gate(0, None, 0.32, -0.50, 0.31), "generated_nonemotion_quality_sensitive")
        self.assertEqual(classify_generated_gate(0, None, 0.50, -0.80, 0.10), "generated_nonemotion_weak")

    def test_recommendation_requires_perceptual_support_for_headline(self):
        self.assertEqual(
            classify_recommendation("source_separable", "generated_strong", "supported")[0],
            "headline_control",
        )
        self.assertEqual(
            classify_recommendation("source_separable", "generated_strong", "needs_review")[0],
            "candidate_headline_pending_listening",
        )
        self.assertEqual(
            classify_recommendation("source_separable", "generated_strong", "blocked")[0],
            "diagnostic_or_limitation",
        )
        self.assertEqual(
            classify_recommendation("source_weak", "generated_strong", "supported")[0],
            "diagnostic_or_limitation",
        )
        self.assertEqual(
            classify_recommendation(
                "source_supported_quality_sensitive",
                "generated_nonemotion_supported",
                "supported",
            )[0],
            "supported_but_quality_sensitive",
        )

    def test_parse_and_mean_helpers(self):
        self.assertEqual(to_float("0.25"), 0.25)
        self.assertIsNone(to_float(""))
        self.assertAlmostEqual(mean([1.0, 2.0, 3.0]), 2.0)
        self.assertIsNone(mean([]))

    def test_markdown_next_queue_handles_no_pending_headline_rows(self):
        rows = [
            {
                "style": "neutral",
                "recommendation_bucket": "headline_control",
                "source_gate": "source_separable",
                "generated_gate": "generated_strong",
                "generated_direct_recall": 0.9,
                "generated_mean_wer": 0.1,
                "generated_mean_mos_delta": -0.1,
                "generated_mean_external_novelty": 0.2,
                "perceptual_status": "supported",
                "paper_claim_status": "paper-ready headline control",
                "perceptual_summary": "Joe confirmed neutral.",
            }
        ]
        args = argparse.Namespace(
            source="source.csv",
            emotion="emotion.csv",
            wer="wer.csv",
            mos="mos.csv",
            novelty="novelty.csv",
            collapse="collapse.csv",
            collapse_condition="condition",
            perceptual="perceptual.csv",
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "recommendation.md"
            write_markdown(rows, out, args)
            text = out.read_text(encoding="utf-8")

        self.assertIn("No candidate headline rows are waiting", text)
        self.assertIn("Joe confirmed neutral.", text)


if __name__ == "__main__":
    unittest.main()
