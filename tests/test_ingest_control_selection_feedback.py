import argparse
import tempfile
import unittest
from pathlib import Path

from scripts.ingest_control_selection_feedback import (
    build_updates,
    parse_mapping,
    summarize_ratings,
    upsert_ledger_rows,
)


class IngestControlSelectionFeedbackTest(unittest.TestCase):
    def test_summarize_empty_ratings_needs_review(self):
        summary = summarize_ratings([], "neutral")

        self.assertEqual(summary["status"], "needs_review")
        self.assertIn("target=n/a", summary["summary"])

    def test_summarize_high_scores_supported(self):
        rows = [
            {
                "neutral_sounds_neutral_1_5": "5",
                "neutral_intelligibility_1_5": "4",
                "neutral_naturalness_1_5": "5",
                "paper_demo_candidate": "yes",
                "notes": "clear neutral control",
            }
        ]

        summary = summarize_ratings(rows, "neutral")

        self.assertEqual(summary["status"], "supported")
        self.assertIn("clear neutral control", summary["summary"])

    def test_summarize_low_target_score_blocks(self):
        rows = [
            {
                "sad_sounds_sad_1_5": "2",
                "sad_intelligibility_1_5": "5",
                "sad_naturalness_1_5": "5",
                "paper_demo_candidate": "maybe",
            }
        ]

        summary = summarize_ratings(rows, "sad")

        self.assertEqual(summary["status"], "blocked")

    def test_parse_mapping_rejects_invalid_status(self):
        with self.assertRaises(ValueError):
            parse_mapping(["neutral=magic"], allowed_statuses={"supported"})

    def test_upsert_preserves_canonical_style_order(self):
        ledger = [{"style": "sad"}, {"style": "anger"}]
        updates = [{"style": "neutral"}, {"style": "custom"}]

        rows = upsert_ledger_rows(ledger, updates)

        self.assertEqual([row["style"] for row in rows], ["anger", "neutral", "sad", "custom"])

    def test_build_updates_can_apply_plain_text_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            feedback_path = Path(tmp) / "joe_feedback.txt"
            feedback_path.write_text("sounds clearly neutral", encoding="utf-8")
            args = argparse.Namespace(
                ratings_csv=None,
                feedback_text=str(feedback_path),
                listener="Joe",
                styles="neutral",
                style_status=["neutral=supported"],
                style_summary=[],
                evidence_path=None,
            )

            updates = build_updates(args)

        self.assertEqual(updates[0]["style"], "neutral")
        self.assertEqual(updates[0]["perceptual_status"], "supported")
        self.assertEqual(updates[0]["evidence_path"], str(feedback_path))
        self.assertIn("sounds clearly neutral", updates[0]["summary"])


if __name__ == "__main__":
    unittest.main()
