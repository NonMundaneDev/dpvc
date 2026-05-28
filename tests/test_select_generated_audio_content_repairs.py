import unittest

from scripts.select_generated_audio_content_repairs import (
    Thresholds,
    classify_candidate_row,
    parse_strength_from_candidate_tag,
    summarize_by_style,
)


def base_row(**overrides):
    row = {
        "style": "anger",
        "source_stem": "speaker_1",
        "reference_match": "0",
        "candidate_match": "1",
        "match_delta": "1",
        "reference_wer": "0.0000",
        "candidate_wer": "0.0000",
        "wer_delta": "0.0000",
        "reference_mos_delta": "-0.2000",
        "candidate_mos_delta": "-0.1000",
        "mos_delta_delta": "0.1000",
        "reference_novelty": "0.2000",
        "candidate_novelty": "0.2500",
        "novelty_delta": "0.0500",
        "quality_risk": "0",
        "severe_risk": "0",
        "candidate_tag": "mixed_teacher_cvrare_strength_grid_anger_s10",
    }
    row.update(overrides)
    return row


class SelectGeneratedAudioContentRepairsTests(unittest.TestCase):
    def test_parse_strength_from_candidate_tag_handles_decimal_tokens(self):
        self.assertEqual(
            parse_strength_from_candidate_tag("mixed_teacher_cvrare_strength_grid_fear_s7p5"),
            7.5,
        )
        self.assertEqual(
            parse_strength_from_candidate_tag("mixed_teacher_cvrare_strength_grid_anger_s10"),
            10.0,
        )

    def test_objective_pass_without_rating_needs_listening(self):
        decision = classify_candidate_row(base_row(), rating=None, thresholds=Thresholds())

        self.assertEqual(decision["objective_gate"], "pass")
        self.assertEqual(decision["perceptual_gate"], "unreviewed")
        self.assertEqual(decision["decision"], "needs_listening")

    def test_perceptual_tie_blocks_promotion_even_when_objective_passes(self):
        rating = {"prefer_reference_or_candidate": "tie"}
        decision = classify_candidate_row(base_row(), rating=rating, thresholds=Thresholds())

        self.assertEqual(decision["objective_gate"], "pass")
        self.assertEqual(decision["perceptual_gate"], "tie")
        self.assertEqual(decision["decision"], "blocked_by_perceptual_tie")

    def test_candidate_rating_promotes_only_when_objective_passes(self):
        rating = {"prefer_reference_or_candidate": "candidate"}
        decision = classify_candidate_row(base_row(), rating=rating, thresholds=Thresholds())

        self.assertEqual(decision["objective_gate"], "pass")
        self.assertEqual(decision["perceptual_gate"], "candidate")
        self.assertEqual(decision["decision"], "promote_candidate")

    def test_quality_risk_rejects_candidate_before_perceptual_gate(self):
        rating = {"prefer_reference_or_candidate": "candidate"}
        decision = classify_candidate_row(
            base_row(candidate_wer="0.7500", wer_delta="0.7500"),
            rating=rating,
            thresholds=Thresholds(max_candidate_wer=0.30, max_wer_delta=0.15),
        )

        self.assertEqual(decision["objective_gate"], "fail")
        self.assertEqual(decision["decision"], "reject_quality")
        self.assertIn("candidate_wer", decision["objective_reasons"])

    def test_no_target_gain_rejects_candidate(self):
        decision = classify_candidate_row(
            base_row(reference_match="1", candidate_match="1", match_delta="0"),
            rating={"prefer_reference_or_candidate": "candidate"},
            thresholds=Thresholds(),
        )

        self.assertEqual(decision["objective_gate"], "fail")
        self.assertEqual(decision["decision"], "reject_no_target_gain")

    def test_style_summary_keeps_objective_and_perceptual_outcomes_separate(self):
        rows = [
            {**base_row(style="anger"), **classify_candidate_row(base_row(style="anger"), None, Thresholds())},
            {
                **base_row(style="fear"),
                **classify_candidate_row(
                    base_row(style="fear"),
                    {"prefer_reference_or_candidate": "tie"},
                    Thresholds(),
                ),
            },
            {
                **base_row(style="disgust", candidate_wer="0.9000"),
                **classify_candidate_row(
                    base_row(style="disgust", candidate_wer="0.9000"),
                    {"prefer_reference_or_candidate": "candidate"},
                    Thresholds(),
                ),
            },
        ]

        summary = summarize_by_style(rows)

        self.assertEqual(summary["anger"]["style_decision"], "needs_more_listening")
        self.assertEqual(summary["fear"]["style_decision"], "diagnostic_only")
        self.assertEqual(summary["disgust"]["style_decision"], "needs_content_repair")


if __name__ == "__main__":
    unittest.main()
