import unittest

import numpy as np

from scripts.audit_training_style_separability import (
    SourceClip,
    build_direct_metrics,
    cap_rows,
    canonical_label,
    nearest_centroid_predict,
    precision_recall_f1,
    style_verdict,
)


class TrainingStyleSeparabilityAuditTest(unittest.TestCase):
    def test_canonical_label_strips_bilingual_prefix(self):
        self.assertEqual(canonical_label("生气/angry"), "angry")
        self.assertEqual(canonical_label(" Disgusted "), "disgusted")

    def test_cap_rows_is_deterministic_per_label(self):
        clips = [
            SourceClip("CREMA-D", index, f"clip_{index}", "speaker", style, style, {})
            for style in ["anger", "sad"]
            for index in range(5)
        ]
        capped_a = cap_rows(clips, max_per_label=2, max_total=0, seed=42)
        capped_b = cap_rows(clips, max_per_label=2, max_total=0, seed=42)
        self.assertEqual(capped_a, capped_b)
        self.assertEqual(len(capped_a), 4)
        self.assertEqual(
            {style: sum(1 for clip in capped_a if clip.unified_style == style) for style in ["anger", "sad"]},
            {"anger": 2, "sad": 2},
        )

    def test_nearest_centroid_predicts_simple_clusters(self):
        train_x = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
        train_y = ["low", "low", "high", "high"]
        test_x = np.array([[0.2, 0.1], [9.8, 10.2]])
        self.assertEqual(nearest_centroid_predict(train_x, train_y, test_x), ["low", "high"])

    def test_precision_recall_f1_reports_per_label_values(self):
        metrics = precision_recall_f1(
            true_y=["anger", "anger", "sad", "sad"],
            pred_y=["anger", "sad", "sad", "sad"],
        )
        self.assertAlmostEqual(metrics["anger"]["precision"], 1.0)
        self.assertAlmostEqual(metrics["anger"]["recall"], 0.5)
        self.assertAlmostEqual(metrics["anger"]["f1"], 2 / 3)
        self.assertEqual(metrics["sad"]["support"], 2)

    def test_build_direct_metrics_treats_string_flags_explicitly(self):
        direct, confusion = build_direct_metrics(
            [
                {
                    "unified_style": "anger",
                    "target_e2v": "angry",
                    "predicted": "angry",
                    "direct_evaluable": "1",
                    "direct_match": "1",
                },
                {
                    "unified_style": "anger",
                    "target_e2v": "angry",
                    "predicted": "neutral",
                    "direct_evaluable": "1",
                    "direct_match": "0",
                },
                {
                    "unified_style": "whisper",
                    "target_e2v": "",
                    "predicted": "neutral",
                    "direct_evaluable": "0",
                    "direct_match": "",
                },
            ]
        )
        self.assertEqual(direct["by_style"]["anger"]["direct_support"], 2)
        self.assertEqual(direct["by_style"]["anger"]["direct_correct"], 1)
        self.assertEqual(direct["by_style"]["whisper"]["direct_support"], 0)
        self.assertEqual(len(confusion), 2)

    def test_style_verdict_uses_best_available_signal(self):
        self.assertEqual(style_verdict(0.75, None), "headline candidate")
        self.assertEqual(style_verdict(None, 0.50), "supported but quality-sensitive")
        self.assertEqual(style_verdict(0.25, 0.10), "diagnostic / weak")
        self.assertEqual(style_verdict(0.05, None), "limitation / do not force")
        self.assertEqual(style_verdict(None, None), "needs non-emotion perceptual review")


if __name__ == "__main__":
    unittest.main()
