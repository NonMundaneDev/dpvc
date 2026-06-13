import unittest

from scripts.build_mixed_training_set import accepted_commonvoice_style


class BuildMixedTrainingSetCommonVoiceTest(unittest.TestCase):
    def test_missing_pseudo_style_fields_are_safe_for_unlabeled_commonvoice_rows(self):
        cv_data = {
            "speaker_ids": ["speaker-a", "speaker-b"],
            "clip_paths": ["clips/a.mp3", "clips/b.mp3"],
        }

        for row_idx in range(2):
            style, confidence, reason = accepted_commonvoice_style(
                cv_data,
                row_idx,
                threshold_map={},
                acceptance_policy="confidence_only",
            )

            self.assertIsNone(style)
            self.assertIsNone(confidence)
            self.assertEqual(reason, "missing_style_or_confidence")


if __name__ == "__main__":
    unittest.main()
