import unittest

from scripts.preflight_commonvoice_gender_followup import (
    gender_known_rows,
    normalize_gender,
    recommendation,
    select_speakers,
    summarize_counts,
)


class PreflightCommonVoiceGenderFollowupTest(unittest.TestCase):
    def test_normalize_gender_maps_commonvoice_values(self):
        self.assertEqual(normalize_gender("female_feminine"), "female")
        self.assertEqual(normalize_gender("male_masculine"), "male")
        self.assertEqual(normalize_gender("female"), "female")
        self.assertIsNone(normalize_gender(""))

    def test_gender_known_rows_requires_binary_gender_speaker_and_local_clip(self):
        rows = [
            {"gender": "female_feminine", "path": "a.mp3", "client_id": "speaker-a"},
            {"gender": "other", "path": "b.mp3", "client_id": "speaker-b"},
            {"gender": "male_masculine", "path": "missing.mp3", "client_id": "speaker-c"},
            {"gender": "male_masculine", "path": "c.mp3", "client_id": ""},
        ]

        known = gender_known_rows(rows, {"a.mp3", "c.mp3"})

        self.assertEqual(len(known), 1)
        self.assertEqual(known[0]["gender_normalized"], "female")

    def test_select_speakers_is_deterministic_and_prefers_multi_clip_speakers(self):
        rows = [
            {"gender_normalized": "female", "client_id": "f1", "path": "f1a.mp3"},
            {"gender_normalized": "female", "client_id": "f2", "path": "f2a.mp3"},
            {"gender_normalized": "female", "client_id": "f2", "path": "f2b.mp3"},
            {"gender_normalized": "male", "client_id": "m1", "path": "m1a.mp3"},
            {"gender_normalized": "male", "client_id": "m2", "path": "m2a.mp3"},
            {"gender_normalized": "male", "client_id": "m2", "path": "m2b.mp3"},
        ]

        selected = select_speakers(rows, clips_per_speaker=2, speaker_cap_per_gender=1, seed=42)

        self.assertEqual([row["speaker_id"] for row in selected], ["f2", "m2"])
        self.assertEqual([row["selected_clips"] for row in selected], [2, 2])

    def test_summarize_counts_and_recommendation(self):
        rows = [
            {"gender_normalized": "female", "client_id": f"f{i}", "path": f"f{i}.mp3"}
            for i in range(250)
        ] + [
            {"gender_normalized": "male", "client_id": f"m{i}", "path": f"m{i}.mp3"}
            for i in range(250)
        ]
        counts = summarize_counts(rows)
        summary = {
            "selected_speaker_counts_by_gender": {"female": 250, "male": 250},
            "selected_clip_counts_by_gender": {"female": 250, "male": 250},
        }

        self.assertEqual(counts["speaker_counts_by_gender"]["female"], 250)
        self.assertTrue(recommendation(summary).startswith("GO:"))


if __name__ == "__main__":
    unittest.main()
