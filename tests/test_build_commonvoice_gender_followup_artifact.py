import tempfile
import unittest
from pathlib import Path

import torch

from scripts.build_commonvoice_gender_followup_artifact import (
    build_subset_artifact,
    read_speaker_manifest,
)


class CommonVoiceGenderFollowupArtifactTest(unittest.TestCase):
    def test_read_speaker_manifest_expands_clip_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "speakers.csv"
            path.write_text(
                "gender,speaker_id,available_clips,selected_clips,clip_paths,ages,accents\n"
                "female,spk-a,3,2,a.mp3;b.mp3,twenties,us\n",
                encoding="utf-8",
            )

            rows = read_speaker_manifest(path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["speaker_id"], "spk-a")
        self.assertEqual(rows[0]["gender"], "female")
        self.assertEqual(rows[0]["clip_name"], "a.mp3")
        self.assertEqual(rows[1]["clip_name"], "b.mp3")

    def test_build_subset_artifact_preserves_metadata_and_reports_coverage(self):
        source = {
            "data": torch.arange(12, dtype=torch.float32).reshape(3, 4, 1),
            "speaker_ids": ["spk-a", "spk-b", "spk-c"],
            "clip_paths": ["clips/a.mp3", "clips/b.mp3", "clips/c.mp3"],
            "age": ["twenties", "thirties", None],
            "gender": ["female", "male", None],
            "accent": ["us", "uk", None],
            "metadata_report": {"gender": {"known": 2, "total": 3}},
        }
        selected = [
            {"gender": "female", "speaker_id": "spk-a", "clip_name": "a.mp3"},
            {"gender": "male", "speaker_id": "spk-b", "clip_name": "b.mp3"},
        ]

        artifact, report = build_subset_artifact(source, selected)

        self.assertEqual(tuple(artifact["data"].shape), (2, 4, 1))
        self.assertEqual(artifact["speaker_ids"], ["spk-a", "spk-b"])
        self.assertEqual(artifact["clip_paths"], ["clips/a.mp3", "clips/b.mp3"])
        self.assertEqual(artifact["gender"], ["female", "male"])
        self.assertEqual(report["selected_clips"], 2)
        self.assertEqual(report["matched_clips"], 2)
        self.assertEqual(report["missing_clips"], 0)
        self.assertEqual(report["matched_by_gender"], {"female": 1, "male": 1})

    def test_build_subset_artifact_fails_on_missing_manifest_clip_by_default(self):
        source = {
            "data": torch.zeros((1, 4, 1)),
            "speaker_ids": ["spk-a"],
            "clip_paths": ["clips/a.mp3"],
            "age": ["twenties"],
            "gender": ["female"],
            "accent": ["us"],
        }
        selected = [
            {"gender": "female", "speaker_id": "spk-a", "clip_name": "a.mp3"},
            {"gender": "male", "speaker_id": "spk-b", "clip_name": "missing.mp3"},
        ]

        with self.assertRaisesRegex(ValueError, "missing manifest clips"):
            build_subset_artifact(source, selected)

        artifact, report = build_subset_artifact(source, selected, allow_missing=True)
        self.assertEqual(tuple(artifact["data"].shape), (1, 4, 1))
        self.assertEqual(report["missing_clips"], 1)
        self.assertEqual(report["missing_by_gender"], {"male": 1})


if __name__ == "__main__":
    unittest.main()
