import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.eval_external_speaker_verifier import (
    build_proxy_trials,
    compute_eer,
    cosine,
    load_audio_mono,
    summarize_by_style,
    summarize_verifier_rows,
)


class ExternalSpeakerVerifierTests(unittest.TestCase):
    def test_cosine_handles_zero_vectors(self):
        self.assertEqual(cosine([0.0, 0.0], [1.0, 0.0]), 0.0)
        self.assertEqual(round(cosine([1.0, 0.0], [1.0, 0.0]), 6), 1.0)

    def test_compute_eer_finds_threshold_for_separated_scores(self):
        result = compute_eer(
            scores=[0.95, 0.85, 0.20, 0.10],
            labels=[1, 1, 0, 0],
        )

        self.assertEqual(result["num_positive_trials"], 2)
        self.assertEqual(result["num_negative_trials"], 2)
        self.assertEqual(result["eer"], 0.0)
        self.assertGreaterEqual(result["threshold"], 0.20)
        self.assertLessEqual(result["threshold"], 0.85)

    def test_build_proxy_trials_uses_baselines_without_self_impostors(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_a = root / "a.wav"
            source_b = root / "b.wav"
            base_a = root / "a_baseline.wav"
            base_b = root / "b_baseline.wav"
            for path in [source_a, source_b, base_a, base_b]:
                path.write_bytes(b"fake")

            records = [
                {
                    "source_stem": "speaker_a",
                    "style": "baseline",
                    "source_file": str(source_a),
                    "generated_file": str(base_a),
                },
                {
                    "source_stem": "speaker_b",
                    "style": "baseline",
                    "source_file": str(source_b),
                    "generated_file": str(base_b),
                },
            ]
            baseline_map = {"speaker_a": str(base_a), "speaker_b": str(base_b)}

            trials = build_proxy_trials(records, baseline_map)

            positive = [trial for trial in trials if trial.label == 1]
            negative = [trial for trial in trials if trial.label == 0]
            self.assertEqual(len(positive), 2)
            self.assertEqual(len(negative), 2)
            self.assertTrue(
                all(Path(trial.enroll_file).name[0] != Path(trial.test_file).name[0] for trial in negative)
            )

    def test_summarize_verifier_rows_counts_acceptance_and_novelty_gain(self):
        rows = [
            {
                "style": "baseline",
                "external_similarity": "0.8000",
                "external_novelty_gain_vs_baseline": "",
                "accepted_as_source_at_threshold": "1",
            },
            {
                "style": "anger",
                "external_similarity": "0.5000",
                "external_novelty_gain_vs_baseline": "0.3000",
                "accepted_as_source_at_threshold": "0",
            },
            {
                "style": "sad",
                "external_similarity": "0.7000",
                "external_novelty_gain_vs_baseline": "0.1000",
                "accepted_as_source_at_threshold": "1",
            },
        ]

        summary = summarize_verifier_rows(rows)

        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["styled_rows"], 2)
        self.assertEqual(summary["accepted_as_source_count"], 2)
        self.assertEqual(summary["styled_accept_rate"], 0.5)
        self.assertEqual(summary["mean_styled_novelty_gain_vs_baseline"], 0.2)

    def test_summarize_by_style_reports_mean_gain_and_accept_rate(self):
        rows = [
            {
                "style": "anger",
                "external_similarity": "0.5000",
                "external_novelty_gain_vs_baseline": "0.3000",
                "accepted_as_source_at_threshold": "0",
            },
            {
                "style": "anger",
                "external_similarity": "0.7000",
                "external_novelty_gain_vs_baseline": "0.1000",
                "accepted_as_source_at_threshold": "1",
            },
            {
                "style": "baseline",
                "external_similarity": "0.8000",
                "external_novelty_gain_vs_baseline": "",
                "accepted_as_source_at_threshold": "1",
            },
        ]

        summary = summarize_by_style(rows)

        self.assertEqual(summary[0]["style"], "anger")
        self.assertEqual(summary[0]["rows"], 2)
        self.assertEqual(summary[0]["mean_external_similarity"], 0.6)
        self.assertEqual(summary[0]["mean_novelty_gain_vs_baseline"], 0.2)
        self.assertEqual(summary[0]["accept_as_source_rate"], 0.5)

    def test_load_audio_mono_uses_soundfile_and_resamples(self):
        import numpy as np
        import soundfile as sf

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stereo.wav"
            audio = np.stack(
                [
                    np.linspace(-0.5, 0.5, 800, dtype=np.float32),
                    np.linspace(0.5, -0.5, 800, dtype=np.float32),
                ],
                axis=1,
            )
            sf.write(path, audio, 8000)

            waveform = load_audio_mono(str(path), target_sample_rate=16000)

            self.assertEqual(waveform.ndim, 2)
            self.assertEqual(waveform.shape[0], 1)
            self.assertGreater(waveform.shape[1], 800)


if __name__ == "__main__":
    unittest.main()
