import sys
import types
import unittest
from unittest import mock

openvoice_stub = types.ModuleType("openvoice")
openvoice_stub.se_extractor = types.SimpleNamespace()
openvoice_api_stub = types.ModuleType("openvoice.api")
openvoice_api_stub.BaseSpeakerTTS = object
openvoice_api_stub.ToneColorConverter = object
sys.modules.setdefault("openvoice", openvoice_stub)
sys.modules.setdefault("openvoice.api", openvoice_api_stub)

from examples import openvoice_infer_controllable


class OpenVoiceInferControllableBaselineOnlyTest(unittest.TestCase):
    def test_parse_args_accepts_baseline_only_without_style(self):
        argv = [
            "openvoice_infer_controllable.py",
            "--source",
            "examples/source_speakers/cremad_1003.wav",
            "--out",
            "output/baseline.wav",
            "--vae-checkpoint",
            "embeddings/checkpoint.pt",
            "--baseline-only",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = openvoice_infer_controllable.parse_args()

        self.assertTrue(args.baseline_only)
        self.assertIsNone(args.style)
        self.assertFalse(args.all_styles)


if __name__ == "__main__":
    unittest.main()
