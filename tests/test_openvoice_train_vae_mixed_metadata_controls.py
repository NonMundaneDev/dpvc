import argparse
import sys
import types
import unittest

import torch

openvoice_stub = types.ModuleType("openvoice")
openvoice_stub.se_extractor = types.SimpleNamespace()
openvoice_api_stub = types.ModuleType("openvoice.api")
openvoice_api_stub.BaseSpeakerTTS = object
openvoice_api_stub.ToneColorConverter = object
sys.modules.setdefault("openvoice", openvoice_stub)
sys.modules.setdefault("openvoice.api", openvoice_api_stub)

from examples.openvoice_train_vae_mixed import build_metadata_controls


class OpenVoiceTrainVaeMixedMetadataControlsTest(unittest.TestCase):
    def _data(self):
        return {
            "metadata_gender_scalar": torch.tensor([[-1.0], [1.0]]),
            "metadata_gender_mask": torch.tensor([[1.0], [1.0]]),
            "metadata_age_ordinal_scalar": torch.tensor([[-0.75], [0.0]]),
            "metadata_age_mask": torch.tensor([[1.0], [0.0]]),
        }

    def _args(self, targets):
        return argparse.Namespace(
            metadata_control_weight=0.25,
            metadata_control_targets=targets,
            metadata_gender_dim=9,
            metadata_age_dim=10,
            latent_dims=15,
        )

    def test_gender_only_metadata_control_uses_only_gender_target_and_dim(self):
        targets, mask, dims, report = build_metadata_controls(
            self._data(),
            self._args("gender"),
            supported_styles=["anger", "neutral", "sad"],
            device="cpu",
        )

        self.assertEqual(dims, [9])
        self.assertEqual(tuple(targets.shape), (2, 1))
        self.assertEqual(tuple(mask.shape), (2, 1))
        self.assertEqual(report["targets"], ["gender"])
        self.assertEqual(report["labeled_rows"], {"gender": 2})

    def test_age_only_metadata_control_uses_only_age_target_and_dim(self):
        targets, mask, dims, report = build_metadata_controls(
            self._data(),
            self._args("age"),
            supported_styles=["anger", "neutral", "sad"],
            device="cpu",
        )

        self.assertEqual(dims, [10])
        self.assertEqual(tuple(targets.shape), (2, 1))
        self.assertEqual(tuple(mask.shape), (2, 1))
        self.assertEqual(report["targets"], ["age"])
        self.assertEqual(report["labeled_rows"], {"age": 1})


if __name__ == "__main__":
    unittest.main()
