import unittest
from argparse import Namespace

from examples.openvoice_train_vae_mixed import apply_generated_audio_objective_plan
from scripts.plan_generated_audio_calibrated_objective import (
    ObjectiveConfig,
    build_objective_plan,
)


GATE_SUMMARY = {
    "style_summary": {
        "anger": {"style_decision": "diagnostic_only"},
        "disgust": {"style_decision": "needs_content_repair"},
        "fear": {"style_decision": "diagnostic_only"},
    }
}

FAILURE_SUMMARY = {
    "reference_condition": "mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard",
    "ready_styles": ["anger", "disgust"],
    "blocked_styles": [{"style": "fear", "selected": 0, "total": 11}],
}


class GeneratedAudioCalibratedObjectivePlanTests(unittest.TestCase):
    def test_builds_conservative_plan_from_gate_and_clean_failure_targets(self):
        plan = build_objective_plan(
            GATE_SUMMARY,
            FAILURE_SUMMARY,
            ObjectiveConfig(
                diagnostic_weight=2.0,
                content_repair_weight=3.0,
                training_strength=5.0,
            ),
        )

        self.assertEqual(plan["selected_styles"], ["anger", "disgust"])
        self.assertEqual(plan["blocked_styles"], ["fear"])
        self.assertEqual(plan["style_plan"]["anger"]["action"], "train_conservative_repair")
        self.assertEqual(plan["style_plan"]["disgust"]["action"], "train_content_repair")
        self.assertEqual(plan["style_plan"]["fear"]["action"], "blocked_no_clean_targets")
        self.assertEqual(
            plan["trainer_overrides"]["style_teacher_style_weights"],
            "anger=2,confused=0,disgust=3,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0",
        )
        self.assertEqual(
            plan["trainer_overrides"]["decoder_prototype_style_strengths"],
            "anger=5,confused=0,disgust=5,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0",
        )
        self.assertEqual(
            plan["trainer_overrides"]["decoder_prototype_style_weights"],
            "anger=2,confused=0,disgust=3,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0",
        )
        self.assertEqual(plan["trainer_overrides"]["anti_neutral_styles"], "anger,disgust")
        self.assertEqual(
            plan["trainer_overrides"]["anti_neutral_style_weights"],
            "anger=2,confused=0,disgust=3,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0",
        )
        self.assertEqual(
            plan["trainer_overrides"]["anti_neutral_style_strengths"],
            "anger=5,confused=0,disgust=5,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0",
        )

    def test_trainer_applies_generated_audio_plan_overrides(self):
        args = Namespace(
            style_teacher_target_mode="all_dims",
            style_teacher_require_label=False,
            style_teacher_style_weights="",
            decoder_prototype_style_weights="",
            decoder_prototype_style_strengths="",
            anti_neutral_styles="anger,disgust",
            anti_neutral_style_weights="",
            anti_neutral_style_strengths="",
        )
        plan = {
            "objective_name": "generated_audio_calibrated_hard_style_repair",
            "selected_styles": ["anger", "disgust"],
            "blocked_styles": ["fear"],
            "trainer_overrides": {
                "style_teacher_target_mode": "target_dim",
                "style_teacher_require_label": True,
                "style_teacher_style_weights": "anger=2,confused=0,disgust=3,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0",
                "decoder_prototype_style_weights": "anger=2,confused=0,disgust=3,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0",
                "decoder_prototype_style_strengths": "anger=5,confused=0,disgust=5,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0",
                "anti_neutral_styles": "anger,disgust",
                "anti_neutral_style_weights": "anger=2,confused=0,disgust=3,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0",
                "anti_neutral_style_strengths": "anger=5,confused=0,disgust=5,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0",
            },
        }

        report = apply_generated_audio_objective_plan(args, plan)

        self.assertEqual(args.style_teacher_target_mode, "target_dim")
        self.assertTrue(args.style_teacher_require_label)
        self.assertEqual(args.anti_neutral_styles, "anger,disgust")
        self.assertEqual(
            args.decoder_prototype_style_weights,
            "anger=2,confused=0,disgust=3,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0",
        )
        self.assertEqual(
            args.anti_neutral_style_weights,
            "anger=2,confused=0,disgust=3,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0",
        )
        self.assertEqual(report["selected_styles"], ["anger", "disgust"])
        self.assertEqual(report["blocked_styles"], ["fear"])
        self.assertEqual(
            report["applied_overrides"]["decoder_prototype_style_strengths"],
            "anger=5,confused=0,disgust=5,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0",
        )


if __name__ == "__main__":
    unittest.main()
