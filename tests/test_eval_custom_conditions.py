import importlib.util
import unittest
from pathlib import Path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).parents[1]


class CustomEvaluationConditionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wer = load_module("eval_wer_custom_conditions", ROOT / "examples" / "eval_wer.py")
        cls.mos = load_module("eval_mos_custom_conditions", ROOT / "examples" / "eval_mos.py")

    def test_wer_parser_accepts_metadata_suffix(self):
        speaker, condition = self.wer.parse_filename(
            Path("cremad_1004_female_s2.wav"), ["male_s2", "female_s2"]
        )
        self.assertEqual((speaker, condition), ("cremad_1004", "female_s2"))

    def test_mos_parser_keeps_baseline_available(self):
        speaker, condition = self.mos.parse_filename(
            Path("cremad_1004_baseline.wav"), ["male_s1", "female_s1"]
        )
        self.assertEqual((speaker, condition), ("cremad_1004", "baseline"))


if __name__ == "__main__":
    unittest.main()
