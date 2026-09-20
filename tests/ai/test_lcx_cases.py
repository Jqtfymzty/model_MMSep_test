"""Static checks for lcx's eight-case work package."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image


AI_ROOT = Path(__file__).resolve().parent
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from data.image_generators import generate_image  # noqa: E402


class LcxCaseDefinitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = AI_ROOT / "cases" / "lcx_cases.json"
        cls.cases = json.loads(path.read_text(encoding="utf-8"))

    def test_has_eight_uniquely_named_lcx_cases(self) -> None:
        self.assertEqual(len(self.cases), 8)
        self.assertEqual(len({case["case_id"] for case in self.cases}), 8)
        self.assertEqual({case["owner"] for case in self.cases}, {"lcx"})

    def test_dimension_split_matches_the_agreed_work_package(self) -> None:
        self.assertEqual(
            Counter(case["dimension"] for case in self.cases),
            Counter({"functional": 2, "robustness": 2, "security": 1, "fairness": 1, "performance": 2}),
        )

    def test_performance_cases_define_two_warmups_and_five_measurements(self) -> None:
        performance_cases = [case for case in self.cases if case["dimension"] == "performance"]
        for case in performance_cases:
            variant = case["input_ref"]["variants"][0]
            self.assertEqual(variant["warmup"], 2)
            self.assertEqual(variant["measurements"], 5)
            self.assertGreaterEqual(variant["max_new_tokens"], 96)

    def test_every_generator_writes_a_valid_rgb_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case in self.cases:
                for variant in case["input_ref"]["variants"]:
                    target = root / variant["file"]
                    generate_image(case["input_ref"]["generator"], variant, target)
                    with Image.open(target) as image:
                        self.assertEqual(image.mode, "RGB")
                        self.assertEqual(image.size, (672, 448))


if __name__ == "__main__":
    unittest.main()
