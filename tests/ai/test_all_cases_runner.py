"""CPU-only tests for the unified fifteen-case runner."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


AI_ROOT = Path(__file__).resolve().parent
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from runners.run_all_cases import (  # noqa: E402
    case_execution_plan,
    compare_modes,
    load_cases,
    timing_summary,
    validate_suite,
)


class CompleteSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_cases(
            [AI_ROOT / "cases" / "wjn_cases.json", AI_ROOT / "cases" / "lcx_cases.json"]
        )

    def test_suite_has_exactly_fifteen_cases_and_valid_distribution(self) -> None:
        validate_suite(self.cases)
        self.assertEqual(len(self.cases), 15)

    def test_formal_performance_plan_has_two_warmups_and_five_measurements(self) -> None:
        case = next(item for item in self.cases if item["case_id"] == "M2-PER-201")
        plan = case_execution_plan(case, quick=False)
        self.assertEqual(sum(item["phase"] == "warmup" for item in plan), 2)
        self.assertEqual(sum(item["phase"] == "measurement" for item in plan), 5)

    def test_timing_summary_reports_median_and_iqr(self) -> None:
        metrics = timing_summary([10.0, 12.0, 14.0, 16.0, 18.0])
        self.assertEqual(metrics["median_ms"], 14.0)
        self.assertEqual(metrics["iqr_ms"], 4.0)

    def test_performance_comparison_applies_all_thresholds(self) -> None:
        case = next(item for item in self.cases if item["case_id"] == "M2-PER-202")
        baseline = {
            "case_id": case["case_id"],
            "verdict": "PASS",
            "oracle_metrics": {"pass_rate": 1.0},
            "timing": {"median_ms": 120.0},
            "peak_memory_gib": 5.0,
            "mmsep": None,
        }
        compressed = {
            "case_id": case["case_id"],
            "verdict": "PASS",
            "oracle_metrics": {"pass_rate": 1.0},
            "timing": {"median_ms": 90.0},
            "peak_memory_gib": 5.1,
            "mmsep": {"effective_kv_ratio": 0.6},
        }
        comparison = compare_modes(
            [case],
            {"baseline": [baseline], "mmsep": [compressed]},
        )[0]
        self.assertTrue(comparison["performance_thresholds_passed"])
        self.assertAlmostEqual(comparison["speedup"], 4 / 3, places=5)
        self.assertEqual(comparison["kv_reduction"], 0.4)


if __name__ == "__main__":
    unittest.main()
