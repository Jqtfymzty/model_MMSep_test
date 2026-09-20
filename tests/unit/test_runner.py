from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mmsep_testkit.adapters import AdapterError, LlavaAdapter, MockAdapter
from mmsep_testkit.adapters.llava import _generated_token_start
from mmsep_testkit.config import load_project_config
from mmsep_testkit.contracts import InferenceRequest
from mmsep_testkit.runlog import StructuredRunLogger
from mmsep_testkit.runner import build_adapter, execute, load_cases, main, write_results


REPO_ROOT = Path(__file__).resolve().parents[2]
MOCK_CONFIG = REPO_ROOT / "configs" / "experiments" / "smoke.mock.json"
MOCK_CASES = REPO_ROOT / "tests" / "fixtures" / "mock_cases.jsonl"
BASELINE_CONFIG = REPO_ROOT / "configs" / "experiments" / "baseline.llava.example.json"


class RunnerTests(unittest.TestCase):
    def test_llava_generated_only_output_is_not_sliced(self) -> None:
        start = _generated_token_start(
            output_length=32,
            input_length=55,
            prefix_matches=False,
        )
        self.assertEqual(start, 0)

    def test_llava_legacy_prompt_prefix_is_sliced(self) -> None:
        start = _generated_token_start(
            output_length=87,
            input_length=55,
            prefix_matches=True,
        )
        self.assertEqual(start, 55)

    def test_llava_nonmatching_long_output_is_not_sliced(self) -> None:
        start = _generated_token_start(
            output_length=87,
            input_length=55,
            prefix_matches=False,
        )
        self.assertEqual(start, 0)

    def test_builds_llava_adapter_without_importing_heavy_dependencies(self) -> None:
        adapter = build_adapter(load_project_config(BASELINE_CONFIG))
        self.assertIsInstance(adapter, LlavaAdapter)
        self.assertFalse(adapter.healthcheck()["loaded"])

    def test_mock_adapter_requires_load(self) -> None:
        config = load_project_config(MOCK_CONFIG)
        adapter = MockAdapter(config)
        request = InferenceRequest("M2-FUN-000", "hello", None, 42)
        with self.assertRaises(AdapterError):
            adapter.infer(request, "run")

    def test_mock_run_produces_traceable_result(self) -> None:
        config = load_project_config(MOCK_CONFIG)
        cases = load_cases(MOCK_CASES)
        results = execute(config, cases)
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.case_id, "M2-FUN-000")
        self.assertEqual(result.mode, "baseline")
        self.assertFalse(result.metadata["formal"])
        self.assertEqual(len(result.config_hash), 64)
        self.assertEqual(len(result.input_hash), 64)

    def test_result_jsonl_round_trip(self) -> None:
        config = load_project_config(MOCK_CONFIG)
        results = execute(config, load_cases(MOCK_CASES))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            write_results(path, results)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["case_id"], "M2-FUN-000")
        self.assertIn("metrics", rows[0])

    def test_cli_writes_incremental_results_and_structured_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "results.jsonl"
            log_path = Path(directory) / "runlog.jsonl"
            return_code = main(
                [
                    "--config",
                    str(MOCK_CONFIG),
                    "--cases",
                    str(MOCK_CASES),
                    "--output",
                    str(output_path),
                    "--log",
                    str(log_path),
                ]
            )
            result_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            log_rows = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(return_code, 0)
        self.assertEqual(len(result_rows), 1)
        events = [row["event"] for row in log_rows]
        self.assertIn("runtime_snapshot", events)
        self.assertIn("case_started", events)
        self.assertIn("case_completed", events)
        self.assertEqual(events[-1], "run_completed")
        self.assertTrue(all(row["run_id"] == result_rows[0]["run_id"] for row in log_rows))

    def test_structured_log_redacts_credentials_but_keeps_token_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runlog.jsonl"
            with StructuredRunLogger(path, "run", console=False) as logger:
                logger.event(
                    "redaction_test",
                    api_key="sk-example-secret-value",
                    output_tokens=12,
                    message="Authorization: Bearer example-secret-value",
                )
            text = path.read_text(encoding="utf-8")
            row = json.loads(text)
        self.assertNotIn("example-secret-value", text)
        self.assertEqual(row["data"]["api_key"], "[REDACTED]")
        self.assertEqual(row["data"]["output_tokens"], 12)


if __name__ == "__main__":
    unittest.main()
