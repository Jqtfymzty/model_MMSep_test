"""Recompute summary metrics from immutable module-two JSONL evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = PROJECT_ROOT / "tests" / "ai"
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from runners.run_all_cases import compare_modes, load_cases, summarize_case, validate_suite  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    if not records:
        raise ValueError(f"原始结果为空：{path}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild a corrected summary without rerunning the model")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--output-name", default="summary.corrected.json")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    raw_path = run_dir / "raw" / "results.jsonl"
    original_summary_path = run_dir / "summary" / "summary.json"
    output_path = run_dir / "summary" / args.output_name
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有汇总：{output_path}")

    original = json.loads(original_summary_path.read_text(encoding="utf-8"))
    records = load_jsonl(raw_path)
    cases = load_cases(
        [AI_ROOT / "cases" / "wjn_cases.json", AI_ROOT / "cases" / "lcx_cases.json"]
    )
    validate_suite(cases)
    selected_ids = {record["case_id"] for record in records}
    selected_cases = [case for case in cases if case["case_id"] in selected_ids]

    modes = sorted({record["mode"] for record in records})
    mode_summaries: dict[str, list[dict[str, Any]]] = {}
    for mode in modes:
        summaries = []
        for case in selected_cases:
            matching = [
                record
                for record in records
                if record["mode"] == mode and record["case_id"] == case["case_id"]
            ]
            if matching:
                summaries.append(summarize_case(case, mode, matching))
        mode_summaries[mode] = summaries

    comparisons = compare_modes(selected_cases, mode_summaries)
    configuration_count = sum(len(items) for items in mode_summaries.values())
    passed = sum(
        item["verdict"] == "PASS"
        for items in mode_summaries.values()
        for item in items
    )
    dimensions = sorted({case["dimension"] for case in selected_cases})
    dimension_results: dict[str, dict[str, int | float]] = {}
    for dimension in dimensions:
        items = [
            item
            for summaries in mode_summaries.values()
            for item in summaries
            if item["dimension"] == dimension
        ]
        count_passed = sum(item["verdict"] == "PASS" for item in items)
        dimension_results[dimension] = {
            "configurations": len(items),
            "passed": count_passed,
            "pass_rate": count_passed / len(items) if items else 0.0,
        }

    corrected = {
        **original,
        "recomputed_at": datetime.now(timezone.utc).isoformat(),
        "analysis_commit": args.analysis_commit,
        "recomputed_from": {
            "raw_results": "raw/results.jsonl",
            "raw_results_sha256": sha256_file(raw_path),
            "original_summary": "summary/summary.json",
            "original_summary_sha256": sha256_file(original_summary_path),
        },
        "metric_definition": {
            "throughput_tokens_per_s": "generated output tokens divided by end-to-end generation seconds",
            "decode_speedup": "MMSep median tokens/s divided by Baseline median tokens/s",
            "latency_ratio": "Baseline median duration divided by MMSep median duration",
        },
        "total_configurations": configuration_count,
        "passed_configurations": passed,
        "failed_configurations": configuration_count - passed,
        "configuration_pass_rate": passed / configuration_count if configuration_count else 0.0,
        "execution_errors": [record["error"] for record in records if record.get("error")],
        "dimension_results": dimension_results,
        "mode_results": mode_summaries,
        "comparisons": comparisons,
    }
    output_path.write_text(json.dumps(corrected, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = run_dir / "manifest.corrected.json"
    manifest = {
        "batch_id": original["batch_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_commit": args.analysis_commit,
        "files": {},
    }
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file() and item != manifest_path):
        relative = str(path.relative_to(run_dir)).replace("\\", "/")
        manifest["files"][relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
