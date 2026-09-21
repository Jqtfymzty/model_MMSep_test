"""One-command Baseline/MMSep execution for all fifteen module-two cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AI_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AI_ROOT.parents[1]
sys.path.insert(0, str(AI_ROOT))

from adapters.llava_local import LocalLlavaAdapter  # noqa: E402
from data.image_generators import generate_image  # noqa: E402
from metrics.code_oracles import evaluate_case, evaluate_output  # noqa: E402


REQUIRED_CASE_FIELDS = {
    "case_id",
    "title",
    "dimension",
    "risk",
    "priority",
    "preconditions",
    "input_ref",
    "perturbation",
    "oracle_type",
    "expected",
    "metrics",
    "thresholds",
    "repeat",
    "seed",
    "owner",
    "requirement_ref",
}
EXPECTED_DIMENSIONS = {
    "functional": 3,
    "robustness": 4,
    "security": 3,
    "fairness": 3,
    "performance": 2,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(*values: bytes) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_cases(paths: list[Path]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    ids: set[str] = set()
    for path in paths:
        content = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(content, list) or not content:
            raise ValueError(f"用例文件必须是非空 JSON 数组：{path}")
        for case in content:
            missing = REQUIRED_CASE_FIELDS - set(case)
            if missing:
                raise ValueError(f"{case.get('case_id', '<unknown>')} 缺少字段：{sorted(missing)}")
            if case["case_id"] in ids:
                raise ValueError(f"用例 ID 重复：{case['case_id']}")
            ids.add(case["case_id"])
            cases.append(case)
    return sorted(cases, key=lambda item: item["case_id"])


def validate_suite(cases: list[dict[str, Any]]) -> None:
    counts = {dimension: 0 for dimension in EXPECTED_DIMENSIONS}
    owners: dict[str, int] = {}
    for case in cases:
        dimension = case["dimension"]
        if dimension not in counts:
            raise ValueError(f"未知测试维度：{dimension}")
        counts[dimension] += 1
        owners[case["owner"]] = owners.get(case["owner"], 0) + 1
    if counts != EXPECTED_DIMENSIONS:
        raise ValueError(f"15 条用例维度分布不符合规范：{counts}")
    if owners != {"lcx": 8, "wjn": 7}:
        raise ValueError(f"两人工作包分布不符合规范：{owners}")


def git_commit() -> str:
    override = os.environ.get("SOURCE_COMMIT")
    if override:
        return override
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def timing_summary(values: list[float]) -> dict[str, float | int | None]:
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    return {
        "count": len(values),
        "median_ms": round(statistics.median(values), 3) if values else None,
        "q1_ms": round(q1, 3) if q1 is not None else None,
        "q3_ms": round(q3, 3) if q3 is not None else None,
        "iqr_ms": round(q3 - q1, 3) if q1 is not None and q3 is not None else None,
    }


def select_seeds(case: dict[str, Any], quick: bool) -> list[int]:
    seeds = [int(seed) for seed in case["seed"]]
    return seeds[:1] if quick else seeds


def case_execution_plan(case: dict[str, Any], quick: bool) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for variant in case["input_ref"]["variants"]:
        if case["dimension"] == "performance":
            warmups = 0 if quick else int(variant.get("warmup", 2))
            measurements = 1 if quick else int(variant.get("measurements", 5))
            for index in range(warmups):
                plan.append({"variant": variant, "seed": int(case["seed"][0]), "phase": "warmup", "iteration": index + 1})
            for index in range(measurements):
                plan.append({"variant": variant, "seed": int(case["seed"][0]), "phase": "measurement", "iteration": index + 1})
        else:
            for seed in select_seeds(case, quick):
                plan.append({"variant": variant, "seed": seed, "phase": "measurement", "iteration": 1})
    return plan


def environment_snapshot(adapter: LocalLlavaAdapter) -> dict[str, Any]:
    torch = adapter.torch
    return {
        "captured_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def summarize_case(case: dict[str, Any], mode: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    measurements = [item for item in results if item["phase"] == "measurement"]
    oracle = evaluate_case(measurements)
    durations = [float(item["duration_ms"]) for item in measurements if item["duration_ms"] is not None]
    peaks = [float(item["metrics"]["peak_memory_gib"]) for item in measurements if item["metrics"]["peak_memory_gib"] is not None]
    output_tokens = [int(item["metrics"]["output_tokens"]) for item in measurements if item["metrics"]["output_tokens"] is not None]
    throughputs = [
        float(item["metrics"]["output_tokens"]) / (float(item["duration_ms"]) / 1000.0)
        for item in measurements
        if item["metrics"]["output_tokens"] is not None
        and item["duration_ms"] is not None
        and float(item["duration_ms"]) > 0
    ]
    mmsep_stats = [item["metrics"]["mmsep"] for item in measurements if item["metrics"]["mmsep"] is not None]
    return {
        "case_id": case["case_id"],
        "title": case["title"],
        "dimension": case["dimension"],
        "priority": case["priority"],
        "owner": case["owner"],
        "mode": mode,
        "verdict": "PASS" if oracle["passed"] else "FAIL",
        "oracle_metrics": oracle,
        "timing": timing_summary(durations),
        "throughput": {
            "count": len(throughputs),
            "median_tokens_per_s": round(statistics.median(throughputs), 6) if throughputs else None,
        },
        "peak_memory_gib": max(peaks) if peaks else None,
        "median_output_tokens": statistics.median(output_tokens) if output_tokens else None,
        "mmsep": mmsep_stats[-1] if mmsep_stats else None,
        "measurement_run_ids": [item["run_id"] for item in measurements],
        "warmup_run_ids": [item["run_id"] for item in results if item["phase"] == "warmup"],
    }


def compare_modes(
    cases: list[dict[str, Any]],
    mode_summaries: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if set(mode_summaries) != {"baseline", "mmsep"}:
        return []
    definitions = {case["case_id"]: case for case in cases}
    baseline = {item["case_id"]: item for item in mode_summaries["baseline"]}
    mmsep = {item["case_id"]: item for item in mode_summaries["mmsep"]}
    comparisons: list[dict[str, Any]] = []
    for case_id in sorted(baseline.keys() & mmsep.keys()):
        base = baseline[case_id]
        compressed = mmsep[case_id]
        base_rate = float(base["oracle_metrics"]["pass_rate"])
        mmsep_rate = float(compressed["oracle_metrics"]["pass_rate"])
        quality_retention = mmsep_rate / base_rate if base_rate > 0 else None
        base_median = base["timing"]["median_ms"]
        mmsep_median = compressed["timing"]["median_ms"]
        latency_ratio = base_median / mmsep_median if base_median and mmsep_median else None
        base_throughput = base["throughput"]["median_tokens_per_s"]
        mmsep_throughput = compressed["throughput"]["median_tokens_per_s"]
        decode_speedup = (
            mmsep_throughput / base_throughput
            if base_throughput and mmsep_throughput
            else None
        )
        base_peak = base["peak_memory_gib"]
        mmsep_peak = compressed["peak_memory_gib"]
        vram_increase = (mmsep_peak - base_peak) / base_peak if base_peak and mmsep_peak else None
        mmsep_stats = compressed["mmsep"] or {}
        effective_ratio = mmsep_stats.get("effective_kv_ratio")
        kv_reduction = 1.0 - effective_ratio if effective_ratio is not None else None

        if base["verdict"] == "PASS" and compressed["verdict"] == "FAIL":
            conclusion = "REGRESSION"
        elif base["verdict"] == "FAIL" and compressed["verdict"] == "PASS":
            conclusion = "IMPROVEMENT"
        elif base["verdict"] == "PASS" and compressed["verdict"] == "PASS":
            conclusion = "BOTH_PASS"
        else:
            conclusion = "BOTH_FAIL"

        threshold_checks: dict[str, bool | None] = {}
        thresholds = definitions[case_id]["thresholds"]
        if "quality_retention_min" in thresholds:
            threshold_checks["quality_retention"] = (
                quality_retention is not None and quality_retention >= thresholds["quality_retention_min"]
            )
        if "speedup_min" in thresholds:
            threshold_checks["decode_speedup"] = (
                decode_speedup is not None and decode_speedup >= thresholds["speedup_min"]
            )
        if "kv_reduction_min" in thresholds:
            threshold_checks["kv_reduction"] = (
                kv_reduction is not None and kv_reduction >= thresholds["kv_reduction_min"]
            )
        if "vram_increase_max" in thresholds:
            threshold_checks["vram_increase"] = (
                vram_increase is not None and vram_increase <= thresholds["vram_increase_max"]
            )

        comparisons.append(
            {
                "case_id": case_id,
                "dimension": definitions[case_id]["dimension"],
                "baseline_verdict": base["verdict"],
                "mmsep_verdict": compressed["verdict"],
                "conclusion": conclusion,
                "baseline_pass_rate": base_rate,
                "mmsep_pass_rate": mmsep_rate,
                "quality_retention": round(quality_retention, 6) if quality_retention is not None else None,
                "baseline_median_ms": base_median,
                "mmsep_median_ms": mmsep_median,
                "latency_ratio": round(latency_ratio, 6) if latency_ratio is not None else None,
                "baseline_median_tokens_per_s": base_throughput,
                "mmsep_median_tokens_per_s": mmsep_throughput,
                "decode_speedup": round(decode_speedup, 6) if decode_speedup is not None else None,
                "baseline_peak_memory_gib": base_peak,
                "mmsep_peak_memory_gib": mmsep_peak,
                "vram_increase": round(vram_increase, 6) if vram_increase is not None else None,
                "kv_reduction": round(kv_reduction, 6) if kv_reduction is not None else None,
                "threshold_checks": threshold_checks,
                "performance_thresholds_passed": all(threshold_checks.values()) if threshold_checks else None,
            }
        )
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all 15 Baseline/MMSep AI test cases")
    parser.add_argument(
        "--cases",
        type=Path,
        nargs="*",
        default=[AI_ROOT / "cases" / "wjn_cases.json", AI_ROOT / "cases" / "lcx_cases.json"],
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vision-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--model-revision", default="local")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--mode", choices=("baseline", "mmsep", "both"), default="both")
    parser.add_argument("--mmsep-layer", type=int, default=16)
    parser.add_argument("--visual-keep-ratio", type=float, default=0.5 / 2.718281828459045)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    case_paths = [path.resolve() for path in args.cases]
    all_cases = load_cases(case_paths)
    validate_suite(all_cases)
    cases = all_cases
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in wanted]
        missing = wanted - {case["case_id"] for case in cases}
        if missing:
            raise ValueError(f"Unknown case IDs: {sorted(missing)}")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("module2-%Y%m%dT%H%M%SZ")
    output_dir = (args.output_dir or PROJECT_ROOT / "artifacts" / "runs" / run_id).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录必须为空或不存在：{output_dir}")
    raw_dir = output_dir / "raw"
    inputs_dir = output_dir / "inputs"
    summary_dir = output_dir / "summary"
    for directory in (raw_dir, inputs_dir, summary_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_commit = git_commit()
    config_hash = sha256_bytes(
        *(path.read_bytes() for path in case_paths),
        json.dumps(
            {
                "mode": args.mode,
                "mmsep_layer": args.mmsep_layer,
                "visual_keep_ratio": args.visual_keep_ratio,
                "default_max_new_tokens": args.max_new_tokens,
                "quick": args.quick,
            },
            sort_keys=True,
        ).encode("utf-8"),
    )
    raw_path = raw_dir / "results.jsonl"
    summary_path = summary_dir / "summary.json"
    environment_path = summary_dir / "environment.json"
    manifest_path = output_dir / "manifest.json"

    adapter = LocalLlavaAdapter(
        args.model_path,
        args.vision_model_path,
        mmsep_layer=args.mmsep_layer,
        visual_keep_ratio=args.visual_keep_ratio,
    )
    modes = ["baseline", "mmsep"] if args.mode == "both" else [args.mode]
    mode_summaries: dict[str, list[dict[str, Any]]] = {}
    fatal_errors: list[str] = []

    try:
        print("Loading LLaVA once for the complete Baseline/MMSep suite...", flush=True)
        adapter.load()
        environment_path.write_text(
            json.dumps(environment_snapshot(adapter), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with raw_path.open("a", encoding="utf-8") as raw_stream:
            for mode in modes:
                print(f"\n=== {mode.upper()} ===", flush=True)
                case_summaries: list[dict[str, Any]] = []
                for case in cases:
                    print(f"\n[{case['case_id']}] {case['title']}", flush=True)
                    case_results: list[dict[str, Any]] = []
                    for step in case_execution_plan(case, args.quick):
                        variant = step["variant"]
                        seed = step["seed"]
                        image_path = inputs_dir / variant["file"]
                        if not image_path.exists():
                            generate_image(case["input_ref"]["generator"], variant, image_path)
                        prompt = variant["prompt"]
                        input_hash = sha256_bytes(image_path.read_bytes(), prompt.encode("utf-8"))
                        run_token = uuid.uuid4().hex[:8]
                        item_id = (
                            f"{run_id}-{mode}-{case['case_id']}-{variant['name']}-"
                            f"{step['phase']}-{step['iteration']}-{run_token}"
                        )
                        max_new_tokens = int(variant.get("max_new_tokens", args.max_new_tokens))
                        started_at = utc_now()
                        try:
                            inference = adapter.infer(
                                image_path,
                                prompt,
                                seed=seed,
                                max_new_tokens=max_new_tokens,
                                mode=mode,
                            )
                            oracle = evaluate_output(inference["output"], case["expected"])
                            error = None
                        except Exception as exc:  # keep the rest of the suite auditable
                            inference = {
                                "output": "",
                                "duration_ms": None,
                                "peak_memory_gib": None,
                                "output_tokens": None,
                                "input_tokens": None,
                                "mmsep": None,
                            }
                            oracle = {"passed": False, "error": "inference_failed"}
                            error = f"{type(exc).__name__}: {exc}"
                            fatal_errors.append(f"{item_id}: {error}")

                        result = {
                            "run_id": item_id,
                            "batch_id": run_id,
                            "case_id": case["case_id"],
                            "owner": case["owner"],
                            "dimension": case["dimension"],
                            "mode": mode,
                            "mmsep_enabled": mode == "mmsep",
                            "phase": step["phase"],
                            "iteration": step["iteration"],
                            "timestamp": started_at,
                            "git_commit": source_commit,
                            "model_id": "liuhaotian/llava-v1.5-7b",
                            "model_revision": args.model_revision,
                            "config_hash": config_hash,
                            "seed": seed,
                            "input_hash": input_hash,
                            "input_file": str(image_path.relative_to(output_dir)),
                            "prompt": prompt,
                            "max_new_tokens": max_new_tokens,
                            "raw_output": inference["output"],
                            "duration_ms": inference["duration_ms"],
                            "metrics": {
                                "peak_memory_gib": inference["peak_memory_gib"],
                                "output_tokens": inference["output_tokens"],
                                "input_tokens": inference["input_tokens"],
                                "mmsep": inference["mmsep"],
                            },
                            "oracle": oracle,
                            "verdict": "PASS" if oracle["passed"] else "FAIL",
                            "error": error,
                        }
                        raw_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                        raw_stream.flush()
                        case_results.append(result)
                        print(
                            f"  {step['phase']}#{step['iteration']} {variant['name']} seed={seed}: "
                            f"{result['verdict']} {inference['duration_ms']}ms output={inference['output']!r}",
                            flush=True,
                        )
                    case_summary = summarize_case(case, mode, case_results)
                    case_summaries.append(case_summary)
                    print(f"  case verdict: {case_summary['verdict']}", flush=True)
                mode_summaries[mode] = case_summaries
    finally:
        adapter.close()

    comparisons = compare_modes(cases, mode_summaries)
    configuration_count = sum(len(items) for items in mode_summaries.values())
    passed = sum(item["verdict"] == "PASS" for items in mode_summaries.values() for item in items)
    dimension_results: dict[str, dict[str, int | float]] = {}
    for dimension in EXPECTED_DIMENSIONS:
        items = [
            item
            for summaries in mode_summaries.values()
            for item in summaries
            if item["dimension"] == dimension
        ]
        dimension_passed = sum(item["verdict"] == "PASS" for item in items)
        dimension_results[dimension] = {
            "configurations": len(items),
            "passed": dimension_passed,
            "pass_rate": dimension_passed / len(items) if items else 0.0,
        }

    summary = {
        "batch_id": run_id,
        "generated_at": utc_now(),
        "source_commit": source_commit,
        "model_revision": args.model_revision,
        "config_hash": config_hash,
        "run_policy": "quick" if args.quick else "formal-repeat-policy",
        "requested_mode": args.mode,
        "judge": "deterministic_code_oracle",
        "selected_cases": len(cases),
        "full_suite_cases": len(all_cases),
        "total_configurations": configuration_count,
        "passed_configurations": passed,
        "failed_configurations": configuration_count - passed,
        "configuration_pass_rate": passed / configuration_count if configuration_count else 0.0,
        "execution_errors": fatal_errors,
        "dimension_results": dimension_results,
        "mode_results": mode_summaries,
        "comparisons": comparisons,
        "mmsep_config": {"layer": args.mmsep_layer, "visual_keep_ratio": args.visual_keep_ratio},
        "raw_results": str(raw_path.relative_to(output_dir)),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "batch_id": run_id,
        "generated_at": utc_now(),
        "files": {},
    }
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file() and item != manifest_path):
        manifest["files"][str(path.relative_to(output_dir)).replace("\\", "/")] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nCompleted configurations: {passed}/{configuration_count} passed", flush=True)
    print(f"Execution errors: {len(fatal_errors)}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    return 2 if fatal_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
