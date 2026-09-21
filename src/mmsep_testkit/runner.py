"""Execute validated cases through a configured inference adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .adapters import LlavaAdapter, MockAdapter
from .config import ConfigError, ProjectConfig, load_project_config
from .contracts import InferenceRequest, InferenceResult, TestCase
from .preflight import run_preflight
from .runlog import (
    StructuredRunLogger,
    disk_snapshot,
    file_sha256,
    gpu_snapshot,
    runtime_snapshot,
)


def load_cases(path: str | Path) -> list[TestCase]:
    case_path = Path(path)
    cases: list[TestCase] = []
    seen: set[str] = set()
    with case_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                case = TestCase.from_dict(raw)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid case at line {line_number}: {exc}") from exc
            if case.case_id in seen:
                raise ValueError(f"duplicate case_id: {case.case_id}")
            seen.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError("case file contains no cases")
    return cases


def build_adapter(config: ProjectConfig):
    if config.model.backend == "mock":
        return MockAdapter(config)
    if config.model.backend == "llava":
        if config.model.mmsep_enabled:
            raise NotImplementedError(
                "MMSep mode is owned by the wjn work package; the lcx adapter "
                "currently provides the Baseline path only"
            )
        return LlavaAdapter(config)
    raise NotImplementedError(f"unsupported backend: {config.model.backend}")


def execute(
    config: ProjectConfig,
    cases: Iterable[TestCase],
    *,
    run_id: str | None = None,
    event_logger: StructuredRunLogger | None = None,
    result_callback: Callable[[InferenceResult], None] | None = None,
) -> list[InferenceResult]:
    case_list = list(cases)
    _event(
        event_logger,
        "preflight_started",
        message="Running configuration, dependency, model, and GPU checks",
    )
    checks = run_preflight(config)
    for check in checks:
        _event(
            event_logger,
            "preflight_check",
            level="INFO" if check.ok else "ERROR",
            check=check.name,
            ok=check.ok,
            detail=check.detail,
        )
    failed = [check for check in checks if not check.ok]
    if failed:
        summary = "; ".join(f"{check.name}: {check.detail}" for check in failed)
        raise RuntimeError(f"preflight failed: {summary}")
    _event(event_logger, "preflight_completed", checks=len(checks), ok=True)

    config_text = json.dumps(config.as_dict(), sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    git_commit = _git_commit()
    results: list[InferenceResult] = []
    effective_run_id = run_id or uuid.uuid4().hex
    _event(
        event_logger,
        "experiment_configured",
        model_id=config.model.canonical_model_id,
        model_revision=config.model.model_revision,
        backend=config.model.backend,
        mode=config.experiment.mode,
        mmsep_enabled=config.model.mmsep_enabled,
        quantization=config.model.quantization,
        configured_device=config.model.device,
        generation=config.as_dict()["generation"],
        seeds=list(config.experiment.seeds),
        repetitions=config.experiment.repetitions,
        case_count=len(case_list),
        config_hash=config_hash,
        source_revision=git_commit,
    )
    adapter = build_adapter(config)
    load_started = time.perf_counter()
    _event(
        event_logger,
        "adapter_load_started",
        message="Loading model adapter",
        gpu=gpu_snapshot(),
    )
    try:
        adapter.load()
        _event(
            event_logger,
            "adapter_load_completed",
            load_duration_ms=round((time.perf_counter() - load_started) * 1000, 3),
            health=dict(adapter.healthcheck()),
            gpu=gpu_snapshot(),
        )
        for case in case_list:
            for repeat_index in range(config.experiment.repetitions):
                for seed in config.experiment.seeds:
                    request = InferenceRequest(
                        case_id=case.case_id,
                        text=case.input_text,
                        image_ref=case.image_ref,
                        seed=seed,
                    )
                    input_hash = hashlib.sha256(
                        json.dumps(
                            {
                                "text": request.text,
                                "image_ref": request.image_ref,
                                "seed": request.seed,
                            },
                            sort_keys=True,
                        ).encode("utf-8")
                    ).hexdigest()
                    _event(
                        event_logger,
                        "case_started",
                        case_id=case.case_id,
                        dimension=case.dimension,
                        formal=case.formal,
                        seed=seed,
                        repeat_index=repeat_index,
                        input_hash=input_hash,
                        gpu=gpu_snapshot(),
                    )
                    try:
                        result = adapter.infer(request, effective_run_id)
                        completed_result = replace(
                            result,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            git_commit=git_commit,
                            config_hash=config_hash,
                            device=config.model.device,
                            input_hash=input_hash,
                            metadata={
                                **dict(result.metadata),
                                "dimension": case.dimension,
                                "formal": case.formal,
                                "repeat_index": repeat_index,
                            },
                        )
                        results.append(completed_result)
                        if result_callback is not None:
                            result_callback(completed_result)
                        _event(
                            event_logger,
                            "case_completed",
                            case_id=case.case_id,
                            dimension=case.dimension,
                            seed=seed,
                            repeat_index=repeat_index,
                            duration_ms=completed_result.duration_ms,
                            metrics=dict(completed_result.metrics),
                            output_characters=len(completed_result.output_text),
                            output_sha256=hashlib.sha256(
                                completed_result.output_text.encode("utf-8")
                            ).hexdigest(),
                            gpu=gpu_snapshot(),
                        )
                    except Exception as exc:
                        if event_logger is not None:
                            event_logger.exception(
                                "case_failed",
                                exc,
                                case_id=case.case_id,
                                dimension=case.dimension,
                                seed=seed,
                                repeat_index=repeat_index,
                                input_hash=input_hash,
                                gpu=gpu_snapshot(),
                            )
                        raise
    finally:
        close_started = time.perf_counter()
        _event(event_logger, "adapter_close_started")
        try:
            adapter.close()
            _event(
                event_logger,
                "adapter_close_completed",
                close_duration_ms=round(
                    (time.perf_counter() - close_started) * 1000, 3
                ),
                gpu=gpu_snapshot(),
            )
        except Exception as exc:
            if event_logger is not None:
                event_logger.exception("adapter_close_failed", exc)
            raise
    return results


def write_results(path: str | Path, results: Iterable[InferenceResult]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for result in results:
            stream.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


def initialize_result_file(path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.flush()
        os.fsync(stream.fileno())


def append_result(path: str | Path, result: InferenceResult) -> None:
    """Durably persist one result so interruptions do not lose prior cases."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git_commit() -> str:
    override = os.environ.get("MMSEP_SOURCE_REVISION", "").strip()
    if override:
        return override
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--log",
        help="structured JSONL log path; defaults to <output>.runlog.jsonl",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = uuid.uuid4().hex
    log_path = Path(args.log) if args.log else Path(f"{args.output}.runlog.jsonl")
    with StructuredRunLogger(log_path, run_id) as event_logger:
        event_logger.event(
            "run_started",
            message="MMSep experiment runner started",
            config_path=str(Path(args.config).resolve()),
            cases_path=str(Path(args.cases).resolve()),
            output_path=str(Path(args.output).resolve()),
            log_path=str(log_path.resolve()),
        )
        event_logger.event(
            "runtime_snapshot",
            runtime=runtime_snapshot(),
            gpu=gpu_snapshot(),
            disk=disk_snapshot(log_path.parent),
        )
        try:
            config = load_project_config(args.config)
            cases = load_cases(args.cases)
            event_logger.event(
                "input_files_verified",
                config_sha256=file_sha256(args.config),
                cases_sha256=file_sha256(args.cases),
                case_count=len(cases),
            )
            initialize_result_file(args.output)
            results = execute(
                config,
                cases,
                run_id=run_id,
                event_logger=event_logger,
                result_callback=lambda result: append_result(args.output, result),
            )
        except KeyboardInterrupt as exc:
            event_logger.exception(
                "run_interrupted",
                exc,
                completed_results=_count_result_rows(args.output),
            )
            print(
                json.dumps(
                    {"ok": False, "error": "interrupted", "log": str(log_path)},
                    ensure_ascii=False,
                )
            )
            return 130
        except (ConfigError, OSError, RuntimeError, ValueError, NotImplementedError) as exc:
            event_logger.exception(
                "run_failed",
                exc,
                completed_results=_count_result_rows(args.output),
            )
            print(
                json.dumps(
                    {"ok": False, "error": str(exc), "log": str(log_path)},
                    ensure_ascii=False,
                )
            )
            return 2
        except Exception as exc:
            event_logger.exception(
                "run_failed_unexpectedly",
                exc,
                completed_results=_count_result_rows(args.output),
            )
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": f"unexpected {type(exc).__name__}: {exc}",
                        "log": str(log_path),
                    },
                    ensure_ascii=False,
                )
            )
            return 1
        event_logger.event(
            "run_completed",
            message="All requested cases completed",
            results=len(results),
            output_path=str(Path(args.output).resolve()),
            output_sha256=file_sha256(args.output),
            disk=disk_snapshot(log_path.parent),
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "run_id": run_id,
                    "results": len(results),
                    "output": args.output,
                    "log": str(log_path),
                }
            )
        )
        return 0


def _event(
    logger: StructuredRunLogger | None,
    event: str,
    *,
    level: str = "INFO",
    **data: object,
) -> None:
    if logger is not None:
        logger.event(event, level=level, **data)


def _count_result_rows(path: str | Path) -> int:
    output_path = Path(path)
    if not output_path.is_file():
        return 0
    try:
        return sum(1 for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
