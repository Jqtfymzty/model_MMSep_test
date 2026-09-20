"""Model-independent preflight checks for mock and future LLaVA runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .config import ConfigError, ProjectConfig, load_project_config
from .runlog import visible_gpu_identifiers


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def run_preflight(config: ProjectConfig) -> list[Check]:
    checks = [
        Check("config_schema", True, "schema_version=1"),
        Check(
            "unified_mode_switch",
            config.model.mmsep_enabled == (config.experiment.mode == "mmsep"),
            f"mode={config.experiment.mode}, mmsep={config.model.mmsep_enabled}",
        ),
    ]
    if config.model.backend == "mock":
        checks.append(Check("mock_backend", True, "model weights are not required"))
        return checks

    path_text = os.environ.get(config.model.model_path_env)
    if not path_text:
        checks.append(
            Check(
                "model_path_env",
                False,
                f"environment variable {config.model.model_path_env} is not set",
            )
        )
        return checks

    model_path = Path(path_text).expanduser()
    checks.append(Check("model_directory", model_path.is_dir(), str(model_path)))
    if not model_path.is_dir():
        return checks

    checks.extend(
        [
            _required_file(model_path, "config.json"),
            _required_file(model_path, "pytorch_model.bin.index.json"),
            _one_of_files(
                model_path,
                "tokenizer",
                ("tokenizer.model", "tokenizer.json"),
            ),
            _weight_check(model_path),
            _model_fingerprint_check(model_path, config.model.model_revision),
        ]
    )
    revision_ok = config.model.model_revision != "TO_BE_FROZEN_AFTER_DOWNLOAD"
    checks.append(
        Check(
            "model_revision",
            revision_ok,
            config.model.model_revision
            if revision_ok
            else "freeze an exact revision before formal runs",
        )
    )
    vision_env = config.model.vision_tower_path_env
    vision_text = os.environ.get(vision_env) if vision_env else None
    if not vision_text:
        checks.append(
            Check(
                "vision_tower_path_env",
                False,
                f"environment variable {vision_env} is not set",
            )
        )
    else:
        vision_path = Path(vision_text).expanduser()
        checks.append(Check("vision_tower_directory", vision_path.is_dir(), str(vision_path)))
        if vision_path.is_dir():
            checks.extend(
                [
                    _required_file(vision_path, "config.json"),
                    _one_of_files(
                        vision_path,
                        "vision_preprocessor",
                        ("preprocessor_config.json", "processor_config.json"),
                    ),
                    _vision_weight_check(vision_path),
                ]
            )
    checks.extend(_runtime_checks(config))
    return checks


def _required_file(directory: Path, name: str) -> Check:
    path = directory / name
    return Check(name, path.is_file() and path.stat().st_size > 0, str(path))


def _one_of_files(directory: Path, name: str, candidates: Sequence[str]) -> Check:
    existing = [candidate for candidate in candidates if (directory / candidate).is_file()]
    return Check(name, bool(existing), ", ".join(existing) or f"missing: {candidates}")


def _weight_check(directory: Path) -> Check:
    patterns = ("*.safetensors", "pytorch_model*.bin")
    files = sorted({path for pattern in patterns for path in directory.glob(pattern)})
    total = sum(path.stat().st_size for path in files if path.is_file())
    ok = bool(files) and total >= 1_000_000_000
    detail = f"{len(files)} files, {total / 1_000_000_000:.2f} GB"
    if not files:
        detail = "no weight shards found"
    return Check("model_weights", ok, detail)


def _vision_weight_check(directory: Path) -> Check:
    patterns = ("*.safetensors", "pytorch_model*.bin")
    files = sorted({path for pattern in patterns for path in directory.glob(pattern)})
    total = sum(path.stat().st_size for path in files if path.is_file())
    return Check(
        "vision_tower_weights",
        bool(files) and total >= 500_000_000,
        f"{len(files)} files, {total / 1_000_000_000:.2f} GB",
    )


def _model_fingerprint_check(directory: Path, revision: str) -> Check:
    index_path = directory / "pytorch_model.bin.index.json"
    expected_marker = "index-sha256-"
    if expected_marker not in revision:
        return Check(
            "model_fingerprint",
            False,
            "model_revision must contain index-sha256-<digest>",
        )
    expected = revision.rsplit(expected_marker, 1)[1].lower()
    if not index_path.is_file():
        return Check("model_fingerprint", False, f"missing {index_path}")
    actual = hashlib.sha256(index_path.read_bytes()).hexdigest()
    return Check(
        "model_fingerprint",
        actual == expected,
        f"expected={expected}, actual={actual}",
    )


def _runtime_checks(config: ProjectConfig) -> list[Check]:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks: list[Check] = [
        Check(
            "python_version",
            sys.version_info[:2] == (3, 10),
            f"detected={version}, required=3.10.x",
        )
    ]
    for module in (
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "sentencepiece",
        "google.protobuf",
        "PIL",
    ):
        available = _module_available(module)
        checks.append(
            Check(
                f"dependency_{module.lower()}",
                available,
                "available" if available else "missing",
            )
        )

    llava_available = importlib.util.find_spec("llava") is not None
    code_env = config.model.llava_code_path_env
    code_value = os.environ.get(code_env) if code_env else None
    if not llava_available and code_value:
        llava_available = (Path(code_value).expanduser() / "llava").is_dir()
    checks.append(
        Check(
            "dependency_llava",
            llava_available,
            "installed/importable"
            if llava_available
            else f"install LLaVA or set {code_env} to its source checkout",
        )
    )
    checks.append(_gpu_memory_check(config.model.device, config.model.quantization))
    return checks


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _gpu_memory_check(device: str, quantization: str) -> Check:
    if device == "cpu":
        return Check(
            "gpu_memory",
            False,
            "CPU loading is disabled for this 7B checkpoint on a 16 GB host",
        )
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return Check("gpu_memory", False, "nvidia-smi is unavailable")
    command = [
        executable,
        "--query-gpu=memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    selected = visible_gpu_identifiers()
    if selected == ():
        return Check("gpu_memory", False, "CUDA_VISIBLE_DEVICES disables GPUs")
    if selected:
        command.extend(["--id", selected[0]])
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        first = completed.stdout.strip().splitlines()[0]
        total_mib, free_mib = (int(part.strip()) for part in first.split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        return Check("gpu_memory", False, f"cannot query CUDA memory: {exc}")
    minimum_mib = 6 * 1024 if quantization == "4bit" else 14 * 1024
    ok = total_mib >= minimum_mib and free_mib >= minimum_mib
    selected_detail = selected[0] if selected else "first-visible"
    return Check(
        "gpu_memory",
        ok,
        f"gpu={selected_detail}, total={total_mib} MiB, free={free_mib} MiB, "
        f"required={minimum_mib} MiB",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to experiment JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_project_config(args.config)
    except ConfigError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    checks = run_preflight(config)
    report = {
        "ok": all(check.ok for check in checks),
        "checks": [asdict(check) for check in checks],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
