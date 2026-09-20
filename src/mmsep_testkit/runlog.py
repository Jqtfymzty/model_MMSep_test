"""Structured, secret-safe runtime logging for shared-server experiments."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


LOG_SCHEMA_VERSION = 1
_REDACTED = "[REDACTED]"
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "passwd",
    "private_key",
    "refresh_token",
    "secret",
    "token",
}
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\b(?:sk|ak)-[a-z0-9_-]{8,}"),
)
_PACKAGE_NAMES = (
    "torch",
    "torchvision",
    "transformers",
    "tokenizers",
    "accelerate",
    "bitsandbytes",
    "sentencepiece",
    "protobuf",
    "Pillow",
    "llava-torch",
)


class StructuredRunLogger:
    """Append durable JSONL events and concise progress messages."""

    def __init__(self, path: str | Path, run_id: str, *, console: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.console = console
        self._started = time.perf_counter()
        self._stream = self.path.open("a", encoding="utf-8", newline="\n")

    def event(self, event: str, *, level: str = "INFO", **data: Any) -> None:
        record = {
            "schema_version": LOG_SCHEMA_VERSION,
            "timestamp": _utc_now(),
            "elapsed_ms": round((time.perf_counter() - self._started) * 1000, 3),
            "level": level.upper(),
            "event": event,
            "run_id": self.run_id,
            "pid": os.getpid(),
            "data": _sanitize(data),
        }
        self._stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        if self.console:
            message = str(record["data"].get("message", "")).strip()
            suffix = f" - {message}" if message else ""
            print(
                f"[{record['timestamp']}] {record['level']} {event}{suffix}",
                file=sys.stderr,
                flush=True,
            )

    def exception(self, event: str, exc: BaseException, **data: Any) -> None:
        self.event(
            event,
            level="ERROR",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback="".join(traceback.format_exception(exc)),
            **data,
        )

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()

    def __enter__(self) -> "StructuredRunLogger":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def runtime_snapshot() -> dict[str, Any]:
    """Return a non-secret snapshot suitable for experiment evidence."""

    packages: dict[str, str] = {}
    for name in _PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "working_directory": str(Path.cwd()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "packages": packages,
    }


def gpu_snapshot() -> dict[str, Any]:
    """Query visible GPUs without exposing other users' process commands."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "error": "nvidia-smi not found", "gpus": []}
    command = [
        executable,
        "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,"
        "utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    selected = visible_gpu_identifiers()
    if selected == ():
        return {"available": False, "error": "CUDA_VISIBLE_DEVICES disables GPUs", "gpus": []}
    if selected:
        command.extend(["--id", ",".join(selected)])
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        rows = list(csv.reader(completed.stdout.strip().splitlines()))
        gpus = []
        for row in rows:
            if len(row) != 8:
                continue
            values = [item.strip() for item in row]
            gpus.append(
                {
                    "index": int(values[0]),
                    "name": values[1],
                    "driver_version": values[2],
                    "memory_total_mib": int(values[3]),
                    "memory_used_mib": int(values[4]),
                    "memory_free_mib": int(values[5]),
                    "utilization_percent": int(values[6]),
                    "temperature_c": int(values[7]),
                }
            )
        return {"available": bool(gpus), "error": None, "gpus": gpus}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"available": False, "error": str(exc), "gpus": []}


def disk_snapshot(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    usage = shutil.disk_usage(target if target.exists() else target.parent)
    return {
        "path": str(target),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visible_gpu_identifiers() -> tuple[str, ...] | None:
    """Return requested physical GPU identifiers, or None when unrestricted."""

    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None or not value.strip():
        return None
    identifiers = tuple(item.strip() for item in value.split(",") if item.strip())
    if identifiers == ("-1",):
        return ()
    return identifiers


def _sanitize(value: Any, key: str | None = None) -> Any:
    if key and key.lower() in _SENSITIVE_KEYS:
        return _REDACTED
    if isinstance(value, Mapping):
        return {str(item_key): _sanitize(item_value, str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        sanitized = value
        for pattern in _SECRET_PATTERNS:
            sanitized = pattern.sub(_REDACTED, sanitized)
        return sanitized
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return repr(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
