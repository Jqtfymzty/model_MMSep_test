"""Portable one-command entry point for the complete module-two experiment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def required_path(cli_value: str | None, environment_name: str) -> Path:
    value = cli_value or os.environ.get(environment_name)
    if not value:
        raise SystemExit(f"请通过参数或环境变量 {environment_name} 提供路径")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"目录不存在：{path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all 15 module-two AI cases")
    parser.add_argument("--model-path")
    parser.add_argument("--vision-model-path")
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--model-revision", default=os.environ.get("MODEL_REVISION", "local"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--case-id", action="append")
    args = parser.parse_args()

    model_path = required_path(args.model_path, "LLAVA_MODEL_PATH")
    vision_path = required_path(args.vision_model_path, "LLAVA_VISION_TOWER_PATH")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("module2-%Y%m%dT%H%M%SZ")
    runs_root = (args.runs_root or Path(os.environ.get("MMSEP_RUNS_ROOT", PROJECT_ROOT / "artifacts" / "runs"))).resolve()
    output_dir = runs_root / run_id

    command = [
        sys.executable,
        str(PROJECT_ROOT / "tests" / "ai" / "runners" / "run_all_cases.py"),
        "--model-path",
        str(model_path),
        "--vision-model-path",
        str(vision_path),
        "--output-dir",
        str(output_dir),
        "--run-id",
        run_id,
        "--model-revision",
        args.model_revision,
        "--mode",
        "both",
    ]
    if args.quick:
        command.append("--quick")
    for case_id in args.case_id or []:
        command.extend(("--case-id", case_id))

    print(f"Source root: {PROJECT_ROOT}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
