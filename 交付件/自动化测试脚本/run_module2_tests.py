"""从交付件目录启动模块二全部 Baseline/MMSep 自动化测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = PROJECT_ROOT / "scripts" / "run_module2_experiment.py"
REQUIRED_FILES = (
    PROJECT_ROOT / "tests" / "ai" / "cases" / "lcx_cases.json",
    PROJECT_ROOT / "tests" / "ai" / "cases" / "wjn_cases.json",
    PROJECT_ROOT / "tests" / "ai" / "runners" / "run_all_cases.py",
)


def main() -> int:
    missing = [path for path in (ENTRYPOINT, *REQUIRED_FILES) if not path.is_file()]
    if missing:
        print("项目文件不完整，无法启动自动化测试：", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 2

    command = [sys.executable, str(ENTRYPOINT), *sys.argv[1:]]
    print("模块二自动化测试入口")
    print(f"项目根目录：{PROJECT_ROOT}")
    print("执行命令：" + subprocess.list2cmdline(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
