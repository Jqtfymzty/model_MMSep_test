from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mmsep_testkit.config import ConfigError, load_project_config
from mmsep_testkit.preflight import run_preflight


REPO_ROOT = Path(__file__).resolve().parents[2]
MOCK_CONFIG = REPO_ROOT / "configs" / "experiments" / "smoke.mock.json"
BASELINE_CONFIG = REPO_ROOT / "configs" / "experiments" / "baseline.llava.example.json"
MMSEP_CONFIG = REPO_ROOT / "configs" / "experiments" / "mmsep.llava.example.json"


class ConfigTests(unittest.TestCase):
    def test_loads_valid_mock_config(self) -> None:
        config = load_project_config(MOCK_CONFIG)
        self.assertEqual(config.model.backend, "mock")
        self.assertEqual(config.experiment.mode, "baseline")
        self.assertFalse(config.model.mmsep_enabled)

    def test_rejects_disagreement_between_mode_and_switch(self) -> None:
        raw = json.loads(MOCK_CONFIG.read_text(encoding="utf-8"))
        raw["experiment"]["mode"] = "mmsep"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "disagree"):
                load_project_config(path)

    def test_llava_preflight_requires_model_path_environment_variable(self) -> None:
        raw = json.loads(MOCK_CONFIG.read_text(encoding="utf-8"))
        raw["model"]["backend"] = "llava"
        raw["model"]["canonical_model_id"] = "liuhaotian/llava-v1.5-7b"
        raw["model"]["model_revision"] = "TO_BE_FROZEN_AFTER_DOWNLOAD"
        raw["model"]["vision_tower_path_env"] = "LLAVA_VISION_TOWER_PATH"
        raw["model"]["llava_code_path_env"] = "LLAVA_CODE_PATH"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llava.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            config = load_project_config(path)
        with patch.dict(os.environ, {}, clear=True):
            checks = run_preflight(config)
        self.assertTrue(any(c.name == "model_path_env" and not c.ok for c in checks))

    def test_llava_pair_differs_only_by_unified_switch(self) -> None:
        baseline = load_project_config(BASELINE_CONFIG).as_dict()
        mmsep = load_project_config(MMSEP_CONFIG).as_dict()
        baseline["model"]["mmsep"]["enabled"] = True
        baseline["experiment"]["mode"] = "mmsep"
        self.assertEqual(baseline, mmsep)


if __name__ == "__main__":
    unittest.main()
