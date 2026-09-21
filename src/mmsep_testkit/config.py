"""Validated configuration shared by Baseline and MMSep execution modes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when a project configuration violates the shared contract."""


VALID_BACKENDS = {"mock", "llava"}
VALID_MODES = {"baseline", "mmsep"}
VALID_QUANTIZATION = {"none", "4bit", "8bit", "fp16", "bf16", "fp32"}


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be an object")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True)
class ModelConfig:
    backend: str
    canonical_model_id: str
    model_revision: str
    model_path_env: str
    vision_tower_path_env: str | None
    llava_code_path_env: str | None
    device: str
    quantization: str
    mmsep_enabled: bool
    download_source: str | None = None


@dataclass(frozen=True)
class GenerationConfig:
    do_sample: bool
    temperature: float
    num_beams: int
    max_new_tokens: int


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str
    seeds: tuple[int, ...]
    repetitions: int


@dataclass(frozen=True)
class ProjectConfig:
    schema_version: int
    model: ModelConfig
    generation: GenerationConfig
    experiment: ExperimentConfig

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model": {
                "backend": self.model.backend,
                "canonical_model_id": self.model.canonical_model_id,
                "download_source": self.model.download_source,
                "model_revision": self.model.model_revision,
                "model_path_env": self.model.model_path_env,
                "vision_tower_path_env": self.model.vision_tower_path_env,
                "llava_code_path_env": self.model.llava_code_path_env,
                "device": self.model.device,
                "quantization": self.model.quantization,
                "mmsep": {"enabled": self.model.mmsep_enabled},
            },
            "generation": {
                "do_sample": self.generation.do_sample,
                "temperature": self.generation.temperature,
                "num_beams": self.generation.num_beams,
                "max_new_tokens": self.generation.max_new_tokens,
            },
            "experiment": {
                "mode": self.experiment.mode,
                "seeds": list(self.experiment.seeds),
                "repetitions": self.experiment.repetitions,
            },
        }


def load_project_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {config_path}: {exc}") from exc

    root = _mapping(raw, "root")
    schema_version = root.get("schema_version")
    if schema_version != 1:
        raise ConfigError("schema_version must be 1")

    model_raw = _mapping(root.get("model"), "model")
    backend = model_raw.get("backend")
    if backend not in VALID_BACKENDS:
        raise ConfigError(f"model.backend must be one of {sorted(VALID_BACKENDS)}")

    quantization = model_raw.get("quantization")
    if quantization not in VALID_QUANTIZATION:
        raise ConfigError(
            f"model.quantization must be one of {sorted(VALID_QUANTIZATION)}"
        )

    mmsep_raw = _mapping(model_raw.get("mmsep"), "model.mmsep")
    mmsep_enabled = mmsep_raw.get("enabled")
    if not isinstance(mmsep_enabled, bool):
        raise ConfigError("model.mmsep.enabled must be a boolean")

    model = ModelConfig(
        backend=backend,
        canonical_model_id=_required_text(
            model_raw.get("canonical_model_id"), "model.canonical_model_id"
        ),
        download_source=_optional_text(
            model_raw.get("download_source"), "model.download_source"
        ),
        model_revision=_required_text(
            model_raw.get("model_revision"), "model.model_revision"
        ),
        model_path_env=_required_text(
            model_raw.get("model_path_env"), "model.model_path_env"
        ),
        vision_tower_path_env=_optional_text(
            model_raw.get("vision_tower_path_env"),
            "model.vision_tower_path_env",
        ),
        llava_code_path_env=_optional_text(
            model_raw.get("llava_code_path_env"),
            "model.llava_code_path_env",
        ),
        device=_required_text(model_raw.get("device"), "model.device"),
        quantization=quantization,
        mmsep_enabled=mmsep_enabled,
    )
    if backend == "llava" and model.vision_tower_path_env is None:
        raise ConfigError(
            "model.vision_tower_path_env is required for the llava backend"
        )

    generation_raw = _mapping(root.get("generation"), "generation")
    do_sample = generation_raw.get("do_sample")
    if not isinstance(do_sample, bool):
        raise ConfigError("generation.do_sample must be a boolean")
    temperature = generation_raw.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ConfigError("generation.temperature must be numeric")
    temperature = float(temperature)
    if temperature < 0:
        raise ConfigError("generation.temperature cannot be negative")
    if do_sample and temperature == 0:
        raise ConfigError("sampled decoding requires temperature > 0")

    generation = GenerationConfig(
        do_sample=do_sample,
        temperature=temperature,
        num_beams=_positive_int(generation_raw.get("num_beams"), "generation.num_beams"),
        max_new_tokens=_positive_int(
            generation_raw.get("max_new_tokens"), "generation.max_new_tokens"
        ),
    )

    experiment_raw = _mapping(root.get("experiment"), "experiment")
    mode = experiment_raw.get("mode")
    if mode not in VALID_MODES:
        raise ConfigError(f"experiment.mode must be one of {sorted(VALID_MODES)}")
    expected_mmsep = mode == "mmsep"
    if mmsep_enabled != expected_mmsep:
        raise ConfigError(
            "experiment.mode and model.mmsep.enabled disagree; the unified switch "
            "must be the only mode difference"
        )

    seeds_raw = experiment_raw.get("seeds")
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ConfigError("experiment.seeds must be a non-empty list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds_raw):
        raise ConfigError("experiment.seeds must contain integers only")

    experiment = ExperimentConfig(
        mode=mode,
        seeds=tuple(seeds_raw),
        repetitions=_positive_int(
            experiment_raw.get("repetitions"), "experiment.repetitions"
        ),
    )
    return ProjectConfig(
        schema_version=schema_version,
        model=model,
        generation=generation,
        experiment=experiment,
    )


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field)
