"""Stable case and result contracts used by all adapters and runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


VALID_DIMENSIONS = {
    "functional",
    "robustness",
    "security",
    "fairness",
    "performance",
}


@dataclass(frozen=True)
class TestCase:
    case_id: str
    title: str
    dimension: str
    input_text: str
    image_ref: str | None = None
    formal: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TestCase":
        case_id = _text(raw.get("case_id"), "case_id")
        title = _text(raw.get("title"), "title")
        dimension = _text(raw.get("dimension"), "dimension")
        if dimension not in VALID_DIMENSIONS:
            raise ValueError(f"unsupported dimension: {dimension}")
        input_raw = raw.get("input")
        if not isinstance(input_raw, Mapping):
            raise ValueError("input must be an object")
        image_ref = input_raw.get("image_ref")
        if image_ref is not None:
            image_ref = _text(image_ref, "input.image_ref")
        formal = raw.get("formal", True)
        if not isinstance(formal, bool):
            raise ValueError("formal must be a boolean")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        return cls(
            case_id=case_id,
            title=title,
            dimension=dimension,
            input_text=_text(input_raw.get("text"), "input.text"),
            image_ref=image_ref,
            formal=formal,
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class InferenceRequest:
    case_id: str
    text: str
    image_ref: str | None
    seed: int


@dataclass(frozen=True)
class InferenceResult:
    run_id: str
    case_id: str
    model_id: str
    model_revision: str
    mode: str
    seed: int
    output_text: str
    duration_ms: float
    metrics: Mapping[str, Any]
    timestamp: str = ""
    git_commit: str = "unknown"
    config_hash: str = ""
    device: str = ""
    input_hash: str = ""
    evidence_refs: tuple[str, ...] = ()
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()
