"""Interface that real Baseline and MMSep adapters must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from ..config import ProjectConfig
from ..contracts import InferenceRequest, InferenceResult


class AdapterError(RuntimeError):
    """Raised when an inference adapter cannot fulfill its contract."""


class InferenceAdapter(ABC):
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model resources and mark the adapter ready."""

    @abstractmethod
    def infer(self, request: InferenceRequest, run_id: str) -> InferenceResult:
        """Run one inference request and return structured evidence."""

    @abstractmethod
    def healthcheck(self) -> Mapping[str, Any]:
        """Return non-secret runtime health information."""

    def close(self) -> None:
        self._loaded = False

    def __enter__(self) -> "InferenceAdapter":
        self.load()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

