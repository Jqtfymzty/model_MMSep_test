"""Deterministic backend for developing the runner without model weights."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Mapping

from .base import AdapterError, InferenceAdapter
from ..contracts import InferenceRequest, InferenceResult


class MockAdapter(InferenceAdapter):
    def load(self) -> None:
        self._loaded = True

    def healthcheck(self) -> Mapping[str, Any]:
        return {
            "ok": self._loaded,
            "backend": "mock",
            "mode": self.config.experiment.mode,
        }

    def infer(self, request: InferenceRequest, run_id: str) -> InferenceResult:
        if not self._loaded:
            raise AdapterError("adapter must be loaded before inference")
        start = time.perf_counter()
        digest_input = "|".join(
            [
                request.case_id,
                request.text,
                request.image_ref or "",
                str(request.seed),
                self.config.experiment.mode,
            ]
        )
        digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:16]
        output_text = f"mock-response:{digest}"
        duration_ms = (time.perf_counter() - start) * 1000
        return InferenceResult(
            run_id=run_id,
            case_id=request.case_id,
            model_id=self.config.model.canonical_model_id,
            model_revision=self.config.model.model_revision,
            mode=self.config.experiment.mode,
            seed=request.seed,
            output_text=output_text,
            duration_ms=duration_ms,
            metrics={"mock": True, "output_sha256_prefix": digest},
            metadata={"backend": "mock"},
        )
