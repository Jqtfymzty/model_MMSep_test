"""Offline, traceable Baseline adapter for the official LLaVA v1.5 code."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from ..config import ProjectConfig
from ..contracts import InferenceRequest, InferenceResult
from .base import InferenceAdapter


class LlavaAdapter(InferenceAdapter):
    """Load LLaVA lazily so mock tests never require ML dependencies."""

    def __init__(self, config: ProjectConfig):
        super().__init__(config)
        self.model: Any = None
        self.tokenizer: Any = None
        self.image_processor: Any = None
        self.torch: Any = None
        self._constants: dict[str, Any] = {}

    def load(self) -> None:
        model_path = _env_directory(self.config.model.model_path_env)
        vision_env = self.config.model.vision_tower_path_env
        if vision_env is None:
            raise RuntimeError("vision_tower_path_env is missing from configuration")
        vision_path = _env_directory(vision_env)

        code_env = self.config.model.llava_code_path_env
        if code_env and os.environ.get(code_env):
            code_path = str(_env_directory(code_env))
            if code_path not in sys.path:
                sys.path.insert(0, code_path)

        # Imports stay local: the lightweight test harness works without torch.
        import torch
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.model.language_model.llava_llama import LlavaConfig

        model_config = LlavaConfig.from_pretrained(
            str(model_path), local_files_only=True
        )
        # The ModelScope snapshot does not bundle CLIP. This override prevents
        # an implicit Hugging Face download during a supposedly offline run.
        model_config.mm_vision_tower = str(vision_path)

        quantization = self.config.model.quantization
        device = _resolved_device(self.config.model.device, torch)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=str(model_path),
            model_base=None,
            model_name="llava-v1.5-7b",
            load_8bit=quantization == "8bit",
            load_4bit=quantization == "4bit",
            device=device,
            config=model_config,
        )
        self.model.eval()
        self.torch = torch
        self._constants = {
            "default_image_token": DEFAULT_IMAGE_TOKEN,
            "image_token_index": IMAGE_TOKEN_INDEX,
            "conv_templates": conv_templates,
            "process_images": process_images,
            "tokenizer_image_token": tokenizer_image_token,
        }
        self._loaded = True

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        self._loaded = False

    def healthcheck(self) -> dict[str, Any]:
        return {
            "loaded": self._loaded,
            "backend": "llava",
            "mode": self.config.experiment.mode,
            "device": str(getattr(self.model, "device", "not-loaded")),
        }

    def infer(self, request: InferenceRequest, run_id: str) -> InferenceResult:
        if self.model is None or self.torch is None:
            raise RuntimeError("LLaVA adapter must be entered before inference")
        if not request.image_ref:
            raise ValueError("LLaVA Baseline cases must provide input.image_ref")

        image_path = Path(request.image_ref).expanduser()
        if not image_path.is_file():
            raise FileNotFoundError(f"input image not found: {image_path}")

        from PIL import Image

        torch = self.torch
        torch.manual_seed(request.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request.seed)
            torch.cuda.reset_peak_memory_stats()

        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image_size = image.size
            image_tensor = self._constants["process_images"](
                [image], self.image_processor, self.model.config
            )

        prompt = self._build_prompt(request.text)
        input_ids = self._constants["tokenizer_image_token"](
            prompt,
            self.tokenizer,
            self._constants["image_token_index"],
            return_tensors="pt",
        ).unsqueeze(0)
        input_ids = input_ids.to(self.model.device)
        if isinstance(image_tensor, list):
            image_tensor = [
                item.to(self.model.device, dtype=torch.float16)
                for item in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        generation = self.config.generation
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=generation.do_sample,
                temperature=(generation.temperature if generation.do_sample else None),
                num_beams=generation.num_beams,
                max_new_tokens=generation.max_new_tokens,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration_ms = (time.perf_counter() - started) * 1000
        # Current LLaVA releases call Hugging Face generation with
        # ``inputs_embeds`` and therefore return generated token IDs only.
        # Older releases may return ``prompt + generated`` IDs.  Strip the
        # prompt only when it is actually present; slicing unconditionally
        # turns valid short answers into an empty tensor.
        output_length = int(output_ids.shape[1])
        input_length = int(input_ids.shape[1])
        prefix_matches = (
            output_length >= input_length
            and torch.equal(output_ids[:, :input_length], input_ids)
        )
        generated_start = _generated_token_start(
            output_length=output_length,
            input_length=input_length,
            prefix_matches=prefix_matches,
        )
        new_tokens = output_ids[:, generated_start:]
        output_text = self.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0].strip()
        peak_vram = (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else 0
        )
        return InferenceResult(
            run_id=run_id,
            case_id=request.case_id,
            model_id=self.config.model.canonical_model_id,
            model_revision=self.config.model.model_revision,
            mode=self.config.experiment.mode,
            seed=request.seed,
            output_text=output_text,
            duration_ms=duration_ms,
            metrics={
                "prompt_tokens": int(input_ids.shape[1]),
                "output_tokens": int(new_tokens.shape[1]),
                "peak_vram_bytes": peak_vram,
            },
            evidence_refs=(str(image_path),),
            metadata={"adapter": "official-llava", "offline": True},
        )

    def _build_prompt(self, text: str) -> str:
        question = self._constants["default_image_token"] + "\n" + text
        conversation = self._constants["conv_templates"]["llava_v1"].copy()
        conversation.append_message(conversation.roles[0], question)
        conversation.append_message(conversation.roles[1], None)
        return conversation.get_prompt()


def _env_directory(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"environment variable {name} is not set")
    path = Path(value).expanduser()
    if not path.is_dir():
        raise RuntimeError(f"{name} does not point to a directory: {path}")
    return path


def _resolved_device(configured: str, torch: Any) -> str:
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return configured


def _generated_token_start(
    *, output_length: int, input_length: int, prefix_matches: bool
) -> int:
    """Return the first generated-token index for old and new LLaVA APIs."""

    if output_length < 0 or input_length < 0:
        raise ValueError("token lengths must be non-negative")
    if prefix_matches and output_length >= input_length:
        return input_length
    return 0
