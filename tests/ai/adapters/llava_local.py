"""Small local adapter shared by code-based AI tests."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class LocalLlavaAdapter:
    """Load local LLaVA once and expose deterministic single-image inference."""

    def __init__(self, model_path: Path, vision_model_path: Path) -> None:
        self.model_path = model_path.resolve()
        self.vision_model_path = vision_model_path.resolve()
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        import torch
        from llava.model.builder import load_pretrained_model
        from llava.model.language_model.llava_llama import LlavaConfig
        from llava.utils import disable_torch_init

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch 未检测到 CUDA，无法用 RTX 3060 加载 7B 模型")
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"模型目录不存在：{self.model_path}")
        if not self.vision_model_path.is_dir():
            raise FileNotFoundError(f"CLIP 目录不存在：{self.vision_model_path}")

        disable_torch_init()
        with (self.model_path / "config.json").open("r", encoding="utf-8") as stream:
            model_config = json.load(stream)
        model_name = str(model_config.get("_name_or_path", "llava-v1.5-7b"))

        config = LlavaConfig.from_pretrained(str(self.model_path))
        config.mm_vision_tower = str(self.vision_model_path)

        self.tokenizer, self.model, self.image_processor, self.context_length = (
            load_pretrained_model(
                str(self.model_path),
                None,
                model_name,
                load_4bit=True,
                device_map="auto",
                config=config,
            )
        )
        self.torch = torch
        self._loaded = True

    def infer(
        self,
        image_path: Path,
        prompt: str,
        *,
        seed: int,
        max_new_tokens: int = 24,
    ) -> dict[str, Any]:
        self.load()

        from PIL import Image
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        torch = self.torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        image = Image.open(image_path).convert("RGB")
        image_sizes = [image.size]
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        if self.model.config.mm_use_im_start_end:
            image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        else:
            image_token = DEFAULT_IMAGE_TOKEN

        conversation = conv_templates["llava_v1"].copy()
        conversation.append_message(conversation.roles[0], image_token + "\n" + prompt)
        conversation.append_message(conversation.roles[1], None)
        full_prompt = conversation.get_prompt()
        input_ids = tokenizer_image_token(
            full_prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.model.device)

        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA 显存不足；请关闭占用显存的软件或减小 --max-new-tokens"
            ) from exc

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        peak_memory_gib = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return {
            "output": output,
            "duration_ms": duration_ms,
            "peak_memory_gib": peak_memory_gib,
            "seed": seed,
        }

    def close(self) -> None:
        if not self._loaded:
            return
        del self.model
        self.torch.cuda.empty_cache()
        self._loaded = False
