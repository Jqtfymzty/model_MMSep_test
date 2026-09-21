"""Runtime integration of MMSep visual separators with Transformers 4.36.

The model weights are unchanged. During prefill, a configured decoder layer
ranks the image tokens. KV caches at that layer and later layers retain only
the selected visual separators, while the current prefill attention still sees
the complete prompt. Decoding then attends to the compressed cache.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
from transformers.cache_utils import Cache

from .mmsep_selectors import rank_visual_separator_positions


@dataclass(frozen=True)
class MMSepConfig:
    enabled: bool = True
    layer: int = 16
    visual_keep_ratio: float = 0.5 / math.e

    def validate(self, layer_count: int, image_token_length: int) -> None:
        if not 0 <= self.layer < layer_count:
            raise ValueError(f"mmsep layer must be in [0, {layer_count - 1}]")
        if not 0.0 < self.visual_keep_ratio <= 1.0:
            raise ValueError("visual_keep_ratio must be in (0, 1]")
        if image_token_length <= 0:
            raise ValueError("image_token_length must be positive")


class MMSepGenerationCache(Cache):
    """Transformers-compatible cache with MMSep visual-token compression."""

    def __init__(
        self,
        *,
        layer_count: int,
        image_start: int,
        image_token_length: int,
        mmsep_layer: int,
    ) -> None:
        self.layer_count = int(layer_count)
        self.image_start = int(image_start)
        self.image_token_length = int(image_token_length)
        self.mmsep_layer = int(mmsep_layer)
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self.visual_separator_positions: torch.Tensor | None = None
        self.original_prefill_tokens: int | None = None
        self._seen_tokens = 0

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    def get_max_length(self) -> None:
        return None

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        layer_idx = 0 if layer_idx is None else int(layer_idx)
        if layer_idx >= len(self.key_cache):
            return 0
        return int(self.key_cache[layer_idx].shape[-2])

    def get_usable_length(self, new_seq_length: int, layer_idx: int | None = 0) -> int:
        del new_seq_length
        return self.get_seq_length(layer_idx)

    def set_visual_separator_positions(self, positions: torch.Tensor) -> None:
        positions = positions.detach().to(dtype=torch.long)
        image_end = self.image_start + self.image_token_length
        if positions.ndim != 1 or positions.numel() == 0:
            raise ValueError("visual separator positions must be a non-empty 1D tensor")
        if int(positions.min()) < self.image_start or int(positions.max()) >= image_end:
            raise ValueError("visual separator position is outside the image-token range")
        self.visual_separator_positions = positions.sort().values

    def _compression_indices(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        if self.visual_separator_positions is None:
            return torch.arange(sequence_length, device=device)
        image_end = self.image_start + self.image_token_length
        prefix = torch.arange(0, self.image_start, device=device)
        selected = self.visual_separator_positions.to(device=device)
        suffix = torch.arange(image_end, sequence_length, device=device)
        return torch.cat((prefix, selected, suffix)).to(dtype=torch.long)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        layer_idx = int(layer_idx)
        is_prefill = layer_idx >= len(self.key_cache)

        if layer_idx == 0:
            self._seen_tokens += int(key_states.shape[-2])
            if is_prefill:
                self.original_prefill_tokens = int(key_states.shape[-2])

        if is_prefill:
            if layer_idx != len(self.key_cache):
                raise RuntimeError("MMSep cache layers must be filled in order")
            stored_keys = key_states
            stored_values = value_states
            if layer_idx >= self.mmsep_layer and self.visual_separator_positions is not None:
                indices = self._compression_indices(key_states.shape[-2], key_states.device)
                stored_keys = key_states.index_select(-2, indices)
                stored_values = value_states.index_select(-2, indices)
            self.key_cache.append(stored_keys)
            self.value_cache.append(stored_values)
            # Prefill attention must still see every prompt and image token.
            return key_states, value_states

        self.key_cache[layer_idx] = torch.cat((self.key_cache[layer_idx], key_states), dim=-2)
        self.value_cache[layer_idx] = torch.cat((self.value_cache[layer_idx], value_states), dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def select_attention_mask(self, attention_mask: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if (
            layer_idx < self.mmsep_layer
            or self.visual_separator_positions is None
            or attention_mask.shape[-1] == self.get_seq_length(layer_idx) + attention_mask.shape[-2]
        ):
            return attention_mask
        indices = self._compression_indices(attention_mask.shape[-1], attention_mask.device)
        return attention_mask.index_select(-1, indices)

    def stats(self) -> dict[str, int | float | None]:
        selected = (
            int(self.visual_separator_positions.numel())
            if self.visual_separator_positions is not None
            else None
        )
        original = self.original_prefill_tokens
        effective = None
        ratio = None
        if original is not None and selected is not None:
            effective = original - self.image_token_length + selected
            ratio = round(effective / original, 6)
        return {
            "image_tokens": self.image_token_length,
            "visual_separators": selected,
            "original_prefill_tokens": original,
            "effective_prefill_tokens": effective,
            "effective_kv_ratio": ratio,
        }


class MMSepRuntime:
    """Install reversible MMSep hooks on one loaded LLaVA model."""

    def __init__(self, model: Any, config: MMSepConfig) -> None:
        self.model = model
        self.config = config
        self.core = model.get_model()
        self._installed = False
        self._original_attention_forwards: list[Any] = []
        self._original_prepare_inputs_for_generation: Any = None

    def install(self) -> None:
        if self._installed:
            return
        if getattr(self.core, "_use_flash_attention_2", False):
            raise RuntimeError("The local MMSep integration currently requires eager or SDPA attention")

        layers = self.core.layers
        self.config.validate(len(layers), 1)
        self._original_attention_forwards = [layer.self_attn.forward for layer in layers]

        for layer_index, layer in enumerate(layers):
            attention = layer.self_attn
            original_forward = attention.forward

            def patched_forward(
                attention_self,
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                *,
                _layer_index=layer_index,
                _original_forward=original_forward,
                **kwargs,
            ):
                if isinstance(past_key_value, MMSepGenerationCache):
                    is_prefill = hidden_states.shape[1] > 1 and past_key_value.get_seq_length(_layer_index) == 0
                    if (
                        is_prefill
                        and _layer_index == self.config.layer
                        and past_key_value.visual_separator_positions is None
                    ):
                        keep_count = max(
                            1,
                            round(
                                past_key_value.image_token_length
                                * self.config.visual_keep_ratio
                            ),
                        )
                        positions = rank_visual_separator_positions(
                            attention_self,
                            hidden_states,
                            position_ids,
                            past_key_value.image_start,
                            past_key_value.image_token_length,
                            keep_count,
                        )
                        past_key_value.set_visual_separator_positions(positions)

                    if attention_mask is not None and not is_prefill:
                        attention_mask = past_key_value.select_attention_mask(
                            attention_mask, _layer_index
                        )

                original_rotary_forward = None
                if (
                    isinstance(past_key_value, MMSepGenerationCache)
                    and not is_prefill
                    and position_ids is not None
                ):
                    # Compressed caches contain fewer entries, but their keys
                    # retain original absolute RoPE positions. Transformers
                    # normally sizes the RoPE table from the compact KV length,
                    # which can be smaller than the next absolute position.
                    original_rotary_forward = attention_self.rotary_emb.forward
                    minimum_length = int(position_ids.max().item()) + 1

                    def expanded_rotary_forward(
                        rotary_self,
                        values,
                        seq_len=None,
                        *,
                        _original=original_rotary_forward,
                        _minimum=minimum_length,
                        **rotary_kwargs,
                    ):
                        requested = _minimum if seq_len is None else max(int(seq_len), _minimum)
                        return _original(values, seq_len=requested, **rotary_kwargs)

                    attention_self.rotary_emb.forward = MethodType(
                        expanded_rotary_forward, attention_self.rotary_emb
                    )

                try:
                    return _original_forward(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        **kwargs,
                    )
                finally:
                    if original_rotary_forward is not None:
                        attention_self.rotary_emb.forward = original_rotary_forward

            attention.forward = MethodType(patched_forward, attention)

        self._original_prepare_inputs_for_generation = self.model.prepare_inputs_for_generation

        def patched_prepare_inputs_for_generation(
            model_self,
            input_ids,
            past_key_values=None,
            inputs_embeds=None,
            **kwargs,
        ):
            if isinstance(past_key_values, MMSepGenerationCache) and not past_key_values.key_cache:
                # LLaVA prepares the expanded image/text embeddings before it
                # enters GenerationMixin. An empty custom cache is still a
                # non-None object, so the stock LLaMA helper would otherwise
                # discard these first-step embeddings.
                model_inputs = {
                    "inputs_embeds": inputs_embeds,
                    "position_ids": kwargs.get("position_ids"),
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache", True),
                    "attention_mask": kwargs.get("attention_mask"),
                }
                return model_inputs

            if not isinstance(past_key_values, MMSepGenerationCache):
                return self._original_prepare_inputs_for_generation(
                    input_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    **kwargs,
                )

            attention_mask = kwargs.get("attention_mask")
            position_ids = kwargs.get("position_ids")
            input_ids = input_ids[:, -1:]
            if attention_mask is not None and position_ids is None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids[:, -1:]
            model_inputs = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
                "attention_mask": attention_mask,
            }
            if kwargs.get("images") is not None:
                model_inputs["images"] = kwargs["images"]
            if kwargs.get("image_sizes") is not None:
                model_inputs["image_sizes"] = kwargs["image_sizes"]
            return model_inputs

        self.model.prepare_inputs_for_generation = MethodType(
            patched_prepare_inputs_for_generation, self.model
        )
        self._installed = True

    def make_cache(self, *, image_start: int, image_token_length: int) -> MMSepGenerationCache:
        self.config.validate(len(self.core.layers), image_token_length)
        return MMSepGenerationCache(
            layer_count=len(self.core.layers),
            image_start=image_start,
            image_token_length=image_token_length,
            mmsep_layer=self.config.layer,
        )

    def uninstall(self) -> None:
        if not self._installed:
            return
        for layer, original in zip(self.core.layers, self._original_attention_forwards):
            layer.self_attn.forward = original
        self.model.prepare_inputs_for_generation = self._original_prepare_inputs_for_generation
        self._installed = False
