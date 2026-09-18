"""Visual-separator selection helpers for the local MMSep integration."""

from __future__ import annotations

import math

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def rank_visual_separator_positions(
    attention_module,
    hidden_states,
    position_ids,
    image_start: int,
    image_token_length: int,
    keep_count: int,
):
    """Return absolute positions of the most relevant visual tokens.

    MMSep ranks visual keys using the attention paid by the final prompt token.
    This side-effect-free helper is kept separate from the upstream
    ``mm_separators.py`` source file.
    """
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError("MMSep currently supports batch_size=1")
    if image_start < 0 or image_token_length <= 0:
        raise ValueError("image_start and image_token_length must be positive")
    image_end = image_start + image_token_length
    if image_end > hidden_states.shape[1]:
        raise ValueError("visual token range exceeds the input sequence")

    keep_count = max(1, min(int(keep_count), image_token_length))
    num_heads = attention_module.num_heads
    num_key_value_heads = attention_module.num_key_value_heads
    head_dim = attention_module.head_dim
    query_states = attention_module.q_proj(hidden_states)
    key_states = attention_module.k_proj(hidden_states)
    query_states = query_states.view(1, -1, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(1, -1, num_key_value_heads, head_dim).transpose(1, 2)

    if hasattr(attention_module, "rotary_emb") and position_ids is not None:
        rotary_length = max(hidden_states.shape[1], int(position_ids.max().item()) + 1)
        cos, sin = attention_module.rotary_emb(key_states, seq_len=rotary_length)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

    if num_key_value_heads != num_heads:
        if num_heads % num_key_value_heads != 0:
            raise ValueError("num_heads must be divisible by num_key_value_heads")
        key_states = key_states.repeat_interleave(num_heads // num_key_value_heads, dim=1)

    query = query_states[0, :, -1:, :]
    visual_keys = key_states[0, :, image_start:image_end, :]
    scores = torch.matmul(query, visual_keys.transpose(1, 2)) / math.sqrt(head_dim)
    scores = torch.softmax(scores, dim=-1).mean(dim=0).squeeze(0)
    relative = torch.topk(scores, k=keep_count, largest=True, sorted=False).indices
    return (relative.sort().values + image_start).to(dtype=torch.long)
