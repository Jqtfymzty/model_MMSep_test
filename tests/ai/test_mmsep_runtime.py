"""CPU tests for the MMSep integration; the 7B model is not loaded here."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mmsep_selectors import rank_visual_separator_positions  # noqa: E402
from src.mmsep_runtime import MMSepConfig, MMSepGenerationCache  # noqa: E402

AI_ROOT = PROJECT_ROOT / "tests" / "ai"
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from runners.run_wjn_cases import compare_modes  # noqa: E402


class FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = 1
        self.num_key_value_heads = 1
        self.head_dim = 2
        self.q_proj = nn.Linear(2, 2, bias=False)
        self.k_proj = nn.Linear(2, 2, bias=False)
        self.q_proj.weight.data.copy_(torch.eye(2))
        self.k_proj.weight.data.copy_(torch.eye(2))


class VisualSeparatorTests(unittest.TestCase):
    def test_ranks_visual_tokens_from_last_prompt_query(self) -> None:
        hidden = torch.tensor(
            [[[0.0, 0.0], [0.0, 1.0], [3.0, 0.0], [2.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]]
        )
        selected = rank_visual_separator_positions(
            FakeAttention(),
            hidden,
            position_ids=None,
            image_start=1,
            image_token_length=4,
            keep_count=2,
        )
        self.assertEqual(selected.tolist(), [2, 3])

    def test_rejects_visual_range_outside_sequence(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            rank_visual_separator_positions(
                FakeAttention(),
                torch.zeros(1, 3, 2),
                position_ids=None,
                image_start=1,
                image_token_length=4,
                keep_count=1,
            )


class MMSepCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = MMSepGenerationCache(
            layer_count=4,
            image_start=2,
            image_token_length=4,
            mmsep_layer=2,
        )
        self.cache.set_visual_separator_positions(torch.tensor([2, 4]))

    @staticmethod
    def states(length: int) -> tuple[torch.Tensor, torch.Tensor]:
        keys = torch.arange(length, dtype=torch.float32).view(1, 1, length, 1)
        return keys, keys + 100

    def test_prefill_keeps_full_attention_but_compresses_later_layer_cache(self) -> None:
        keys, values = self.states(8)
        for layer_idx in range(4):
            attention_keys, attention_values = self.cache.update(keys, values, layer_idx)
            self.assertEqual(attention_keys.shape[-2], 8)
            self.assertEqual(attention_values.shape[-2], 8)

        self.assertEqual(self.cache.get_seq_length(1), 8)
        self.assertEqual(self.cache.get_seq_length(2), 6)
        self.assertEqual(self.cache.key_cache[2].flatten().tolist(), [0, 1, 2, 4, 6, 7])

    def test_decode_appends_to_compressed_cache(self) -> None:
        keys, values = self.states(8)
        for layer_idx in range(4):
            self.cache.update(keys, values, layer_idx)
        new_key, new_value = self.states(1)
        returned_keys, returned_values = self.cache.update(new_key, new_value, 2)
        self.assertEqual(returned_keys.shape[-2], 7)
        self.assertEqual(returned_values.shape[-2], 7)

    def test_attention_mask_uses_the_same_visual_positions(self) -> None:
        keys, values = self.states(8)
        for layer_idx in range(4):
            self.cache.update(keys, values, layer_idx)
        full_mask = torch.arange(9, dtype=torch.float32).view(1, 1, 1, 9)
        selected = self.cache.select_attention_mask(full_mask, layer_idx=2)
        self.assertEqual(selected.flatten().tolist(), [0, 1, 2, 4, 6, 7, 8])

    def test_stats_report_effective_kv_ratio(self) -> None:
        keys, values = self.states(8)
        self.cache.update(keys, values, 0)
        stats = self.cache.stats()
        self.assertEqual(stats["visual_separators"], 2)
        self.assertEqual(stats["effective_prefill_tokens"], 6)
        self.assertEqual(stats["effective_kv_ratio"], 0.75)


class MMSepConfigTests(unittest.TestCase):
    def test_rejects_invalid_layer(self) -> None:
        with self.assertRaisesRegex(ValueError, "layer"):
            MMSepConfig(layer=4).validate(layer_count=4, image_token_length=576)


class ModeComparisonTests(unittest.TestCase):
    def test_marks_mmsep_failure_as_regression(self) -> None:
        summaries = {
            "baseline": [{"case_id": "M2-ROB-101", "verdict": "PASS"}],
            "mmsep": [{"case_id": "M2-ROB-101", "verdict": "FAIL"}],
        }
        self.assertEqual(compare_modes(summaries)[0]["conclusion"], "REGRESSION")


if __name__ == "__main__":
    unittest.main()
