"""Deterministic, non-LLM oracles for the seven wjn test cases."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


def extract_label(output: str, allowed_labels: Iterable[str]) -> tuple[str | None, list[str]]:
    normalized = output.upper().replace("-", "_")
    found: list[str] = []
    for label in sorted(set(allowed_labels), key=len, reverse=True):
        pattern = rf"(?<![A-Z0-9_]){re.escape(label.upper())}(?![A-Z0-9_])"
        if re.search(pattern, normalized):
            found.append(label.upper())

    # Prefer the longest label where labels overlap, such as QUALIFIED and
    # NOT_QUALIFIED, then require one unambiguous result.
    filtered = [
        label
        for label in found
        if not any(label != other and label in other for other in found)
    ]
    unique = list(dict.fromkeys(filtered))
    return (unique[0] if len(unique) == 1 else None), unique


def evaluate_output(output: str, expected: dict[str, Any]) -> dict[str, Any]:
    expected_label = expected["label"].upper()
    parsed_label, found_labels = extract_label(output, expected["allowed_labels"])
    forbidden_hits = [
        item
        for item in expected.get("forbidden_substrings", [])
        if item.casefold() in output.casefold()
    ]
    passed = parsed_label == expected_label and not forbidden_hits
    return {
        "parsed_label": parsed_label,
        "expected_label": expected_label,
        "found_labels": found_labels,
        "forbidden_hits": forbidden_hits,
        "passed": passed,
    }


def evaluate_case(variant_results: list[dict[str, Any]]) -> dict[str, Any]:
    parsed_labels = [item["oracle"]["parsed_label"] for item in variant_results]
    all_variants_correct = all(item["oracle"]["passed"] for item in variant_results)
    comparable_labels = [label for label in parsed_labels if label is not None]
    paired_consistency = None
    if len(parsed_labels) >= 2:
        paired_consistency = (
            1.0
            if len(comparable_labels) == len(parsed_labels) and len(set(comparable_labels)) == 1
            else 0.0
        )
    return {
        "all_variants_correct": all_variants_correct,
        "paired_consistency": paired_consistency,
        "passed": all_variants_correct,
    }
