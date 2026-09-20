"""Deterministic, non-LLM oracles for all module-two test cases."""

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
    if "required_substrings" in expected:
        required_hits = [
            item
            for item in expected["required_substrings"]
            if item.casefold() in output.casefold()
        ]
        forbidden_hits = [
            item
            for item in expected.get("forbidden_substrings", [])
            if item.casefold() in output.casefold()
        ]
        word_count = len(re.findall(r"\b[\w'-]+\b", output, flags=re.UNICODE))
        min_words = int(expected.get("min_words", 0))
        passed = (
            len(required_hits) == len(expected["required_substrings"])
            and word_count >= min_words
            and not forbidden_hits
        )
        return {
            "required_hits": required_hits,
            "required_count": len(expected["required_substrings"]),
            "word_count": word_count,
            "min_words": min_words,
            "forbidden_hits": forbidden_hits,
            "passed": passed,
        }

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
    parsed_labels = [item["oracle"].get("parsed_label") for item in variant_results]
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
        "passed_runs": sum(item["oracle"]["passed"] for item in variant_results),
        "total_runs": len(variant_results),
        "pass_rate": (
            sum(item["oracle"]["passed"] for item in variant_results) / len(variant_results)
            if variant_results
            else 0.0
        ),
        "paired_consistency": paired_consistency,
        "passed": all_variants_correct,
    }
