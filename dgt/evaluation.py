"""Evaluation utilities for DGT."""

from __future__ import annotations

import re


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_matches(prediction: str, gold_answers: list[str]) -> bool:
    pred = normalize(prediction)
    if not pred:
        return False
    for gold in gold_answers:
        gold_norm = normalize(gold)
        if gold_norm and (gold_norm in pred or pred in gold_norm):
            return True
    return False
