"""Phase 2: generative answer determination."""

from __future__ import annotations

import re

from .backends import LLMBackend
from .data import format_triples
from .prompts import answer_prompt


def determine_answer(
    question: str,
    evidence: list[tuple[str, str, str]],
    backend: LLMBackend,
) -> tuple[str, str]:
    prompt = answer_prompt(question, format_triples(evidence))
    generated = backend.generate_answer(prompt, evidence)
    prediction = extract_braced_answer(generated) or generated.strip()
    return prediction, generated


def extract_braced_answer(text: str) -> str:
    matches = re.findall(r"\{([^}]+)\}", text or "")
    if matches:
        return matches[-1].strip()
    return ""
