"""LLM backend abstractions for DGT."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 2}


class LLMBackend(ABC):
    @abstractmethod
    def score_yes_no(self, prompts: list[str]) -> list[float]:
        """Return logit-style YES minus NO scores for each prompt."""

    @abstractmethod
    def generate_answer(self, prompt: str, evidence: list[tuple[str, str, str]]) -> str:
        """Generate the answer-determination output."""


class MockBackend(LLMBackend):
    """Deterministic backend for smoke tests.

    It is not intended to reproduce paper numbers. It implements the same
    interface as an LLM backend so the full DGT control flow can be tested.
    """

    def score_yes_no(self, prompts: list[str]) -> list[float]:
        return [self._score_prompt(prompt) for prompt in prompts]

    def generate_answer(self, prompt: str, evidence: list[tuple[str, str, str]]) -> str:
        question = self._field(prompt, "Question")
        q_tokens = _tokens(question)
        best_tail = ""
        best_score = -1
        for head, relation, tail in evidence:
            rel_tokens = _tokens(relation)
            tail_tokens = _tokens(tail)
            score = len(q_tokens & rel_tokens) + len(q_tokens & tail_tokens)
            if score > best_score:
                best_score = score
                best_tail = tail
        if not best_tail and evidence:
            best_tail = evidence[-1][2]
        return f"The answer is {{ {best_tail} }}."

    def _score_prompt(self, prompt: str) -> float:
        question = self._field(prompt, "Question")
        candidate = self._field(prompt, "Candidate relation")
        entities = self._field(prompt, "Entities reached by this path")
        q_tokens = _tokens(question)

        if candidate:
            rel_tokens = _tokens(candidate.replace(".", " ").replace("_", " "))
            score = float(len(q_tokens & rel_tokens))
            # Small lexical hints for the bundled sample data.
            if "music" in q_tokens and {"genre", "type", "form"} & rel_tokens:
                score += 3.0
            if "currency" in q_tokens and "currency" in rel_tokens:
                score += 3.0
            if {"country", "nation"} & q_tokens and {"country", "nation"} & rel_tokens:
                score += 2.0
            return score - 0.5

        entity_tokens = _tokens(entities)
        if q_tokens & entity_tokens:
            return 2.0
        if "currency" in q_tokens and "shilling" in entity_tokens:
            return 4.0
        if "music" in q_tokens and {"classical", "romantic"} & entity_tokens:
            return 4.0
        return -2.0

    @staticmethod
    def _field(prompt: str, field: str) -> str:
        match = re.search(rf"^{re.escape(field)}:\s*(.*)$", prompt, flags=re.MULTILINE)
        return match.group(1).strip() if match else ""


class OpenAICompatibleBackend(LLMBackend):
    """OpenAI-compatible chat backend for real model runs."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", concurrency: int = 16):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional 'openai' package to use this backend.") from exc
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.concurrency = max(1, concurrency)

    def score_yes_no(self, prompts: list[str]) -> list[float]:
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(self._score_one, prompts))

    def _score_one(self, prompt: str) -> float:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=10,
        )
        top = response.choices[0].logprobs.content[0].top_logprobs
        token_to_logprob = {item.token: item.logprob for item in top}
        return _best_logprob(token_to_logprob, [" YES", "YES", " Yes", "Yes"]) - _best_logprob(
            token_to_logprob, [" NO", "NO", " No", "No"]
        )

    def generate_answer(self, prompt: str, evidence: list[tuple[str, str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=128,
        )
        return response.choices[0].message.content or ""


def _best_logprob(token_to_logprob: dict[str, float], variants: Iterable[str]) -> float:
    vals = [token_to_logprob[v] for v in variants if v in token_to_logprob]
    return max(vals) if vals else -100.0
