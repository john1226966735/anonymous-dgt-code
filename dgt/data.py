"""Data loading and graph utilities for DGT."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class KGQAExample:
    """A single KGQA example with a question-specific local KG."""

    qid: str
    question: str
    topic_entities: list[str]
    answers: list[str]
    triples: list[tuple[str, str, str]]


class LocalGraph:
    """Directed local graph used by DGT traversal."""

    def __init__(self, triples: Iterable[tuple[str, str, str]]):
        self.triples = list(triples)
        self._out: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for head, relation, tail in self.triples:
            self._out[head][relation].append(tail)

    def candidate_relations(self, entities: Iterable[str]) -> list[str]:
        relations = set()
        for entity in entities:
            relations.update(self._out.get(entity, {}).keys())
        return [r for r in sorted(relations) if keep_relation(r)]

    def advance(self, entities: Iterable[str], relation: str) -> list[str]:
        next_entities = []
        for entity in entities:
            next_entities.extend(self._out.get(entity, {}).get(relation, []))
        return sorted(set(next_entities))

    def instantiate_path(
        self,
        topic_entities: Iterable[str],
        relation_path: list[str],
        max_paths: int = 20,
    ) -> list[list[tuple[str, str, str]]]:
        """Instantiate a relation path into concrete entity-level triple paths."""

        paths = [[(entity, None, entity)] for entity in topic_entities]
        for relation in relation_path:
            new_paths: list[list[tuple[str, str, str]]] = []
            for path in paths:
                current_entity = path[-1][2]
                for tail in self._out.get(current_entity, {}).get(relation, []):
                    new_paths.append(path + [(current_entity, relation, tail)])
                    if len(new_paths) >= max_paths:
                        break
                if len(new_paths) >= max_paths:
                    break
            paths = new_paths
            if not paths:
                break

        instantiated = []
        for path in paths[:max_paths]:
            instantiated.append([edge for edge in path[1:] if edge[1] is not None])
        return instantiated


def load_jsonl(path: str | Path) -> list[KGQAExample]:
    examples: list[KGQAExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            examples.append(
                KGQAExample(
                    qid=str(item["id"]),
                    question=item["question"],
                    topic_entities=list(item["topic_entities"]),
                    answers=list(item["answers"]),
                    triples=[tuple(x) for x in item["triples"]],
                )
            )
    return examples


def format_triples(triples: Iterable[tuple[str, str, str]]) -> str:
    return "\n".join(f"{h}, {r}, {t}" for h, r, t in triples)


def keep_relation(relation: str) -> bool:
    """Filter uninformative Freebase relations before LLM scoring."""
    if relation in {"type.object.type", "type.object.name"}:
        return False
    if relation.startswith("common.") or relation.startswith("freebase."):
        return False
    if "sameAs" in relation:
        return False
    return True
