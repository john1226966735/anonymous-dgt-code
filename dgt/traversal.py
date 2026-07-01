"""Phase 1: discriminative graph traversal."""

from __future__ import annotations

from dataclasses import dataclass

from .backends import LLMBackend
from .data import KGQAExample, LocalGraph
from .prompts import relation_prompt, stop_prompt

STOP = "__STOP__"


@dataclass
class Beam:
    relation_path: list[str]
    current_entities: list[str]
    score: float
    stopped: bool = False


def run_discriminative_traversal(
    example: KGQAExample,
    backend: LLMBackend,
    beam_size: int = 3,
    max_hops: int = 3,
    return_top_k: int = 10,
) -> list[Beam]:
    """Run DGT Phase 1 and return high-scoring relation paths."""

    graph = LocalGraph(example.triples)
    active = [Beam([], sorted(set(example.topic_entities)), 0.0)]
    completed: list[Beam] = []

    for _hop in range(max_hops):
        prompts: list[str] = []
        expansions: list[tuple[Beam, str]] = []

        for beam in active:
            candidates = graph.candidate_relations(beam.current_entities)
            candidates.append(STOP)
            for relation in candidates:
                if relation == STOP:
                    prompt = stop_prompt(example.question, beam.relation_path, beam.current_entities)
                else:
                    prompt = relation_prompt(example.question, beam.relation_path, relation)
                prompts.append(prompt)
                expansions.append((beam, relation))

        if not prompts:
            break

        scores = backend.score_yes_no(prompts)
        next_active: list[Beam] = []

        for (beam, relation), local_score in zip(expansions, scores):
            total_score = beam.score + local_score
            if relation == STOP:
                if beam.relation_path:
                    completed.append(
                        Beam(beam.relation_path, beam.current_entities, total_score, stopped=True)
                    )
                continue

            next_entities = graph.advance(beam.current_entities, relation)
            if next_entities:
                next_active.append(
                    Beam(
                        relation_path=beam.relation_path + [relation],
                        current_entities=next_entities,
                        score=total_score,
                    )
                )

        next_active.sort(key=lambda b: b.score, reverse=True)
        active = next_active[:beam_size]
        if not active:
            break

    all_beams = completed + active
    all_beams.sort(key=lambda b: b.score, reverse=True)
    return all_beams[:return_top_k]


def instantiate_evidence(
    example: KGQAExample,
    beams: list[Beam],
    max_paths_per_beam: int = 30,
) -> list[tuple[str, str, str]]:
    graph = LocalGraph(example.triples)
    evidence: list[tuple[str, str, str]] = []
    seen = set()
    for beam in beams:
        paths = graph.instantiate_path(example.topic_entities, beam.relation_path, max_paths=max_paths_per_beam)
        for path in paths:
            for triple in path:
                if triple not in seen:
                    evidence.append(triple)
                    seen.add(triple)
    return evidence
