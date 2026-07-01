"""Prompts used by DGT."""

from __future__ import annotations


def relation_prompt(question: str, relation_path: list[str], candidate_relation: str) -> str:
    prefix = " -> ".join(relation_path) if relation_path else "(start)"
    return (
        "You are selecting the next relation for knowledge graph question answering.\n"
        f"Question: {question}\n"
        f"Current relation path: {prefix}\n"
        f"Candidate relation: {candidate_relation}\n"
        "Is this candidate relation useful for answering the question? Answer YES or NO.\n"
        "Answer:"
    )


def stop_prompt(question: str, relation_path: list[str], current_entities: list[str]) -> str:
    prefix = " -> ".join(relation_path) if relation_path else "(start)"
    entities = ", ".join(current_entities[:10]) if current_entities else "(none)"
    return (
        "You are deciding whether to stop knowledge graph traversal.\n"
        f"Question: {question}\n"
        f"Current relation path: {prefix}\n"
        f"Entities reached by this path: {entities}\n"
        "Do these entities provide enough evidence to answer the question? Answer YES or NO.\n"
        "Answer:"
    )


def answer_prompt(question: str, evidence_triples: str) -> str:
    return (
        "Given a question and retrieved knowledge graph triples, answer the question.\n"
        "Use the triples as evidence. If several triples are relevant, reason over them and "
        "synthesize a concise final answer.\n\n"
        f"Question: {question}\n"
        f"Knowledge graph triples:\n{evidence_triples if evidence_triples else '(none)'}\n\n"
        "Final answer:"
    )
