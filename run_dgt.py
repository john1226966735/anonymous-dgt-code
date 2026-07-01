#!/usr/bin/env python3
"""Run DGT on a JSONL dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dgt.answering import determine_answer
from dgt.backends import MockBackend, OpenAICompatibleBackend
from dgt.data import load_jsonl
from dgt.evaluation import answer_matches
from dgt.traversal import instantiate_evidence, run_discriminative_traversal


def build_backend(args):
    if args.backend == "mock":
        return MockBackend()
    return OpenAICompatibleBackend(
        base_url=args.api_base,
        model=args.model,
        api_key=args.api_key,
        concurrency=args.concurrency,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_kgqa.jsonl")
    parser.add_argument("--output", default="outputs/sample_predictions.jsonl")
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--top_relation_paths", type=int, default=10)
    parser.add_argument("--max_entity_paths_per_relation_path", type=int, default=30)
    parser.add_argument("--backend", choices=["mock", "openai"], default="mock")
    parser.add_argument("--api_base", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--model", default="")
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    backend = build_backend(args)
    examples = load_jsonl(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_correct = 0
    rows = []
    for example in examples:
        beams = run_discriminative_traversal(
            example,
            backend,
            beam_size=args.beam_size,
            max_hops=args.max_hops,
            return_top_k=args.top_relation_paths,
        )
        evidence = instantiate_evidence(
            example,
            beams,
            max_paths_per_beam=args.max_entity_paths_per_relation_path,
        )
        prediction, generated_text = determine_answer(example.question, evidence, backend)
        correct = answer_matches(prediction, example.answers)
        n_correct += int(correct)

        rows.append(
            {
                "id": example.qid,
                "question": example.question,
                "prediction": prediction,
                "gold_answers": example.answers,
                "correct": correct,
                "relation_paths": [
                    {
                        "relations": beam.relation_path,
                        "score": round(beam.score, 4),
                        "entities": beam.current_entities,
                        "stopped": beam.stopped,
                    }
                    for beam in beams
                ],
                "evidence": evidence,
                "generated_text": generated_text,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(examples)
    accuracy = n_correct / total if total else 0.0
    print(f"Examples: {total}")
    print(f"Correct: {n_correct}")
    print(f"Hits@1: {accuracy:.3f}")
    print(f"Wrote predictions to {output_path}")


if __name__ == "__main__":
    main()
