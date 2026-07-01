# DGT Code Release

This directory contains the implementation of Discriminative Graph Traversal (DGT) for knowledge graph question answering.

## Contents

- `run_dgt.py`: end-to-end runner for DGT.
- `dgt/traversal.py`: Phase 1 discriminative graph traversal with beam search and STOP.
- `dgt/backends.py`: LLM backend interface, deterministic mock backend, and OpenAI-compatible backend.
- `dgt/prompts.py`: relation scoring, STOP, and answer determination prompts.
- `dgt/answering.py`: Phase 2 generative answer determination.
- `dgt/evaluation.py`: normalized answer matching.
- `data/sample_kgqa.jsonl`: a sample KGQA dataset for testing the code path.

## Quick Start

The default command uses the deterministic mock backend, so it does not require installing extra packages, a GPU, Freebase endpoint, or API key.

```bash
cd ACTIVE/anonymous_code
python run_dgt.py \
  --data data/sample_kgqa.jsonl \
  --output outputs/sample_predictions.jsonl \
  --backend mock
```

Expected output:

```text
Examples: 2
Correct: 2
Hits@1: 1.000
Wrote predictions to outputs/sample_predictions.jsonl
```

The mock backend verifies that the code runs end to end. It is not used to report paper numbers.

## Data Format

Each line in the input JSONL file is one question:

```json
{
  "id": "sample-1",
  "question": "What kind of music did Franz Liszt compose?",
  "topic_entities": ["Franz Liszt"],
  "answers": ["Classical music"],
  "triples": [
    ["Franz Liszt", "music.artist.genre", "Classical music"]
  ]
}
```

The `triples` field is the local KG subgraph for that question. In full experiments, this local graph is constructed from Freebase around the linked topic entities.

## Running with a Real LLM Backend

The code can call a vLLM server or any OpenAI-compatible chat-completions endpoint. Install the real-model inference dependencies first:

```bash
pip install -r requirements.txt

python run_dgt.py \
  --data data/sample_kgqa.jsonl \
  --output outputs/real_model_predictions.jsonl \
  --backend openai \
  --api_base http://localhost:8000/v1 \
  --api_key EMPTY \
  --model YOUR_MODEL_NAME
```

The OpenAI-compatible backend requests one-token YES/NO outputs with log probabilities for Phase 1 and a short generative answer for Phase 2.

To run a local model, start a vLLM OpenAI-compatible server separately and set `--api_base` to that server address.

## Method Summary

DGT separates KGQA into two stages:

1. Discriminative traversal: for each candidate relation at a hop, the model answers whether the relation is useful for the question. The local score is computed as a YES-vs-NO score, and beam search keeps high-scoring relation paths. A STOP action allows early termination.
2. Answer determination: selected relation paths are instantiated into KG triples. The model then uses generative reasoning to synthesize the final answer from the retrieved evidence.
