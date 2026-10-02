# Reproducibility notes

## What is included

The public portfolio repository includes:

- all main Python pipeline scripts
- the 50-question benchmark
- a 50-row sample of the filtered parliamentary data
- the final question-level evaluation metrics
- aggregate result tables
- generated evaluation figures

## What is not included

The repository intentionally excludes:

- the full ParlSpeech source archive
- the full filtered 39k-speech dataset
- the persistent ChromaDB vector store
- the generated 6.7 MB GraphML graph
- full generated answer/retrieval dumps
- local Ollama model weights

Those artefacts are either third-party data, derived files that can be regenerated, or unnecessarily large for a recruiter-facing repository.

## Expected paths

`src/01_data_loading.py` expects the ParlSpeech archive at the project root as:

```text
archive.zip
```

It reads the House of Commons source files configured in the script and writes the filtered dataset into `data/`.

The remaining scripts use project-relative paths, so run them from the repository root.

## Local models

Generation uses Ollama through an OpenAI-compatible local endpoint. Pull the required models before running retrieval/generation:

```bash
ollama pull qwen2.5:3b
ollama pull phi3:mini
```

## Runtime

The full pipeline is not intended to be a quick unit-test-sized demo. Embedding more than 200k chunks and evaluating both retrieval systems across 50 questions takes substantially longer than running the small sample data.

For a quick inspection of the implementation, start with the source code, `data/questions.json`, `results/aggregate_metrics.csv`, and the figures rather than rebuilding the full corpus.


## Evaluation interpretation

The committed MRR and nDCG@5 values depend on the target speaker/party sets
stored in the generated pooled-reference evaluation metadata. They should be
reproduced with the same question file and pooled-reference procedure.

Because these targets are not independent human passage-relevance labels,
re-running reference generation with different model outputs may change both
the reference answers and the derived entity targets.
