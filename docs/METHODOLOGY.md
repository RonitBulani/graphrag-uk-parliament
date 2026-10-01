# Methodology

## Research question

The experiment asks whether explicit graph structure improves retrieval for questions that require evidence to be connected across multiple speakers, parties, topics or time periods.

## Corpus preparation

The data-loading stage filters ParlSpeech material to Brexit-related UK parliamentary speeches between 2015 and 2020. It normalises speaker names, removes procedural speakers, drops very short records and exact text duplicates, and writes a cleaned corpus for downstream use.

The public repository includes only a small sample for inspection. The full derived corpus is not committed.

## Chunking and vector retrieval

Speeches are segmented on sentence boundaries into overlapping chunks of roughly 200 words with a 75-word overlap. Chunks retain speaker, party, date and agenda metadata and are embedded with `BAAI/bge-small-en-v1.5` into a persistent ChromaDB collection.

## Knowledge graph

A NetworkX graph represents speakers, parties, topics and years. Relationships include:

- speaker -> party membership
- speaker -> topic participation
- topic -> year discussion links
- speaker <-> speaker co-debate relationships

Topic edges retain speech counts and aggregate sentiment values, while speaker nodes retain speech-frequency metadata.

## Baseline

The hybrid baseline combines dense retrieval, HyDE, BM25, party-aware retrieval and temporal retrieval. Candidate rankings are fused with reciprocal-rank fusion before cross-encoder reranking.

This baseline is intentionally strong so the comparison measures the value of graph guidance on top of a realistic retrieval stack.

## GraphRAG pipeline

The graph-guided pipeline first extracts graph-relevant entities and semantically matched topics from each question. It then uses weighted graph traversal to identify relevant speakers, including community bridges and temporal signals where applicable.

Graph-derived candidates are combined with vector candidates, reranked, and optionally followed by an iterative retrieval pass that uses an initial draft to identify additional entities/evidence.

## Evaluation design

The question set contains 50 items divided into 1-hop, 2-hop and 3-hop categories. Both systems are evaluated over the same question file.

Metrics cover:

- MRR
- nDCG@5
- BERTScore
- NLI-based faithfulness
- RAGAS answer relevancy
- RAGAS context precision
- retrieved speaker diversity

### Pooled references

Reference answers are generated from the deduplicated union of evidence retrieved by the two systems. This follows the idea of TREC-style pooling to avoid constructing a reference from only one system's evidence.

These references are generated rather than independently human annotated. For that reason the repository uses the term **pooled references** instead of treating them as unquestionable ground truth.
