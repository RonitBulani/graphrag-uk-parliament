# Evaluation results

This folder contains the compact evaluation artefacts used in the public repository.

## Overall comparison

| Metric | Hybrid RAG | GraphRAG | Difference |
|---|---:|---:|---:|
| MRR | 0.7249 | **0.7623** | +0.0374 |
| nDCG@5 | 0.8013 | **0.8179** | +0.0166 |
| BERTScore | **0.7990** | 0.7863 | -0.0127 |
| NLI faithfulness | 0.1138 | **0.1223** | +0.0085 |
| Answer relevancy | **0.5900** | 0.5384 | -0.0517 |
| Context precision | 0.4464 | **0.4622** | +0.0157 |
| Speaker diversity | 4.1400 | **4.9800** | +0.8400 |

GraphRAG was numerically higher on five metrics. The largest and most consistent gain was retrieved-speaker diversity.

## Paired statistical checks

Paired Wilcoxon signed-rank tests over the 50 questions gave approximately:

| Metric | p-value | Interpretation in this experiment |
|---|---:|---|
| MRR | 0.304 | no clear paired difference |
| nDCG@5 | 0.653 | no clear paired difference |
| BERTScore | 0.008 | baseline was higher overall |
| Faithfulness | 0.711 | no clear paired difference |
| Answer relevancy | 0.345 | no clear paired difference |
| Context precision | 0.808 | no clear paired difference |
| Speaker diversity | < 0.001 | clear increase for GraphRAG |

These tests are included to avoid overstating small average differences. The benchmark is only 50 questions and some metrics rely on model-based evaluation, so the results should be read as evidence from this experiment rather than a general claim that one retrieval architecture is always superior.

## By reasoning complexity

GraphRAG's average MRR was higher for all three groups:

- 1-hop: 0.818 vs 0.780
- 2-hop: 0.627 vs 0.598
- 3-hop: 0.842 vs 0.798

Average nDCG@5 followed the same pattern:

- 1-hop: 0.848 vs 0.838
- 2-hop: 0.731 vs 0.724
- 3-hop: 0.873 vs 0.844

This pattern is directionally consistent with the motivation for graph-guided retrieval, but the aggregate retrieval improvements were not statistically significant in this benchmark.

## Files

- `evaluation_results.json` — question-level metric output for both systems
- `aggregate_metrics.csv` — overall averages
- `metrics_by_hop.csv` — averages split by reasoning complexity
- `../figures/` — publication-style plots generated from the evaluation output

The much larger generated answer/retrieval dumps are intentionally not committed in the portfolio version. They can be recreated by running the retrieval pipelines.
