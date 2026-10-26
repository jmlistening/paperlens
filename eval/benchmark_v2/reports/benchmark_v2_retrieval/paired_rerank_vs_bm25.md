# Paired bootstrap comparison

Control: `bm25_top6` (60 questions)
Treatment: `hybrid_rrf_rerank_top6` (60 questions)

| Metric | Control | Treatment | Delta | 95% CI | W/L/T |
|---|---:|---:|---:|---|---|
| recall@6 | 0.8306 | 0.9083 | +0.0778 | [+0.0111, +0.1472] | 10/1/49 |
| ndcg@10 | 0.6743 | 0.8471 | +0.1727 | [+0.0909, +0.2526] | 29/6/25 |
| mrr@10 | 0.6658 | 0.8394 | +0.1736 | [+0.0694, +0.2750] | 24/6/30 |

Descriptive paired bootstrap on a fixed benchmark; no pre-registration or multiple-comparison correction.
