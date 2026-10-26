# Paired bootstrap comparison

Control: `hybrid_rrf_top6` (60 questions)
Treatment: `hybrid_rrf_rerank_top6` (60 questions)

| Metric | Control | Treatment | Delta | 95% CI | W/L/T |
|---|---:|---:|---:|---|---|
| recall@6 | 0.8083 | 0.8722 | +0.0639 | [-0.0333, +0.1639] | 10/4/46 |
| ndcg@10 | 0.6552 | 0.8296 | +0.1744 | [+0.0750, +0.2725] | 29/7/24 |
| mrr@10 | 0.6381 | 0.8381 | +0.2000 | [+0.0833, +0.3136] | 25/6/29 |

Descriptive paired bootstrap on a fixed benchmark; no pre-registration or multiple-comparison correction.
