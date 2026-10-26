# Paired bootstrap comparison

Control: `hybrid_rrf_top6` (60 questions)
Treatment: `hybrid_rrf_rerank_top6` (60 questions)

| Metric | Control | Treatment | Delta | 95% CI | W/L/T |
|---|---:|---:|---:|---|---|
| recall@6 | 0.8083 | 0.9083 | +0.1000 | [+0.0167, +0.1833] | 9/1/50 |
| ndcg@10 | 0.6609 | 0.8471 | +0.1862 | [+0.0969, +0.2751] | 27/7/26 |
| mrr@10 | 0.6408 | 0.8394 | +0.1986 | [+0.0944, +0.3033] | 23/6/31 |

Descriptive paired bootstrap on a fixed benchmark; no pre-registration or multiple-comparison correction.
