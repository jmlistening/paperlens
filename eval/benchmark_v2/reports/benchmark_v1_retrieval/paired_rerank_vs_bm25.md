# Paired bootstrap comparison

Control: `bm25_top6` (31 questions)
Treatment: `hybrid_rrf_rerank_top6` (31 questions)

| Metric | Control | Treatment | Delta | 95% CI | W/L/T |
|---|---:|---:|---:|---|---|
| recall@6 | 0.8387 | 0.9032 | +0.0645 | [-0.0161, +0.1613] | 4/1/26 |
| ndcg@10 | 0.6577 | 0.8475 | +0.1898 | [+0.0777, +0.3060] | 15/5/11 |
| mrr@10 | 0.6194 | 0.8602 | +0.2409 | [+0.0995, +0.3833] | 14/4/13 |

Descriptive paired bootstrap on a fixed benchmark; no pre-registration or multiple-comparison correction.
