# Paired bootstrap comparison

Control: `bm25_top6` (60 questions)
Treatment: `hybrid_rrf_rerank_top6` (60 questions)

| Metric | Control | Treatment | Delta | 95% CI | W/L/T |
|---|---:|---:|---:|---|---|
| recall@6 | 0.8222 | 0.8722 | +0.0500 | [-0.0444, +0.1389] | 10/4/46 |
| ndcg@10 | 0.6597 | 0.8296 | +0.1700 | [+0.0728, +0.2628] | 29/7/24 |
| mrr@10 | 0.6447 | 0.8381 | +0.1933 | [+0.0800, +0.3039] | 24/7/29 |

Descriptive paired bootstrap on a fixed benchmark; no pre-registration or multiple-comparison correction.
