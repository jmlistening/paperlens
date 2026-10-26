# Paired bootstrap comparison

Control: `hybrid_rrf_top6` (31 questions)
Treatment: `hybrid_rrf_rerank_top6` (31 questions)

| Metric | Control | Treatment | Delta | 95% CI | W/L/T |
|---|---:|---:|---:|---|---|
| recall@6 | 0.7581 | 0.9032 | +0.1452 | [+0.0323, +0.2742] | 6/1/24 |
| ndcg@10 | 0.6055 | 0.8475 | +0.2420 | [+0.1215, +0.3716] | 16/3/12 |
| mrr@10 | 0.5672 | 0.8602 | +0.2930 | [+0.1505, +0.4355] | 16/2/13 |

Descriptive paired bootstrap on a fixed benchmark; no pre-registration or multiple-comparison correction.
