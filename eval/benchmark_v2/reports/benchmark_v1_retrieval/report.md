# PaperLens Eval-lite Report

Descriptive statistics only. Failed or invalid system outputs are scored 0 and stay in every denominator; they are never silently dropped.

Bootstrap intervals: 2000 percentile resamples of the question-level scores, 95% interval, seed 20261025.

No significance testing was performed. These are descriptive statistics; one fixed benchmark run cannot establish a difference between systems.

## Main table

| System | Status | n | Failed | Tokens | Recall@3 (95% CI) | Recall@6 (95% CI) | Recall@10 (95% CI) | AllGold@6 (95% CI) | MRR@10 (95% CI) | nDCG@10 (95% CI) | Schema success (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25_top6 | completed | 31 | 0 | 0 | 0.661 [0.516, 0.790] | 0.839 [0.710, 0.935] | 0.839 [0.710, 0.935] | 0.774 [0.613, 0.903] | 0.619 [0.489, 0.758] | 0.658 [0.542, 0.770] | 1.000 [1.000, 1.000] |
| dense_top6 | completed | 31 | 0 | 0 | 0.532 [0.371, 0.694] | 0.726 [0.581, 0.871] | 0.726 [0.581, 0.855] | 0.645 [0.452, 0.806] | 0.510 [0.370, 0.653] | 0.545 [0.407, 0.669] | 1.000 [1.000, 1.000] |
| dense_zh_top6 | completed | 31 | 0 | 0 | 0.145 [0.032, 0.274] | 0.177 [0.065, 0.323] | 0.177 [0.065, 0.306] | 0.161 [0.032, 0.290] | 0.114 [0.029, 0.220] | 0.123 [0.041, 0.230] | 1.000 [1.000, 1.000] |
| hybrid_rrf_top6 | completed | 31 | 0 | 0 | 0.677 [0.532, 0.806] | 0.758 [0.613, 0.887] | 0.758 [0.613, 0.903] | 0.710 [0.548, 0.871] | 0.567 [0.433, 0.694] | 0.606 [0.482, 0.731] | 1.000 [1.000, 1.000] |

## Recall@6 overview

Mean recall@6 (0 to 1):

```text
bm25_top6        ####################....  0.839
dense_top6       #################.......  0.726
dense_zh_top6    ####....................  0.177
hybrid_rrf_top6  ##################......  0.758
```

## Limitations

- Retrieval-first Eval-lite: no answer-quality, claim-level or LLM-judge metric is included.
- A failed, crashed or schema-invalid system output counts as 0; it is never dropped.
- Questions are a fixed benchmark, not a random sample, so bootstrap intervals are descriptive.
- No hypothesis tests were run. Do not present these rows as a confirmed improvement.
- Systems marked `skipped` are declared stubs and contribute no measurement.
- `n` counts attempted questions; metric-level `n` can be smaller when a question has no gold evidence.

