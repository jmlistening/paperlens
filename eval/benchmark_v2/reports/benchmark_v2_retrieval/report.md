# PaperLens Eval-lite Report

Descriptive statistics only. Failed or invalid system outputs are scored 0 and stay in every denominator; they are never silently dropped.

Bootstrap intervals: 2000 percentile resamples of the question-level scores, 95% interval, seed 20261025.

No significance testing was performed. These are descriptive statistics; one fixed benchmark run cannot establish a difference between systems.

## Main table

| System | Status | n | Failed | Tokens | Recall@3 (95% CI) | Recall@6 (95% CI) | Recall@10 (95% CI) | AllGold@6 (95% CI) | MRR@10 (95% CI) | nDCG@10 (95% CI) | Schema success (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25_top6 | completed | 60 | 0 | 0 | 0.650 [0.539, 0.750] | 0.831 [0.747, 0.906] | 0.831 [0.747, 0.906] | 0.750 [0.633, 0.867] | 0.666 [0.570, 0.752] | 0.674 [0.597, 0.752] | 1.000 [1.000, 1.000] |
| dense_top6 | completed | 60 | 0 | 0 | 0.542 [0.425, 0.661] | 0.719 [0.614, 0.819] | 0.719 [0.614, 0.822] | 0.633 [0.500, 0.750] | 0.590 [0.483, 0.700] | 0.586 [0.491, 0.682] | 1.000 [1.000, 1.000] |
| dense_zh_top6 | completed | 60 | 0 | 0 | 0.108 [0.042, 0.192] | 0.175 [0.092, 0.267] | 0.175 [0.092, 0.275] | 0.150 [0.067, 0.250] | 0.090 [0.042, 0.150] | 0.106 [0.053, 0.172] | 1.000 [1.000, 1.000] |
| hybrid_rrf_top6 | completed | 60 | 0 | 0 | 0.675 [0.564, 0.781] | 0.808 [0.719, 0.894] | 0.808 [0.708, 0.889] | 0.750 [0.633, 0.850] | 0.641 [0.541, 0.739] | 0.661 [0.575, 0.748] | 1.000 [1.000, 1.000] |

## Recall@6 overview

Mean recall@6 (0 to 1):

```text
bm25_top6        ####################....  0.831
dense_top6       #################.......  0.719
dense_zh_top6    ####....................  0.175
hybrid_rrf_top6  ###################.....  0.808
```

## Limitations

- Retrieval-first Eval-lite: no answer-quality, claim-level or LLM-judge metric is included.
- A failed, crashed or schema-invalid system output counts as 0; it is never dropped.
- Questions are a fixed benchmark, not a random sample, so bootstrap intervals are descriptive.
- No hypothesis tests were run. Do not present these rows as a confirmed improvement.
- Systems marked `skipped` are declared stubs and contribute no measurement.
- `n` counts attempted questions; metric-level `n` can be smaller when a question has no gold evidence.

