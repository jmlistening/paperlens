# PaperLens Eval-lite Report

Descriptive statistics only. Failed or invalid system outputs are scored 0 and stay in every denominator; they are never silently dropped.

Bootstrap intervals: 2000 percentile resamples of the question-level scores, 95% interval, seed 20261025.

No significance testing was performed. These are descriptive statistics; one fixed benchmark run cannot establish a difference between systems.

## Main table

| System | Status | n | Failed | Tokens | Recall@3 (95% CI) | Recall@6 (95% CI) | Recall@10 (95% CI) | AllGold@6 (95% CI) | MRR@10 (95% CI) | nDCG@10 (95% CI) | Schema success (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25_top6 | completed | 60 | 0 | 0 | 0.650 [0.539, 0.750] | 0.822 [0.733, 0.903] | 0.822 [0.733, 0.900] | 0.750 [0.633, 0.850] | 0.645 [0.547, 0.733] | 0.660 [0.581, 0.740] | 1.000 [1.000, 1.000] |
| dense_top6 | completed | 60 | 0 | 0 | 0.542 [0.425, 0.661] | 0.678 [0.569, 0.783] | 0.678 [0.564, 0.786] | 0.600 [0.483, 0.717] | 0.578 [0.468, 0.689] | 0.567 [0.468, 0.670] | 1.000 [1.000, 1.000] |
| dense_zh_top6 | completed | 60 | 0 | 0 | 0.083 [0.025, 0.150] | 0.142 [0.067, 0.233] | 0.142 [0.067, 0.233] | 0.117 [0.050, 0.200] | 0.069 [0.028, 0.119] | 0.082 [0.037, 0.135] | 1.000 [1.000, 1.000] |
| hybrid_rrf_top6 | completed | 60 | 0 | 0 | 0.692 [0.592, 0.786] | 0.808 [0.719, 0.894] | 0.808 [0.708, 0.889] | 0.750 [0.633, 0.850] | 0.638 [0.541, 0.736] | 0.655 [0.570, 0.739] | 1.000 [1.000, 1.000] |

## Recall@6 overview

Mean recall@6 (0 to 1):

```text
bm25_top6        ####################....  0.822
dense_top6       ################........  0.678
dense_zh_top6    ###.....................  0.142
hybrid_rrf_top6  ###################.....  0.808
```

## Limitations

- Retrieval-first Eval-lite: no answer-quality, claim-level or LLM-judge metric is included.
- A failed, crashed or schema-invalid system output counts as 0; it is never dropped.
- Questions are a fixed benchmark, not a random sample, so bootstrap intervals are descriptive.
- No hypothesis tests were run. Do not present these rows as a confirmed improvement.
- Systems marked `skipped` are declared stubs and contribute no measurement.
- `n` counts attempted questions; metric-level `n` can be smaller when a question has no gold evidence.

