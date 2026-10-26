# PaperLens Eval-lite Report

Descriptive statistics only. Failed or invalid system outputs are scored 0 and stay in every denominator; they are never silently dropped.

Bootstrap intervals: 2000 percentile resamples of the question-level scores, 95% interval, seed 20261025.

No significance testing was performed. These are descriptive statistics; one fixed benchmark run cannot establish a difference between systems.

## Main table

| System | Status | n | Failed | Tokens | Recall@3 (95% CI) | Recall@6 (95% CI) | Recall@10 (95% CI) | AllGold@6 (95% CI) | MRR@10 (95% CI) | nDCG@10 (95% CI) | Schema success (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_rrf_rerank_top6 | completed | 60 | 1 | 408643 | 0.850 [0.761, 0.931] | 0.908 [0.831, 0.967] | 0.908 [0.836, 0.969] | 0.883 [0.800, 0.950] | 0.839 [0.756, 0.917] | 0.847 [0.770, 0.917] | 0.983 [0.950, 1.000] |

## Failed or invalid questions

- `hybrid_rrf_rerank_top6`: 1 of 60 questions scored 0 (`q037`).

## Recall@6 overview

Mean recall@6 (0 to 1):

```text
hybrid_rrf_rerank_top6  ######################..  0.908
```

## Limitations

- Retrieval-first Eval-lite: no answer-quality, claim-level or LLM-judge metric is included.
- A failed, crashed or schema-invalid system output counts as 0; it is never dropped.
- Questions are a fixed benchmark, not a random sample, so bootstrap intervals are descriptive.
- No hypothesis tests were run. Do not present these rows as a confirmed improvement.
- Systems marked `skipped` are declared stubs and contribute no measurement.
- `n` counts attempted questions; metric-level `n` can be smaller when a question has no gold evidence.

