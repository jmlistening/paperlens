# PaperLens Answer-Level Evaluation (citation-level)

> **Citation-level evaluation only: this measures JSON schema validity, exact quote copying, and whether the answer cited the gold parent chunks and gold-point quotes. It is NOT answer accuracy, correctness, completeness, or faithfulness. There is no LLM judge and no human review.**

## Configuration

- Retrieval run: `eval\benchmark_v2\runs\eval_refusal\hybrid_rrf_top6`
- Retrieval system: `hybrid_rrf_top6`
- Benchmark: `eval\benchmark_v2\refusal`
- Corpus: `eval\benchmark_v2\refusal\index\corpus_clean.json`
- Answer model: `deepseek-flash`
- Questions: 12 (failed or invalid: 0)
- Bootstrap: 2000 percentile resamples, 95% interval, deterministic seed

## Metrics

Higher is better. A schema-invalid answer contributes 0 to every metric. Empty-gold questions are excluded from ratio metrics; their `n` is shown.

| Metric | Mean [95% CI] |
|---|---|
| Schema success | 1.000 [1.000, 1.000] (n=12) |
| Quote integrity | 1.000 [1.000, 1.000] (n=12) |
| Cited gold parent recall | n/a |
| Gold point citation recall | n/a |

## Answer status counts

| Status | Count |
|---|---|
| answered | 0 |
| partial | 0 |
| insufficient | 12 |
| invalid | 0 |

## Failures

None.

## Cost and latency

- Tokens: total 22267, mean 1855.6
- Latency (ms): mean 1004.6, p50 1029.5, p95 1282.1

## Per question

| Question | Pipeline | Answer | Schema | Quote | Cited gold | Gold points | Tokens | Latency ms |
|---|---|---|---|---|---|---|---|---|
| r001 | ok | insufficient | 1.000 | 1.000 | n/a | n/a | 1855 | 1031.3 |
| r002 | ok | insufficient | 1.000 | 1.000 | n/a | n/a | 1257 | 639.1 |
| r003 | ok | insufficient | 1.000 | 1.000 | n/a | n/a | 2301 | 884.2 |
| r004 | ok | insufficient | 1.000 | 1.000 | n/a | n/a | 2566 | 866.2 |
| r005 | ok | insufficient | 1.000 | 1.000 | n/a | n/a | 2398 | 1027.6 |
| r006 | ok | insufficient | 1.000 | 1.000 | n/a | n/a | 1296 | 715.8 |
| r007 | ok | insufficient | 1.000 | 1.000 | n/a | n/a | 2328 | 1170.2 |
| r008 | ok | insufficient | 1.000 | 1.000 | n/a | n/a | 2191 | 1140.3 |
| r009 | ok | insufficient | 1.000 | 1.000 | n/a | n/a | 2126 | 1082.0 |
| r010 | ok | insufficient | 1.000 | 1.000 | n/a | n/a | 1430 | 1403.8 |
| r011 | ok | insufficient | 1.000 | 1.000 | n/a | n/a | 1453 | 911.6 |
| r012 | ok | insufficient | 1.000 | 1.000 | n/a | n/a | 1066 | 1182.5 |

## Limitations

- This is citation-level scoring, not answer accuracy or faithfulness.
- Schema success means the JSON passed structural and quote validation; it does not mean the explanation is correct.
- `gold_point_citation_recall` only checks that the gold quote was copied inside a cited evidence text; it does not check that the answer used or interpreted that quote.
- There is no LLM judge, no human review and no claim-level entailment metric.
- Bootstrap intervals are descriptive because the benchmark is a fixed set, not a random sample.
