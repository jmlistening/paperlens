# PaperLens Answer-Level Evaluation (citation-level)

> **Citation-level evaluation only: this measures JSON schema validity, exact quote copying, and whether the answer cited the gold parent chunks and gold-point quotes. It is NOT answer accuracy, correctness, completeness, or faithfulness. There is no LLM judge and no human review.**

## Configuration

- Retrieval run: `eval\benchmark_v2\runs\eval120_rerank\hybrid_rrf_rerank_top6`
- Retrieval system: `hybrid_rrf_rerank_top6`
- Benchmark: `eval\benchmark_v2\eval120`
- Corpus: `eval\benchmark_v2\eval120\index\corpus_clean.json`
- Answer model: `deepseek-flash`
- Questions: 60 (failed or invalid: 8)
- Bootstrap: 2000 percentile resamples, 95% interval, deterministic seed

## Metrics

Higher is better. A schema-invalid answer contributes 0 to every metric. Empty-gold questions are excluded from ratio metrics; their `n` is shown.

| Metric | Mean [95% CI] |
|---|---|
| Schema success | 0.867 [0.767, 0.950] (n=60) |
| Quote integrity | 0.867 [0.783, 0.950] (n=60) |
| Cited gold parent recall | 0.764 [0.661, 0.867] (n=60) |
| Gold point citation recall | 0.725 [0.619, 0.829] (n=60) |

## Answer status counts

| Status | Count |
|---|---|
| answered | 44 |
| partial | 8 |
| insufficient | 0 |
| invalid | 8 |

## Failures

Questions with a failed model call, a failed/unavailable retrieval record, or failed answer validation:

- `q005`
- `q007`
- `q009`
- `q018`
- `q101`
- `q108`
- `q114`
- `q127`

## Cost and latency

- Tokens: total 154495, mean 2574.9
- Latency (ms): mean 2254.4, p50 1914.9, p95 3297.6

## Per question

| Question | Pipeline | Answer | Schema | Quote | Cited gold | Gold points | Tokens | Latency ms |
|---|---|---|---|---|---|---|---|---|
| q001 | ok | answered | 1.000 | 1.000 | 1.000 | 0.500 | 2304 | 1372.3 |
| q002 | ok | partial | 1.000 | 1.000 | 1.000 | 1.000 | 3163 | 2903.2 |
| q004 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2774 | 1745.6 |
| q005 | failed | invalid | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0.0 |
| q007 | failed | invalid | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0.0 |
| q009 | failed | invalid | 0.000 | 0.000 | 0.000 | 0.000 | 2774 | 18989.6 |
| q010 | ok | answered | 1.000 | 1.000 | 1.000 | 0.500 | 2713 | 1574.1 |
| q011 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2955 | 1877.9 |
| q012 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2307 | 1494.2 |
| q013 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2883 | 1834.2 |
| q014 | ok | answered | 1.000 | 1.000 | 0.000 | 0.000 | 2682 | 1998.6 |
| q015 | ok | partial | 1.000 | 1.000 | 1.000 | 1.000 | 2848 | 1313.8 |
| q016 | ok | partial | 1.000 | 1.000 | 1.000 | 1.000 | 2686 | 2134.7 |
| q017 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2812 | 1946.5 |
| q018 | ok | invalid | 0.000 | 0.000 | 0.000 | 0.000 | 3247 | 3078.4 |
| q019 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2455 | 1908.0 |
| q020 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 3041 | 1921.8 |
| q021 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2722 | 2132.9 |
| q023 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2404 | 2014.5 |
| q025 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 1980 | 1662.4 |
| q027 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2425 | 1342.7 |
| q028 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2695 | 2349.5 |
| q029 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2755 | 2038.5 |
| q030 | ok | answered | 1.000 | 1.000 | 0.500 | 0.750 | 2390 | 1424.0 |
| q032 | ok | partial | 1.000 | 1.000 | 1.000 | 1.000 | 2589 | 2966.9 |
| q033 | ok | partial | 1.000 | 1.000 | 0.000 | 0.000 | 2738 | 2465.4 |
| q035 | ok | answered | 1.000 | 1.000 | 1.000 | 0.333 | 2084 | 1779.9 |
| q036 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2192 | 1211.1 |
| q037 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2455 | 1215.4 |
| q038 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2684 | 943.4 |
| q040 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2346 | 1221.1 |
| q101 | ok | invalid | 0.000 | 0.000 | 0.000 | 0.000 | 2905 | 3360.4 |
| q102 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2931 | 3294.3 |
| q103 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2458 | 2600.8 |
| q104 | ok | answered | 1.000 | 1.000 | 1.000 | 0.750 | 2622 | 2808.1 |
| q105 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2744 | 2163.7 |
| q106 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2628 | 2150.1 |
| q107 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2487 | 931.9 |
| q108 | ok | invalid | 0.000 | 0.000 | 0.000 | 0.000 | 2885 | 2862.6 |
| q109 | ok | answered | 1.000 | 1.000 | 0.000 | 0.000 | 2780 | 1551.0 |
| q110 | ok | partial | 1.000 | 1.000 | 0.000 | 0.000 | 2617 | 2834.6 |
| q111 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2870 | 2745.9 |
| q112 | ok | answered | 1.000 | 1.000 | 0.000 | 0.000 | 2900 | 3109.3 |
| q113 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2280 | 1374.3 |
| q114 | ok | invalid | 0.000 | 0.000 | 0.000 | 0.000 | 3409 | 2860.1 |
| q115 | ok | answered | 1.000 | 1.000 | 1.000 | 0.500 | 2547 | 1329.8 |
| q116 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2609 | 1435.4 |
| q117 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 3243 | 1159.2 |
| q118 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 3304 | 1553.1 |
| q119 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2614 | 1781.0 |
| q121 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2480 | 2018.8 |
| q122 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2670 | 1747.6 |
| q123 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2792 | 3009.3 |
| q124 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2382 | 1354.5 |
| q125 | ok | partial | 1.000 | 1.000 | 1.000 | 1.000 | 2500 | 2247.2 |
| q126 | ok | answered | 1.000 | 1.000 | 1.000 | 0.500 | 2463 | 2352.6 |
| q127 | ok | invalid | 0.000 | 0.000 | 0.000 | 0.000 | 2529 | 2474.0 |
| q128 | ok | answered | 1.000 | 1.000 | 1.000 | 1.000 | 2747 | 1818.0 |
| q129 | ok | answered | 1.000 | 1.000 | 0.333 | 0.667 | 2647 | 3711.9 |
| q130 | ok | partial | 1.000 | 1.000 | 1.000 | 1.000 | 2349 | 1764.4 |

## Limitations

- This is citation-level scoring, not answer accuracy or faithfulness.
- Schema success means the JSON passed structural and quote validation; it does not mean the explanation is correct.
- `gold_point_citation_recall` only checks that the gold quote was copied inside a cited evidence text; it does not check that the answer used or interpreted that quote.
- There is no LLM judge, no human review and no claim-level entailment metric.
- Bootstrap intervals are descriptive because the benchmark is a fixed set, not a random sample.
