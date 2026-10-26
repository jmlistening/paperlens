# PaperLens Agent A/B — fixed_rag

> **Citation-level evaluation only: schema_success, quote_integrity, cited_gold_parent_recall and gold_point_citation_recall measure JSON validity, exact quote copying and whether gold parents/quotes were present in cited evidence. They are NOT answer accuracy, correctness, completeness or faithfulness. No LLM judge and no human review are used. Harness-generated abstentions (no answer-model call) are excluded from schema/quote means. fixed_rag consumes a precomputed retrieval run; agent systems retrieve locally. Both use the same six-unit evidence budget and answer prompt.**

## Configuration

- System: `fixed_rag`
- Benchmark: `eval\benchmark_v2\eval120`
- Retrieval run: `eval\benchmark_v2\runs\eval120_rerank\hybrid_rrf_rerank_top6`
- Retrieval system: `hybrid_rrf_rerank_top6`
- Benchmark fingerprint: `cb60fa504e7fd6cc39314c2b4fc3b0746c78c9686d18c0a149b24a1097aaa497`
- Answer model: `deepseek-flash`
- Questions: 5 (failed or invalid: 1)
- Evidence budget: 6

## Metrics

| Metric | Mean (n) |
|---|---|
| Schema success | 0.800 (n=5) |
| Quote integrity | 0.800 (n=5) |
| Cited gold parent recall | 0.800 (n=5) |
| Gold point citation recall | 0.600 (n=5) |

## Behavior

| Measure | Value |
|---|---|
| Tool calls (total / mean) | 0 / 0.00 |
| Retrieval rounds (total / mean) | 0 / 0.00 |
| Tokens (total / mean) | 12967 / 2593.4 |
| Latency ms (mean / p50 / p95) | 1752.3 / 1631.271 / 2350.590 |
| Answer latency ms (mean) | 1752.3 |
| Retrieval latency ms (mean) | 0.0 |

## Answer status

| Status | Count |
|---|---|
| answered | 4 |
| partial | 0 |
| insufficient | 0 |
| invalid | 1 |

## Stop reasons

| Reason | Count |
|---|---|
| precomputed_retrieval | 5 |

## Failures

- `q009`

## Per question

| Question | Pipeline | Answer | Stop reason | Schema | Quote | Cited gold | Gold points | Tools | Rounds | Tokens | Latency ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q001 | ok | answered | precomputed_retrieval | 1.000 | 1.000 | 1.000 | 0.500 | 0 | 0 | 2285 | 1741.2 |
| q002 | ok | answered | precomputed_retrieval | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 | 3031 | 2502.9 |
| q004 | ok | answered | precomputed_retrieval | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 | 2833 | 1380.1 |
| q009 | ok | invalid | precomputed_retrieval | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 2157 | 1505.8 |
| q010 | ok | answered | precomputed_retrieval | 1.000 | 1.000 | 1.000 | 0.500 | 0 | 0 | 2661 | 1631.3 |

## Limitations

- Citation-level scoring only; not answer accuracy or faithfulness.
- fixed_rag retrieval comes from a precomputed run, so its retrieval latency is 0 and its tool calls are 0.
- Agent retrieval returns indexed child passages; fixed evidence uses parent chunks. Text granularity differs by pipeline.
- Agent answers are scored exactly as returned; production's coverage-demotion/limitation merging is not applied, so fixed and agent share one scoring path.
- `quote_integrity` is not independent of `schema_success`: the shared validator already rejects non-substring quotes, so the two agree by construction.
- A per-question failure (invalid decision, answer schema failure) scores 0 and stays in the denominator.
- Harness-generated abstentions (no selected evidence, no answer-model call) are marked `answer_source=harness_abstention`; their schema/quote values are null and excluded from those means, while status and citation metrics remain.
- The fixed control's precomputed run used the benchmark `query_en`; agent searches use `question_zh`, matching the production agent. Query language and retrieval method differ by design.
- `tokens` counts decision + answer calls only; the fixed control's precomputed rerank cost is excluded, so this is not an end-to-end cost comparison.
- Agent answer calls use `answer_question.PROMPT` only; production's `ANSWER_EXTRA`/coverage payload is not applied.
- `gold_point_citation_recall` is bounded by qrels coverage: 13/145 v2 gold points do not appear in the frozen corpus.
- No paired bootstrap CI or significance test is computed; n is descriptive.
- Error handling is eval-specific: a failed question is scored 0 and the run continues; a supplemental tool error keeps prior evidence and still answers, whereas production aborts the run.
