# PaperLens Agent A/B — agent_multi

> **Citation-level evaluation only: schema_success, quote_integrity, cited_gold_parent_recall and gold_point_citation_recall measure JSON validity, exact quote copying and whether gold parents/quotes were present in cited evidence. They are NOT answer accuracy, correctness, completeness or faithfulness. No LLM judge and no human review are used. Harness-generated abstentions (no answer-model call) are excluded from schema/quote means. fixed_rag consumes a precomputed retrieval run; agent systems retrieve locally. Both use the same six-unit evidence budget and answer prompt.**

## Configuration

- System: `agent_multi`
- Benchmark: `eval\benchmark_v2\eval120`
- Retrieval run: `n/a`
- Retrieval system: `local_tools`
- Benchmark fingerprint: `cb60fa504e7fd6cc39314c2b4fc3b0746c78c9686d18c0a149b24a1097aaa497`
- Answer model: `deepseek-flash`
- Questions: 5 (failed or invalid: 0)
- Evidence budget: 6

## Metrics

| Metric | Mean (n) |
|---|---|
| Schema success | 1.000 (n=5) |
| Quote integrity | 1.000 (n=5) |
| Cited gold parent recall | 0.600 (n=5) |
| Gold point citation recall | 0.500 (n=5) |

## Behavior

| Measure | Value |
|---|---|
| Tool calls (total / mean) | 8 / 1.60 |
| Retrieval rounds (total / mean) | 8 / 1.60 |
| Tokens (total / mean) | 58418 / 11683.6 |
| Latency ms (mean / p50 / p95) | 4237.8 / 4480.352 / 5456.921 |
| Answer latency ms (mean) | 1720.0 |
| Retrieval latency ms (mean) | 553.7 |

## Answer status

| Status | Count |
|---|---|
| answered | 4 |
| partial | 1 |
| insufficient | 0 |
| invalid | 0 |

## Stop reasons

| Reason | Count |
|---|---|
| sufficient | 5 |

## Failures

None.

## Per question

| Question | Pipeline | Answer | Stop reason | Schema | Quote | Cited gold | Gold points | Tools | Rounds | Tokens | Latency ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q001 | ok | answered | sufficient | 1.000 | 1.000 | 1.000 | 0.500 | 2 | 2 | 12818 | 4480.4 |
| q002 | ok | answered | sufficient | 1.000 | 1.000 | 1.000 | 1.000 | 1 | 1 | 8300 | 3782.1 |
| q004 | ok | answered | sufficient | 1.000 | 1.000 | 1.000 | 1.000 | 2 | 2 | 16873 | 4755.6 |
| q009 | ok | partial | sufficient | 1.000 | 1.000 | 0.000 | 0.000 | 2 | 2 | 14160 | 5632.2 |
| q010 | ok | answered | sufficient | 1.000 | 1.000 | 0.000 | 0.000 | 1 | 1 | 6267 | 2538.7 |

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
