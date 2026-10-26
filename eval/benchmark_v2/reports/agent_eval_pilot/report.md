# PaperLens Agent A/B Evaluation

> **Citation-level evaluation only: schema_success, quote_integrity, cited_gold_parent_recall and gold_point_citation_recall measure JSON validity, exact quote copying and whether gold parents/quotes were present in cited evidence. They are NOT answer accuracy, correctness, completeness or faithfulness. No LLM judge and no human review are used. Harness-generated abstentions (no answer-model call) are excluded from schema/quote means. fixed_rag consumes a precomputed retrieval run; agent systems retrieve locally. Both use the same six-unit evidence budget and answer prompt.**

## Configuration

- Benchmark: `eval\benchmark_v2\eval120`
- Retrieval run: `eval\benchmark_v2\runs\eval120_rerank`
- Retrieval system: `hybrid_rrf_rerank_top6`
- Benchmark fingerprint: `cb60fa504e7fd6cc39314c2b4fc3b0746c78c9686d18c0a149b24a1097aaa497`
- Evidence budget: 6
- Systems: fixed_rag, agent_single, agent_multi

## Comparison

| System | n | Failed | Schema | Quote | Cited gold | Gold points | Tool calls | Rounds | Tokens | Latency ms |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed_rag | 5 | 1 | 0.800 (n=5) | 0.800 (n=5) | 0.800 (n=5) | 0.600 (n=5) | 0.0 | 0.0 | 2593.4 | 1752.3 |
| agent_single | 5 | 0 | 1.000 (n=3) | 1.000 (n=3) | 0.200 (n=5) | 0.200 (n=5) | 1.0 | 1.0 | 5859.4 | 6235.3 |
| agent_multi | 5 | 0 | 1.000 (n=5) | 1.000 (n=5) | 0.600 (n=5) | 0.500 (n=5) | 1.6 | 1.6 | 11683.6 | 4237.8 |

## Notes

- All systems receive the same evidence budget and the same answer prompt.
- Gold evidence never enters a decision or answer prompt; it is used only for scoring.
- fixed_rag consumes the precomputed retrieval run and makes one answer call per question.
- agent_single allows one decision round and 2 model calls; agent_multi allows 3 rounds and 4 calls.
- Empty-gold questions are excluded from ratio metrics; failures remain in the denominator.
- `quote_integrity` is not independent of `schema_success`; the shared validator enforces the same exact-substring rule.
- Harness-generated abstentions are excluded from schema/quote means (marked `answer_source=harness_abstention`).
- The fixed run used the benchmark `query_en`; agent searches use `question_zh` (production behavior), so query language and retrieval method differ.
- `tokens` excludes the fixed control's precomputed rerank cost, so it is not an end-to-end cost comparison.
- Agent answer calls use `answer_question.PROMPT` only; production's `ANSWER_EXTRA`/coverage payload is not applied.
- `gold_point_citation_recall` is bounded by qrels coverage (13/145 v2 gold points are unmatched).
- No paired bootstrap CI or significance test is computed; n is descriptive.
- A supplemental tool error keeps prior evidence and still answers, whereas production aborts the run.
- These metrics are citation-level only; see the disclaimer above.
