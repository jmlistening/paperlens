# PaperLens Agent A/B Evaluation

`eval/agent_eval.py` is an offline-testable A/B harness for the **fixed RAG
control** versus the **bounded retrieval agent**. It scores answers with
`eval.answer_eval.evaluate_answer` and reports citation-level metrics plus
behavioral cost fields. It does not call an LLM judge and does not inspect
answer correctness, faithfulness or claim-level entailment.

> **What this measures:** JSON schema validity and exact quote copying **for
> model-generated answers** (harness-generated abstentions are excluded from
> those two means), whether the answer cited the gold parent chunks, whether
> gold-point quotes appear in cited evidence, plus tool calls, retrieval rounds,
> tokens and latency. `quote_integrity` is **not independent** of
> `schema_success`: `validate_answer` already enforces the exact-substring rule,
> so the two columns agree by construction.
>
> **What this does NOT measure:** answer accuracy, correctness, completeness,
> faithfulness or claim-level entailment. A high score means the structured
> output passed mechanical checks; it does not mean the Chinese explanation is
> right. Every `aggregate.json` and `report.md` carries this disclaimer.

## Systems

All systems use the same evidence budget (`EVIDENCE_BUDGET = 6`) and the same
answer prompt, `answer_question.PROMPT`, verbatim. Gold data is loaded for
scoring only; it never enters a decision or answer prompt, and
`evaluate_answer` runs after the answer is generated.

| System | Retrieval | Model-call budget | Rounds |
|---|---|---|---|
| `fixed_rag` | Precomputed rerank run (`retrieved_parent_ids` read from `<retrieval-system>/<qid>.json`, mapped through `index/corpus_clean.json`), then capped at six units; the default top-6 run already has six | 1 answer call | n/a |
| `agent_single` | One dense top-20 search, one coverage decision, at most 6 selected units | 2 (decision + answer) | 1 |
| `agent_multi` | Dense top-20, then up to 2 supplemental searches with `search_dense` / `search_bm25`, at most 6 selected units | 4 | up to 3 |

The agent decision contract is the production one: `agent_policy.DECISION_PROMPT`
and `agent_policy.validate_decision`. It cannot select more than six units,
request an unknown tool, or search without a query. The fixed control and the
agent both send exactly this answer payload shape to the answer model:

```json
{
  "question": "<question_zh>",
  "evidence": [
    {"id": "E1", "title": "...", "text": "..."}
  ]
}
```

No `coverage`, gold IDs, gold quotes, qrels or `gold_points` are included.
Answers are scored exactly as returned by the shared prompt; production's
coverage-demotion / limitation-merging post-processing is not applied, so fixed
and agent answers go through one identical scoring path. Agent answer calls also
omit production's `ANSWER_EXTRA` and the `coverage` object, not only its
post-processing. This is deliberate: it keeps fixed and agent under one answer
contract and isolates retrieval policy, but the agent row is **not** the
production agent's full prompt or quality.

### Metric / field definitions

| Field | Definition |
|---|---|
| `schema_success` | Response parsed and passed `answer_question.validate_answer`; else 0. `null` for a harness-generated abstention (no answer-model call). |
| `quote_integrity` | All citation quotes are exact whitespace-normalized substrings of cited evidence; 0 if schema fails; `null` for a harness abstention. Not independent: `validate_answer` already enforces the same rule, so this is 1 exactly when schema succeeds. |
| `cited_gold_parent_recall` | `|cited parent IDs ∩ gold_parent_ids| / |gold_parent_ids|`; `null` for questions with no gold parents. |
| `gold_point_citation_recall` | Fraction of gold points whose quote appears in at least one cited evidence text; `null` when there are no gold points. |
| `tool_calls` | Successful retrieval-tool invocations (initial dense search plus supplements). `fixed_rag` is 0. |
| `retrieval_rounds` | Completed agent coverage-decision rounds (the "1 round" / "3 rounds" budget). `fixed_rag` is 0. |
| `tokens` | Total decision + answer tokens from the backend `usage` / `answer_fn.last_usage` (0 if unavailable). |
| `latency_ms` | Whole-question wall time. `answer_latency_ms` and `retrieval_latency_ms` are also recorded separately, because the fixed control's retrieval happened before the run. |

Failed or schema-invalid model answers stay in the denominator and score 0.
Harness-generated abstentions are marked `answer_source=harness_abstention`;
their schema/quote values are null and excluded from those means, but they
remain in status/citation counts and the per-question table. Empty-gold
questions are excluded only from the ratio metrics, not from `schema_success`.
Each checkpoint records `answer_source` as `model`, `harness_abstention`, or
`null` for a failed question.

## Evaluation-mode plumbing

Two production defaults are preserved exactly.

### `agent_tools.LocalTools`

```python
LocalTools(root, paper=None, rag_dir=None, verify_sources=True)
```

- Defaults (`rag_dir=None`, `verify_sources=True`) behave like the original:
  `search_vector.load_store(root)` on `<root>/data/rag`, the library journal
  check, and every source PDF under `<root>/data/pdfs` is hash-verified.
- Evaluation mode (`rag_dir=<frozen index>`, `verify_sources=False`):
  - loads `vector_store.npz` + `corpus_clean.json` from `rag_dir` through
    `retrieval_core.load_index` (same structural checks as `load_store`);
  - skips the pipeline journal/lock check;
  - skips `data/pdfs` source verification;
  - still hash-checks the frozen corpus/index files on every `check_current()`,
    so a mid-run change is detected.
- A `rag_dir` that resolves to `<root>/data/rag` is the live library, not
  evaluation mode: it uses `load_store` and the journal even when passed
  explicitly. `rag_dir=<frozen>` with `verify_sources=True` is rejected with an
  explicit error; frozen-index mode requires `verify_sources=False`.

The frozen benchmark index is `eval/benchmark_v2/eval120/index` (a symlink to
`eval/benchmark_v2/index120`).

### `run_agent.AgentRun`

```python
AgentRun(..., output_dir=None)
```

- Default (`None`): unchanged production paths
  `<root>/data/rag/agent_runs/<stamp>` and live history
  `<root>/data/rag/answers/<stamp>`.
- `output_dir=<dir>`: all run artifacts, including live-mode history, go under
  `<dir>/<stamp>/` (`history/` subfolder); nothing is written to `data/rag`.
- `eval/agent_eval.py` itself does not use `AgentRun`; it implements an
  eval-only bounded loop so it can inject `tools` / `decision_backend` /
  `answer_fn` directly and avoid any data writes.

## CLI

```bat
python eval/agent_eval.py ^
  --benchmark-dir eval/benchmark_v2/eval120 ^
  --retrieval-run eval/benchmark_v2/runs/eval120_rerank ^
  --retrieval-system hybrid_rrf_rerank_top6 ^
  --systems fixed_rag,agent_single,agent_multi ^
  --limit 5 ^
  --out eval/benchmark_v2/runs/eval120_rerank/agent_eval_pilot ^
  --answer-model <model-id>
```

| Flag | Default | Notes |
|---|---|---|
| `--benchmark-dir` | `eval/benchmark_v2/eval120` | `questions.jsonl` + `qrels.jsonl`; fixed also needs `index/corpus_clean.json`. |
| `--retrieval-run` | `eval/benchmark_v2/runs/eval120_rerank` | Run root or one system directory. |
| `--retrieval-system` | `hybrid_rrf_rerank_top6` | Fixed control's retrieval records. The actual resolved system directory name is recorded; a stale flag label is not trusted. |
| `--systems` | `fixed_rag,agent_single,agent_multi` | Comma-separated subset. |
| `--limit` | none | Cap questions per system; use 5-10 for a pilot. |
| `--question-ids` | none | Optional comma-separated IDs (e.g. `q001,q002`), applied before `--limit`. Useful to avoid known failed retrieval records. |
| `--out` | `<run root>/agent_eval/<retrieval-system-or-local_tools>` | New output root with one subfolder per system; must differ from the retrieval run root and system directory. |
| `--resume` | true | Reuse terminal checkpoints whose run signature matches; `--no-resume` forces fresh calls. |
| `--answer-model` | `DEEPSEEK_MODEL` from `.env` | Used for answer and decision calls. |
| `--rag-dir` | `<benchmark-dir>/index` | Frozen index for agent systems; the harness passes `verify_sources=False` internally and rejects a `corpus_clean.json` that differs from `<benchmark-dir>/index` (fixed evidence and gold would otherwise use a different corpus). |

The `make_deepseek_answer_fn` / `make_deepseek_decision_backend` functions
(used by the CLI) are the only code paths that read `.env` or create OpenAI
clients. No network call happens until a question is actually run. Tests never
make network calls; the answer wrapper's non-stop rejection is exercised with a
mocked `openai.OpenAI`.

## Outputs

```
<out>/<system>/<question_id>.json    # terminal checkpoint (ok/failed + metrics)
<out>/<system>/aggregate.json        # means, status/stop-reason counts, cost
<out>/<system>/report.md             # human-readable per-system report
<out>/summary.json                   # all system aggregates in one document
<out>/report.md                      # combined A/B comparison table
```

Checkpoints are terminal and reused on a re-run (`resume=True`) only when the
stored run signature matches, so an interrupted pilot can skip completed
questions without repeating paid calls. An in-flight question with no terminal
checkpoint is repeated. The signature covers the answer/decision models, both
prompts, benchmark fingerprint (including actual `corpus_clean.json` and
`vector_store.npz` hashes), gold data, question text, evidence budget, call
limits, retrieval configuration (`DEFAULT_TOP_K`, `SUPPLEMENT_TOP_K`,
`MAX_POOL`, `MAX_PAYLOAD_CHARS`, query source), `ANSWER_EVAL_VERSION`,
`answer_schema.VERSION`, the `validate_decision` source hash, the fixed
retrieval record and the full agent-index identity (corpus/vector/manifest
hashes plus model name/revision and query prefix). `summary.json` and each
`aggregate.json` record the benchmark fingerprint and an `agent_index` block.
Pre-signature checkpoints are ignored. Bump `AGENT_EVAL_VERSION` on any behavior
change that is not otherwise captured by this signature.

## Tests

`tests/test_agent_eval.py` runs entirely offline with synthetic questions, a
fake corpus, fake retrieval records, fake tools and fake answer/decision
backends. It covers:

- fixed control reads the precomputed retrieval and uses the shared prompt;
- `agent_single` one-decision flow and `agent_multi` dense+BM25 two-round flow;
- `agent_multi` actually performs two supplemental searches within 4 calls, and
  its third search request stops with `round_budget`;
- a harness-generated abstention is marked `answer_source=harness_abstention`
  and excluded from schema/quote means;
- the six-unit cap on both fixed and agent, and rejection of a seventh selection;
- tool errors become `stop_reason=tool_error` (and an initial tool error fails
  the question without an answer call);
- a mid-question index change, including one detected inside a search, is
  recorded as `index_changed`;
- a missing `finish_reason` is rejected instead of being treated as `stop`;
- stale `last_usage` is cleared before every call, so a failed question is not
  charged the previous question's tokens;
- benchmark fingerprint (including actual corpus/vector hashes) and run
  signature are written, and a model change invalidates resume;
- resume is also invalidated by retrieval-config and scoring-version changes;
- an agent index whose `corpus_clean.json` differs from the benchmark is
  rejected, and the agent-index block is recorded;
- an output root equal to the retrieval run root/system directory is rejected;
- a target question without a qrels record is rejected (an explicit empty
  `gold_parent_ids` is allowed for empty-gold questions);
- per-stage answer latency is null when no answer call ran, and its mean shows
  the measured-call n;
- no gold text/IDs in decision or answer prompts;
- checkpoints, aggregate, combined report, `--question-ids` order and
  filter-before-limit behavior, and resume;
- injected tools are used; production `LocalTools` is not constructed;
- `LocalTools` eval mode skips journal and PDF checks, while defaults still use
  `load_store` and the journal; an explicit live `rag_dir` is treated as
  production, and frozen `rag_dir` plus `verify_sources=True` is rejected;
- `AgentRun(output_dir=...)` writes outside `data/rag` and keeps production
  defaults when omitted.

Run:

```bat
python -I -B -X utf8 -m unittest discover -s tests -p "test_agent_eval.py" -v
```

## Real pilot readiness and 5-question pilot

A 5-question paid pilot was run on 2026-10-25; the full 60-question A/B was not.
Readiness checks:

- `.env` contains `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` and `DEEPSEEK_MODEL`;
  the code reads configuration lazily and makes no call at import.
- The frozen index (`eval/benchmark_v2/index120`) loads in evaluation mode:
  7,746 units, 4,962 eligible after section/short-fragment filtering.
- The local embedding model cache exists at `data/models/models--intfloat--multilingual-e5-small`,
  and `torch`, `sentence_transformers`, `faiss`, `rank_bm25`, `openai` and
  `python-dotenv` are installed in `paperlens`.
- The precomputed retrieval run has all 60 `q*.json` records for
  `hybrid_rrf_rerank_top6` (58 `ok`, 2 `failed`). The harness records those two
  as scored failures rather than dropping them. **Important:** the failed
  records are `q005` and `q007`, and both are inside the first five benchmark
  questions (`q001, q002, q004, q005, q007`). A naive `--limit 5`/`--limit 10`
  therefore gives the fixed control two unavoidable 0-score questions. Use
  `--question-ids` to pick clean IDs, for example
  `--question-ids q001,q002,q004,q009,q010 --limit 5`.

Caveats / blockers to budget for:

1. **Paid traffic is real.** `--limit 5` with all three systems can consume up
   to `1 + 2 + 4 = 7` model calls per question (35 calls), plus output tokens;
   `agent_multi` only spends 4 when it keeps searching.
2. **First dense search loads the local embedding model** (CPU/GPU) and one
   BM25 supplement re-tokenizes ~5k eligible units; expect seconds per search,
   not milliseconds. `check_current()` also re-hashes the frozen corpus/index
   (~118 MB); with a warm OS cache this was ~0.09 s per call in a local check.
3. **Evidence granularity differs.** Agent hits are indexed child passages;
   fixed evidence is parent chunks from `corpus_clean.json`. Both are capped at
   six units, but `gold_point_citation_recall` can differ because a child
   passage may not contain the whole gold quote. This is a property of the
   pipelines, not a harness bug; document it when interpreting a pilot.
4. **No automatic retry.** A provider error fails that question and stays in
   the denominator, matching `answer_eval` and the production agent.
5. **Agent answer prompt is the shared fixed-RAG prompt.** Production's agent
   appends `ANSWER_EXTRA` and sends a `coverage` object; both are omitted so
   fixed and agent use one contract. Read the agent row as a retrieval-policy
   comparison, not the production agent's full quality.
6. **Token counts are not end-to-end cost.** `tokens` counts decision + answer
   calls; the fixed control's precomputed rerank cost (399,725 tokens for 60
   questions in the 120-paper run) is excluded. Fixed retrieval latency is also
   0 because it ran before this harness.
7. **`gold_point_citation_recall` has a qrels ceiling.** 13/145 v2 gold points
   do not appear verbatim in the frozen corpus
   (`eval/benchmark_v2/qrels_coverage_v2.json`), so a perfect answer cannot
   reach 1.0 on those questions. Agent child passages can lower it further.
8. **No paired CI or significance test.** n=5–60 is descriptive; do not write a
   significance or general-improvement claim.
9. **Error handling is eval-specific.** A failed question is scored 0 and the
   run continues. A supplemental tool error keeps prior evidence and still
   answers; production aborts the whole run on a tool/model error.
10. **Small pilots and clean-ID selection are convenience samples.**
    `--question-ids` avoids known failed fixed records; do not generalize a
    5-question rate. The full 60 includes fixed failures `q005`/`q007`; report
    them as failures or compare on a paired subset and say so.
11. **Query paths differ by design.** The fixed control consumes a precomputed
    run built from the benchmark's page-blind `query_en`; agent systems search
    with the Chinese `question_zh`, matching the production agent. This is a
    system comparison, not a same-query retrieval ablation.
12. **Agent index must match the benchmark corpus.** `--rag-dir` is rejected
    when its `corpus_clean.json` hash differs from `<benchmark-dir>/index`.
    A same-corpus re-embedding is recorded in `agent_index` (vector/model
    hashes) but is not rejected; inspect that block before interpreting.
13. **Use a new output root.** `--out` equal to the retrieval run root or its
    system directory is rejected; those directories hold retrieval records and
    generated reports.
14. **Per-stage latency is measured-only.** `answer_latency_ms` and
    `retrieval_latency_ms` are null when that stage did not run, and means use
    the measured-call n. Fixed `latency_ms` is answer-only; agent `latency_ms`
    is whole-question.
15. **Every target question needs a qrels record.** Missing records are
    rejected; use an explicit empty `gold_parent_ids` for empty-gold questions.

### 5-question pilot results (2026-10-25)

Question IDs: `q001,q002,q004,q009,q010`; model `deepseek-flash`; pilot-summary
fingerprint `cb60fa50…7aaa497` (the current actual-file fingerprint for the same
files is `3e1d11df…`; see `PILOT_NOTES.md`). Full details and per-question files:
`eval/benchmark_v2/reports/agent_eval_pilot/`.

| System | Answer status | Schema / quote | Cited gold | Gold point | Tokens |
|---|---|---|---|---|---|
| `fixed_rag` | 4 answered / 1 invalid | 0.800 (n=5) | 0.800 | 0.600 | 12,967 |
| `agent_single` | 2 answered / 3 insufficient | 1.000 (n=3) | 0.200 | 0.200 | 29,297 |
| `agent_multi` | 4 answered / 1 partial | 1.000 (n=5) | 0.600 | 0.500 | 58,418 |

- `fixed_rag` q009 is invalid: the model cited E2 for quotes not in that
  parent chunk. It is a real model citation error, counted as 0.
- `agent_single` q001/q004 stopped at `round_budget` after the decision model
  requested a supplemental search despite `supplementary_search_allowed=false`;
  they are `answer_source=harness_abstention` and excluded from schema/quote
  means. q009 was a model-generated `insufficient`.
- `agent_multi` recovered q001/q004/q009 with a second round, but it did **not**
  beat `fixed_rag` on cited-gold or gold-point recall in this sample (0.6 vs
  0.8; 0.5 vs 0.6). It did beat `agent_single` (0.6 vs 0.2).
- The five-question qrels ceiling is 9/12 gold points; `gold_point_citation_recall`
  cannot reach 1.0. Fixed tokens exclude the precomputed rerank cost; agent
  tokens include decisions and answers. `agent_single`'s mean latency includes
  the first-search model load (p50 2.7s).
- **n=5 convenience sample, no CI or significance test; citation-level only.**
  Do not use these numbers as answer accuracy or a general multi-round gain.
  The full 60-question A/B remains unrun.

The pilot report was generated before the per-stage-latency/provenance
guardrails: its `agent_single` answer-latency mean (916.7 ms) includes two 0 ms
abstentions; the three measured answer calls average ~1,528 ms. The current
harness excludes no-call records, shows n, and records the `agent_index` block.
The pilot omitted `--rag-dir`, so fixed and agent both used `eval120/index`.

The pilot command was:

```bat
... eval/agent_eval.py --question-ids q001,q002,q004,q009,q010 ^
  --systems fixed_rag,agent_single,agent_multi --out <new empty dir>
```

Inspect `<out>/summary.json`, each `<system>/aggregate.json`, and the
per-question checkpoints before deciding whether to spend on the full 60.
