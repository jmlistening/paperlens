# PaperLens Answer-Level Evaluation (citation-level)

`eval/answer_eval.py` evaluates answers produced from an **already computed**
retrieval run. It does not re-run retrieval, does not call an LLM judge, and
does not perform human review.

> **What this measures:** JSON schema validity, exact quote copying, and
> whether the answer cited the gold parent chunks and gold-point quotes.
>
> **What this does NOT measure:** answer accuracy, correctness, completeness,
> faithfulness, or claim-level entailment. A high score here does not mean the
> Chinese explanation is correct; it means the structured answer passed the
> mechanical checks. See `aggregate.json` field `disclaimer` and the top of
> `report.md`.

## Inputs

| Input | Default | Notes |
|---|---|---|
| `--benchmark-dir` | `eval/benchmark_v2/eval120` | Contains `questions.jsonl` + `qrels.jsonl` and `index/corpus_clean.json`. |
| `--retrieval-run` | `eval/benchmark_v2/runs/eval120_rerank` | Run root with one subdirectory per system, **or** a single system directory. |
| `--retrieval-system` | `hybrid_rrf_rerank_top6` | System subdirectory under the run root. |
| `--out` | `<run root>/answer_eval/<system>` | Never allowed to equal the retrieval system directory, so checkpoints cannot overwrite `q*.json`. |
| `--limit` | none | Debugging cap on the number of benchmark questions. |
| `--answer-model` | `DEEPSEEK_MODEL` from `.env` | Optional model override. |

Gold data comes from the benchmark:

- `qrels.jsonl` -> `gold_parent_ids` for `cited_gold_parent_recall`.
- `questions.jsonl` -> `gold_points[*].quote` for
  `gold_point_citation_recall`.

Retrieval records are read from `<retrieval-system>/<question_id>.json`. The
`retrieved_parent_ids` list is mapped, in order, to chunks in
`index/corpus_clean.json`; each evidence unit carries `id`, `title`, `text`,
`page_number` and the original parent chunk ID. The model receives exactly the
payload shape used by `answer_question.main`:

```json
{
  "question": "<question_zh>",
  "evidence": [
    {"id": "E1", "title": "...", "text": "..."}
  ]
}
```

The system prompt is `answer_question.PROMPT`, verbatim.

## Metrics

| Metric | Definition |
|---|---|
| `schema_success` | 1 if the response parsed as a JSON object and passed `answer_question.validate_answer` after `answer_schema.normalize_answer_schema`; otherwise 0. |
| `quote_integrity` | 1 if all citation quotes are exact whitespace-normalized substrings of their cited evidence text; 0 when schema validation fails. (`validate_answer` implies this, so it mirrors schema success.) |
| `cited_gold_parent_recall` | `|cited parent IDs ∩ gold_parent_ids| / |gold_parent_ids|`. 0 for a schema-invalid answer; `null` when the question has no gold parents. |
| `gold_point_citation_recall` | Fraction of gold points whose normalized `quote` appears inside at least one **cited** evidence text. 0 for schema-invalid answers; `null` when there are no gold points. |

Also recorded per question: `answer_status` (`answered` / `partial` /
`insufficient` / `invalid`), `tokens`, `latency_ms`, raw response (or a
sanitized error), retrieved IDs, cited IDs and gold IDs.

`quote_integrity` is not independent of `schema_success` today: validation
requires exact quotes. It is kept separate so a future schema that stops
checking quotes does not silently inherit a perfect integrity score.

Failed or schema-invalid answers stay in the denominator and are scored 0;
they are never dropped. Empty-gold questions are excluded only from the ratio
metrics (their `n` is smaller), not from `schema_success` or status counts.

## Outputs

For each non-resumed question, `eval/answer_eval.py` writes:

```
<out>/<question_id>.json      # checkpoint: raw response/error + metrics
<out>/aggregate.json          # means, 95% bootstrap CIs, status counts, failures
<out>/report.md               # human-readable table + explicit disclaimer
```

`aggregate.json` contains:

- `n`, `n_failed` and `failures` (failed pipeline or `schema_success != 1`).
- `status_counts` over `answered` / `partial` / `insufficient` / `invalid`.
- `metrics.<metric>` = `{mean, ci_low, ci_high, n, resamples}`.
- `tokens` = total and mean.
- `latency_ms` = mean, p50 and p95.
- `per_question` and a fixed `disclaimer`.

Bootstrap: percentile bootstrap over question-level scores, reusing
`eval.analysis.make_report.bootstrap_ci` with 2,000 resamples and a fixed
per-metric SHA-256 seed. Intervals are deterministic for identical inputs and
are **descriptive only** because the benchmark is a fixed set.

## Checkpointing and resume

A checkpoint is terminal when `status` is `ok` or `failed` and its
`question_id` matches. Re-running the same `--out` reuses terminal checkpoints
and makes no model call for them. Delete the question's JSON file (or use a
fresh `--out`) to rescore it.

## Default answer function

`make_deepseek_answer_fn()` reads `.env` with UTF-8-BOM tolerance, requires
`DEEPSEEK_API_KEY` and `DEEPSEEK_MODEL`, and calls DeepSeek with:

- `messages` = system `PROMPT` + user payload,
- `response_format={"type": "json_object"}`,
- `extra_body={"thinking": {"type": "disabled"}}`,
- `max_tokens=4096`,
- `timeout=120`, `max_retries=0`.

It records usage on `answer_fn.last_usage` and the resolved model on
`answer_fn.model`. Tests inject their own `answer_fn(messages) -> str | dict`
and never touch the network.

## CLI

```powershell
python eval/answer_eval.py `
  --benchmark-dir eval/benchmark_v2/eval120 `
  --retrieval-run eval/benchmark_v2/runs/eval120_rerank `
  --retrieval-system hybrid_rrf_rerank_top6 `
  --out eval/benchmark_v2/runs/eval120_rerank/answer_eval/hybrid_rrf_rerank_top6 `
  --limit 2
```

`--limit 2` is a smoke test and still calls the paid API. The real 60-question
run was intentionally not executed here; run it yourself when ready. Omitting
`--limit` scores every question in the benchmark.

## Tests

```powershell
python -I -B -X utf8 -m unittest discover -s tests -p "test_answer_eval.py" -v
```

The tests build a fake benchmark index, a fake retrieval run and a fake
`answer_fn`. They cover a valid gold citation, a JSON string response, invalid
JSON, a fabricated quote, no gold citation, an empty-gold question, an answer
function exception, a failed retrieval record, aggregate bootstrap
determinism, checkpoint resume, retrieval-run path resolution and missing
DeepSeek configuration. No network, no model, no `data/` and no real benchmark
files are touched.

## Limitations

- Not answer accuracy or faithfulness; no judge and no human review.
- Quote matching is whitespace-normalized substring matching only (case and
  punctuation are preserved, matching `validate_answer`).
- `gold_point_citation_recall` checks copying, not interpretation or use.
- `cited_gold_parent_recall` credits a gold parent only if the answer cited at
  least one evidence unit mapped to it.
- Retrieval failures are scored 0 and reported under `failures`; they are not
  silently skipped.
- The benchmark is fixed, so bootstrap intervals are descriptive, not
  inferential.
