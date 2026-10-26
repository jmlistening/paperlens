# PaperLens Answer Repair (one-shot, opt-in)

`eval/answer_repair.py` retries **once** the answer-eval checkpoints that were
`answer_status == "invalid"` or `status == "failed"`. It is opt-in: nothing else
in the pipeline calls it, and a normal `eval/answer_eval.py` run never invokes
it.

It is a schema/citation repair pass, not a correctness pass. A repaired answer
still only proves that the JSON parsed and that every quote is an exact
10-800 character substring of its cited evidence. It does not become a
faithfulness or accuracy metric.

> This module never re-runs retrieval, never changes `data/`, benchmark
> questions/qrels, or the original retrieval run. Tests are fully offline and
> use a temporary copy of a checkpoint and benchmark plus a fake `answer_fn`.

## Usage

```bash
# Preview candidates only: no model call, no files written.
python eval/answer_repair.py --dry-run

# Repair every invalid/failed checkpoint in the default eval directory.
python eval/answer_repair.py --max-tokens 8192

# Repair a subset, with an explicit model override.
python eval/answer_repair.py --ids q018,q101,q108 --answer-model deepseek-chat
```

| Flag | Default | Meaning |
|---|---|---|
| `--eval-dir` | `eval/benchmark_v2/runs/answer_eval120` | Directory with `<question_id>.json` checkpoints. Also receives the reports. |
| `--benchmark-dir` | `eval/benchmark_v2/eval120` | Directory with `questions.jsonl`, `qrels.jsonl` and `index/corpus_clean.json`. |
| `--ids` | all checkpoints | Comma-separated question IDs to consider. Missing IDs are reported as skipped. |
| `--answer-model` | `DEEPSEEK_MODEL` from `.env` | Optional model override for the repair call. |
| `--max-tokens` | `8192` | Completion budget for the repair call. |
| `--dry-run` | off | Classify candidates and print a summary. Makes **no** model call and writes **no** files. |

## Selection rules

A checkpoint is selected only when:

- `answer_status == "invalid"` (the model answered but validation failed), or
- `status == "failed"` (the model call or retrieval failed).

Everything else is left untouched:

- **Already valid** (`answered` / `partial` / `insufficient` with `status` not
  `failed`) is a no-op and listed under `already_valid`.
- **No usable evidence** (for example a retrieval/rerank failure with
  `evidence: []`) is reported under `skipped` and **never repaired**, because
  there is nothing valid to cite.
- A requested `--ids` entry with no checkpoint is reported under `skipped`.

## What the model receives

The original payload is rebuilt exactly, read-only, from the checkpoint's
evidence metadata plus the benchmark `question_zh`:

```json
{
  "question": "<question_zh>",
  "evidence": [
    {"id": "E1", "title": "...", "text": "..."}
  ]
}
```

`parent_chunk_id` is resolved to the chunk text in
`<benchmark-dir>/index/corpus_clean.json`; the evidence order is preserved, so
the payload matches the one used by the original run. Gold data (gold parent
IDs and gold-point quotes) is **not** sent to the model.

Messages are:

1. The verbatim `answer_question.PROMPT` system message.
2. The rebuilt original payload.
3. A repair user message containing the previous validation error and the
   previous raw response, asking for a compact, complete JSON object with
   exact 10-800 character evidence quotes.

**Network-only failure exception:** when `raw_response is None` but evidence is
present (a failed model call with no output), the original messages are retried
unchanged, without appending the repair message.

At most **one** model call is made per selected checkpoint per invocation.

## Checkpoint semantics

Validation uses `eval.answer_eval.evaluate_answer`, so it matches the original
evaluator exactly.

On success (`answered` / `partial` / `insufficient`):

- `status` becomes `ok`;
- `answer_status`, `metrics`, `raw_response`, `usage`, and citation ID lists are
  replaced with the repair result;
- `tokens` is `old_tokens + new_tokens`;
- `latency_ms` is `old_latency + new_latency`;
- `attempt` becomes `attempt + 1`;
- `previous_errors` accumulates the prior error;
- `repaired_at` and an audit `repair` block (previous error/raw response/usage/
  model) are added;
- only that checkpoint file is rewritten.

On failure:

- the original `status`, `answer_status`, `error`, `raw_response`, `metrics`,
  `tokens` and `latency_ms` are preserved unchanged (the failure is not hidden);
- a visible `repaired_attempt` block records the repair attempt's model, mode,
  error, answer status, metrics, raw response, usage, tokens and latency.

## Reports

Unless `--dry-run`, the pass writes:

- `<eval-dir>/repair_report.json` - `repaired`, `skipped`, `still_invalid`,
  `already_valid`, `would_repair` entries with question IDs and reasons, plus
  counts.
- `<eval-dir>/repair_report.md` - the same information as readable tables.

A dry run returns the same report in memory (with `would_repair`) and prints a
summary, but writes nothing.

## Tests

```bash
python -I -B -X utf8 -m unittest discover -s tests -p "test_answer_repair.py" -v
```

The tests cover: a successful repair, a still-invalid repair, a no-evidence
skip, an already-valid no-op, a dry run with no writes, token/latency
accumulation, the JSON/Markdown reports, and network-only retry of the original
messages. Every test uses a temporary synthetic benchmark/checkpoint copy and a
fake `answer_fn`; no network call, paid API, `data/` access or real checkpoint
write occurs.

## Notes

- The module builds its own DeepSeek answer function because
  `eval.answer_eval.make_deepseek_answer_fn` hardcodes `max_tokens=4096`; the
  repair pass exposes `--max-tokens` (default `8192`). The prompt, response
  format and thinking-disabled setting are otherwise identical.
- `--dry-run` is safe to run against the real checkpoint directory: it only
  reads files.
- There is no automatic retry loop. Re-running the CLI is a new one-shot pass;
  each run makes at most one call per selected checkpoint.
