# PaperLens Phase 1 Implementation Plan (M0 + Push Preview + Eval-Lite Retrieval)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the existing PaperLens state, then build two independently testable vertical slices in parallel: (P) a 163-first arXiv push pipeline that can preview a real digest without sending, and (E) a 40-paper held-out test index with BM25/dense/RRF retrieval metrics.

**Architecture:** Track P adds new push modules on top of the existing fetch/rank/summarize/send_digest code; it never mutates old data and defaults to `enabled=false`. Track E re-parses 40 held-out PDFs with the same `prepare_corpus`/`clean_corpus` pipeline as the dev index, freezes a separate `eval/benchmark_v2/index`, and measures deterministic retrieval metrics. The two tracks share only M0 scaffolding, `tests/__init__.py`, `requirements.txt`, `.env.example`, `README.md`, and `eval/CLAIMS.md`.

**Tech Stack:** Python 3.11.17 (`python`), Streamlit, SQLite, FAISS, sentence-transformers E5-small, rank-bm25, PyMuPDF, unittest, Windows Task Scheduler.

**Spec:** `docs/superpowers/specs/2026-10-25-paperlens-eval-and-push-design.md`

## Global Constraints

- Work only in `<repo>`; never modify `data/dataset_v1`, `data/dataset_history_v1`, `data/rag`, `data/reading.sqlite3`, `data/delivery.sqlite3`, or old failure records.
- All new runtime state goes under `data/push/` or `eval/benchmark_v2/`; test output goes under `tests/.tmp/`.
- Push default is `enabled=false`; `preview` never connects to SMTP; real send requires `--confirm` and `first_send_approved_at`.
- Never read, print, or commit `.env` values; `.env.example` contains placeholders only.
- Never fabricate metrics. Until a manifest-backed measurement exists, use `TODO`, not a number.
- Python commands use `python`; run from `<repo>`.
- Each task ends with `python -m unittest` for its test module and a git commit.
- 163 is the primary provider; QQ is a documented preset. SMTP accepted ≠ inbox delivery.

## Review Focus

- Test index must be built from PDFs through the same `prepare_corpus` + `clean_corpus` path as the dev index, not from `prepared_v1` pages; otherwise retrieval numbers are not comparable.
- `push_run` must exit before any network/model/SMTP call when `enabled=false`, `paused=true`, or `quiet_mode=true` (quiet may render preview only).
- Deduplication must use the version-stripped arXiv base ID; the same family must never appear twice in a digest or in both dev/test.
- SMTP credentials and recipient addresses must never appear in logs, status JSON, git, or test fixtures.
- An experiment run must count a failed/schema-invalid answer as a failure, not silently drop it.
- 40 questions can only support descriptive statistics; the report must not claim significance.

---

## Task 0: M0 freeze, repo hygiene, and test harness

**Files:**
- Create: `.gitignore`, `requirements.txt`, `.env.example`, `tests/__init__.py`, `README.md`, `eval/CLAIMS.md`, `eval/snapshot_baseline.py`, `eval/baseline_snapshot_20261025.json`
- Modify: `tests/test_answer_quality.py`, `tests/test_coverage_display.py`, `tests/test_stage1a_coordination.py` (import/fixture robustness only)
- Test: `tests/test_repo_hygiene.py`

**Interfaces:**
- Produces: `eval/snapshot_baseline.py: main(out_path: Path) -> dict` (read-only counts/hashes)
- Produces: a repo where `python -m unittest discover -s tests` collects the full offline suite.

- [ ] **Step 1: Write `tests/test_repo_hygiene.py`** asserting: `.env` and `data/` are ignored by `git check-ignore`; `tests/__init__.py` exists; `requirements.txt` contains `streamlit`, `faiss-cpu` or `faiss`, `sentence-transformers`; `.env.example` contains `SMTP_PASSWORD=` with no real-looking secret; `eval/CLAIMS.md` exists.
- [ ] **Step 2: Run `python -m unittest tests.test_repo_hygiene -v`; expect FAIL.**
- [ ] **Step 3: Create the M0 files.** `.gitignore` must exclude at least `data/`, `.env`, `__pycache__/`, `.idea/`, `.vscode/`, `backups/`, `*.pdf`, `*.zip`, `docs/handoff/evidence_*/`, `docs/stage1*/**/deliveries/`, `.pytest_cache/`, `tests/.tmp/`. Keep `docs/superpowers/` tracked. `requirements.txt` pins the installed versions relevant to runtime (from `pip list`), with a comment that GPU torch is optional. `.env.example` lists `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`, `SMTP_HOST`, `SMTP_PORT`, `SMTP_SECURITY`, `SMTP_USER`, `SMTP_PASSWORD`, `MAIL_TO` with placeholders.
- [ ] **Step 4: Fix test imports.** Change `from test_agent import ...` in `tests/test_answer_quality.py` to `from tests.test_agent import ...`; make `tests/test_coverage_display.py` and `tests/test_stage1a_coordination.py` skip with `unittest.SkipTest` if their `docs/` or `eval/development_cases` fixture is absent. Add `tests/__init__.py`.
- [ ] **Step 5: Run `python -m unittest discover -s tests -v`; expect all offline tests to pass (record the exact count in `README.md`; do not claim CI yet).**
- [ ] **Step 6: Implement `eval/snapshot_baseline.py`** to read (read-only) `data/rag/library_manifest.json`, `data/rag/answers/`, `data/rag/qa_runs/`, `data/dataset_v1/state.json`, `data/dataset_history_v1/state.json`, and write counts + SHA-256 of key manifests to `eval/baseline_snapshot_20261025.json`. It must not import project business modules or touch `.env`.
- [ ] **Step 7: Run `python eval/snapshot_baseline.py`; expect exit 0 and a JSON with `papers_indexed=8`, `vectors=460`, `qa_answers=11`, `qa_runs=5` (adjust only if the read-only count differs, and record why).**
- [ ] **Step 8: `git init`, `git add` only code/docs/config templates/tests, `git commit -m "chore: freeze PaperLens baseline and repo hygiene"`.** Verify `git status --ignored` shows `data/` and `.env` ignored.

---

## Track P: 163-first Push Preview

### Task P1: push config and env validation

**Files:**
- Create: `push_config.py`, `push_config.json`, `tests/test_push_config.py`
- Modify: `.env.example`

**Interfaces:**
- Produces: `PushConfig` dataclass with fields from the spec §7.1.
- Produces: `load_push_config(path: Path = DEFAULT_CONFIG_PATH) -> PushConfig`
- Produces: `validate_push_config(config: PushConfig, env: Mapping[str, str]) -> list[str]` returning human-readable missing/invalid problems; never returns secrets.
- Produces: `SMTP_PRESETS: dict[str, SmtpPreset]` with `163` and `qq`.

- [ ] **Step 1: Write failing tests** for: default `enabled is False`; unknown provider rejected; `enabled=true` with missing `SMTP_USER`/`SMTP_PASSWORD`/`MAIL_TO` returns errors; a config containing a `password` key is rejected; `max_per_digest` ≤ `top_k`; `provider=163` maps to `smtp.163.com:465` SSL.
- [ ] **Step 2: Run `python -m unittest tests.test_push_config -v`; expect FAIL.**
- [ ] **Step 3: Implement `PushConfig`, `load_push_config`, `validate_push_config`, `SMTP_PRESETS`.** Use `dataclasses`, `json`, `pathlib`; read `.env` only through `os.environ` or a passed mapping, never print values. Reject unknown keys and a literal password field.
- [ ] **Step 4: Run the test; expect PASS.**
- [ ] **Step 5: Commit.**

### Task P2: frontier state and dedup ledger

**Files:**
- Create: `push_state.py`, `tests/test_push_state.py`

**Interfaces:**
- Produces: `PushState(path: Path)` with `init_schema()`, `claim_slot(slot_key: str, trigger: str) -> bool`, `record_paper(paper: dict, status: str) -> None`, `mark_status(paper_key: str, status: str) -> None`, `is_seen(paper_key: str) -> bool`, `eligible(limit: int) -> list[dict]`, `set_watermark(name: str, value: str) -> None`, `get_watermark(name: str) -> str | None`, `record_impression(run_id: str, paper_key: str, position: int, score: float, channel: str) -> None`, `status_counts() -> dict[str, int]`, `reap_stale(now: datetime, max_age_hours: int = 2) -> int`.
- Produces: tables `papers`, `runs`, `watermarks`, `impressions`.

- [ ] **Step 1: Write failing tests** for: `paper_key` strips `vN` and dedups `2601.12345v1` vs `v2`; `claim_slot` is unique and returns False on duplicate; `set_watermark` only changes when called; `eligible` excludes `delivered`/`ignored` and respects FIFO; `reap_stale` marks `running` > 2h as `interrupted`; impressions persist.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement SQLite schema + methods** with `PRAGMA journal_mode=WAL`, `BEGIN IMMEDIATE` for `claim_slot`, and `paper_key = re.sub(r'v\\d+$', '', arxiv_id)`. Store timestamps as UTC ISO strings.
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit.**

### Task P3: incremental fetch window

**Files:**
- Create: `push_fetch.py`, `tests/test_push_fetch.py`
- Modify: `fetch_papers.py` only if a small, backward-compatible hook is needed (do not change default CLI behavior).

**Interfaces:**
- Produces: `compute_window(now: datetime, watermark: str | None, config: PushConfig) -> tuple[datetime, datetime]`
- Produces: `fetch_candidates(profile: dict, start: datetime, end: datetime, config: PushConfig, client: FakeArxivClient | None = None) -> list[dict]`
- Produces: `build_queries(profile: dict, categories: list[str]) -> list[dict]`

- [ ] **Step 1: Write failing tests** with a fake client: first run uses `initial_lookback_days`; later run uses `max(watermark - overlap, now - catchup)`; categories are added as `cat:` clauses; 429 raises a typed `RateLimited` and does not advance the watermark; candidate cap writes overflow to `queued`.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement** by importing and reusing `fetch_papers.read_profile`, `normalize_result`, `merge_groups`, `term_clause`, `check_cooldown`, `record_cooldown`; do not copy/paste the query logic.
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit.**

### Task P4: ranking and cold start

**Files:**
- Create: `push_rank.py`, `tests/test_push_rank.py`
- Modify: `rank_papers.py` only to extract `rank_batch(...)` without changing existing CLI behavior.

**Interfaces:**
- Produces: `rank_batch(papers: list[dict], profile: dict, feedback: dict, model_config: dict, cache_dir: Path) -> list[dict]` (each item gets `llm_score`, `favorite_similarity`, `final_score`, `reason`, `feedback_refs`).
- Produces: `select_digest(ranked: list[dict], config: PushConfig, ignored_keys: set[str]) -> list[dict]`.
- Produces: `cold_start(feedback: dict) -> bool`.

- [ ] **Step 1: Write failing tests** for: cold start returns `final == llm_score`; `min_score` applies only to `llm_score`; `favorite_weight=0` equals llm-only; `ignored` keys are hard-excluded; category diversity caps at `max_per_primary_category`; rank cache key includes profile/model/prompt hash.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement** by extracting the existing prompt/validation/select logic; E5 similarity loads `data/models` with `local_files_only=True` and falls back to lexical similarity with a warning.
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit.**

### Task P5: 163/QQ mail transport

**Files:**
- Create: `mail_transport.py`, `tests/test_mail_transport.py`

**Interfaces:**
- Produces: `SmtpSettings` dataclass (`host`, `port`, `security`, `user`, `password`, `mail_to`, `from_name`).
- Produces: `resolve_smtp(provider: str, env: Mapping[str, str]) -> SmtpSettings`
- Produces: `test_login(settings: SmtpSettings) -> None`, `noop(settings: SmtpSettings) -> None`, `send_message(settings: SmtpSettings, msg: EmailMessage) -> None`.

- [ ] **Step 1: Write failing tests** with monkeypatched `smtplib`: 163 → `smtp.163.com:465` SSL; QQ → `smtp.qq.com:465` SSL; 587 uses `SMTP` + `starttls()`; 25 with non-localhost is rejected; `repr(settings)` and exception messages contain no password; `test_login` sends no message.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement** with `smtplib.SMTP_SSL` / `smtplib.SMTP`, 60s timeout, certificate verification on; never log response bodies.
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit.**

### Task P6: digest rendering

**Files:**
- Create: `push_digest.py`, `tests/test_push_digest.py`

**Interfaces:**
- Produces: `Digest` dataclass (`subject`, `html`, `text`, `attachments`, `paper_keys`).
- Produces: `build_digest(selected: list[dict], summaries: dict, config: PushConfig) -> Digest`
- Produces: `render_html(digest: Digest) -> str`, `render_text(digest: Digest) -> str`.

- [ ] **Step 1: Write failing tests** for: links-only by default; no `<img>`/tracking pixel; title/abstract HTML-escaped; only `arxiv.org`/`www.arxiv.org` links; footer contains the disable path; attachment over 10MB degrades to a link; digest contains no notes/reading history.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement** with `html.escape`, a minimal inline-CSS template, and the existing `send_digest.checked_url` helper if reusable.
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit.**

### Task P7: push orchestration (preview only)

**Files:**
- Create: `push_run.py`, `tests/test_push_run.py`

**Interfaces:**
- Produces: `RunResult` dataclass (`run_id`, `status`, `candidate_count`, `selected_count`, `sent`, `errors`).
- Produces: `run_push(mode: Literal['scheduled','preview','send'], config_path: Path, now: datetime | None = None, deps: PushDeps | None = None) -> RunResult`
- Produces: `PushDeps` injection point for fake arXiv/LLM/SMTP/clock.

- [ ] **Step 1: Write failing tests** for: `enabled=false` → no network/model/SMTP calls; `paused=true` → no calls; `preview` → fetch/rank/digest but zero SMTP; `send` without `first_send_approved_at` → refuses; fetch failure → no send and watermark unchanged; SMTP `uncertain` → auto-pause; `slot_key` duplicate → busy; `resume` reuses cached rank/summary.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement** gates first, then the pipeline; reuse `run_pipeline.pipeline_lock`, `Tee`, `validate_candidates`; write `data/push/runs/<run_id>/run.json`, `digest_preview.html`, `pipeline.log`.
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit.**

### Task P8: control plane and Streamlit page

**Files:**
- Create: `push_ctl.py`, `push_status.py`, `pages/3_推送设置.py`, `tests/test_push_status.py`, `tests/test_push_ui_contract.py`

**Interfaces:**
- Produces: CLI `init`, `doctor`, `enable`, `disable`, `pause`, `resume`, `preview`, `send --confirm`, `test-email [--send --confirm]`, `approve-first-send --confirm`, `schedule print|install|remove`, `history`, `status --json`.
- Produces: `status_json(config_path: Path) -> dict` with masked recipient (`a***@163.com`), `last_success`, `next_expected`, `queue`, `counts`.

- [ ] **Step 1: Write failing tests** for: `disable` makes `push_run` a no-op; `status` masks the email; `status` never contains `SMTP_PASSWORD`; `approve-first-send` requires `--confirm`; the Streamlit page imports without side effects.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement** the CLI with `argparse`, status as JSON, and a Streamlit page using the same config/status functions.
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit.**

### Task P9: Task Scheduler scripts and docs

**Files:**
- Create: `scripts/register_push_task.ps1`, `scripts/unregister_push_task.ps1`, `scripts/run_push.bat`, `docs/PUSH.md`, `tests/test_scheduler_contract.py`

**Interfaces:**
- Produces: a scheduled task named `PaperLens Frontier Push` that runs `run_push.bat` daily at `time_local`, with `StartWhenAvailable`, `MultipleInstances IgnoreNew`, `RestartCount 2`, `RestartInterval 10m`, `ExecutionTimeLimit 2h`.

- [ ] **Step 1: Write failing contract tests** that read the scripts and assert: absolute Python path, `--scheduled`, task name, no password, `StartWhenAvailable`, `IgnoreNew`, and `-WhatIf` support.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement the PS1/bat scripts and `docs/PUSH.md`** covering 163/QQ authorization codes, install/remove, on-time vs catch-up, and how to disable.
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit.**

### Task P10: offline integration and 163 checklist

**Files:**
- Create: `tests/test_push_integration.py`, `tests/fixtures/push/` (fake arXiv/LLM/SMTP JSON)
- Modify: `docs/PUSH.md`

- [ ] **Step 1: Write an integration test** that runs `push_run --preview` end-to-end with fake deps: 3 candidates → 1 selected → digest HTML exists → no SMTP call → `run.json` status `preview_ready`.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Add the fixtures and wire `PushDeps`; make the test pass.**
- [ ] **Step 4: Run the full offline suite; expect all pass.**
- [ ] **Step 5: Add the real 163 checklist to `docs/PUSH.md`**: user fills `SMTP_PASSWORD` in `.env`; run `push_ctl.py test-email` (login only); then `test-email --send --confirm`; confirm receipt; `approve-first-send --confirm`; `push_run.py --preview`; `push_run.py --send --confirm`. Never automate these without the user present.
- [ ] **Step 6: Commit.**

---

## Track E: Eval-Lite Retrieval

### Task E1: select 40 held-out papers and freeze splits

**Files:**
- Create: `eval/benchmark_v2/select_corpus.py`, `eval/benchmark_v2/splits.json`, `eval/benchmark_v2/CORPUS_CARD.md`, `tests/test_select_corpus.py`

**Interfaces:**
- Produces: `select_test_papers(catalog: list[dict], dev_families: set[str], recent_n: int, history_n: int, seed: int = 20261025) -> list[dict]`
- Produces: `splits.json` with `dev`, `test`, `reserve`, `excluded` lists and a `manifest_hash`.

- [ ] **Step 1: Write failing tests** with a synthetic catalog: excludes 8 dev families, `out_of_scope`/`uncertain`, flagged papers, and near-duplicate titles; selects exactly 24 recent + 16 history; family unique; deterministic with the same seed.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement** by reading the prepared manifests/state read-only; write `splits.json`; `CORPUS_CARD.md` lists IDs, years, core/adjacent, exclusion reasons, and license status `not_verified`.
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit.**

### Task E2: build a separate test index from 40 PDFs

**Files:**
- Create: `eval/benchmark_v2/build_test_index.py`, `tests/test_build_test_index.py`
- Modify: none in `prepare_corpus.py`/`clean_corpus.py` unless a backward-compatible `out_dir` parameter is required.

**Interfaces:**
- Produces: `build_test_index(pdf_paths: list[Path], out_dir: Path, chunker_config: dict, embedder: object) -> dict` writing `corpus.json`, `corpus_clean.json`, `vector_store.npz`, `manifest.json`.
- Produces: `manifest.json` with paper/family IDs, PDF SHA-256, chunker hash, model name/revision, vector count/dimension, index hash.

- [ ] **Step 1: Write failing tests** with 2 tiny fake PDFs: uses `prepare_corpus.extract_paper` + `clean_corpus` (not `prepared_v1` pages); parent/child offsets are preserved; `data/rag` is untouched; `search_vector.load_store` accepts the output; manifest hashes exist.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement** by calling the existing prepare/clean/build functions with an explicit output directory; do not rebuild the dev index. If the existing functions hardcode paths, add optional parameters with defaults preserving old behavior.
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit.**

### Task E3: leakage audit

**Files:**
- Create: `eval/benchmark_v2/leakage_audit.py`, `tests/test_leakage_audit.py`

**Interfaces:**
- Produces: `audit_leakage(splits: dict, corpus: list[dict], index_manifest: dict) -> dict` with `dev_test_family_overlap`, `near_duplicate_pairs`, `title_collisions`, `index_namespace_ok`.

- [ ] **Step 1: Write failing tests** for: an injected family overlap is detected; title Jaccard ≥0.8 is flagged; a clean split returns zero overlaps.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement** with normalized titles/abstracts, shingle/embedding optional, and a clear `review_required` list.
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit.**

### Task E4: retrieval core with BM25, dense, and RRF

**Files:**
- Create: `retrieval_core.py`, `tests/test_retrieval_core.py`
- Modify: `search_evidence.py`, `search_bm25.py` only for thin adapters; defaults unchanged.

**Interfaces:**
- Produces: `Hit` dataclass (`chunk_id`, `parent_chunk_id`, `paper_id`, `score`, `source`).
- Produces: `retrieve_dense(index, query: str, k: int, paper_filter: str | None = None) -> list[Hit]`
- Produces: `retrieve_bm25(corpus, query_en: str, k: int, paper_filter: str | None = None) -> list[Hit]`
- Produces: `rrf_fuse(dense: list[Hit], bm25: list[Hit], k: int = 60, top_n: int = 6) -> list[Hit]`
- Produces: `retrieve(mode: Literal['dense','bm25','hybrid'], query_zh: str, query_en: str, k: int, ...) -> list[Hit]`

- [ ] **Step 1: Write failing tests** for: RRF ordering on a hand-built example; dedup by `parent_chunk_id`; paper filter; `k=6`; BM25 rejects a Chinese query; dense top-6 on a fake index.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement** the adapters and RRF; keep `search_evidence.py` default behavior unchanged.
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit.**

### Task E5: deterministic metrics

**Files:**
- Create: `eval/metrics_v2/metrics.py`, `eval/metrics_v2/score_run.py`, `tests/test_metrics_v2.py`

**Interfaces:**
- Produces: `recall_at_k(gold: set[str], retrieved: list[str], k: int) -> float`
- Produces: `all_gold_at_k(gold: set[str], retrieved: list[str], k: int) -> float`
- Produces: `mrr_at_k(gold: set[str], retrieved: list[str], k: int) -> float`
- Produces: `ndcg_at_k(gold: set[str], retrieved: list[str], k: int) -> float`
- Produces: `quote_integrity(citations: list[dict]) -> float`, `schema_success(runs: list[dict]) -> float`
- Produces: `score_run(run_dir: Path, gold_path: Path) -> dict`

- [ ] **Step 1: Write failing tests** with hand-calculated values: recall@3 = 2/3; MRR@10 = 1/2; nDCG@10 for a known ranking; quote integrity 1.0/0.0; a failed run counts as 0.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement** pure functions; no LLM calls.
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit.**

### Task E5b: question generation, validation, qrels, and freeze

**Files:**
- Create: `eval/benchmark_v2/generate_questions.py`, `eval/benchmark_v2/questions.draft.jsonl`, `eval/benchmark_v2/validate_questions.py`, `eval/benchmark_v2/build_qrels.py`, `eval/benchmark_v2/freeze.py`, `eval/benchmark_v2/ANNOTATION.md`, `tests/test_question_generation.py`, `tests/test_qrels.py`

**Interfaces:**
- `generate_candidate(paper: dict, pages: list[dict], chat_fn: Callable[[list[dict]], str], *, model: str) -> dict`
- `validate_candidate(candidate: dict, pages: list[dict], abstract: str) -> list[str]`
- `generate_for_split(splits_path: Path, out_path: Path, chat_fn, *, limit: int | None = None) -> dict` (writes JSONL + usage JSON)
- `build_qrels(questions_path: Path, clean_corpus_path: Path, out_path: Path) -> dict`
- Question schema: `question_id`, `paper_ids`, `question_zh`, `query_en`, `type`, `answerability`, `gold_points[{requirement, quote, pdf_page, section}]`, `missing_aspects`, `generator`, `review`.
- qrels schema: `question_id`, `gold_parent_ids`, `answerability`, `missing_aspects`.

- [ ] **Step 1: Write failing tests** with a fake `chat_fn`: a valid quote is accepted; a quote not present on the page is rejected; an abstract-only quote is rejected; a numeric question without a digit/unit is rejected; duplicate question text is rejected; qrels maps a quote to the parent chunk containing it and returns an empty gold list for unanswerable questions.
- [ ] **Step 2: Run the new test modules; expect FAIL.**
- [ ] **Step 3: Implement generation with page-marked text from `prepared_v1/pages/*.jsonl`, a Chinese prompt, injectable `chat_fn`, and a default DeepSeek client that records model/prompt hash/usage under `eval/benchmark_v2/generation_usage.json`. Never call the real API in tests.
- [ ] **Step 4: Implement validation and qrels.** Quotes must be whitespace-normalized substrings of a page; page numbers must exist; ≥70% of answerable gold quotes must not appear in the abstract; qrels must map to parent chunks in `eval/benchmark_v2/index/corpus_clean.json`.
- [ ] **Step 5: Run the tests; expect PASS.**
- [ ] **Step 6: Run a 3-paper pilot** with the real model and inspect the JSONL; do not freeze until the user reviews the questions.
- [ ] **Step 7: Commit the generator, validator, qrels builder, tests, and pilot output.**

### Task E6: experiment runner and report

**Files:**
- Create: `eval/run_experiment.py`, `eval/experiments/matrix.yaml`, `eval/analysis/make_report.py`, `tests/test_run_experiment.py`, `README_EVAL.md`

**Interfaces:**
- Produces: `run_experiment(matrix: dict, benchmark_dir: Path, out_dir: Path, deps: ExperimentDeps) -> dict`
- Produces: `make_report(results: list[dict], out_dir: Path) -> str` (Markdown + `results.json`)
- Produces: per-system `run.json` with `status`, `n`, `metrics`, `failures`, `tokens`.

- [ ] **Step 1: Write failing tests** with a fake system: 3 questions → 3 results; one failure counts as 0; `--budget` stops cleanly; resume skips completed questions; report contains `n` and `95% CI` but no significance claim for n=40.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement** the runner with checkpointing in `eval/benchmark_v2/runs/<system>/<question_id>.json`; `matrix.yaml` lists `bm25_top6`, `dense_top6`, `hybrid_rrf_top6`, `hybrid_rrf_rerank_top6`, `fixed_rag_matched`, `agent_single_round`.
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit.**

### Task E7: first real benchmark run (manual gate)

**Files:**
- Create: `eval/benchmark_v2/runs/20261025_retrieval_v1/` (results)
- Modify: `eval/CLAIMS.md`, `README_EVAL.md`

- [ ] **Step 1: Run `python eval/benchmark_v2/select_corpus.py` and inspect `splits.json`; manually verify 40 paper IDs and zero dev family overlap.**
- [ ] **Step 2: Run `python eval/benchmark_v2/build_test_index.py`; verify `manifest.json` vector count/dimension and that `data/rag` hashes are unchanged.**
- [ ] **Step 3: Run `python eval/run_experiment.py --matrix eval/experiments/matrix.yaml --systems bm25_top6,dense_top6,hybrid_rrf_top6 --budget 200000`; expect results for 40 questions per system.**
- [ ] **Step 4: Run `python eval/analysis/make_report.py`; inspect the Markdown table. Do not fill resume numbers until the run manifest hash is recorded.**
- [ ] **Step 5: Commit the results and update `eval/CLAIMS.md` with measured/not-measured status.**

---

## Task X1: integration, claims, and README

**Files:**
- Modify: `README.md`, `eval/CLAIMS.md`, `requirements.txt`, `.gitignore`
- Create: `docs/PHASE1_REPORT.md`

- [ ] **Step 1: Run the full offline suite (`python -m unittest discover -s tests -v`) and record the exact pass count.**
- [ ] **Step 2: Run `python push_ctl.py status --json` and `python eval/analysis/make_report.py`; capture outputs.**
- [ ] **Step 3: Update `README.md` with three sections: “Implemented (no metric yet)”, “Measured (manifest-backed)”, “Not measured / limitations”. Do not put `[X%]` placeholders in README.**
- [ ] **Step 4: Update `eval/CLAIMS.md`: every measured number points to a result file + manifest hash + n; every unimplemented claim says `TODO`.**
- [ ] **Step 5: Commit.**

---

## Execution Notes

- Track P and Track E touch disjoint files after Task 0 and can run in parallel.
- Do not run a real 163 send inside CI or any subagent. The real send is a user-present, explicit `--confirm` action.
- If the 40-paper test index build fails because of a PDF parsing issue, record the paper as excluded and replace it from the reserve pool; do not silently shrink the test set.
- If a retrieval metric is null or negative, keep it in the report. Do not tune on the test set; tune only on the 8 dev papers.
- The first resume numbers may be written only after Task E7 and only with the run manifest hash.
