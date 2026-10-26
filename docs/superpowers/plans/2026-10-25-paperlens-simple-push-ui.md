# PaperLens Simple Push UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the default Streamlit screen a simple personal arXiv digest tool (topics, email, count, preview, manual send, auto-push toggle, time) while moving research/eval/evidence and push history into an advanced mode, without publishing personal data.

**Architecture:** A new simple page and shared core modules sit above the existing push/rank/digest/state code. The existing pages move under an advanced navigation; a forecast preview copies the production frontier DB read-only, manual send can bypass auto-enable gates but never the first-send/verification gates, and every run writes structured digest history. Private profile/recipient state lives under `data/`, public examples replace personal files in git.

**Tech Stack:** Python 3.11 (`python`), Streamlit 1.66 (`st.navigation`/`st.Page`/`st.page_link`), SQLite, unittest, existing push modules, DeepSeek/OpenAI client.

**Spec:** `docs/superpowers/specs/2026-10-25-paperlens-simple-push-ui-design.md`

## Global Constraints

- Work in `<repo>`; do not modify `data/`, `.env`, real delivery DBs, or existing eval reports except by the tasks below.
- Tests run offline: no real arXiv/SMTP/DeepSeek calls. Use fakes and temp directories under `tests/.tmp/`.
- Preview never creates SMTP; manual/auto send never bypasses first-send approval or recipient verification.
- No real email, API key, authorization code, or personal profile in tracked files, logs, status JSON, or test fixtures. Active profile: `data/push/profile.json`; public example: `interests.example.json`.
- Simple page must not show evidence, trace, token, latency, schema/quote, rerank, or eval tables.
- Existing 476-test suite must keep passing; new tests use `python -m unittest discover -s tests -p "test_<name>.py" -v`.
- `push_config.json` is public and must never contain `MAIL_TO` or `SMTP_PASSWORD`.
- Keep baseline/eval/evidence/CLAIMS assets; only move them behind advanced navigation.
- Time is local HH:mm; count is 1–10, default 3; preview is a forecast of the next send; history lists only `status=sent` runs.

## Review Focus

- A recipient entered/changed in the UI must never be sent to (auto, scheduled, manual, or test) before its own verification; changing back also requires a new test.
- Preview must not advance production watermark/status and must never touch SMTP, even when auto-push is off/paused/quiet.
- Manual send with auto-push off must work, must not flip the auto switch, and must still require first-send approval/confirmation.
- If topic-to-query generation fails, the new topic must not be saved while old query groups remain (no mismatched candidate pool).
- Moving pages behind navigation must not trigger network/model/SMTP/state writes at import and must not leave a second enable path that bypasses verification.

---

### Task 1: Public profile and repository privacy

**Files:**
- Modify: `profile_config.py`
- Create: `interests.example.json`
- Modify: `.gitignore`, `tests/test_profile_config.py`, `tests/test_repo_hygiene.py`
- Modify: `push_run.py`, `pages/4_兴趣配置.py` (active profile path; page may move in Task 9)
- Git index: remove `interests.json`, `docs/handoff/ACCEPTANCE_REPORT.md`, `CURRENT_STATUS.md`, `NEXT_STAGE_PLAN.md`, `PROJECT_ROADMAP.md`

**Interfaces:**
- Produces: `profile_config.ACTIVE_PROFILE_REL = Path("data/push/profile.json")`; `EXAMPLE_PROFILE`, `LEGACY_PROFILE`; `profile_path(root: Path = ROOT) -> Path`; `load_active_profile(root: Path = ROOT) -> dict[str, Any]`; `save_active_profile(profile: dict[str, Any], root: Path = ROOT) -> dict[str, Any]`; `profile_sha256(profile: dict[str, Any]) -> str`.
- `profile_path` precedence: active if exists, else legacy if exists, else example.
- `profile_sha256` is the canonical JSON hash (`sort_keys=True`, `ensure_ascii=False`, validated profile); use it in `push_run`/`simple_ui` for preview freshness.

- [ ] **Step 1: Write failing tests** in `tests/test_active_profile.py`: active path wins over legacy; legacy wins over example; save writes `data/push/profile.json`; `interests.example.json` validates; changing active does not touch legacy.

```python
def test_active_profile_precedence():
    root = tmp_root()
    (root / "data/push").mkdir(parents=True)
    (root / "data/push/profile.json").write_text(json.dumps(PROFILE))
    (root / "interests.json").write_text(json.dumps({**PROFILE, "top_k": 9}))
    self.assertEqual(load_active_profile(root)["top_k"], 3)
```

- [ ] **Step 2: Run and verify failure**
Run: `python -m unittest tests.test_active_profile -v`
Expected: FAIL (`ImportError`/attribute missing).

- [ ] **Step 3: Implement the profile resolver and example**
Copy the current personal `interests.json` to `data/push/profile.json` locally (gitignored); replace tracked `interests.json` with a generic `interests.example.json`; add `interests.json`, `data/push/profile.json`, old handoff/answer artifacts, `docs/handoff/`, `docs/stage1a/`, `docs/stage1b/`, `docs/stage1c/`, `eval/results/`, `eval/reviews/`, `paperlens_answer*`, `review_notes/`, and the old root handoff files to `.gitignore`. `git rm --cached` the 4 tracked handoff files and `interests.json`; keep local copies. Update `push_run.py` and `pages/4_兴趣配置.py` to use `profile_path`.

- [ ] **Step 4: Run tests and repo checks**
Run: `python -m unittest tests.test_active_profile tests.test_profile_config tests.test_repo_hygiene -v`
Expected: PASS; `git check-ignore -q interests.json data/push/profile.json` returns 0.

- [ ] **Step 5: Commit**

```bash
git add profile_config.py interests.example.json .gitignore tests/test_active_profile.py tests/test_profile_config.py tests/test_repo_hygiene.py push_run.py pages/4_兴趣配置.py
git commit -m "chore: keep the active profile private and publish an example"
```

### Task 2: UI state and recipient verification

**Files:**
- Create: `ui_state.py`, `tests/test_ui_state.py`
- Modify: `mail_transport.py` only if an email normalizer is needed

**Interfaces:**
- Produces: `STATE_REL = Path("data/push/ui_state.json")`; `load_state(root: Path) -> dict`; `save_state(root: Path, state: Mapping) -> dict`; `set_recipient(root: Path, email: str) -> dict`; `recipient_hash(email: str) -> str`; `resolve_recipient(root: Path, env: Mapping[str, str]) -> dict` with `{recipient, verified, source, state_exists}`; `mail_env(root: Path, env: Mapping[str, str]) -> dict`; `mark_verified(root: Path, email: str, verified_at: str) -> dict`; `set_preview(root: Path, preview: Mapping) -> dict`; `get_preview(root: Path) -> dict | None`; `record_manual_send(root: Path, run_id: str, at: str, count: int) -> dict`.
- `recipient_hash` is `sha256(email.strip().casefold())`; never store the raw email in preview references.
- `resolve_recipient`: if state file exists, return its recipient and `verified_at` state, `source="ui_state"`; do not fall back to env. If no state file, return env `MAIL_TO`, `verified=True` (legacy compatibility), `source="env"`.
- `set_recipient` clears `verified_at` and `preview` when the email changes; same email keeps them.

- [ ] **Step 1: Write failing tests** for atomic write, email change clearing verification/preview, same email preserving, state precedence over env, legacy env fallback, and masked display via `mail_transport.mask_email`.
- [ ] **Step 2: Run and verify failure** with `python -m unittest tests.test_ui_state -v`.
- [ ] **Step 3: Implement `ui_state.py`** with a JSON email regex (`user@host.tld`), `os.replace` atomic writes, and no raw email in any log/exception.
- [ ] **Step 4: Run tests** with `python -m unittest tests.test_ui_state -v`; expected PASS.
- [ ] **Step 5: Commit** `feat: add private UI recipient state and verification`.

### Task 3: Shared auto-push gate, status, and scheduled recipient

**Files:**
- Modify: `push_ctl.py`, `push_status.py`, `push_run.py`, `tests/test_push_status.py`, `tests/test_push_ctl.py`, `tests/test_push_run.py`
- Modify: `pages/0_总览.py` and `pages/3_推送设置.py` to call the shared gate (pages move in Task 9)
- Create: `tests/test_auto_push_gate.py`

**Interfaces:**
- Produces: `push_ctl.set_auto_push(enabled: bool, root: Path = ROOT, config_path: Path = DEFAULT_CONFIG_PATH, env: Mapping | None = None) -> dict`.
  - `enabled=True`: require `ui_state.resolve_recipient` verified (or no UI state and legacy env recipient); reject `auto_paused=true`; reject `quiet_mode=true`; write `enabled=true, paused=false`; validate with the resolved `MAIL_TO`.
  - `enabled=False`: write `enabled=false`; do not clear `auto_paused`/`quiet_mode`.
- Produces: `push_ctl.task_installed() -> bool`, `push_ctl.register_task(root: Path, time_local: str) -> dict`, `push_ctl.unregister_task() -> dict` (subprocess wrappers; contract tests only).
- `push_status.status`/`push_ctl.status_json`: use `ui_state.resolve_recipient`; add `recipient_verified`, `recipient_source`; keep `mail_to` masked; no raw email.
- `run_push`: if `ui_state` exists and the recipient is unverified, return `status="recipient_unverified"` before network/model/SMTP; add status to fatal CLI statuses. If no UI state, legacy `.env` behavior remains.

- [ ] **Step 1: Write failing tests**: enabling with unverified UI recipient raises; auto-paused/quiet raises; enabling legacy env succeeds; disabling leaves safety flags; status masks email and reports verification; scheduled run with unverified UI state returns `recipient_unverified` and calls no client/SMTP; task wrapper contract strings.
- [ ] **Step 2: Run and verify failure** with `python -m unittest tests.test_auto_push_gate tests.test_push_status tests.test_push_ctl tests.test_push_run -v`.
- [ ] **Step 3: Implement `set_auto_push`, status resolver, scheduled fail-closed, and task wrappers.**
- [ ] **Step 4: Run tests**; expected PASS, existing push tests updated for the new fields.
- [ ] **Step 5: Commit** `feat: enforce verified recipients across auto-push paths`.

### Task 4: Forecast preview without SMTP, and structured digest history

**Files:**
- Modify: `push_run.py`, `push_config.py`, `tests/test_push_run.py`
- Create: `tests/test_forecast_preview.py`

**Interfaces:**
- `run_push(mode, config_path=..., now=None, deps=None, *, force: bool = False)`.
  - `force=True` only with `mode="preview"`: bypass `enabled`/`paused` gates; do not treat `quiet_mode` as send; still preview-only.
  - Preview validates config with `validate_push_config(..., require_smtp=False)` so missing SMTP credentials do not block it.
  - Preview snapshots production `data/push/frontier.sqlite3` into `data/push/preview.sqlite3` using `sqlite3.Connection.backup` before running; if production is absent, start empty. The preview may write only to the snapshot; production rows/watermark must be unchanged.
  - After `build_digest`, write `digest.json` with `{run_id, mode, created_at, subject, profile_sha256, papers:[{arxiv_id,paper_key,title,title_zh,reason,limitation,summary,url,score,categories}], warnings}`. `profile_sha256` is `profile_config.profile_sha256(profile)`; `title_zh` is parsed from summary line `中文标题：...`; missing -> `""`.
- `push_config.validate_push_config(config, env, require_smtp: bool = True)`; preview passes `False`; scheduled/send keep `True`.

- [ ] **Step 1: Write failing tests**: preview runs with `enabled=false`/`paused=true` when `force=True`; no SMTP; production `frontier.sqlite3` rows/watermark hash unchanged; preview snapshot has selected/impressions; missing SMTP env no longer blocks preview; `digest.json` has Chinese title/summary for fake summaries.
- [ ] **Step 2: Run and verify failure** with `python -m unittest tests.test_forecast_preview tests.test_push_run -v`.
- [ ] **Step 3: Implement snapshot, force-preview gate, SMTP-optional validation, and `digest.json`.**
- [ ] **Step 4: Run tests**; expected PASS; update the old missing-SMTP preview test to assert preview succeeds while send remains `config_error`.
- [ ] **Step 5: Commit** `feat: make preview a read-only forecast and persist digest history`.

### Task 5: Manual send while auto-push is off

**Files:**
- Modify: `push_run.py`, `tests/test_push_run.py`
- Create: `tests/test_manual_send.py`

**Interfaces:**
- `run_push(mode, ..., manual: bool = False, preview_run_id: str | None = None, force: bool = False)`.
  - `manual=True` only with `mode="send"`: bypass `enabled`/`paused`; still require `first_send_approved_at`, verified recipient, and `auto_paused != "true"`; ignore `quiet_mode` (explicit manual send is a real send).
  - Manual run uses a unique `manual-send-<UTC stamp>` slot; writes `run.json` with `"manual": true`; leaves `enabled/paused` unchanged.
  - If `preview_run_id` is supplied and the preview run's `digest.json` matches current profile hash/count/recipient hash/date, send that preview's papers/summaries via `_send_digest` without fetch/rank; otherwise run the normal send pipeline.
- `RunResult` and `run.json` expose `manual`; history uses it for a badge.

- [ ] **Step 1: Write failing tests**: manual send with `enabled=false,paused=true` works with fake SMTP and approved/verified state; it does not change config `enabled/paused`; unverified/unapproved/auto-paused manual send fails before SMTP; fresh preview sends exactly its papers; stale preview regenerates.
- [ ] **Step 2: Run and verify failure** with `python -m unittest tests.test_manual_send -v`.
- [ ] **Step 3: Implement the manual gate/slot/run field and preview-send path.**
- [ ] **Step 4: Run tests**; expected PASS.
- [ ] **Step 5: Commit** `feat: support manual digest sends when auto-push is off`.

### Task 6: Topic-to-query generation

**Files:**
- Create: `topic_queries.py`, `tests/test_topic_queries.py`

**Interfaces:**
- Produces: `class TopicQueryError(ValueError)`; `generate_query_groups(topics_text: str, chat_fn: Callable[[list[dict]], str], *, max_groups: int = 3) -> list[dict]`.
- `chat_fn` returns JSON `{"groups":[{"name":...,"any_terms":[...],"require_any_terms":[...]}]}`; validate 1–3 groups, non-empty name, 1–5 non-empty terms, max 5 required terms; strip duplicates; no unknown fields; raise `TopicQueryError` on any failure.
- The prompt must say translate/expand the user's own topics into arXiv search terms; do not add unrelated directions.

- [ ] **Step 1: Write failing tests** with fake `chat_fn`: valid Chinese topic -> groups; invalid JSON/empty/long -> `TopicQueryError`; no network.
- [ ] **Step 2: Run and verify failure** with `python -m unittest tests.test_topic_queries -v`.
- [ ] **Step 3: Implement the module and default DeepSeek `chat_fn` factory; tests inject fakes.**
- [ ] **Step 4: Run tests**; expected PASS.
- [ ] **Step 5: Commit** `feat: generate arXiv query groups from plain-language topics`.

### Task 7: Sent push history

**Files:**
- Create: `push_history.py`, `advanced/history.py`, `tests/test_push_history.py`

**Interfaces:**
- Produces: `load_sent_history(root: Path, *, limit: int = 50) -> list[dict]`.
  - Read `data/push/runs/*/run.json`; include only `status == "sent"` and `mode in {"send","scheduled","manual"}`; exclude preview/failed/no_digest.
  - Sort by `finished_at` (fallback `started_at`) descending.
  - Each item: `{run_id, sent_at, mode, manual, subject, paper_count, papers, digest_html, legacy}`.
  - `papers` from `digest.json` when present; otherwise `[]` and `legacy=True`, `digest_html` path to `digest_preview.html` for display.
  - Never read/return recipient or SMTP values.
- `advanced/history.py`: read-only Streamlit page with date/time, manual/auto badge, N papers; expandable cards with English title, Chinese title/summary/reason, arXiv link; `看邮件版` HTML; empty state when none.

- [ ] **Step 1: Write failing tests** with temp run dirs: sent/preview/failed filtering, date order, manual badge, Chinese fields, legacy fallback, no email keys.
- [ ] **Step 2: Run and verify failure** with `python -m unittest tests.test_push_history -v`.
- [ ] **Step 3: Implement `push_history.py` and the advanced page.**
- [ ] **Step 4: Run tests and an AppTest render** with a temp sent run; expected PASS, no model/SMTP calls.
- [ ] **Step 5: Commit** `feat: show sent push history in advanced mode`.

### Task 8: Simple page core logic and Streamlit view

**Files:**
- Create: `simple_ui.py`, `simple_page.py`, `tests/test_simple_ui.py`, `tests/test_simple_page_ui.py`
- Rewrite: `app.py` (navigation entry; see Task 9)

**Interfaces:**
- `simple_ui.save_settings(root, config_path, *, topics_text, email, count, time_local, chat_fn) -> dict`:
  - Save recipient via `ui_state.set_recipient`; save count to active profile `top_k` and `push_config.max_per_digest`; save `time_local`; if topics changed, generate groups first and save active profile only on success; on `TopicQueryError` do not save the new topic, raise a user-facing error.
  - If the saved recipient changes while `enabled=true`, call `set_auto_push(False)` first and return `auto_disabled=True`; the UI shows “邮箱已改，自动推送已关闭，验证后重新开启”.
  - If `time_local` changes while `enabled=true`, call `register_task(root, time_local)`; on failure return `task_error` and keep the config value; the UI shows “计划任务时间未更新”.
  - If the email field differs from the saved recipient (not yet previewed/saved), the UI disables auto/manual actions until `save_settings`/preview saves it.
- `simple_ui.preview(root, config_path, env, *, deps=None, now=None) -> RunResult`: call `run_push("preview", force=True)`; record a preview reference `{run_id, created_at, profile_sha256, count, recipient_hash, local_date}` in `ui_state`.
- `simple_ui.manual_send(root, config_path, env, *, deps=None, now=None) -> RunResult`: require verified recipient + first approval + confirmation; use fresh preview `run_id` if present else `run_push("send", manual=True)`; never call `set_auto_push`.
- `simple_ui.test_email(root, config_path, env) -> dict`: call `push_ctl.test_email(send=True, confirm=True)` with `ui_state.mail_env`; do not mark verified.
- `simple_ui.confirm_test_and_enable(root, config_path, env, *, register_task_fn) -> dict`: mark verified, `approve_first_send(confirm=True)`, install task, `set_auto_push(True)`; on failure leave disabled.
- `simple_ui.disable(root, config_path, env) -> dict`: `set_auto_push(False)`.
- `simple_ui.status(root, config_path, env) -> dict`: masked recipient, verification, enabled/paused/auto_paused/quiet, next time, last manual send.
- `simple_page.py`: render the labels from the spec (`我想关注什么`, `推送到哪个邮箱`, `每次推送几篇`, `先预览这周会推什么`, `现在推送今天的`, `每天自动推送`, `每天几点`, `高级 / 研究模式`); no evidence/eval/token.

- [ ] **Step 1: Write failing logic tests** for: count/time/recipient save; topics unchanged skips LLM; changed topics success saves groups; failure saves no new topic; preview references; manual send doesn't change enabled; test email doesn't verify until confirmation; confirm enables; auto-paused/quiet blocks; email change while enabled auto-disables and clears verification; time change calls task registration and surfaces failure; unsaved email disables auto/manual actions; empty preview shows a no-papers state.
- [ ] **Step 2: Run and verify failure** with `python -m unittest tests.test_simple_ui -v`.
- [ ] **Step 3: Implement `simple_ui.py` and `simple_page.py`; use fake deps in tests.**
- [ ] **Step 4: Run logic tests + AppTest**: `python -m unittest tests.test_simple_ui tests.test_simple_page_ui -v`; expected PASS, no network/SMTP at render.
- [ ] **Step 5: Commit** `feat: add the simple subscription page`.

### Task 9: Advanced navigation and page migration

**Files:**
- Move: `pages/0_总览.py -> advanced/overview.py`, `1_论文发现.py -> advanced/discovery.py`, `2_论文库.py -> advanced/library.py`, `3_推送设置.py -> advanced/push_settings.py`, `4_兴趣配置.py -> advanced/interests.py`
- Move: Q&A content from `app.py` to `advanced/evidence_qa.py`
- Create: `advanced/home.py`, new `app.py` navigation entry
- Modify: old pages' `st.set_page_config` calls, overview push toggle, tests `test_dashboard_ui.py`, `test_push_ui_contract.py`, `test_profile_ui_contract.py`, `test_start_bat.py` if paths
- Modify: `pages/0_总览.py` renamed; remove duplicate auto toggle or link to simple page; use `set_auto_push` if kept

**Interfaces:**
- `app.py`: `st.set_page_config(...)`, then `st.navigation([st.Page("simple_page.py", title="每日推送", default=True), st.Page("advanced/home.py", title="高级 / 研究模式"), *hidden advanced pages], position="sidebar").run()`.
- `advanced/home.py`: links to hidden advanced pages (overview/eval, evidence QA, push settings, precise interests, library, discovery, history); no research content on the simple page.
- Advanced pages no longer call `st.set_page_config`; the entry sets it once. `ui_theme.apply_theme` remains per page.
- Update every moved page's root/path calculation (the Q&A page's `ROOT` must become `Path(__file__).resolve().parents[1]`; the other pages already use `parents[1]`). Verify `push_config.json`, `interests.example.json`, `data/rag`, and scripts resolve from the project root, not `advanced/`.

- [ ] **Step 1: Write/adjust contract tests**: simple page is default; advanced home links exist (overview/eval, evidence QA, push settings, interests, library, discovery, history); overview has no independent auto-push toggle; old page imports/mains still render via AppTest with no side effects; the advanced pages do not call `st.set_page_config` a second time.
- [ ] **Step 2: Run and verify failure** with the relevant UI tests.
- [ ] **Step 3: Move pages, add navigation/links, strip duplicate `set_page_config`/toggle, update paths/imports.**
- [ ] **Step 4: Run UI tests and full suite** (`python -m unittest discover -s tests`); expected PASS.
- [ ] **Step 5: Commit** `refactor: move research pages behind an advanced entry`.

### Task 10: Docs, privacy scan, and full verification

**Files:**
- Modify: `README.md`, `docs/PUSH.md`, `.gitignore`, `tests/test_repo_hygiene.py`
- Create: `scripts/check_public_repo.py` (or equivalent), `PRIVACY.md` if useful
- Verify: full 60/pilot reports have no email; full test suite; git diff

**Interfaces:**
- `check_public_repo.py`: scan tracked files (`git ls-files`) and history (`git log -p`) for real email-like addresses, `sk-`, `SMTP_PASSWORD=` with non-placeholder values; exit non-zero with paths, never print full secrets.
- README: simple page, first test email, manual send, count/time, advanced mode, push history, privacy.
- `docs/PUSH.md`: first enable/test email, manual send, time/count, advanced setup.

- [ ] **Step 1: Write/adjust repo hygiene tests** for `interests.example.json`, ignored private paths, no real email in tracked fixtures; run expected failure.
- [ ] **Step 2: Implement the scan and docs; update `.gitignore` and any remaining page paths.**
- [ ] **Step 3: Run the scan** `python scripts/check_public_repo.py`; expected 0; run `git grep` checks.
- [ ] **Step 4: Run the full offline suite** `python -m unittest discover -s tests` and `python -m unittest discover -s . -p "test_eval.py"`; record exact counts.
- [ ] **Step 5: Commit** `docs: document simple push and public-repo privacy`.

## Out of Scope / Follow-up

- Full 60-question A/B results are already copied to `eval/benchmark_v2/reports/agent_eval_full60/`; update its `CLAIMS.md`/README rows in a separate commit so the UI plan does not mix eval claims with UI changes.
- Per-day count calendar, multiple profiles, email-embedded feedback links, and weekday-frequency schedules are not in v1.
