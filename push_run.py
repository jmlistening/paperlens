"""Push orchestration for PaperLens (preview + gated 163 send).

The pipeline is deliberately gated:
- ``enabled=False`` exits before any network/model/SMTP call;
- ``preview`` renders a digest and never touches SMTP;
- ``send`` requires an explicit first-send approval stored in the frontier DB;
- SMTP errors after submission are recorded as ``uncertain`` and auto-pause.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import Any, Callable, Mapping

from dotenv import dotenv_values

import ui_state
from fetch_papers import CooldownActive, read_profile
from profile_config import profile_path, profile_sha256
from push_config import DEFAULT_CONFIG_PATH, load_push_config, validate_push_config
from push_digest import build_digest, format_summary
from push_fetch import RateLimited, compute_window, fetch_candidates
from push_rank import rank_batch, select_digest
from push_state import PushState, paper_key
from reading_feedback import load_feedback
from run_pipeline import pipeline_lock

ROOT = Path(__file__).resolve().parent


@dataclass
class RunResult:
    run_id: str | None
    status: str
    candidate_count: int = 0
    selected_count: int = 0
    sent: bool = False
    errors: list[str] = field(default_factory=list)
    run_dir: str | None = None
    manual: bool = False


@dataclass
class PushDeps:
    root: Path = ROOT
    env: Mapping[str, str] | None = None
    arxiv_client: Any | None = None
    rank_fn: Callable[..., Any] | None = None
    summarize_fn: Callable[..., dict[str, str]] | None = None
    smtp_send: Callable[..., Any] | None = None


def _load_env() -> dict[str, str]:
    values = dotenv_values(ROOT / ".env", encoding="utf-8-sig")
    return {key: str(value) for key, value in values.items() if value is not None}


def _dedupe(papers: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result: list[dict] = []
    for paper in papers:
        key = paper_key(paper.get("arxiv_id", ""))
        if key in seen:
            continue
        seen.add(key)
        result.append(paper)
    return result


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


_TITLE_ZH_RE = re.compile(r"^\s*中文标题\s*[：:]\s*(.+?)\s*$", re.MULTILINE)


def _title_zh(summary: str) -> str:
    """Extract the Chinese title line from a summary; missing yields ``""``."""
    match = _TITLE_ZH_RE.search(summary or "")
    return match.group(1).strip() if match else ""


def _summary_for(paper: Mapping[str, Any], summaries: Mapping[str, str]) -> str:
    key = paper_key(paper["arxiv_id"])
    return format_summary(str(summaries.get(key, paper.get("summary", ""))))


def _write_digest_json(
    run_dir: Path,
    *,
    run_id: str,
    mode: str,
    created_at: datetime,
    digest: Any,
    selected: list[dict],
    summaries: Mapping[str, str],
    profile: dict,
) -> None:
    """Persist the structured digest used by the advanced history page.

    Only paper content and run metadata are stored; recipients, credentials,
    and SMTP settings never appear here.
    """
    papers: list[dict[str, Any]] = []
    for paper in selected:
        summary = _summary_for(paper, summaries)
        assessment = paper.get("assessment") or {}
        arxiv_id = str(paper.get("arxiv_id") or "")
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "paper_key": paper_key(arxiv_id),
                "title": str(paper.get("title") or ""),
                "title_zh": _title_zh(summary),
                "reason": str(assessment.get("reason") or ""),
                "limitation": str(assessment.get("limitation") or ""),
                "summary": summary,
                "url": paper.get("url")
                or f"https://arxiv.org/abs/{arxiv_id}",
                "score": assessment.get("score", ""),
                "categories": list(paper.get("categories") or []),
            }
        )
    _write_json(
        run_dir / "digest.json",
        {
            "run_id": run_id,
            "mode": mode,
            "created_at": created_at.isoformat(),
            "subject": digest.subject,
            "profile_sha256": profile_sha256(profile),
            "papers": papers,
            "warnings": list(digest.warnings),
        },
    )


def _snapshot_preview_state(root: Path) -> Path:
    """Copy the production frontier ledger into ``preview.sqlite3``.

    A preview may write only to its snapshot; production rows and the
    watermark stay untouched. When production is absent the snapshot starts
    empty (stale preview rows are never reused).
    """
    push_dir = root / "data" / "push"
    push_dir.mkdir(parents=True, exist_ok=True)
    snapshot = push_dir / "preview.sqlite3"
    for suffix in ("", "-wal", "-shm"):
        Path(f"{snapshot}{suffix}").unlink(missing_ok=True)
    production = push_dir / "frontier.sqlite3"
    if production.is_file():
        source = sqlite3.connect(production, timeout=15)
        try:
            destination = sqlite3.connect(snapshot, timeout=15)
            try:
                source.backup(destination)
            finally:
                destination.close()
        finally:
            source.close()
    return snapshot


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _progress(message: str) -> None:
    """Print a human progress line for the interactive task window.

    The scheduled batch sets ``PAPERLENS_PROGRESS=1``; library/test calls do
    not, so this stays out of normal stdout and test output.
    """
    if os.environ.get("PAPERLENS_PROGRESS") == "1":
        print(f"[PaperLens] {message}", file=sys.stderr, flush=True)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


_SENT_OUTCOMES = frozenset({"sent", "already_sent", "uncertain"})


def _release_unsent_scheduled_slot(
    state: PushState, root: Path, slot_key: str
) -> None:
    """Free today's scheduled slot when the previous attempt sent nothing.

    The daily slot exists to prevent two automatic digests on one day, not to
    prevent retries. A run that ended with ``no_digest``/``failed``/
    ``quiet_preview`` is archived under a unique slot key so a later time that
    day can try again. A run that sent, was already sent, or is uncertain keeps
    the slot so the day cannot deliver a second digest.
    """
    previous = state.slot_run(slot_key)
    if previous is None:
        return
    if previous.get("status") == "running":
        return
    if previous.get("status") not in {"completed", "failed", "interrupted"}:
        return
    run_id = str(previous.get("run_id") or "")
    if not run_id:
        return
    payload = _read_json_object(
        root / "data" / "push" / "runs" / run_id / "run.json"
    )
    if payload is None:
        return
    if str(payload.get("status") or "") in _SENT_OUTCOMES:
        return
    state.archive_slot_key(run_id, f"{slot_key}:attempt:{run_id}")


def _fresh_preview(
    root: Path,
    preview_run_id: str,
    *,
    profile: dict,
    digest_limit: int,
    now: datetime,
    recipient: str,
) -> tuple[list[dict], dict[str, str]] | None:
    """Return a fresh preview's papers and summaries, or ``None`` when stale.

    A preview is fresh only when its ``digest.json`` was rendered for the
    current profile, paper count, and local day, and the private UI state holds
    a preview reference binding it to the current recipient. ``digest.json``
    deliberately stores no recipient, so without that reference freshness
    cannot be proven and the caller must use the normal send pipeline.

    The requested paper count lives in the preview run's ``run.json``
    (``counts.digest_limit``); falling back to the digest's actual paper count
    keeps hand-written/legacy digest files conservative. An underfilled but
    otherwise fresh preview is reusable: the send then carries the papers the
    user actually previewed.
    """
    if not isinstance(preview_run_id, str) or not _RUN_ID_RE.fullmatch(
        preview_run_id
    ):
        return None
    run_dir = root / "data" / "push" / "runs" / preview_run_id
    payload = _read_json_object(run_dir / "digest.json")
    if payload is None or payload.get("mode") != "preview":
        return None
    if payload.get("run_id") != preview_run_id:
        return None
    if payload.get("profile_sha256") != profile_sha256(profile):
        return None

    papers = payload.get("papers")
    if not isinstance(papers, list) or not papers:
        return None
    if len(papers) > digest_limit:
        return None
    requested = len(papers)
    run_payload = _read_json_object(run_dir / "run.json")
    if run_payload is not None:
        counts = run_payload.get("counts")
        if isinstance(counts, dict):
            stored = counts.get("digest_limit")
            if isinstance(stored, int) and not isinstance(stored, bool):
                requested = stored
    if requested != digest_limit:
        return None

    created_at = payload.get("created_at")
    if not isinstance(created_at, str):
        return None
    try:
        rendered_at = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    if rendered_at.astimezone().date() != now.astimezone().date():
        return None

    reference = ui_state.get_preview(root)
    if reference is None:
        # No private reference means no recipient/date binding: never fail
        # open by reusing a preview we cannot prove belongs to this send.
        return None
    if reference.get("run_id") != preview_run_id:
        return None
    if not recipient:
        return None
    if reference.get("recipient_hash") != ui_state.recipient_hash(recipient):
        return None
    if reference.get("local_date") != now.astimezone().date().isoformat():
        return None
    if reference.get("created_at") != created_at:
        return None
    reference_profile = reference.get("profile_sha256")
    if (
        reference_profile is not None
        and reference_profile != payload.get("profile_sha256")
    ):
        return None

    selected: list[dict] = []
    summaries: dict[str, str] = {}
    for item in papers:
        if not isinstance(item, Mapping):
            return None
        arxiv_id = item.get("arxiv_id")
        title = item.get("title")
        summary = item.get("summary")
        if not isinstance(arxiv_id, str) or not arxiv_id:
            return None
        if not isinstance(title, str) or not isinstance(summary, str):
            return None
        try:
            key = paper_key(arxiv_id)
        except ValueError:
            return None
        if key in summaries:
            return None
        categories = item.get("categories")
        if not isinstance(categories, list):
            categories = []
        selected.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "url": item.get("url") or f"https://arxiv.org/abs/{arxiv_id}",
                "categories": list(categories),
                "assessment": {
                    "score": item.get("score", ""),
                    "reason": str(item.get("reason") or ""),
                    "limitation": str(item.get("limitation") or ""),
                    "feedback_refs": [],
                },
            }
        )
        summaries[key] = summary
    return selected, summaries


def _write_digest_artifacts(
    run_dir: Path,
    *,
    run_id: str,
    mode: str,
    created_at: datetime,
    digest: Any,
    selected: list[dict],
    summaries: Mapping[str, str],
    profile: dict,
) -> None:
    """Write the HTML/TXT/meta/JSON digest from one exact paper set.

    Called again after ``_send_digest`` filters already-delivered papers, so
    every persisted artifact describes what was actually sent rather than the
    pre-filter selection.
    """
    (run_dir / "digest_preview.html").write_text(digest.html, encoding="utf-8")
    (run_dir / "digest_preview.txt").write_text(digest.text, encoding="utf-8")
    _write_json(
        run_dir / "digest_meta.json",
        {
            "subject": digest.subject,
            "paper_keys": digest.paper_keys,
            "warnings": digest.warnings,
        },
    )
    _write_digest_json(
        run_dir,
        run_id=run_id,
        mode=mode,
        created_at=created_at,
        digest=digest,
        selected=selected,
        summaries=summaries,
        profile=profile,
    )


def _record_preview_ledger(
    state: PushState, run_id: str, selected: list[dict]
) -> None:
    """Materialise previewed papers in the production ledger before sending.

    A preview writes only to its snapshot, so production may have no row for a
    previewed paper. ``_send_digest`` marks rows delivered with an UPDATE-only
    statement, which would silently affect zero rows; the next automatic run
    would then re-fetch and re-select those papers, receive ``already_sent``
    forever, and never advance. Recording the papers here (resetting any stale
    status) makes the send visible to the production ledger and keeps the
    normal fetch/select path unblocked. Impressions mirror the channel the
    normal send path records.
    """
    for position, paper in enumerate(selected, start=1):
        state.record_paper(paper, "queued")
        # ``record_paper`` does not overwrite an existing row's status, so a
        # stale ``selected`` row from an interrupted run is reset explicitly.
        state.mark_status(paper["arxiv_id"], "queued")
        score = (paper.get("assessment") or {}).get("score")
        try:
            numeric_score: float | None = float(score)
        except (TypeError, ValueError):
            numeric_score = None
        state.record_impression(
            run_id, paper["arxiv_id"], position, numeric_score, "email"
        )


def _write_run(
    run_dir: Path,
    *,
    run_id: str,
    mode: str,
    status: str,
    started_at: datetime,
    errors: list[str],
    counts: dict[str, Any],
    paper_keys: list[str] | None = None,
    digest_subject: str | None = None,
    manual: bool = False,
) -> None:
    _write_json(
        run_dir / "run.json",
        {
            "run_id": run_id,
            "mode": mode,
            "status": status,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "errors": errors,
            "counts": counts,
            "paper_keys": paper_keys or [],
            "digest_subject": digest_subject,
            "manual": bool(manual),
        },
    )
    (run_dir / "pipeline.log").write_text(
        "\n".join(
            [
                f"run_id={run_id}",
                f"mode={mode}",
                f"status={status}",
                f"manual={'true' if manual else 'false'}",
                f"counts={json.dumps(counts, ensure_ascii=False)}",
                f"errors={json.dumps(errors, ensure_ascii=False)}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _default_summarize_fn(
    papers: list[dict], env: Mapping[str, str], root: Path
) -> dict[str, str]:
    from openai import OpenAI

    from summarize_selected import PROMPT

    api_key = (env.get("DEEPSEEK_API_KEY") or "").strip()
    base_url = (env.get("DEEPSEEK_BASE_URL") or "").strip()
    model = (env.get("DEEPSEEK_MODEL") or "").strip()
    if not all((api_key, base_url, model)):
        raise ValueError(".env missing DeepSeek configuration")

    cache_dir = root / "data" / "summary_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, str] = {}
    with OpenAI(
        api_key=api_key, base_url=base_url, timeout=120, max_retries=0
    ) as client:
        for paper in papers:
            cache_input = {
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "abstract": paper["abstract"],
                "model": model,
                "base_url": base_url,
                "prompt": PROMPT,
            }
            cache_key = hashlib.sha256(
                json.dumps(cache_input, ensure_ascii=False, sort_keys=True).encode(
                    "utf-8"
                )
            ).hexdigest()
            cache_file = cache_dir / f"{cache_key}.json"
            summary = None
            if cache_file.exists():
                try:
                    cached = json.loads(cache_file.read_text(encoding="utf-8"))
                    value = cached.get("summary")
                    if isinstance(value, str) and value.strip():
                        summary = value.strip()
                except (ValueError, AttributeError):
                    pass
            if summary is None:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "title": paper["title"],
                                    "abstract": paper["abstract"],
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    max_tokens=4096,
                )
                if not response.choices:
                    raise ValueError("summary model returned no choices")
                choice = response.choices[0]
                if choice.finish_reason != "stop":
                    raise ValueError("summary model did not finish cleanly")
                summary = (choice.message.content or "").strip()
                if not summary:
                    raise ValueError("summary model returned empty content")
                cache_file.write_text(
                    json.dumps({"summary": summary}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            summaries[paper_key(paper["arxiv_id"])] = summary
    return summaries


def _payload_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    detail: dict[str, Any] = {}
    try:
        parsed = json.loads(row.get("detail") or "{}")
        if isinstance(parsed, dict):
            detail = parsed
    except (TypeError, ValueError):
        detail = {}
    assessment = detail.get("assessment")
    if not isinstance(assessment, dict):
        assessment = {
            "score": int(row.get("llm_score") or 0),
            "reason": "previously selected in a preview",
            "limitation": "",
            "feedback_refs": [],
        }
    return {
        "arxiv_id": row["arxiv_id"],
        "title": row.get("title") or "",
        "abstract": row.get("abstract") or "",
        "url": detail.get("url") or f"https://arxiv.org/abs/{row['arxiv_id']}",
        "categories": detail.get("categories") or [],
        "assessment": assessment,
        "llm_score": row.get("llm_score"),
        "final_score": row.get("final_score"),
    }


def _selected_detail(paper: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "assessment": paper.get("assessment") or {},
            "url": paper.get("url"),
            "categories": paper.get("categories") or [],
        },
        ensure_ascii=False,
    )


def _send_digest(
    *,
    root: Path,
    state: PushState,
    config: Any,
    env: Mapping[str, str],
    digest: Any,
    selected: list[dict],
    summaries: Mapping[str, str],
    deps: PushDeps,
) -> tuple[str, list[str], int, Any, list[dict], dict[str, str]]:
    """Send a digest and return the exact post-filter digest and paper set.

    ``eligible_papers`` may drop papers that another attempt already reserved
    or delivered. The returned digest is rebuilt from that filtered set (or is
    empty when nothing remains), so callers can rewrite every run artifact and
    ``paper_keys`` from what really went out.
    """
    from mail_transport import resolve_smtp, send_message as default_send

    import send_digest as delivery

    recipient = (env.get("MAIL_TO") or "").strip()
    if not recipient:
        return (
            "config_error",
            ["MAIL_TO is missing"],
            0,
            digest,
            list(selected),
            dict(summaries),
        )
    db = delivery.open_db(root)
    try:
        eligible = delivery.eligible_papers(db, recipient, selected, verbose=False)
        if not eligible:
            return (
                "already_sent",
                [],
                0,
                build_digest([], {}, config),
                [],
                {},
            )
        if len(eligible) != len(selected):
            # Rebuild the digest so an already-delivered paper is not sent again.
            selected = eligible
            keys = {paper_key(paper["arxiv_id"]) for paper in selected}
            summaries = {key: value for key, value in summaries.items() if key in keys}
            digest = build_digest(selected, summaries, config)
        settings = resolve_smtp(config.provider, env)
        message = EmailMessage()
        message["Subject"] = digest.subject
        message["From"] = settings.user
        message["To"] = recipient
        message["Date"] = formatdate(localtime=True)
        message["Message-ID"] = make_msgid()
        message.set_content(digest.text)
        message.add_alternative(digest.html, subtype="html")
        for filename, content in digest.attachments:
            message.add_attachment(
                content, maintype="application", subtype="pdf", filename=filename
            )
        attempt_id = delivery.reserve(
            db, recipient, selected, message_id=message["Message-ID"]
        )
        delivery.set_status(db, attempt_id, "sending", "SMTP submit", {"pending"})
        sender = deps.smtp_send or default_send
        try:
            sender(settings, message)
        except Exception as exc:
            delivery.set_status(
                db, attempt_id, "uncertain", type(exc).__name__, {"sending"}
            )
            state.set_setting("auto_paused", "true")
            return (
                "uncertain",
                [type(exc).__name__],
                0,
                digest,
                selected,
                dict(summaries),
            )
        delivery.set_status(db, attempt_id, "accepted", "SMTP accepted", {"sending"})
        for paper in selected:
            state.mark_status(paper["arxiv_id"], "delivered")
        return "sent", [], len(selected), digest, selected, dict(summaries)
    except Exception as exc:
        return "failed", [type(exc).__name__], 0, digest, selected, dict(summaries)
    finally:
        db.close()


def _result(
    *, run_id: str | None, status: str, run_dir: Path | None, counts: dict[str, Any],
    errors: list[str], sent: bool = False,
) -> RunResult:
    return RunResult(
        run_id=run_id,
        status=status,
        candidate_count=int(counts.get("candidates", 0)),
        selected_count=int(counts.get("selected", 0)),
        sent=sent,
        errors=list(errors),
        run_dir=str(run_dir) if run_dir else None,
    )


def run_push(
    mode: str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    now: datetime | None = None,
    deps: PushDeps | None = None,
    *,
    manual: bool = False,
    preview_run_id: str | None = None,
    force: bool = False,
    lookback_days: int | None = None,
) -> RunResult:
    """Run one push slot.

    ``manual`` is honored only for ``mode="send"``: an explicit user action may
    run while automatic push is disabled or paused, but it still requires a
    verified recipient, first-send approval, and a non-auto-paused ledger.
    ``force`` remains preview-only. A fresh ``preview_run_id`` lets a manual
    send reuse that preview instead of fetching and ranking again.

    ``lookback_days`` is used by the simple page's preview/manual actions: it
    replaces the incremental watermark window with ``[now - days, now]`` and
    re-opens non-delivered papers evaluated inside that window, so a changed
    topic can be pushed the same day. Scheduled runs omit it and keep the
    incremental window.
    """
    manual_send = bool(manual) and mode == "send"
    result = _run_push(
        mode,
        config_path=config_path,
        now=now,
        deps=deps,
        force=force,
        manual=manual_send,
        preview_run_id=preview_run_id,
        lookback_days=lookback_days,
    )
    result.manual = manual_send
    return result


def _run_push(
    mode: str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    now: datetime | None = None,
    deps: PushDeps | None = None,
    *,
    force: bool = False,
    manual: bool = False,
    preview_run_id: str | None = None,
    lookback_days: int | None = None,
) -> RunResult:
    if mode not in {"scheduled", "preview", "send"}:
        raise ValueError("mode must be scheduled, preview, or send")
    deps = deps or PushDeps()
    root = Path(deps.root)
    env = dict(deps.env) if deps.env is not None else _load_env()
    current = now or datetime.now(timezone.utc)

    config = load_push_config(Path(config_path))
    # ``force`` is preview-only: the simple page may forecast while automatic
    # push is disabled or paused. ``manual`` is send-only: the page may send an
    # explicit digest while auto-push is off. Both still enforce recipient
    # verification, first-send approval, and the auto-paused safety flag below.
    bypass_gates = (force and mode == "preview") or manual
    if not config.enabled and not bypass_gates:
        _progress("自动推送已关闭，本次跳过（没有发邮件）。")
        return _result(
            run_id=None, status="disabled", run_dir=None,
            counts={}, errors=[],
        )
    if config.paused and not bypass_gates:
        _progress("自动推送已暂停，本次跳过（没有发邮件）。")
        return _result(
            run_id=None, status="paused", run_dir=None,
            counts={}, errors=[],
        )

    # The scheduled/send paths must use the same recipient as the UI. An
    # existing private state is authoritative: an unverified, missing, or
    # unreadable one fails closed here, before fetch/rank/SMTP and without
    # falling back to a stale .env value. Previews never send, so they remain
    # available while a newly saved address is still awaiting verification.
    resolved = ui_state.resolve_recipient(root, env)
    if mode in {"scheduled", "send"} and resolved["state_exists"]:
        if not resolved["verified"]:
            _progress("收件邮箱尚未验证，本次不发送。")
            return _result(
                run_id=None, status="recipient_unverified", run_dir=None,
                counts={}, errors=["recipient is not verified"],
            )
        env = dict(env)
        env["MAIL_TO"] = resolved["recipient"]

    # A manual send is a real send even while automatic push is off, so its
    # transport settings are validated as if push were enabled. This fails
    # before fetch/rank/SMTP when credentials are missing.
    validation_config = replace(config, enabled=True) if manual else config
    problems = validate_push_config(
        validation_config, env, require_smtp=mode != "preview"
    )
    if problems:
        _progress("推送配置不完整，本次不发送。")
        return _result(
            run_id=None, status="config_error", run_dir=None,
            counts={}, errors=problems,
        )
    if lookback_days is not None and (
        not isinstance(lookback_days, int)
        or isinstance(lookback_days, bool)
        or not 1 <= lookback_days <= 365
    ):
        # Reject before the slot is claimed: an invalid window otherwise
        # raises inside the pipeline and leaves a stuck "running" run.
        return _result(
            run_id=None, status="config_error", run_dir=None,
            counts={}, errors=["lookback_days must be an integer from 1 to 365"],
        )

    effective_mode = (
        "preview"
        if mode == "preview" or (config.quiet_mode and not manual)
        else "send"
    )
    # quiet_mode turns automatic runs into previews. An explicit manual send is
    # a real send, so it ignores quiet_mode.
    quiet = config.quiet_mode and mode != "preview" and not manual

    try:
        with pipeline_lock(root):
            if mode == "preview":
                # Previews must not advance the production watermark or mark
                # papers as selected. They run against a snapshot copy of the
                # production ledger and write only to that copy.
                state_path = _snapshot_preview_state(root)
            else:
                state_path = root / "data" / "push" / "frontier.sqlite3"
            state = PushState(state_path)
            state.init_schema()
            state.reap_stale(current)
            if manual:
                # One unique slot per explicit send; the microsecond UTC stamp
                # keeps two manual sends from sharing a slot.
                utc_stamp = current.astimezone(timezone.utc).strftime(
                    "%Y%m%dT%H%M%S%f"
                )
                slot_key = f"manual-send-{utc_stamp}"
            elif mode == "preview":
                slot_key = f"manual-{current.strftime('%Y%m%dT%H%M%S%f')}"
            else:
                # One automatic digest per configured time per day. Changing
                # the time is an explicit new attempt and gets its own slot;
                # the delivery ledger still prevents duplicate papers.
                slot_key = (
                    f"{current.astimezone().date().isoformat()}"
                    f"T{config.time_local}"
                )
            if mode == "scheduled":
                _release_unsent_scheduled_slot(state, root, slot_key)
            if not state.claim_slot(slot_key, mode, started_at=current.isoformat()):
                _progress("这个时间今天已经执行过，跳过重复运行。")
                return _result(
                    run_id=None, status="busy", run_dir=None,
                    counts={}, errors=[],
                )
            run_id = state.last_run_id
            run_dir = root / "data" / "push" / "runs" / str(run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            errors: list[str] = []
            counts: dict[str, Any] = {
                "candidates": 0, "ranked": 0, "selected": 0, "sent": 0
            }

            if effective_mode == "send" and state.get_setting("auto_paused") == "true":
                status = "auto_paused"
                _write_run(
                    run_dir, run_id=str(run_id), mode=mode, status=status,
                    started_at=current, errors=[], counts=counts,
                    manual=manual,
                )
                state.finish_run(str(run_id), "completed", counts)
                return _result(
                    run_id=str(run_id), status=status, run_dir=run_dir,
                    counts=counts, errors=[],
                )

            if effective_mode == "send" and not state.get_setting(
                "first_send_approved_at"
            ):
                status = "send_not_approved"
                _write_run(
                    run_dir, run_id=str(run_id), mode=mode, status=status,
                    started_at=current, errors=[], counts=counts,
                    manual=manual,
                )
                state.finish_run(str(run_id), "completed", counts)
                return _result(
                    run_id=str(run_id), status=status, run_dir=run_dir,
                    counts=counts, errors=[],
                )

            try:
                summarize = deps.summarize_fn or (
                    lambda papers, env: _default_summarize_fn(papers, env, root)
                )
                profile, _parsed, _profile_hash = read_profile(
                    profile_path(root)
                )
                digest_limit = max(
                    1,
                    min(
                        int(config.max_per_digest),
                        int(profile.get("top_k", config.max_per_digest)),
                    ),
                )
                rank_limit = max(
                    1,
                    min(
                        int(config.rank_max_candidates),
                        int(
                            (profile.get("search") or {}).get(
                                "max_candidates", config.rank_max_candidates
                            )
                        ),
                    ),
                )
                counts["digest_limit"] = digest_limit
                counts["rank_limit"] = rank_limit
                render_config = replace(config, max_per_digest=digest_limit)
                selected: list[dict] = []
                summaries: dict[str, str] | None = None
                reused_preview = False
                if mode == "send" and preview_run_id:
                    fresh = _fresh_preview(
                        root,
                        preview_run_id,
                        profile=profile,
                        digest_limit=digest_limit,
                        now=current,
                        recipient=resolved["recipient"],
                    )
                    if fresh is not None:
                        # Reuse the previewed papers and summaries exactly: no
                        # fetch, rank, or summarize call for this send.
                        selected, summaries = fresh
                        counts["selected"] = len(selected)
                        reused_preview = True

                # A lookback run re-opens papers an earlier profile only
                # evaluated (or skipped/selected) inside the window. It runs
                # before the ``state.selected`` fallback so a changed topic can
                # never send a stale selection made under the previous profile.
                # Inside the try: a requeue failure must finish the claimed run
                # as failed instead of leaving it "running". A reused preview
                # already ranked inside its snapshot, so requeueing every
                # evaluated row would leave queued rows nothing consumes; the
                # next scheduled run would then spend its pool on that backlog.
                # Still clear leftover ``selected`` rows: otherwise a scheduled
                # run could carry an old profile's pick over without ranking it.
                lookback_start = (
                    current - timedelta(days=lookback_days)
                    if lookback_days is not None
                    else None
                )
                if reused_preview:
                    state.requeue_selected()
                elif lookback_start is not None:
                    state.requeue_for_lookback(lookback_start)

                if (
                    not selected
                    and effective_mode == "send"
                    and lookback_days is None
                ):
                    # The incremental path may carry over an earlier selection.
                    # A lookback run must re-rank inside its wide window instead
                    # of reusing a selection made for another profile/day.
                    rows = state.selected(digest_limit)
                    if rows:
                        selected = [_payload_from_row(row) for row in rows]
                        counts["selected"] = len(selected)

                if not selected:
                    if lookback_days is None:
                        watermark = state.get_watermark("frontier")
                        start, end = compute_window(current, watermark, config)
                    else:
                        # Manual/preview on the simple page forecasts the next
                        # explicit send, so it searches the recent catch-up
                        # window rather than the automatic incremental window.
                        end = current
                        start = current - timedelta(days=lookback_days)
                    _progress("正在从 arXiv 抓取候选论文……")
                    new_papers = fetch_candidates(
                        profile,
                        start,
                        end,
                        config,
                        client=deps.arxiv_client,
                        state=state,
                        now=current,
                        data_dir=root / "data",
                    )
                    _progress(f"抓取完成：{len(new_papers)} 篇。")
                    for paper in new_papers:
                        state.record_paper(paper, "queued")
                    # Only new/queued rows are eligible; this filters out papers
                    # already delivered/ignored. This fetch's papers come first
                    # so a backlog of older queued or requeued rows cannot
                    # consume the whole rank pool on either path.
                    fresh_keys = [
                        paper_key(str(paper["arxiv_id"]))
                        for paper in new_papers
                        if paper.get("arxiv_id")
                    ]
                    pool = state.eligible_prioritizing(fresh_keys, rank_limit)
                    counts["candidates"] = len(pool)
                    if len(pool) < config.min_new_papers:
                        _progress("没有新的候选论文，本次不发送。")
                        status = "no_digest"
                        _write_run(
                            run_dir, run_id=str(run_id), mode=mode, status=status,
                            started_at=current, errors=[], counts=counts,
                            manual=manual,
                        )
                        state.finish_run(str(run_id), "completed", counts)
                        return _result(
                            run_id=str(run_id), status=status, run_dir=run_dir,
                            counts=counts, errors=[],
                        )

                    feedback = load_feedback(root)
                    model_config = {
                        "api_key": env.get("DEEPSEEK_API_KEY", ""),
                        "base_url": env.get("DEEPSEEK_BASE_URL", ""),
                        "model": env.get("DEEPSEEK_MODEL", ""),
                        "embedding_model_path": str(root / "data" / "models"),
                    }
                    _progress(f"正在排序 {len(pool)} 篇候选论文……")
                    ranked = rank_batch(
                        pool,
                        profile,
                        feedback,
                        model_config,
                        root / "data" / "push" / "rank_cache",
                        rank_fn=deps.rank_fn,
                    )
                    counts["ranked"] = len(ranked)
                    _progress(f"排序完成：{len(ranked)} 篇。")
                    for paper in ranked:
                        state.mark_status(
                            paper["arxiv_id"],
                            "evaluated",
                            llm_score=paper.get("llm_score"),
                            final_score=paper.get("final_score"),
                        )
                    digest_config = {
                        "min_score": int(profile.get("min_score", 60)),
                        "max_per_digest": digest_limit,
                        "max_per_primary_category": config.max_per_primary_category,
                    }
                    selected = select_digest(
                        ranked,
                        digest_config,
                        set(feedback.get("ignored_keys") or []),
                    )
                    counts["selected"] = len(selected)
                    for paper in selected:
                        state.mark_status(
                            paper["arxiv_id"],
                            "selected",
                            _selected_detail(paper),
                            llm_score=paper.get("llm_score"),
                            final_score=paper.get("final_score"),
                        )
                    for position, paper in enumerate(selected, start=1):
                        state.record_impression(
                            str(run_id),
                            paper["arxiv_id"],
                            position,
                            paper.get("final_score"),
                            "preview" if effective_mode == "preview" else "email",
                        )

                if not selected:
                    _progress("没有达到推荐分数的论文，本次不发送。")
                    status = "no_digest"
                    _write_run(
                        run_dir, run_id=str(run_id), mode=mode, status=status,
                        started_at=current, errors=[], counts=counts,
                        manual=manual,
                    )
                    state.finish_run(str(run_id), "completed", counts)
                    return _result(
                        run_id=str(run_id), status=status, run_dir=run_dir,
                        counts=counts, errors=[],
                    )

                _progress(f"已选中 {len(selected)} 篇，正在生成中文摘要……")
                if summaries is None:
                    summaries = summarize(selected, env)
                digest = build_digest(selected, summaries, render_config)
                _write_digest_artifacts(
                    run_dir,
                    run_id=str(run_id),
                    mode=mode,
                    created_at=current,
                    digest=digest,
                    selected=selected,
                    summaries=summaries,
                    profile=profile,
                )

                if effective_mode == "preview":
                    _progress(
                        f"预览完成：{len(selected)} 篇（不会发送邮件）。"
                    )
                    status = "quiet_preview" if quiet else "preview_ready"
                    _write_run(
                        run_dir, run_id=str(run_id), mode=mode, status=status,
                        started_at=current, errors=[], counts=counts,
                        paper_keys=digest.paper_keys,
                        digest_subject=digest.subject,
                        manual=manual,
                    )
                    state.finish_run(str(run_id), "completed", counts)
                    return _result(
                        run_id=str(run_id), status=status, run_dir=run_dir,
                        counts=counts, errors=[],
                    )

                if reused_preview:
                    _record_preview_ledger(state, str(run_id), selected)
                # Keep the consumed preview set for the ledger repair below;
                # ``_send_digest`` returns a possibly smaller delivery-filtered
                # set that must only drive the persisted artifacts.
                previewed_selected = selected if reused_preview else []
                _progress(f"正在发送邮件（{len(selected)} 篇）……")
                (
                    status,
                    send_errors,
                    sent_count,
                    digest,
                    selected,
                    final_summaries,
                ) = _send_digest(
                    root=root,
                    state=state,
                    config=render_config,
                    env=env,
                    digest=digest,
                    selected=selected,
                    summaries=summaries,
                    deps=deps,
                )
                summaries = final_summaries
                counts["sent"] = sent_count
                errors.extend(send_errors)
                _progress(f"发送结果：{status}，{sent_count} 篇。")
                # The delivery ledger may have dropped already-delivered
                # papers. Persist exactly the set that was attempted/sent so
                # run.json paper_keys, digest.json, and the email preview never
                # describe a superset of the real email.
                _write_digest_artifacts(
                    run_dir,
                    run_id=str(run_id),
                    mode=mode,
                    created_at=current,
                    digest=digest,
                    selected=selected,
                    summaries=summaries,
                    profile=profile,
                )
                if reused_preview and status in {
                    "sent", "already_sent", "uncertain"
                }:
                    # The previewed papers are consumed (sent, already
                    # delivered, or reserved with an unknown outcome). Record
                    # that in production so the next run can fetch ahead
                    # instead of re-selecting them forever.
                    for paper in previewed_selected:
                        state.mark_status(paper["arxiv_id"], "delivered")
                _write_run(
                    run_dir, run_id=str(run_id), mode=mode, status=status,
                    started_at=current, errors=errors, counts=counts,
                    paper_keys=digest.paper_keys,
                    digest_subject=digest.subject,
                    manual=manual,
                )
                state.finish_run(
                    str(run_id),
                    "completed" if status in {"sent", "already_sent"} else "failed",
                    counts,
                    None if status in {"sent", "already_sent"} else status,
                )
                return _result(
                    run_id=str(run_id), status=status, run_dir=run_dir,
                    counts=counts, errors=errors, sent=status == "sent",
                )
            except (RateLimited, CooldownActive) as exc:
                status = "rate_limited" if isinstance(exc, RateLimited) else "cooldown"
                errors.append(type(exc).__name__)
                _write_run(
                    run_dir, run_id=str(run_id), mode=mode, status=status,
                    started_at=current, errors=errors, counts=counts,
                    manual=manual,
                )
                state.finish_run(str(run_id), "failed", counts, type(exc).__name__)
                return _result(
                    run_id=str(run_id), status=status, run_dir=run_dir,
                    counts=counts, errors=errors,
                )
            except Exception as exc:
                errors.append(type(exc).__name__)
                _write_run(
                    run_dir, run_id=str(run_id), mode=mode, status="failed",
                    started_at=current, errors=errors, counts=counts,
                    manual=manual,
                )
                state.finish_run(str(run_id), "failed", counts, type(exc).__name__)
                return _result(
                    run_id=str(run_id), status="failed", run_dir=run_dir,
                    counts=counts, errors=errors,
                )
    except RuntimeError as exc:
        # pipeline_lock raises RuntimeError when another pipeline holds the lock.
        return _result(
            run_id=None, status="busy", run_dir=None,
            counts={}, errors=[str(exc)],
        )


# Statuses that should make a scheduled run exit non-zero so Task Scheduler can
# record a failure/restart. ``busy`` and ``no_digest`` are normal outcomes.
_FATAL_CLI_STATUSES = frozenset(
    {
        "failed",
        "config_error",
        "rate_limited",
        "cooldown",
        "uncertain",
        "recipient_unverified",
    }
)


def _parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one PaperLens push slot (default: --scheduled)."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--scheduled", action="store_true", help="run the scheduled slot (default)"
    )
    mode.add_argument(
        "--preview", action="store_true", help="render a preview and never send"
    )
    mode.add_argument(
        "--send", action="store_true", help="send a digest (requires --confirm)"
    )
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI used by ``scripts/run_push.bat`` and manual previews."""
    args = _parse_cli(argv)
    mode = "preview" if args.preview else "send" if args.send else "scheduled"
    if mode == "send" and not args.confirm:
        print("ERROR: --send requires --confirm", file=sys.stderr)
        return 2
    _progress(f"PaperLens 定时推送开始（{mode}）。")
    try:
        result = run_push(mode, config_path=args.config, deps=PushDeps(root=ROOT))
    except (ValueError, OSError) as exc:
        _progress(f"运行失败：{type(exc).__name__}。")
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    outcome = "已发送" if result.sent else "未发送"
    _progress(
        f"PaperLens 结束：{result.status}（{outcome}）；"
        f"候选 {result.candidate_count} 篇，选中 {result.selected_count} 篇。"
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 1 if result.status in _FATAL_CLI_STATUSES else 0


if __name__ == "__main__":
    raise SystemExit(main())
