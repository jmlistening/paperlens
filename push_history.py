"""Read-only sent push history for the advanced Streamlit view.

Only real sends are exposed: ``status == "sent"`` runs whose mode is one of
``send``, ``scheduled``, or ``manual``. Previews, quiet previews, failures, and
runs without a digest are never treated as delivered history.

New runs carry a structured ``digest.json`` (written by :mod:`push_run`); old
runs only have the rendered ``digest_preview.html``/``.txt`` and are returned
as ``legacy`` items whose papers are unavailable in structured form.

This module only reads ``run.json`` and ``digest.json`` below
``<root>/data/push/runs``. It never reads ``.env``, SMTP settings, or recipient
state, and the returned dictionaries deliberately contain only display-safe
fields.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LIMIT = 50
SENT_MODES = frozenset({"send", "scheduled", "manual"})
_EPOCH = datetime.min.replace(tzinfo=timezone.utc)

# Whitelisted digest paper fields. Unknown keys (for example a recipient or a
# credential accidentally added upstream) are dropped instead of being passed
# through to the UI.
_PAPER_FIELDS = (
    "arxiv_id",
    "paper_key",
    "title",
    "title_zh",
    "reason",
    "limitation",
    "summary",
    "url",
    "score",
    "categories",
)


def _read_object(path: Path) -> dict[str, Any] | None:
    """Read a JSON object; missing, unreadable, or malformed files yield None."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed


def _sent_at(payload: dict[str, Any]) -> str:
    """Prefer a parseable ``finished_at``, then ``started_at``.

    A missing or malformed ``finished_at`` must not push an otherwise valid run
    to the bottom of the history, so the fallback is also used for ordering.
    """
    fallback = ""
    for key in ("finished_at", "started_at"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()
        if not fallback:
            fallback = value
        if _parse_time(value) is not None:
            return value
    return fallback


def _subject(payload: dict[str, Any], digest: dict[str, Any] | None) -> str:
    for candidate in (
        payload.get("digest_subject"),
        payload.get("subject"),
        digest.get("subject") if digest is not None else None,
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _digest_papers(digest: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """Return sanitized structured papers, or None when no digest is usable."""
    if digest is None:
        return None
    raw_papers = digest.get("papers")
    if not isinstance(raw_papers, list):
        return None
    papers: list[dict[str, Any]] = []
    for raw in raw_papers:
        if not isinstance(raw, dict):
            continue
        item: dict[str, Any] = {}
        for field in _PAPER_FIELDS:
            if field == "categories":
                value = raw.get(field)
                item[field] = (
                    [str(entry) for entry in value] if isinstance(value, list) else []
                )
            elif field == "score":
                item[field] = raw.get(field, "")
            else:
                value = raw.get(field, "")
                if value is None:
                    value = ""
                item[field] = value if isinstance(value, str) else str(value)
        papers.append(item)
    return papers


def _paper_count(
    payload: dict[str, Any], papers: list[dict[str, Any]], *, legacy: bool
) -> int:
    if not legacy:
        return len(papers)
    paper_keys = payload.get("paper_keys")
    if isinstance(paper_keys, list) and paper_keys:
        return len(paper_keys)
    counts = payload.get("counts")
    if isinstance(counts, dict):
        selected = counts.get("selected")
        if (
            isinstance(selected, int)
            and not isinstance(selected, bool)
            and selected >= 0
        ):
            return selected
    return 0


def load_sent_history(root: Path, *, limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
    """Return the most recent sent push runs, newest first.

    Each item is ``{run_id, sent_at, mode, manual, subject, paper_count,
    papers, digest_html, legacy}``. ``papers`` comes from ``digest.json`` when
    it is present and structured; otherwise the item is ``legacy`` with an
    empty ``papers`` list and ``digest_html`` pointing at the rendered email.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        return []
    runs_dir = Path(root) / "data" / "push" / "runs"
    try:
        entries = sorted(runs_dir.iterdir(), key=lambda path: path.name)
    except OSError:
        return []

    items: list[dict[str, Any]] = []
    for run_dir in entries:
        if not run_dir.is_dir():
            continue
        payload = _read_object(run_dir / "run.json")
        if payload is None:
            continue
        if payload.get("status") != "sent":
            continue
        mode = payload.get("mode")
        if mode not in SENT_MODES:
            continue

        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            run_id = run_dir.name

        digest = _read_object(run_dir / "digest.json")
        papers = _digest_papers(digest)
        legacy = papers is None
        if papers is None:
            papers = []

        html_path = run_dir / "digest_preview.html"
        items.append(
            {
                "run_id": run_id,
                "sent_at": _sent_at(payload),
                "mode": mode,
                "manual": bool(payload.get("manual")) or mode == "manual",
                "subject": _subject(payload, digest),
                "paper_count": _paper_count(payload, papers, legacy=legacy),
                "papers": papers,
                "digest_html": str(html_path),
                "legacy": legacy,
            }
        )

    items.sort(
        key=lambda item: _parse_time(item["sent_at"]) or _EPOCH,
        reverse=True,
    )
    return items[:limit]


__all__ = ["DEFAULT_LIMIT", "SENT_MODES", "load_sent_history"]
