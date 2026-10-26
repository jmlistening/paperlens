"""SQLite ledger for incremental arXiv push state.

The ledger is deliberately separate from `data/delivery.sqlite3`: delivery is
the authoritative record of what was emailed, while this ledger records what
was seen/evaluated/selected and the fetch watermark.
"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

VERSION_RE = re.compile(r"v(\d+)$")


def paper_key(arxiv_id: str) -> str:
    value = (arxiv_id or "").strip()
    if not value:
        raise ValueError("arxiv_id is required")
    key = re.sub(r"v\d+$", "", value)
    if not key:
        raise ValueError(f"invalid arxiv_id: {arxiv_id!r}")
    return key


def _version(arxiv_id: str) -> int:
    match = VERSION_RE.search((arxiv_id or "").strip())
    return int(match.group(1)) if match else 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PushState:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.last_run_id: str | None = None

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    paper_key TEXT PRIMARY KEY,
                    arxiv_id TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 0,
                    title TEXT NOT NULL DEFAULT '',
                    abstract TEXT NOT NULL DEFAULT '',
                    published_at TEXT,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN
                      ('new','queued','evaluated','selected','delivered',
                       'skipped','ignored','expired','interrupted')),
                    profile_sha256 TEXT,
                    llm_score REAL,
                    final_score REAL,
                    delivered_at TEXT,
                    detail TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    slot_key TEXT NOT NULL UNIQUE,
                    trigger TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN
                      ('running','completed','failed','interrupted','busy')),
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    counts_json TEXT NOT NULL DEFAULT '{}',
                    tokens INTEGER NOT NULL DEFAULT 0,
                    error_type TEXT
                );

                CREATE TABLE IF NOT EXISTS watermarks (
                    name TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS impressions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    paper_key TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    score REAL,
                    channel TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS settings (
                    name TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
        finally:
            conn.close()

    def claim_slot(
        self, slot_key: str, trigger: str, started_at: str | None = None
    ) -> bool:
        run_id = uuid.uuid4().hex
        timestamp = started_at or _now()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "INSERT INTO runs(run_id,slot_key,trigger,status,started_at) "
                    "VALUES(?,?,?,?,?)",
                    (run_id, slot_key, trigger, "running", timestamp),
                )
                conn.execute("COMMIT")
            except sqlite3.IntegrityError:
                conn.execute("ROLLBACK")
                return False
        finally:
            conn.close()
        self.last_run_id = run_id
        return True

    def slot_run(self, slot_key: str) -> dict[str, Any] | None:
        """Return the run occupying ``slot_key``, if any (read-only)."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT run_id, status FROM runs WHERE slot_key=?",
                (slot_key,),
            ).fetchone()
        finally:
            conn.close()
        return dict(row) if row is not None else None

    def archive_slot_key(self, run_id: str, slot_key: str) -> None:
        """Park a finished run under a unique slot key.

        Used when a scheduled attempt sent nothing: freeing the daily slot
        lets a later time (or a Task Scheduler retry) run, while the run row
        and its artifacts stay in the history. A sent/uncertain run is never
        archived, so it keeps blocking further automatic sends that day.
        """
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE runs SET slot_key=? WHERE run_id=?",
                (slot_key, run_id),
            )
        finally:
            conn.close()

    def finish_run(
        self,
        run_id: str,
        status: str,
        counts: dict[str, Any] | None = None,
        error_type: str | None = None,
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE runs SET status=?, finished_at=?, counts_json=?, "
                "error_type=? WHERE run_id=?",
                (
                    status,
                    _now(),
                    json.dumps(counts or {}, ensure_ascii=False),
                    error_type,
                    run_id,
                ),
            )
        finally:
            conn.close()

    def record_paper(self, paper: dict[str, Any], status: str) -> None:
        arxiv_id = paper.get("arxiv_id")
        if not arxiv_id:
            raise ValueError("paper is missing arxiv_id")
        key = paper_key(arxiv_id)
        timestamp = _now()
        # Fetch normalization stores the arXiv publication date under
        # ``published``; older callers may pass ``published_at`` directly.
        # Without this, the column stayed NULL and we lost the only timestamp
        # that can decide whether a paper belongs to a recent lookback window.
        published_at = paper.get("published_at") or paper.get("published")
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO papers(
                    paper_key, arxiv_id, version, title, abstract, published_at,
                    first_seen_at, last_seen_at, status
                ) VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(paper_key) DO UPDATE SET
                    arxiv_id=excluded.arxiv_id,
                    version=excluded.version,
                    title=excluded.title,
                    abstract=excluded.abstract,
                    published_at=COALESCE(excluded.published_at, papers.published_at),
                    last_seen_at=excluded.last_seen_at
                """,
                (
                    key,
                    arxiv_id,
                    _version(arxiv_id),
                    paper.get("title", ""),
                    paper.get("abstract", ""),
                    published_at,
                    timestamp,
                    timestamp,
                    status,
                ),
            )
        finally:
            conn.close()

    def requeue_for_lookback(self, since: datetime | None = None) -> int:
        """Re-open recent, not-yet-delivered papers for a lookback run.

        A changed topic needs the papers an earlier profile only evaluated to
        be ranked again. Only ``evaluated``/``skipped``/``selected`` rows are
        reset to ``queued`` (clearing any stale selection detail); delivered,
        uncertain, queued, new, ignored, expired, and interrupted rows are
        never touched, so a manual send can never re-open something the
        delivery ledger already took.

        ``since`` is an inclusive UTC cutoff compared against the paper's
        publication date (falling back to the first-seen time for legacy rows
        recorded before the publication date was persisted). ``None`` re-opens
        every matching row.
        """
        sql = (
            "UPDATE papers SET status='queued', detail='' "
            "WHERE status IN ('evaluated','skipped','selected')"
        )
        params: list[Any] = []
        if since is not None:
            if not isinstance(since, datetime):
                raise TypeError("since must be a datetime")
            if since.tzinfo is None:
                cutoff = since.replace(tzinfo=timezone.utc)
            else:
                cutoff = since.astimezone(timezone.utc)
            sql += " AND COALESCE(published_at, first_seen_at) >= ?"
            params.append(cutoff.isoformat())
        conn = self._connect()
        try:
            cursor = conn.execute(sql, params)
            return cursor.rowcount
        finally:
            conn.close()

    def requeue_selected(self) -> int:
        """Reset leftover ``selected`` rows to ``queued``.

        A reused preview already ranked inside its snapshot, so its manual
        send does not need the full lookback requeue. Production can still
        hold a ``selected`` row from an interrupted run; leaving it selected
        lets the next scheduled run carry that old pick over without ranking
        it under the current profile. Only ``selected`` is touched;
        evaluated/skipped rows and safety states are left alone.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE papers SET status='queued', detail='' "
                "WHERE status='selected'"
            )
            return cursor.rowcount
        finally:
            conn.close()

    def mark_status(
        self,
        arxiv_id: str,
        status: str,
        detail: str = "",
        *,
        llm_score: float | None = None,
        final_score: float | None = None,
    ) -> None:
        key = paper_key(arxiv_id)
        timestamp = _now()
        delivered = timestamp if status == "delivered" else None
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE papers SET status=?, detail=?, "
                "delivered_at=COALESCE(?, delivered_at), "
                "llm_score=COALESCE(?, llm_score), "
                "final_score=COALESCE(?, final_score) WHERE paper_key=?",
                (
                    status,
                    detail,
                    delivered,
                    llm_score,
                    final_score,
                    key,
                ),
            )
        finally:
            conn.close()

    def selected(self, limit: int = 10) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM papers WHERE status='selected' "
                "ORDER BY final_score DESC, first_seen_at ASC LIMIT ?",
                (int(limit),),
            ).fetchall()
        finally:
            conn.close()
        return [dict(row) for row in rows]

    def is_seen(self, arxiv_id: str) -> bool:
        key = paper_key(arxiv_id)
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM papers WHERE paper_key=?", (key,)
            ).fetchone()
        finally:
            conn.close()
        return row is not None

    def eligible(self, limit: int) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM papers WHERE status IN ('new','queued') "
                "ORDER BY first_seen_at ASC, paper_key ASC LIMIT ?",
                (int(limit),),
            ).fetchall()
        finally:
            conn.close()
        return [dict(row) for row in rows]

    def eligible_prioritizing(
        self, fresh_keys: list[str], limit: int
    ) -> list[dict[str, Any]]:
        """Eligible rows with the papers this run fetched ordered first.

        A lookback requeue (or a backlog left by an earlier one) leaves rows
        that keep their older ``first_seen_at``. Ordering only by that column
        lets them fill the whole rank pool and starve the papers the current
        fetch returned. Fetched keys come first; the remaining slots are
        filled by the normal oldest-first eligible rows. The status filter
        still excludes delivered, uncertain, and ignored papers even when
        fetched.
        """
        keys = list(dict.fromkeys(str(key) for key in fresh_keys if key))
        conn = self._connect()
        try:
            if keys:
                placeholders = ",".join("?" for _ in keys)
                rows = conn.execute(
                    "SELECT * FROM papers WHERE status IN ('new','queued') "
                    "ORDER BY CASE WHEN paper_key IN ("
                    f"{placeholders}) THEN 0 ELSE 1 END, "
                    "first_seen_at ASC, paper_key ASC LIMIT ?",
                    (*keys, int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM papers WHERE status IN ('new','queued') "
                    "ORDER BY first_seen_at ASC, paper_key ASC LIMIT ?",
                    (int(limit),),
                ).fetchall()
        finally:
            conn.close()
        return [dict(row) for row in rows]

    def set_watermark(self, name: str, value: str) -> None:
        timestamp = _now()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO watermarks(name,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET value=excluded.value, "
                "updated_at=excluded.updated_at",
                (name, value, timestamp),
            )
        finally:
            conn.close()

    def get_watermark(self, name: str) -> str | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT value FROM watermarks WHERE name=?", (name,)
            ).fetchone()
        finally:
            conn.close()
        return row["value"] if row else None

    def set_setting(self, name: str, value: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO settings(name,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET value=excluded.value, "
                "updated_at=excluded.updated_at",
                (name, value, _now()),
            )
        finally:
            conn.close()

    def get_setting(self, name: str) -> str | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT value FROM settings WHERE name=?", (name,)
            ).fetchone()
        finally:
            conn.close()
        return row["value"] if row else None

    def record_impression(
        self,
        run_id: str,
        arxiv_id: str,
        position: int,
        score: float | None,
        channel: str,
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO impressions(run_id,paper_key,position,score,channel,"
                "created_at) VALUES(?,?,?,?,?,?)",
                (run_id, paper_key(arxiv_id), int(position), score, channel, _now()),
            )
        finally:
            conn.close()

    def status_counts(self) -> dict[str, int]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM papers GROUP BY status"
            ).fetchall()
        finally:
            conn.close()
        return {row["status"]: row["n"] for row in rows}

    def run_status_counts(self) -> dict[str, int]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM runs GROUP BY status"
            ).fetchall()
        finally:
            conn.close()
        return {row["status"]: row["n"] for row in rows}

    def impression_count(self) -> int:
        conn = self._connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM impressions").fetchone()[0]
        finally:
            conn.close()

    def reap_stale(
        self, now: datetime, max_age_hours: int = 2
    ) -> int:
        cutoff = now.astimezone(timezone.utc) - timedelta(hours=max_age_hours)
        finished = now.astimezone(timezone.utc).isoformat(timespec="seconds")
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE runs SET status='interrupted', finished_at=? "
                "WHERE status='running' AND started_at < ?",
                (finished, cutoff.isoformat(timespec="seconds")),
            )
            return cursor.rowcount
        finally:
            conn.close()
