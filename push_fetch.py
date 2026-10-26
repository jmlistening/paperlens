"""Incremental arXiv fetch window for the optional push pipeline (Task P3).

The module is a thin layer over the proven helpers in :mod:`fetch_papers`:
profile validation, term clauses, result normalization, multi-query merging,
and the local 429 cooldown ledger are reused rather than copied.

No network call happens at import time. Callers may inject a fake client (as
the tests do) and a temporary data directory for the cooldown file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from fetch_papers import (
    ROOT,
    check_cooldown,
    is_rate_limit,
    merge_groups,
    normalize_result,
    parse_profile,
    record_cooldown,
    term_clause,
)

# New push state stays under data/push/; the legacy fetch cooldown file is not
# modified by this module.
DEFAULT_COOLDOWN_DIR = ROOT / "data"
DEFAULT_WATERMARK_NAME = "frontier"
DEFAULT_INITIAL_LOOKBACK_DAYS = 7

_CATEGORY_PATTERN = re.compile(
    r"[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z][A-Za-z0-9-]*)?"
)


class RateLimited(RuntimeError):
    """arXiv answered HTTP 429; the caller must stop and honor the cooldown."""

    def __init__(self, message: str, retry_not_before: datetime | None = None):
        super().__init__(message)
        self.status = 429
        self.retry_not_before = retry_not_before


@dataclass(frozen=True)
class SearchRequest:
    """Query handed to a client; fake clients only need ``query``/``max_results``."""

    query: str
    max_results: int


def _as_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("expected a datetime")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_watermark(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("watermark must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(f"invalid watermark timestamp: {value!r}") from exc
    return _as_utc(parsed)


def compute_window(
    now: datetime, watermark: str | None, config: Any
) -> tuple[datetime, datetime]:
    """Return ``(start, end)`` in UTC for the next incremental fetch.

    First run: at least ``initial_lookback_days`` (default 7) and never less
    than ``window_hours``. Later runs: ``max(watermark - overlap_hours,
    now - catchup_days)``.
    """
    end = _as_utc(now)
    if watermark is not None:
        watermark_at = _parse_watermark(watermark)
        overlap = timedelta(hours=int(getattr(config, "overlap_hours", 24)))
        catchup = timedelta(days=int(getattr(config, "catchup_days", 7)))
        start = max(watermark_at - overlap, end - catchup)
        return start, end

    initial_days = int(
        getattr(config, "initial_lookback_days", DEFAULT_INITIAL_LOOKBACK_DAYS)
    )
    window_hours = int(getattr(config, "window_hours", 72))
    start = end - timedelta(hours=max(window_hours, 24 * initial_days))
    return start, end


def _checked_categories(categories: Iterable[str] | None) -> list[str]:
    checked: list[str] = []
    for value in categories or ():
        if not isinstance(value, str):
            raise ValueError("search categories must be strings")
        category = value.strip()
        if not category or not _CATEGORY_PATTERN.fullmatch(category):
            raise ValueError(f"invalid arXiv category: {value!r}")
        if category not in checked:
            checked.append(category)
    return checked


def build_queries(profile: dict, categories: list[str]) -> list[dict]:
    """Build one query per interest group, reusing ``term_clause``.

    Categories become a single ``(cat:A OR cat:B)`` clause joined with the
    term clauses. The submitted-date clause is added by ``fetch_candidates``
    because only that layer knows the incremental window.
    """
    search = parse_profile(profile)
    checked = _checked_categories(categories)
    category_clause = ""
    if checked:
        category_clause = "(" + " OR ".join(f"cat:{item}" for item in checked) + ")"

    queries: list[dict] = []
    for group in search["groups"]:
        clauses = [term_clause(group["any_terms"])]
        if group["require_any_terms"]:
            clauses.append(term_clause(group["require_any_terms"]))
        if category_clause:
            clauses.append(category_clause)
        queries.append({"name": group["name"], "query": " AND ".join(clauses)})
    return queries


def _date_clause(start: datetime, end: datetime) -> str:
    # arXiv submittedDate is UTC and minute-granular.
    return f"submittedDate:[{start:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}]"


class _RealArxivClient:
    """Adapter so the fetch loop always talks to a ``results(request)`` client."""

    def __init__(self, per_query_limit: int):
        import arxiv  # local import: tests never touch the network

        self._arxiv = arxiv
        self._client = arxiv.Client(
            page_size=per_query_limit, delay_seconds=10.0, num_retries=0
        )

    def results(self, request: SearchRequest):
        arxiv = self._arxiv
        search = arxiv.Search(
            query=request.query,
            max_results=request.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        return self._client.results(search)


def fetch_candidates(
    profile: dict,
    start: datetime,
    end: datetime,
    config: Any,
    client: Any | None = None,
    *,
    state: Any | None = None,
    now: datetime | None = None,
    data_dir: Path | str | None = None,
    overflow_out: list[dict] | None = None,
    watermark_name: str = DEFAULT_WATERMARK_NAME,
) -> list[dict]:
    """Fetch and merge candidates inside ``[start, end]``.

    ``client`` is any object exposing ``results(SearchRequest)``. When it is
    omitted, a real ``arxiv.Client`` is built lazily. A 429 raises
    :class:`RateLimited` after recording the local cooldown and never advances
    the watermark. On success, overflow beyond ``fetch_pool_limit`` is recorded
    as ``queued`` in the frontier ledger (when ``state`` is supplied) and the
    watermark moves to ``end``.
    """
    window_start = _as_utc(start)
    window_end = _as_utc(end)
    if window_start > window_end:
        raise ValueError("window start must not be after window end")

    search = parse_profile(profile)
    queries = build_queries(
        profile, list(getattr(config, "search_categories", None) or ())
    )
    date_clause = _date_clause(window_start, window_end)

    current = _as_utc(now) if now is not None else datetime.now(timezone.utc)
    cooldown_dir = Path(data_dir) if data_dir is not None else DEFAULT_COOLDOWN_DIR
    check_cooldown(cooldown_dir, current)

    per_query_limit = int(search["per_query_limit"])
    arxiv_client = client if client is not None else _RealArxivClient(per_query_limit)
    run_id = current.strftime("%Y%m%dT%H%M%S_%fZ")
    buckets: list[tuple[str, list[dict]]] = []

    try:
        for query in queries:
            request = SearchRequest(
                query=f"{query['query']} AND {date_clause}",
                max_results=per_query_limit,
            )
            papers: list[dict] = []
            for item in arxiv_client.results(request):
                paper = normalize_result(item)
                published = _as_utc(datetime.fromisoformat(paper["published"]))
                if window_start <= published <= window_end:
                    papers.append(paper)
            buckets.append((query["name"], papers))
    except Exception as error:
        if isinstance(error, RateLimited) or is_rate_limit(error):
            retry_not_before = record_cooldown(cooldown_dir, current)
            raise RateLimited(
                "arXiv returned HTTP 429; fetch stopped and the watermark was "
                "not advanced. Respect the local cooldown before retrying.",
                retry_not_before=retry_not_before,
            ) from error
        raise

    total_cap = sum(len(papers) for _, papers in buckets)
    merged, _unique_count = merge_groups(
        buckets, total_cap, run_id, current.isoformat()
    )

    limit = max(0, int(getattr(config, "fetch_pool_limit", 40)))
    selected = merged[:limit]
    overflow = merged[limit:]

    if overflow_out is not None:
        overflow_out.extend(overflow)
    if state is not None:
        for paper in overflow:
            state.record_paper(paper, "queued")
        # Advance to the newest paper actually observed, never to wall-clock
        # ``window_end``. arXiv can lag behind the local clock (weekends,
        # indexing delay); advancing on an empty window would skip every paper
        # published later with an earlier submission date.
        if merged:
            latest_published = max(
                _as_utc(datetime.fromisoformat(paper["published"]))
                for paper in merged
            )
            state.set_watermark(
                watermark_name, latest_published.isoformat()
            )

    return selected
