"""Task E1: select 40 held-out papers and produce a DRAFT split.

This module is read-only with respect to ``data/``. It reads the prepared
manifests, the metadata-screening state, and the review queues, then writes a
draft split (``splits.draft.json``) plus a draft corpus card. Nothing here is
network-aware.

Design notes
------------
* Recent pool: dataset_v1 papers first submitted in 2026.
* Historical pool: dataset_history_v1 papers from 2020-2025.
* The historical batch has no metadata screening at all. Those papers are
  included explicitly as ``relevance="unscreened"`` with
  ``review_required=True`` and a blocker in the draft. They are never silently
  treated as screened, and a recent paper with missing screening is excluded.
* Families are mutually exclusive across dev/test/reserve.
* Near-duplicate titles (token Jaccard >= 0.8) are rejected during selection and
  recorded; Task E3 performs the full leakage audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path("eval/benchmark_v2/splits.draft.json")
DEFAULT_CORPUS_CARD = Path("eval/benchmark_v2/CORPUS_CARD.md")
DRAFT_VERSION = "splits_draft_v1"
DRAFT_STATUS = "DRAFT_PENDING_HUMAN_REVIEW"
RECENT_YEAR = 2026
HISTORY_YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
NEAR_DUP_JACCARD = 0.8
RECENT_CORE_SHARE = 0.6  # design target: core:adjacent ~= 3:2
RESERVE_PER_STRATUM = 40
TOPIC_WEIGHTS = {
    "rag": 4,
    "retrieval": 3,
    "scientific_document_qa": 3,
    "evaluation_safety": 1,
    "llm_agent": 1,
    "agent_memory": 1,
    "tool_use_planning": 1,
    "domain_application": -1,
}
CORE_TOPICS = {"rag", "retrieval", "scientific_document_qa"}

DEV_FAMILIES: frozenset[str] = frozenset(
    {
        "2610.19819",
        "2610.19341",
        "2610.22163",
        "2610.20419",
        "2610.21900",
        "2610.22017",
        "2610.19549",
        "2610.19398",
    }
)

DATASET_BATCHES = ("dataset_v1", "dataset_history_v1")
HISTORY_SCREENING_PATHS = (
    "docs/stage1c/20261023_screen_remaining_v3/output/state.json",
    "docs/stage1c/20261023_screen_expand_v2/output/state.json",
    "docs/stage1c/20261023_screen_pilot_v1/output/state.json",
)
QUALITY_QUEUE_NAME = "fulltext_quality_review_queue_29.json"
BOUNDARY_QUEUE_NAME = "boundary_review_queue_131.json"

_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^a-z0-9]+")


class SelectionError(RuntimeError):
    """Raised when the eligible pool cannot satisfy the requested split."""


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------


def normalized_title(title: str | None) -> str:
    """Lowercase, strip accents/punctuation, and collapse whitespace."""
    text = unicodedata.normalize("NFKD", (title or "").casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = _NON_WORD.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def title_tokens(title: str | None) -> set[str]:
    normalized = normalized_title(title)
    return set(normalized.split()) if normalized else set()


def title_jaccard(left: str | None, right: str | None) -> float:
    """Token-set Jaccard similarity of two normalized titles."""
    left_tokens = title_tokens(left)
    right_tokens = title_tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _family_of(record: dict) -> str | None:
    return record.get("family") or record.get("family_id")


def _flags_of(record: dict) -> list[str]:
    flags = record.get("flags")
    if flags is None:
        flags = record.get("quality_flags")
    if not flags:
        return []
    return sorted({str(flag) for flag in flags})


def exclusion_reason_for(record: dict, dev_families: Iterable[str] = DEV_FAMILIES) -> str | None:
    """Return the first rule that removes ``record`` from the test pool.

    ``None`` means the record is eligible. Historical records may remain
    eligible while their screening is missing only when both
    ``screening_missing`` is true and ``source_batch`` is
    ``dataset_history_v1``; they are then visibly marked for review.
    """
    family = _family_of(record)
    if family in set(dev_families):
        return "dev_family"
    if record.get("exclude_from_test"):
        return "exclude_from_test_flag"

    status = record.get("status")
    if status and status != "prepared":
        return f"source_status:{status}"

    flags = _flags_of(record)
    if flags:
        return "quality_flag:" + ",".join(flags)

    relevance = str(record.get("relevance") or "").strip().lower()
    screening_missing = bool(record.get("screening_missing"))
    screening_status = str(record.get("screening_status") or "")

    if relevance in {"out_of_scope", "uncertain"}:
        return f"screening_{relevance}"

    if screening_missing:
        if relevance == "unscreened" and record.get("source_batch") == "dataset_history_v1":
            return None
        if screening_status.startswith("invalid"):
            return "screening_invalid_response"
        return "screening_missing"

    if relevance == "unscreened":
        return "screening_missing"
    if relevance == "missing":
        return "screening_missing"
    if relevance not in {"core", "adjacent"}:
        return f"screening_{relevance or 'missing'}"

    year = record.get("year")
    if year is not None:
        try:
            year_value = int(year)
        except (TypeError, ValueError):
            return f"invalid_year:{year}"
        if year_value != RECENT_YEAR and year_value not in HISTORY_YEARS:
            return f"year_out_of_range:{year_value}"

    if record.get("pdf_exists") is False:
        return "pdf_missing"

    return None


def _eligible(record: dict, dev_families: Iterable[str]) -> bool:
    return exclusion_reason_for(record, dev_families) is None


def _copy_test_record(record: dict, stratum: str, seed: int) -> dict:
    row = dict(record)
    row["flags"] = list(record.get("flags") or [])
    row["quality_flags"] = list(record.get("quality_flags") or row["flags"])
    row["split"] = "test"
    row["exclusion_reason"] = None
    row["selected_stratum"] = stratum
    row["selection_seed"] = seed
    row["near_duplicate_check"] = "unique"
    if row.get("screening_missing"):
        row["review_required"] = True
    return row


def _compact(record: dict, exclusion_reason: str) -> dict:
    return {
        "arxiv_id": record.get("arxiv_id"),
        "family": _family_of(record),
        "title": record.get("title"),
        "year": record.get("year"),
        "relevance": record.get("relevance"),
        "source_batch": record.get("source_batch"),
        "screening_status": record.get("screening_status"),
        "screening_missing": bool(record.get("screening_missing")),
        "license_status": record.get("license_status"),
        "flags": _flags_of(record),
        "exclusion_reason": exclusion_reason,
    }


# ---------------------------------------------------------------------------
# Deterministic selection
# ---------------------------------------------------------------------------


def _shuffled(records: Iterable[dict], rng: random.Random) -> list[dict]:
    ordered = sorted(
        records,
        key=lambda row: (row.get("year") or 0, row.get("arxiv_id") or ""),
    )
    rng.shuffle(ordered)
    return ordered


def topic_score(record: dict) -> int:
    """Prefer papers whose screening topics match the PaperLens scope.

    This is a general, pre-registered topic preference, not per-paper
    cherry-picking. Unscreened records have no topics and sort last.
    """
    topics = {str(topic) for topic in (record.get("screening_topics") or [])}
    score = sum(TOPIC_WEIGHTS.get(topic, 0) for topic in topics)
    if not topics & CORE_TOPICS:
        score -= 2
    return score


def _topic_ordered(rows: Sequence[dict], rng: random.Random) -> list[dict]:
    # Stable sort after shuffling: random within equal topic score.
    return sorted(_shuffled(rows, rng), key=lambda row: -topic_score(row))


def _recent_order(pool: Sequence[dict], recent_n: int, rng: random.Random) -> list[dict]:
    core = _topic_ordered(
        [row for row in pool if row.get("relevance") == "core"], rng
    )
    adjacent = _topic_ordered(
        [row for row in pool if row.get("relevance") == "adjacent"], rng
    )
    target_core = min(round(recent_n * RECENT_CORE_SHARE), len(core))
    # Take the core target first, then adjacent, then any spare core.
    return core[:target_core] + adjacent + core[target_core:]


def _year_quotas(pools: dict[int, list[dict]], history_n: int) -> dict[int, int]:
    total = sum(len(pool) for pool in pools.values())
    quotas = {year: 0 for year in pools}
    if total == 0:
        return quotas

    exact = {year: Fraction(history_n * len(pool), total) for year, pool in pools.items()}
    for year, value in exact.items():
        quotas[year] = min(math.floor(value), len(pools[year]))

    remainder = history_n - sum(quotas.values())
    order = sorted(
        pools,
        key=lambda year: (-(exact[year] - math.floor(exact[year])), year),
    )
    while remainder > 0:
        progressed = False
        for year in order:
            if remainder == 0:
                break
            if quotas[year] < len(pools[year]):
                quotas[year] += 1
                remainder -= 1
                progressed = True
        if not progressed:
            break
    return quotas


def _history_order(pool: Sequence[dict], history_n: int, rng: random.Random) -> list[dict]:
    year_pools = {}
    for year in HISTORY_YEARS:
        rows = [row for row in pool if row.get("year") == year]
        screened = _topic_ordered(
            [row for row in rows if not row.get("screening_missing")], rng
        )
        unscreened = _topic_ordered(
            [row for row in rows if row.get("screening_missing")], rng
        )
        year_pools[year] = screened + unscreened
    quotas = _year_quotas(year_pools, history_n)
    ordered: list[dict] = []
    for year in HISTORY_YEARS:
        ordered.extend(year_pools[year][: quotas[year]])
    for year in HISTORY_YEARS:
        ordered.extend(year_pools[year][quotas[year] :])
    return ordered


def _take(
    ordered: Sequence[dict],
    count: int,
    *,
    stratum: str,
    seed: int,
    selected: list[dict],
    selected_ids: set[str],
    family_to_id: dict[str, str],
    collisions: list[dict],
) -> int:
    before = len(selected)
    for record in ordered:
        if len(selected) - before >= count:
            break
        family = _family_of(record)
        arxiv_id = record.get("arxiv_id")
        if not family or not arxiv_id:
            collisions.append(
                {
                    "reason": "missing_identifier",
                    "stratum": stratum,
                    "rejected_arxiv_id": arxiv_id,
                    "kept_arxiv_id": None,
                    "jaccard": None,
                }
            )
            continue
        if arxiv_id in selected_ids:
            continue
        if family in family_to_id:
            collisions.append(
                {
                    "reason": "family_duplicate",
                    "stratum": stratum,
                    "rejected_arxiv_id": arxiv_id,
                    "kept_arxiv_id": family_to_id[family],
                    "jaccard": None,
                }
            )
            continue

        duplicate = None
        for kept in selected:
            similarity = title_jaccard(record.get("title"), kept.get("title"))
            if similarity >= NEAR_DUP_JACCARD:
                duplicate = (kept, similarity)
                break
        if duplicate is not None:
            kept, similarity = duplicate
            collisions.append(
                {
                    "reason": "near_duplicate_title",
                    "stratum": stratum,
                    "rejected_arxiv_id": arxiv_id,
                    "rejected_title": record.get("title"),
                    "kept_arxiv_id": kept.get("arxiv_id"),
                    "kept_title": kept.get("title"),
                    "jaccard": round(similarity, 4),
                }
            )
            continue

        selected.append(_copy_test_record(record, stratum, seed))
        selected_ids.add(arxiv_id)
        family_to_id[family] = arxiv_id
    return len(selected) - before


def select_test_papers(
    catalog: list[dict],
    dev_families: set[str],
    recent_n: int,
    history_n: int,
    seed: int = 20261025,
    collisions: list[dict] | None = None,
) -> list[dict]:
    """Select ``recent_n`` recent and ``history_n`` historical test papers.

    Raises ``SelectionError`` instead of silently returning fewer papers.
    Near-duplicate rejections and family collisions are appended to
    ``collisions`` when provided.
    """
    if recent_n < 0 or history_n < 0:
        raise SelectionError("recent_n and history_n must be non-negative")
    if collisions is None:
        collisions = []

    dev_families = set(dev_families)
    eligible = [row for row in catalog if _eligible(row, dev_families)]

    recent_pool = [row for row in eligible if row.get("year") == RECENT_YEAR]
    history_pool = [row for row in eligible if row.get("year") in HISTORY_YEARS]

    rng = random.Random(seed)
    selected: list[dict] = []
    selected_ids: set[str] = set()
    family_to_id: dict[str, str] = {}

    recent_taken = _take(
        _recent_order(recent_pool, recent_n, rng),
        recent_n,
        stratum="recent",
        seed=seed,
        selected=selected,
        selected_ids=selected_ids,
        family_to_id=family_to_id,
        collisions=collisions,
    )
    if recent_taken < recent_n:
        raise SelectionError(
            "insufficient eligible recent candidates: "
            f"requested {recent_n}, selected {recent_taken} unique/non-duplicate "
            f"from {len(recent_pool)} eligible recent papers "
            f"({len(collisions)} collision(s) recorded)"
        )

    history_taken = _take(
        _history_order(history_pool, history_n, rng),
        history_n,
        stratum="history",
        seed=seed,
        selected=selected,
        selected_ids=selected_ids,
        family_to_id=family_to_id,
        collisions=collisions,
    )
    if history_taken < history_n:
        raise SelectionError(
            "insufficient eligible historical candidates: "
            f"requested {history_n}, selected {history_taken} unique/non-duplicate "
            f"from {len(history_pool)} eligible historical papers "
            f"({len(collisions)} collision(s) recorded)"
        )

    return selected


# ---------------------------------------------------------------------------
# Catalog construction (read-only)
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"required read-only input not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_optional_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _year_from_published(value) -> int | None:
    text = str(value or "")
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return None


def _load_history_screening(root: Path) -> dict:
    """Merge the read-only stage1c history screening outputs (pilot < v2 < v3)."""
    merged: dict = {}
    for relative in reversed(HISTORY_SCREENING_PATHS):
        state = _read_optional_json(root / relative) or {}
        results = state.get("results") or {}
        if isinstance(results, dict):
            merged.update(results)
    return merged


def build_catalog(root: Path) -> list[dict]:
    """Build the read-only candidate catalog from prepared artifacts.

    Reads (never writes): ``state.json``, ``prepared_v1/manifest.json``, the
    metadata-screening state, and the review-note queues.
    """
    root = Path(root)
    screening_path = (
        root
        / "data"
        / "dataset_v1"
        / "prepared_v1"
        / "screening_v1"
        / "state.json"
    )
    screening_state = _read_optional_json(screening_path) or {}
    screening_results = screening_state.get("results") or {}
    history_screening = _load_history_screening(root)
    history_by_base = {
        str(key).split("v")[0]: value for key, value in history_screening.items()
    }

    quality_queue = _read_optional_json(root / "review_notes" / QUALITY_QUEUE_NAME) or []
    boundary_queue = _read_optional_json(root / "review_notes" / BOUNDARY_QUEUE_NAME) or []
    quality_flags_by_id = {
        item.get("arxiv_id"): list(item.get("flags") or [])
        for item in quality_queue
        if isinstance(item, dict) and item.get("arxiv_id")
    }
    boundary_ids = {
        item.get("paper_id")
        for item in boundary_queue
        if isinstance(item, dict) and item.get("paper_id")
    }

    catalog: list[dict] = []
    for batch in DATASET_BATCHES:
        base = root / "data" / batch
        manifest = _read_json(base / "prepared_v1" / "manifest.json")
        state = _read_optional_json(base / "state.json") or {}
        state_papers = state.get("papers") if isinstance(state.get("papers"), dict) else {}

        for paper in manifest.get("papers") or []:
            arxiv_id = paper.get("arxiv_id")
            family = paper.get("family_id") or paper.get("family")
            if not arxiv_id or not family:
                continue
            year = _year_from_published(paper.get("published"))
            pages_file = paper.get("pages_file") or ""
            pdf_path = f"data/{batch}/pdfs/{arxiv_id}.pdf"
            prepared_path = f"data/{batch}/prepared_v1/{pages_file}" if pages_file else None

            if batch == "dataset_history_v1":
                result = history_screening.get(arxiv_id) or history_by_base.get(
                    arxiv_id.split("v")[0]
                )
            else:
                result = (
                    screening_results.get(arxiv_id)
                    if isinstance(screening_results, dict)
                    else None
                )

            relevance = "unscreened" if batch == "dataset_history_v1" else "missing"
            screening_status = "missing"
            screening_missing = True
            screening_topics: list[str] = []
            if isinstance(result, dict):
                screening_status = str(
                    result.get("classification_status")
                    or result.get("status")
                    or "missing"
                )
                suggestion = result.get("suggestion") or {}
                suggested = str(suggestion.get("relevance") or "").strip().lower()
                if screening_status.startswith("invalid"):
                    relevance = "missing"
                    screening_missing = True
                elif suggested:
                    relevance = suggested
                    screening_missing = False
                    screening_topics = list(suggestion.get("topics") or [])

            flags = sorted(
                set(paper.get("flags") or []) | set(quality_flags_by_id.get(arxiv_id, []))
            )
            boundary_review = arxiv_id in boundary_ids

            record = {
                "arxiv_id": arxiv_id,
                "family": family,
                "family_id": family,
                "title": paper.get("title"),
                "year": year,
                "published": paper.get("published"),
                "relevance": relevance,
                "screening_status": screening_status,
                "screening_missing": screening_missing,
                "screening_topics": screening_topics,
                "source_batch": batch,
                "flags": flags,
                "quality_flags": list(flags),
                "status": paper.get("status"),
                "exclude_from_test": bool(paper.get("exclude_from_test")),
                "pdf_path": pdf_path,
                "pdf_exists": (root / pdf_path).exists(),
                "prepared_path": prepared_path,
                "prepared_exists": bool(prepared_path and (root / prepared_path).exists()),
                "license_status": paper.get("license_status"),
                "categories": paper.get("categories") or [],
                "query_matches": paper.get("query_matches") or [],
                "page_count": paper.get("page_count"),
                "text_chars": paper.get("text_chars"),
                "boundary_review_required": boundary_review,
                "review_required": screening_missing or boundary_review,
            }
            state_paper = state_papers.get(family) if isinstance(state_papers, dict) else None
            if isinstance(state_paper, dict):
                record["state_status"] = state_paper.get("status")
                pdf_info = state_paper.get("pdf") or {}
                record["pdf_sha256"] = pdf_info.get("sha256")

            record["exclusion_reason"] = exclusion_reason_for(record, DEV_FAMILIES)
            catalog.append(record)

    return catalog


# ---------------------------------------------------------------------------
# Draft split + corpus card
# ---------------------------------------------------------------------------


def _manifest_hash(
    *,
    seed: int,
    recent_n: int,
    history_n: int,
    dev_families: Iterable[str],
    test_records: Sequence[dict],
    reserve_records: Sequence[dict],
) -> str:
    payload = {
        "version": DRAFT_VERSION,
        "seed": seed,
        "requested": {"recent": recent_n, "history": history_n},
        "dev_families": sorted(dev_families),
        "test_families": sorted({_family_of(row) for row in test_records}),
        "test_arxiv_ids": sorted(row["arxiv_id"] for row in test_records),
        "reserve_arxiv_ids": sorted(row["arxiv_id"] for row in reserve_records),
    }
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _near_dup_against_test(record: dict, test_records: Sequence[dict]) -> dict | None:
    for selected in test_records:
        similarity = title_jaccard(record.get("title"), selected.get("title"))
        if similarity >= NEAR_DUP_JACCARD:
            return {"kept": selected, "jaccard": similarity}
    return None


def build_splits(
    catalog: list[dict],
    dev_families: set[str],
    recent_n: int,
    history_n: int,
    seed: int = 20261025,
) -> dict:
    """Build the draft split payload (no file I/O)."""
    dev_families = set(dev_families)
    collisions: list[dict] = []
    test_records = select_test_papers(
        catalog,
        dev_families,
        recent_n,
        history_n,
        seed=seed,
        collisions=collisions,
    )
    test_ids = {row["arxiv_id"] for row in test_records}
    test_families = {_family_of(row) for row in test_records}

    dev_rows: list[dict] = []
    seen_dev: set[str] = set()
    for record in catalog:
        family = _family_of(record)
        if family in dev_families and family not in seen_dev:
            seen_dev.add(family)
            dev_rows.append(_compact(record, "dev_family"))

    eligible_remaining = [
        record
        for record in catalog
        if _eligible(record, dev_families)
        and record.get("arxiv_id") not in test_ids
        and _family_of(record) not in test_families
    ]
    recent_remaining = sorted(
        [row for row in eligible_remaining if row.get("year") == RECENT_YEAR],
        key=lambda row: row.get("arxiv_id") or "",
    )
    history_remaining = sorted(
        [row for row in eligible_remaining if row.get("year") in HISTORY_YEARS],
        key=lambda row: (row.get("year") or 0, row.get("arxiv_id") or ""),
    )

    reserve_rows: list[dict] = []
    near_dup_reserve_skipped = 0
    for stratum, pool in (("recent", recent_remaining), ("history", history_remaining)):
        taken = 0
        for record in pool:
            if taken >= RESERVE_PER_STRATUM:
                break
            if _near_dup_against_test(record, test_records) is not None:
                near_dup_reserve_skipped += 1
                continue
            reserve_rows.append(_compact(record, "reserve_not_selected"))
            taken += 1

    excluded_rows = [
        _compact(record, exclusion_reason_for(record, dev_families))
        for record in catalog
        if exclusion_reason_for(record, dev_families) is not None
    ]

    year_counts = Counter(str(row.get("year")) for row in test_records)
    relevance_counts = Counter(str(row.get("relevance")) for row in test_records)
    batch_counts = Counter(str(row.get("source_batch")) for row in test_records)

    missing_total = sum(1 for row in catalog if row.get("screening_missing"))
    missing_selected = sum(1 for row in test_records if row.get("screening_missing"))
    invalid_total = sum(
        1
        for row in catalog
        if str(row.get("screening_status") or "").startswith("invalid")
    )
    flagged_total = sum(1 for row in catalog if _flags_of(row))
    license_unverified = sum(
        1 for row in test_records if row.get("license_status") != "verified"
    )
    pdf_missing_test = [row["arxiv_id"] for row in test_records if row.get("pdf_exists") is False]
    missing_dev = sorted(dev_families - {_family_of(row) for row in dev_rows})
    near_dup_collisions = [
        item for item in collisions if item.get("reason") == "near_duplicate_title"
    ]

    blockers: list[str] = []
    if missing_selected:
        blockers.append(
            "history_screening_missing: "
            f"{missing_selected} selected historical papers have no screening record "
            f"(all {missing_total} catalog papers lacking screening); they are marked "
            "relevance=unscreened and review_required=true"
        )
    if invalid_total:
        blockers.append(
            f"screening_invalid_responses: {invalid_total} recent papers have invalid "
            "screening responses and are excluded"
        )
    if missing_dev:
        blockers.append(f"dev_family_missing_from_catalog: {', '.join(missing_dev)}")
    if near_dup_collisions:
        blockers.append(
            f"near_duplicate_title_collisions: {len(near_dup_collisions)} candidate(s) "
            f"rejected at title Jaccard >= {NEAR_DUP_JACCARD}"
        )
    if pdf_missing_test:
        blockers.append(f"pdf_missing_for_selected: {', '.join(pdf_missing_test)}")

    warnings: list[str] = []
    if flagged_total:
        warnings.append(
            f"quality_flags_excluded: {flagged_total} papers carry quality flags"
        )
    if license_unverified:
        warnings.append(
            f"license_not_verified: {license_unverified}/{len(test_records)} test papers "
            "remain not_verified; do not redistribute PDFs or full text"
        )
    if near_dup_reserve_skipped:
        warnings.append(
            f"reserve_near_duplicates_skipped: {near_dup_reserve_skipped} reserve "
            "candidates were dropped because their titles are near duplicates of test papers"
        )
    warnings.append(
        "draft_only: 40 papers support descriptive statistics only; do not freeze "
        "or run test once until a human reviews this split"
    )
    warnings.append(
        "near_duplicate_scope: title-token Jaccard only; abstract/full-text leakage "
        "audit is Task E3 (leakage_audit.py)"
    )

    test_sorted = sorted(
        test_records,
        key=lambda row: (
            0 if row.get("selected_stratum") == "recent" else 1,
            row.get("year") or 0,
            row.get("arxiv_id") or "",
        ),
    )

    splits = {
        "version": DRAFT_VERSION,
        "status": DRAFT_STATUS,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "eval/benchmark_v2/select_corpus.py --dry-run",
        "seed": seed,
        "requested": {"recent": recent_n, "history": history_n, "total": recent_n + history_n},
        "selection_rules": {
            "recent_year": RECENT_YEAR,
            "history_years": list(HISTORY_YEARS),
            "recent_core_share_target": RECENT_CORE_SHARE,
            "near_duplicate_title_jaccard": NEAR_DUP_JACCARD,
            "excluded_relevance": ["out_of_scope", "uncertain"],
            "dev_families": sorted(dev_families),
        },
        "counts": {
            "total": len(test_records),
            "recent": sum(1 for row in test_records if row.get("year") == RECENT_YEAR),
            "history": sum(1 for row in test_records if row.get("year") in HISTORY_YEARS),
            "dev": len(dev_rows),
            "reserve": len(reserve_rows),
            "excluded": len(excluded_rows),
            "eligible_not_selected": max(
                0, len(eligible_remaining) - len(reserve_rows)
            ),
            "screening_missing_selected": missing_selected,
        },
        "distributions": {
            "year": dict(sorted(year_counts.items())),
            "relevance": dict(sorted(relevance_counts.items())),
            "source_batch": dict(sorted(batch_counts.items())),
            "recent_core_adjacent": {
                "core": sum(
                    1
                    for row in test_records
                    if row.get("year") == RECENT_YEAR and row.get("relevance") == "core"
                ),
                "adjacent": sum(
                    1
                    for row in test_records
                    if row.get("year") == RECENT_YEAR and row.get("relevance") == "adjacent"
                ),
            },
        },
        "screening": {
            "missing_total": missing_total,
            "missing_selected": missing_selected,
            "invalid_total": invalid_total,
            "flagged_total": flagged_total,
        },
        "dev": dev_rows,
        "test": test_sorted,
        "reserve": reserve_rows,
        "excluded": excluded_rows,
        "near_duplicate_collisions": collisions,
        "blockers": blockers,
        "warnings": warnings,
    }
    splits["manifest_hash"] = _manifest_hash(
        seed=seed,
        recent_n=recent_n,
        history_n=history_n,
        dev_families=dev_families,
        test_records=test_sorted,
        reserve_records=reserve_rows,
    )
    return splits


def _reason_summary(excluded: Sequence[dict]) -> list[tuple[str, int]]:
    counter = Counter()
    for row in excluded:
        reason = str(row.get("exclusion_reason") or "unknown")
        counter[reason.split(":", 1)[0]] += 1
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def render_corpus_card(splits: dict) -> str:
    """Render the DRAFT corpus card from a split payload."""
    lines: list[str] = []
    lines.append("# PaperLens Benchmark v2 — Corpus Card (DRAFT)")
    lines.append("")
    lines.append(
        f"**Status:** `{splits['status']}` — not frozen. Generated "
        f"{splits['created_at']} by `{splits['generated_by']}`."
    )
    lines.append(f"**Manifest hash:** `{splits['manifest_hash']}`")
    lines.append(
        f"**Seed:** `{splits['seed']}`  |  **Requested:** "
        f"{splits['requested']['recent']} recent + {splits['requested']['history']} historical"
    )
    lines.append("")
    lines.append("> Human review is required before this split is frozen. Replace parsing")
    lines.append("> failures from `reserve`; never silently shrink the test set.")
    lines.append("")

    counts = splits["counts"]
    lines.append("## Counts")
    lines.append("")
    lines.append("| Split | Count |")
    lines.append("|---|---:|")
    for key in (
        "total",
        "recent",
        "history",
        "dev",
        "reserve",
        "excluded",
        "eligible_not_selected",
        "screening_missing_selected",
    ):
        lines.append(f"| {key} | {counts.get(key, 0)} |")
    lines.append("")

    lines.append("## Distribution")
    lines.append("")
    lines.append(
        f"- Year: {', '.join(f'{year}={count}' for year, count in splits['distributions']['year'].items())}"
    )
    lines.append(
        f"- Relevance: {', '.join(f'{name}={count}' for name, count in splits['distributions']['relevance'].items())}"
    )
    lines.append(
        f"- Source batch: {', '.join(f'{name}={count}' for name, count in splits['distributions']['source_batch'].items())}"
    )
    recent_split = splits["distributions"]["recent_core_adjacent"]
    lines.append(
        f"- Recent core/adjacent: core={recent_split['core']}, adjacent={recent_split['adjacent']}"
    )
    lines.append("")

    lines.append("## Test papers (40)")
    lines.append("")
    lines.append("| # | arXiv ID | family | year | relevance | screening | batch | license | review |")
    lines.append("|---:|---|---|---:|---|---|---|---|---|")
    for index, row in enumerate(splits["test"], start=1):
        screening = (
            "MISSING" if row.get("screening_missing") else str(row.get("screening_status") or "")
        )
        review = "required" if row.get("review_required") else "no"
        lines.append(
            f"| {index} | {row['arxiv_id']} | {row['family']} | {row['year']} | "
            f"{row['relevance']} | {screening} | {row['source_batch']} | "
            f"{row.get('license_status')} | {review} |"
        )
    lines.append("")

    lines.append("## Dev families (excluded)")
    lines.append("")
    for row in splits["dev"]:
        lines.append(f"- `{row['family']}` — {row['title']}")
    lines.append("")

    lines.append("## Exclusion summary")
    lines.append("")
    lines.append("| Reason | Count |")
    lines.append("|---|---:|")
    for reason, count in _reason_summary(splits["excluded"]):
        lines.append(f"| {reason} | {count} |")
    lines.append("")

    collisions = [
        item
        for item in splits.get("near_duplicate_collisions", [])
        if item.get("reason") == "near_duplicate_title"
    ]
    lines.append("## Near-duplicate title collisions")
    lines.append("")
    if collisions:
        lines.append("| Rejected | Kept | Jaccard |")
        lines.append("|---|---|---:|")
        for item in collisions:
            lines.append(
                f"| {item.get('rejected_arxiv_id')} | {item.get('kept_arxiv_id')} | "
                f"{item.get('jaccard')} |"
            )
    else:
        lines.append("No title collisions at Jaccard >= 0.8 were found in the selected pool.")
    lines.append("")

    lines.append("## Blockers")
    lines.append("")
    if splits["blockers"]:
        for blocker in splits["blockers"]:
            lines.append(f"- {blocker}")
    else:
        lines.append("- None recorded.")
    lines.append("")

    lines.append("## Warnings and limitations")
    lines.append("")
    for warning in splits["warnings"]:
        lines.append(f"- {warning}")
    lines.append("- No PDFs or full text are published; this card contains arXiv IDs and metadata only.")
    lines.append("- Labels are model suggestions, not human verification (`human_reviewed=0`).")
    lines.append("")

    lines.append("## Review checklist")
    lines.append("")
    lines.append("- [ ] Verify all 40 arXiv IDs and that no dev family leaked into `test`.")
    lines.append("- [ ] Replace any paper whose PDF fails to parse with one from `reserve`.")
    lines.append("- [ ] Confirm the 16 historical papers (no screening) are in scope.")
    lines.append("- [ ] Run the Task E3 leakage audit before freezing.")
    lines.append("- [ ] Only then rename `splits.draft.json` to `splits.json`.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _summary_payload(splits: dict) -> dict:
    recent_split = splits["distributions"]["recent_core_adjacent"]
    return {
        "status": splits["status"],
        "selected_total": splits["counts"]["total"],
        "recent": splits["counts"]["recent"],
        "history": splits["counts"]["history"],
        "year_distribution": splits["distributions"]["year"],
        "relevance_distribution": splits["distributions"]["relevance"],
        "recent_core_adjacent": recent_split,
        "screening_missing_selected": splits["counts"]["screening_missing_selected"],
        "near_duplicate_collisions": len(
            [
                item
                for item in splits["near_duplicate_collisions"]
                if item.get("reason") == "near_duplicate_title"
            ]
        ),
        "blockers": splits["blockers"],
        "manifest_hash": splits["manifest_hash"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select the held-out PaperLens benchmark corpus (Task E1)."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="PaperLens repository root (default: %(default)s)",
    )
    parser.add_argument("--recent-n", type=int, default=24)
    parser.add_argument("--history-n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write splits.draft.json and CORPUS_CARD.md (draft artifacts)",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--corpus-card", type=Path, default=None)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    out_path = args.out or (root / DEFAULT_OUT)
    card_path = args.corpus_card or (root / DEFAULT_CORPUS_CARD)

    try:
        catalog = build_catalog(root)
        splits = build_splits(
            catalog,
            set(DEV_FAMILIES),
            args.recent_n,
            args.history_n,
            seed=args.seed,
        )
    except (SelectionError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    summary = _summary_payload(splits)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not args.dry_run:
        print(
            "\nNo files written. Re-run with --dry-run to write "
            "eval/benchmark_v2/splits.draft.json and CORPUS_CARD.md.",
            file=sys.stderr,
        )
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(splits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(render_corpus_card(splits), encoding="utf-8")
    print(f"\nWrote draft split: {out_path}")
    print(f"Wrote draft corpus card: {card_path}")
    print("DRAFT ONLY: do not rename to splits.json before human review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
