"""Deterministic retrieval and answer metrics for Eval-lite (Task E5).

Every function here is pure: no model calls, no network, no file access, no
randomness. The same inputs always produce the same floats.

Conventions that callers must not silently change
-------------------------------------------------
* ``retrieved`` is an ordered best-first list of parent-chunk IDs.
* ``k`` is the ranking cutoff. ``k == 0`` yields ``0.0``; a negative ``k``
  raises ``ValueError`` because it can only be a caller bug.
* An empty ``gold`` set is scored ``0.0`` rather than the vacuous ``1.0``.
  Treating "no gold" as perfect would inflate averages over answerable
  questions; callers that evaluate unanswerable questions should exclude them
  from the retrieval denominator (``score_run`` does exactly that).
* nDCG uses exponential gain ``2**rel - 1`` and discount ``log2(rank + 1)``,
  the standard formulation. When ``rel`` is omitted every gold ID has grade 2;
  when ``rel`` is supplied, an explicit grade there overrides the gold default.
  Grades must be non-negative integers.
* Duplicate IDs in ``retrieved`` are scored only at their first occurrence for
  nDCG, so a repeated hit cannot inflate the score. Recall and MRR are
  unaffected by duplicates.
* Quotes are compared after whitespace normalization only (" ".join split);
  case and punctuation are preserved, matching ``answer_question.validate_answer``.
  A citation without a resolvable source text counts as a failure, and an
  empty citation list yields ``0.0`` (no citations means integrity is
  unverified, not perfect).
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

# Statuses that can never be a schema success, even if ``schema_ok`` says True.
FAILED_STATUSES = frozenset(
    {"failed", "error", "timeout", "schema_invalid", "invalid", "missing_run"}
)

# Statuses a successful E6/E7 answer record may use when it omits ``schema_ok``.
SUCCESS_STATUSES = frozenset({"ok", "success", "answered", "partial", "insufficient"})

# Citation keys that may carry the cited evidence text directly.
SOURCE_TEXT_KEYS = ("source_text", "evidence_text", "text")

GOLD_GRADE = 2


def _check_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k < 0:
        raise ValueError("k must be >= 0")
    return k


def _top_k(retrieved: Sequence[str] | None, k: int) -> list[str]:
    _check_k(k)
    if not retrieved:
        return []
    return list(retrieved)[:k]


def _gold_set(gold: Iterable[str] | None) -> set[str]:
    return set(gold or ())


def recall_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    """Fraction of gold parent chunks present in the top-k ranking."""
    gold_set = _gold_set(gold)
    top = _top_k(retrieved, k)
    if not gold_set or not top:
        return 0.0
    hits = len(gold_set.intersection(top))
    return hits / len(gold_set)


def all_gold_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    """1.0 only when every gold chunk is in the top-k ranking, else 0.0."""
    gold_set = _gold_set(gold)
    top = _top_k(retrieved, k)
    if not gold_set or not top:
        return 0.0
    return 1.0 if gold_set.issubset(set(top)) else 0.0


def mrr_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    """Reciprocal rank of the first relevant hit within top-k (0.0 if none)."""
    gold_set = _gold_set(gold)
    top = _top_k(retrieved, k)
    if not gold_set:
        return 0.0
    for rank, item in enumerate(top, 1):
        if item in gold_set:
            return 1.0 / rank
    return 0.0


def _normalized_grades(rel: Mapping[str, int] | None) -> dict[str, int]:
    if rel is None:
        return {}
    if not isinstance(rel, Mapping):
        raise TypeError("rel must be a mapping of id -> non-negative int")
    grades: dict[str, int] = {}
    for key, value in rel.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("relevance grades must be non-negative integers")
        grades[key] = value
    return grades


def _effective_grade(item: str, gold: set[str], rel: Mapping[str, int]) -> int:
    if item in rel:
        return rel[item]
    return GOLD_GRADE if item in gold else 0


def _gain(grade: int) -> float:
    return float(2**grade - 1) if grade > 0 else 0.0


def ndcg_at_k(
    gold: set[str],
    retrieved: list[str],
    k: int,
    rel: dict[str, int] | None = None,
) -> float:
    """Graded nDCG@k with exponential gain and log2 discount.

    ``rel`` maps IDs to relevance grades (2 = gold, 1 = same page/adjacent,
    0 = not relevant). IDs absent from ``rel`` but present in ``gold`` default
    to grade 2; all other retrieved IDs are grade 0. The ideal DCG is built
    from the judged pool (``gold`` plus ``rel`` entries with grade > 0).
    """
    gold_set = _gold_set(gold)
    grades = _normalized_grades(rel)
    top = _top_k(retrieved, k)
    if not top:
        return 0.0

    dcg = 0.0
    seen: set[str] = set()
    for rank, item in enumerate(top, 1):
        if item in seen:
            continue
        seen.add(item)
        grade = _effective_grade(item, gold_set, grades)
        if grade > 0:
            dcg += _gain(grade) / math.log2(rank + 1)

    pool = set(gold_set) | {item for item, grade in grades.items() if grade > 0}
    ideal_grades = sorted(
        (_effective_grade(item, gold_set, grades) for item in pool),
        reverse=True,
    )
    ideal_grades = [grade for grade in ideal_grades if grade > 0][:k]
    if not ideal_grades:
        return 0.0
    idcg = sum(
        _gain(grade) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, 1)
    )
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def _normalized_text(text: str) -> str:
    return " ".join(text.split())


def _citation_source(citation: dict) -> str | None:
    for key in SOURCE_TEXT_KEYS:
        value = citation.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def quote_integrity(citations: list[dict]) -> float:
    """Fraction of citations whose quote is verbatim in its cited evidence.

    Whitespace is normalized on both sides; case and punctuation must match.
    Citations must carry their evidence text in ``source_text``,
    ``evidence_text`` or ``text`` (``score_run`` resolves an evidence map into
    ``source_text`` before calling this). Unresolvable citations count as
    failures, and an empty list returns 0.0.
    """
    items = list(citations or ())
    if not items:
        return 0.0
    valid = 0
    for citation in items:
        if not isinstance(citation, dict):
            continue
        quote = citation.get("quote")
        source = _citation_source(citation)
        if not isinstance(quote, str) or not quote.strip() or source is None:
            continue
        if _normalized_text(quote) in _normalized_text(source):
            valid += 1
    return valid / len(items)


def run_schema_ok(run: dict) -> bool:
    """Whether one run record is a schema-valid answer.

    A failed/error/schema-invalid status always loses, even when ``schema_ok``
    is present and True. Records without ``schema_ok`` are judged by status.
    """
    if not isinstance(run, dict):
        return False
    if run.get("status") in FAILED_STATUSES:
        return False
    if "schema_ok" in run:
        return run["schema_ok"] is True
    return run.get("status") in SUCCESS_STATUSES


def schema_success(runs: list[dict]) -> float:
    """Fraction of runs that produced a schema-valid answer.

    Failed runs stay in the denominator and count as 0; they are never dropped.
    An empty list returns 0.0.
    """
    items = list(runs or ())
    if not items:
        return 0.0
    return sum(1 for run in items if run_schema_ok(run)) / len(items)
