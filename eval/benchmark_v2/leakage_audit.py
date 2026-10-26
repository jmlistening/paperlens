"""Task E3: leakage audit for the held-out split and test index.

The audit is read-only and deterministic. It checks four things:

1. No arXiv family appears in both the dev and test split.
2. No dev/test pair of papers is a near duplicate by title (token Jaccard
   >= 0.8) or abstract (>= 0.75).
3. The held-out index only contains families from the test split, and every
   test paper is present in it.
4. Every test paper has a PDF and an index entry, and every exclusion reason
   is accounted for.

The command line entry point reads the prepared manifests, ``splits.draft.json``
and ``eval/benchmark_v2/index_manifest.json`` (all read-only) and writes
``eval/benchmark_v2/leakage_report.json``. It never builds an index, loads a
model, or touches the network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.benchmark_v2.select_corpus import (  # noqa: E402
    title_jaccard,
    title_tokens,
)

SCHEMA_VERSION = 1
TITLE_JACCARD_THRESHOLD = 0.8
ABSTRACT_JACCARD_THRESHOLD = 0.75
DEFAULT_SPLITS = PROJECT_ROOT / "eval" / "benchmark_v2" / "splits.draft.json"
DEFAULT_INDEX_MANIFEST = (
    PROJECT_ROOT / "eval" / "benchmark_v2" / "index_manifest.json"
)
DEFAULT_OUT = PROJECT_ROOT / "eval" / "benchmark_v2" / "leakage_report.json"
DEFAULT_MANIFESTS = (
    PROJECT_ROOT / "data" / "dataset_v1" / "prepared_v1" / "manifest.json",
    PROJECT_ROOT
    / "data"
    / "dataset_history_v1"
    / "prepared_v1"
    / "manifest.json",
)

_VERSION_SUFFIX = re.compile(r"v\d+$")
_SPLIT_NAMES = ("dev", "test", "reserve", "excluded")


class LeakageAuditError(RuntimeError):
    """Raised when an audit input is missing or malformed."""


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------


def _base_id(value: Any) -> str:
    """Return the version-stripped arXiv ID (the family)."""
    if value is None:
        return ""
    return _VERSION_SUFFIX.sub("", str(value).strip())


def _family_of(record: dict) -> str:
    for key in ("family", "family_id"):
        value = record.get(key)
        if value:
            return _base_id(value)
    for key in ("arxiv_id", "paper_id", "id"):
        value = record.get(key)
        if value:
            return _base_id(value)
    return ""


def _paper_id_of(record: dict) -> str:
    for key in ("arxiv_id", "paper_id", "id"):
        value = record.get(key)
        if value:
            return str(value).strip()
    return _family_of(record)


def _text_jaccard(left: str | None, right: str | None) -> float:
    left_tokens = title_tokens(left)
    right_tokens = title_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _split_records(splits: dict, name: str) -> list[dict]:
    value = splits.get(name) if isinstance(splits, dict) else None
    if value is None:
        return []
    if not isinstance(value, list):
        raise LeakageAuditError(f"splits[{name!r}] must be a list")
    return [record for record in value if isinstance(record, dict)]


# ---------------------------------------------------------------------------
# Corpus indexing and merged record views
# ---------------------------------------------------------------------------


def _index_corpus(corpus: Iterable[dict]) -> tuple[dict[str, dict], list[dict]]:
    """Index corpus records by family and keep any chunk-level records."""
    by_family: dict[str, dict] = {}
    chunk_records: list[dict] = []
    for record in corpus or []:
        if not isinstance(record, dict):
            continue
        if "chunk_id" in record:
            chunk_records.append(record)
        family = _family_of(record)
        if not family:
            continue
        current = by_family.get(family)
        if current is None:
            by_family[family] = record
            continue
        # Prefer the richest record for the family.
        if not current.get("abstract") and record.get("abstract"):
            by_family[family] = record
        elif not current.get("title") and record.get("title"):
            by_family[family] = record
    return by_family, chunk_records


def _merged_views(
    records: Sequence[dict], split: str, corpus_by_family: dict[str, dict]
) -> list[dict]:
    views: list[dict] = []
    for record in records:
        family = _family_of(record)
        base = corpus_by_family.get(family) or {}
        views.append(
            {
                "record": record,
                "split": split,
                "family": family,
                "paper_id": _paper_id_of(record),
                "title": record.get("title") or base.get("title"),
                "abstract": record.get("abstract") or base.get("abstract"),
            }
        )
    return views


# ---------------------------------------------------------------------------
# Index identity
# ---------------------------------------------------------------------------


def _index_identity(
    index_manifest: Any,
) -> tuple[set[str], set[str], list[dict], list[dict]]:
    """Return (paper_ids, families, paper entries, id/family mismatches)."""
    paper_ids: set[str] = set()
    families: set[str] = set()
    entries: list[dict] = []
    mismatches: list[dict] = []

    if isinstance(index_manifest, list):
        raw_papers = index_manifest
        raw_ids: list[Any] = []
        raw_families: list[Any] = []
    elif isinstance(index_manifest, dict):
        raw_papers = index_manifest.get("papers") or []
        raw_ids = list(index_manifest.get("paper_ids") or [])
        raw_families = list(index_manifest.get("families") or [])
    else:
        raise LeakageAuditError("index_manifest must be a dict or a list")
    if not isinstance(raw_papers, list):
        raise LeakageAuditError("index_manifest['papers'] must be a list")

    for value in raw_ids:
        text = str(value).strip()
        if text:
            paper_ids.add(text)
            families.add(_base_id(text))
    for value in raw_families:
        family = _base_id(value)
        if family:
            families.add(family)

    for item in raw_papers:
        if not isinstance(item, dict):
            continue
        paper_id = _paper_id_of(item)
        family = _family_of(item)
        if paper_id:
            paper_ids.add(paper_id)
            families.add(_base_id(paper_id))
        if family:
            families.add(family)
        entries.append({"paper_id": paper_id, "family": family})
        if paper_id and family and _base_id(paper_id) != family:
            mismatches.append(
                {
                    "paper_id": paper_id,
                    "declared_family": family,
                    "implied_family": _base_id(paper_id),
                }
            )

    # A manifest that only lists IDs still gives us one entry per ID.
    if not entries:
        entries = [
            {"paper_id": paper_id, "family": _base_id(paper_id)}
            for paper_id in sorted(paper_ids)
        ]

    return paper_ids, families, entries, mismatches


# ---------------------------------------------------------------------------
# Pair collection
# ---------------------------------------------------------------------------


def _pair_payload(
    left: dict, right: dict, jaccard: float, scope: str
) -> dict:
    ordered = sorted(
        [left, right], key=lambda view: (view["family"], view["paper_id"])
    )
    first, second = ordered
    return {
        "families": [first["family"], second["family"]],
        "paper_ids": [first["paper_id"], second["paper_id"]],
        "titles": [first["title"], second["title"]],
        "jaccard": round(jaccard, 4),
        "scope": scope,
        "cross_split": scope == "dev_test",
    }


def _add_pair(
    pairs: dict[tuple[str, str], dict],
    left: dict,
    right: dict,
    jaccard: float,
    scope: str,
) -> None:
    if not left["family"] or not right["family"]:
        return
    if left["family"] == right["family"]:
        return
    key = tuple(sorted((left["family"], right["family"])))
    existing = pairs.get(key)
    if existing is not None and existing["scope"] == "dev_test":
        return
    pairs[key] = _pair_payload(left, right, jaccard, scope)


def _collect_title_pairs(
    dev: Sequence[dict],
    test: Sequence[dict],
    pool: Sequence[dict],
) -> list[dict]:
    pairs: dict[tuple[str, str], dict] = {}

    def _compare(left: dict, right: dict, scope: str) -> None:
        if not left["title"] or not right["title"]:
            return
        score = title_jaccard(left["title"], right["title"])
        if score >= TITLE_JACCARD_THRESHOLD:
            _add_pair(pairs, left, right, score, scope)

    for left in dev:
        for right in test:
            _compare(left, right, "dev_test")

    for index, left in enumerate(test):
        for right in test[index + 1 :]:
            _compare(left, right, "test_internal")

    for left in test:
        for right in pool:
            _compare(left, right, "test_pool")

    return sorted(
        pairs.values(),
        key=lambda pair: (
            -pair["jaccard"],
            pair["families"][0],
            pair["families"][1],
        ),
    )


def _collect_abstract_pairs(
    dev: Sequence[dict], test: Sequence[dict]
) -> list[dict]:
    pairs: dict[tuple[str, str], dict] = {}

    for left in dev:
        for right in test:
            score = _text_jaccard(left["abstract"], right["abstract"])
            if score >= ABSTRACT_JACCARD_THRESHOLD:
                _add_pair(pairs, left, right, score, "dev_test")

    for index, left in enumerate(test):
        for right in test[index + 1 :]:
            score = _text_jaccard(left["abstract"], right["abstract"])
            if score >= ABSTRACT_JACCARD_THRESHOLD:
                _add_pair(pairs, left, right, score, "test_internal")

    return sorted(
        pairs.values(),
        key=lambda pair: (
            -pair["jaccard"],
            pair["families"][0],
            pair["families"][1],
        ),
    )


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------


def _pdf_missing(record: dict, root: Path | None) -> bool:
    if record.get("pdf_exists") is False:
        return True
    raw_path = record.get("pdf_path")
    if not raw_path:
        return True
    if root is None:
        return False
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    return not path.exists()


def _review_entry(code: str, severity: str, detail: dict) -> dict:
    return {"code": code, "severity": severity, "detail": detail}


def audit_leakage(
    splits: dict,
    corpus: list[dict],
    index_manifest: dict,
    *,
    root: Path | str | None = None,
) -> dict:
    """Audit the held-out split and test index for leakage.

    ``splits`` is the parsed ``splits.draft.json`` (or an equivalent mapping
    with ``dev``/``test``/``reserve``/``excluded`` lists). ``corpus`` is a list
    of prepared-manifest paper records or index chunk records. ``index_manifest``
    is the parsed ``eval/benchmark_v2/index_manifest.json``.

    ``root`` is optional. When supplied, test-paper PDF paths are checked on
    disk relative to it; otherwise the split's ``pdf_exists`` flag is trusted.
    """
    if not isinstance(splits, dict):
        raise LeakageAuditError("splits must be a dict")
    root_path = Path(root) if root is not None else None

    corpus_by_family, chunk_records = _index_corpus(corpus)

    dev = _merged_views(_split_records(splits, "dev"), "dev", corpus_by_family)
    test = _merged_views(_split_records(splits, "test"), "test", corpus_by_family)
    reserve = _merged_views(
        _split_records(splits, "reserve"), "reserve", corpus_by_family
    )
    excluded = _merged_views(
        _split_records(splits, "excluded"), "excluded", corpus_by_family
    )

    dev_families = {view["family"] for view in dev if view["family"]}
    test_families = {view["family"] for view in test if view["family"]}
    reserve_families = {
        view["family"] for view in reserve if view["family"]
    }

    dev_test_family_overlap = sorted(dev_families & test_families)
    test_reserve_family_overlap = sorted(test_families & reserve_families)

    index_ids, index_families, index_entries, index_mismatches = _index_identity(
        index_manifest
    )

    missing_in_index: list[dict] = []
    for view in test:
        paper_id = view["paper_id"]
        family = view["family"]
        if paper_id in index_ids:
            continue
        if family and family in index_families:
            continue
        missing_in_index.append(view)

    missing_index_ids = sorted(
        {view["paper_id"] for view in missing_in_index if view["paper_id"]}
    )

    index_ids_not_in_test = sorted(
        entry["paper_id"]
        for entry in index_entries
        if entry["paper_id"]
        and _base_id(entry["paper_id"]) not in test_families
    )

    foreign_families = sorted(index_families - test_families)
    dev_families_in_index = sorted(index_families & dev_families)
    missing_test_families = sorted(test_families - index_families)

    foreign_chunk_paper_ids: set[str] = set()
    for chunk in chunk_records:
        family = _family_of(chunk)
        if family and family not in test_families:
            foreign_chunk_paper_ids.add(_paper_id_of(chunk))

    chunk_namespace_issue = bool(foreign_chunk_paper_ids)
    namespace_issues = bool(
        foreign_families
        or dev_families_in_index
        or missing_test_families
        or index_mismatches
        or chunk_namespace_issue
        or not index_ids
    )
    test_index_namespace_ok = not namespace_issues

    missing_pdf_views = [
        view for view in test if _pdf_missing(view["record"], root_path)
    ]
    missing_pdf_ids = sorted(
        {view["paper_id"] for view in missing_pdf_views if view["paper_id"]}
    )

    missing_details: dict[str, list[str]] = {}
    for paper_id in missing_pdf_ids:
        missing_details.setdefault(paper_id, []).append("pdf")
    for paper_id in missing_index_ids:
        missing_details.setdefault(paper_id, []).append("index")
    missing_pdf_or_index_ids = sorted(missing_details)

    corpus_families = set(corpus_by_family)
    missing_corpus_ids = sorted(
        {
            view["paper_id"]
            for view in test
            if view["paper_id"]
            and view["family"]
            and view["family"] not in corpus_families
        }
    )

    # Title near duplicates: cross-split pairs are blockers, within-test pairs
    # are blockers, and collisions against the wider paper pool need review.
    pool = [
        view
        for family, record in corpus_by_family.items()
        if family not in test_families and family not in dev_families
        for view in _merged_views([record], "corpus", corpus_by_family)
    ]
    title_pairs = _collect_title_pairs(dev, test, pool)
    abstract_pairs = _collect_abstract_pairs(dev, test)

    excluded_reason_counter = Counter(
        str(view["record"].get("exclusion_reason") or "unspecified")
        for view in excluded
    )

    test_exclusion_leaks = [
        {
            "arxiv_id": view["paper_id"],
            "family": view["family"],
            "exclusion_reason": view["record"].get("exclusion_reason"),
        }
        for view in test
        if view["record"].get("exclusion_reason")
    ]
    test_exclude_from_test = [
        view["paper_id"]
        for view in test
        if view["record"].get("exclude_from_test")
    ]

    review_required: list[dict] = []
    blockers: list[str] = []

    def flag(code: str, severity: str, detail: dict) -> None:
        review_required.append(_review_entry(code, severity, detail))
        if severity == "blocker" and code not in blockers:
            blockers.append(code)

    if dev_test_family_overlap:
        flag(
            "dev_test_family_overlap",
            "blocker",
            {"families": dev_test_family_overlap},
        )
    if test_reserve_family_overlap:
        flag(
            "test_reserve_family_overlap",
            "review",
            {"families": test_reserve_family_overlap},
        )
    if test_exclusion_leaks:
        flag(
            "test_item_excluded",
            "blocker",
            {"items": test_exclusion_leaks},
        )
    if test_exclude_from_test:
        flag(
            "test_item_exclude_from_test",
            "blocker",
            {"paper_ids": sorted(test_exclude_from_test)},
        )
    if title_pairs:
        blocking_pairs = [
            pair for pair in title_pairs if pair["scope"] != "test_pool"
        ]
        pool_pairs = [pair for pair in title_pairs if pair["scope"] == "test_pool"]
        if blocking_pairs:
            flag(
                "near_duplicate_title",
                "blocker",
                {"pairs": blocking_pairs},
            )
        if pool_pairs:
            flag(
                "near_duplicate_title_pool",
                "review",
                {"pairs": pool_pairs},
            )
    if abstract_pairs:
        flag(
            "near_duplicate_abstract",
            "blocker",
            {"pairs": abstract_pairs},
        )
    if missing_pdf_ids:
        flag("missing_pdf", "blocker", {"paper_ids": missing_pdf_ids})
    if missing_index_ids:
        flag(
            "missing_index_id",
            "blocker",
            {"paper_ids": missing_index_ids},
        )
    if namespace_issues and (
        foreign_families
        or dev_families_in_index
        or index_mismatches
        or chunk_namespace_issue
        or not index_ids
    ):
        flag(
            "test_index_namespace",
            "blocker",
            {
                "foreign_families": foreign_families,
                "dev_families": dev_families_in_index,
                "id_family_mismatches": index_mismatches,
                "foreign_chunk_paper_ids": sorted(foreign_chunk_paper_ids),
                "index_paper_count": len(index_ids),
            },
        )
    if chunk_namespace_issue and not (
        foreign_families
        or dev_families_in_index
        or index_mismatches
        or not index_ids
    ):
        flag(
            "test_index_chunk_namespace",
            "blocker",
            {"foreign_chunk_paper_ids": sorted(foreign_chunk_paper_ids)},
        )
    if missing_corpus_ids:
        flag(
            "test_paper_not_in_corpus",
            "review",
            {"paper_ids": missing_corpus_ids},
        )

    index_namespace = {
        "index_paper_count": len(index_ids),
        "foreign_families": foreign_families,
        "dev_families": dev_families_in_index,
        "id_family_mismatches": index_mismatches,
        "foreign_chunk_paper_ids": sorted(foreign_chunk_paper_ids),
        "missing_test_families": missing_test_families,
        "index_ids_not_in_test": index_ids_not_in_test,
    }

    status = "clean"
    if blockers:
        status = "blockers"
    elif review_required:
        status = "review_required"

    summary = {
        "dev_papers": len(dev),
        "test_papers": len(test),
        "reserve_papers": len(reserve),
        "excluded_papers": len(excluded),
        "dev_test_family_overlap": len(dev_test_family_overlap),
        "near_duplicate_title_pairs": len(title_pairs),
        "near_duplicate_abstract_pairs": len(abstract_pairs),
        "missing_pdf_or_index_ids": len(missing_pdf_or_index_ids),
        "test_index_namespace_ok": test_index_namespace_ok,
        "review_items": len(review_required),
        "blockers": len(blockers),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "dev_test_family_overlap": dev_test_family_overlap,
        "test_reserve_family_overlap": test_reserve_family_overlap,
        "near_duplicate_title_pairs": title_pairs,
        "near_duplicate_abstract_pairs": abstract_pairs,
        "title_collisions": [
            pair for pair in title_pairs if pair["jaccard"] >= 1.0
        ],
        "near_duplicate_pairs": title_pairs + abstract_pairs,
        "excluded_reasons": dict(sorted(excluded_reason_counter.items())),
        "excluded_total": len(excluded),
        "test_exclusion_leaks": test_exclusion_leaks,
        "test_index_namespace_ok": test_index_namespace_ok,
        "index_namespace": index_namespace,
        "missing_pdf_or_index_ids": missing_pdf_or_index_ids,
        "missing_details": missing_details,
        "missing_index_ids": missing_index_ids,
        "missing_pdf_ids": missing_pdf_ids,
        "missing_corpus_ids": missing_corpus_ids,
        "review_required": review_required,
        "blockers": blockers,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise LeakageAuditError(f"cannot read {path}: {error}") from error
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LeakageAuditError(f"invalid JSON in {path}: {error}") from error


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_manifests(
    paths: Sequence[Path], root: Path
) -> tuple[list[dict], list[dict]]:
    corpus: list[dict] = []
    metadata: list[dict] = []
    for path in paths:
        payload = _read_json(path)
        if isinstance(payload, dict):
            papers = payload.get("papers")
        elif isinstance(payload, list):
            papers = payload
        else:
            raise LeakageAuditError(f"unrecognised manifest shape: {path}")
        if papers is None:
            raise LeakageAuditError(f"manifest has no 'papers' list: {path}")
        if not isinstance(papers, list):
            raise LeakageAuditError(f"manifest 'papers' is not a list: {path}")
        records = [item for item in papers if isinstance(item, dict)]
        corpus.extend(records)
        metadata.append(
            {
                "path": _display_path(path, root),
                "paper_count": len(records),
                "sha256": _sha256_bytes(path.read_bytes()),
            }
        )
    return corpus, metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the PaperLens held-out split and test index for leakage."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="PaperLens repository root (default: %(default)s)",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=None,
        help="splits JSON (default: <root>/eval/benchmark_v2/splits.draft.json)",
    )
    parser.add_argument(
        "--index-manifest",
        type=Path,
        default=None,
        help=(
            "test index manifest "
            "(default: <root>/eval/benchmark_v2/index_manifest.json)"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=None,
        help=(
            "prepared_v1 manifest; repeatable. Defaults to the dataset_v1 and "
            "dataset_history_v1 prepared manifests."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="report path (default: <root>/eval/benchmark_v2/leakage_report.json)",
    )
    args = parser.parse_args(argv)

    root = args.root
    splits_path = args.splits or root / "eval/benchmark_v2/splits.draft.json"
    index_path = (
        args.index_manifest
        or root / "eval/benchmark_v2/index_manifest.json"
    )
    out_path = args.out or root / "eval/benchmark_v2/leakage_report.json"
    manifest_paths = list(args.manifest) if args.manifest else [
        root / "data" / "dataset_v1" / "prepared_v1" / "manifest.json",
        root
        / "data"
        / "dataset_history_v1"
        / "prepared_v1"
        / "manifest.json",
    ]

    try:
        splits_payload = _read_json(splits_path)
        index_payload = _read_json(index_path)
        corpus, manifest_meta = _load_manifests(manifest_paths, root)
        report = audit_leakage(
            splits_payload,
            corpus,
            index_payload,
            root=root,
        )
    except LeakageAuditError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["generated_by"] = "eval/benchmark_v2/leakage_audit.py"
    report["inputs"] = {
        "splits": {
            "path": _display_path(splits_path, root),
            "sha256": _sha256_bytes(splits_path.read_bytes()),
        },
        "index_manifest": {
            "path": _display_path(index_path, root),
            "sha256": _sha256_bytes(index_path.read_bytes()),
        },
        "manifests": manifest_meta,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": report["status"],
                "summary": report["summary"],
                "blockers": report["blockers"],
                "report": str(out_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
