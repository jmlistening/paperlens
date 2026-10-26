"""Aggregate one experiment run's deterministic Eval-lite metrics (Task E5).

``score_run`` reads only local fixtures: a run directory (or a single JSONL /
JSON file) and a gold file (JSONL / JSON). It never calls a model, a network
service or ``data/``.

Contract
--------
Run records may be keyed as ``question_id`` / ``id`` / ``qid``. A directory is
read from ``answers.jsonl`` / ``results.jsonl`` / ``answers.json`` /
``results.json`` when one exists; otherwise every ``*.json`` / ``*.jsonl`` file
except aggregate files (``run.json``, ``manifest.json``, ``metrics.json``,
``summary.json``) is read, and a filename stem supplies a missing question ID.

Recognized run fields:

* ``status``: ``ok`` / ``success`` / ``answered`` / ``partial`` /
  ``insufficient`` (schema-valid unless ``schema_ok`` is False), or a failure
  status such as ``failed`` / ``error`` / ``timeout`` / ``schema_invalid``.
* ``schema_ok``: explicit boolean; a failure status always wins.
* ``retrieved_parent_ids`` (or ``retrieved_parents`` / ``retrieved`` /
  ``retrieved_ids``): ordered best-first list of parent-chunk IDs. Entries may
  be strings or objects with ``parent_chunk_id`` / ``parent_id`` /
  ``chunk_id`` / ``id``.
* ``citations`` and/or ``answer.points[].citations``: citation objects with
  ``quote`` plus ``source_text`` / ``evidence_text`` / ``text``. A citation
  with only ``id`` is resolved against the record's ``evidence`` /
  ``evidence_texts`` / ``sources`` map.
* ``rel`` / ``relevance_grades`` / ``grades``: optional ID -> grade map for
  nDCG; otherwise grades are derived from ``gold_points[].relevance``.

Gold records may be keyed the same way and use ``gold_parent_ids`` /
``gold_parent_chunk_ids`` / ``gold_chunk_ids`` / ``gold_chunks`` /
``gold_parents`` (strings or objects), or ``gold_points`` with
``parent_chunk_id`` / ``parent_id``.

Failure policy: a failed or schema-invalid run is scored 0 on every metric,
and a gold question with no run record is treated as a failed run
(``status="missing_run"``) rather than silently dropped. Unanswerable questions
(no gold parent chunks) are excluded from the retrieval denominators and
reported separately; they still count toward schema success. Retrieval means
are taken over answerable questions only, with failures included as zeros.
Quote integrity is the micro-average over every emitted citation (failures
included), so a run with no citations reports 0.0, not 1.0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics_v2.metrics import (  # noqa: E402
    all_gold_at_k,
    mrr_at_k,
    ndcg_at_k,
    quote_integrity,
    recall_at_k,
    run_schema_ok,
)

METRICS_VERSION = "metrics_v2_e5"
RETRIEVAL_METRIC_KEYS = (
    "recall@3",
    "recall@6",
    "recall@10",
    "all_gold@6",
    "mrr@10",
    "ndcg@10",
)
AGGREGATE_FILENAMES = frozenset(
    {"run.json", "manifest.json", "metrics.json", "summary.json"}
)
PREFERRED_RUN_FILES = (
    "answers.jsonl",
    "results.jsonl",
    "answers.json",
    "results.json",
)

QUESTION_ID_KEYS = ("question_id", "qid", "id")
RETRIEVED_KEYS = ("retrieved_parent_ids", "retrieved_parents", "retrieved", "retrieved_ids")
GOLD_KEYS = (
    "gold_parent_ids",
    "gold_parent_chunk_ids",
    "gold_chunk_ids",
    "gold_chunks",
    "gold_parents",
)
RETRIEVED_ITEM_KEYS = ("parent_chunk_id", "parent_id", "chunk_id", "id")
GOLD_ITEM_KEYS = ("parent_chunk_id", "parent_id", "chunk_id")
EVIDENCE_KEYS = ("evidence", "evidence_texts", "sources")
RELEVANCE_KEYS = ("rel", "relevance_grades", "grades")


class ScoreRunError(ValueError):
    """Raised when a fixture cannot be understood as run/gold records."""


def _question_id(record: Mapping, fallback: str | None = None) -> str | None:
    for key in QUESTION_ID_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _ids_from_value(value: object, item_keys: tuple[str, ...]) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    ids: list[str] = []
    for item in value:
        if isinstance(item, str):
            if item.strip():
                ids.append(item)
            continue
        if isinstance(item, dict):
            for key in item_keys:
                candidate = item.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    ids.append(candidate)
                    break
    return _dedupe(ids)


def _gold_ids(record: Mapping) -> list[str]:
    for key in GOLD_KEYS:
        if key in record and record[key] is not None:
            ids = _ids_from_value(record[key], GOLD_ITEM_KEYS)
            if ids or record[key] == []:
                return ids
    ids: list[str] = []
    points = record.get("gold_points")
    if isinstance(points, list):
        for point in points:
            if not isinstance(point, dict):
                continue
            for key in GOLD_ITEM_KEYS:
                candidate = point.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    ids.append(candidate)
                    break
    return _dedupe(ids)


def _retrieved_ids(record: Mapping) -> list[str]:
    for key in RETRIEVED_KEYS:
        if key in record and record[key] is not None:
            return _ids_from_value(record[key], RETRIEVED_ITEM_KEYS)
    return []


def _relevance_map(record: Mapping) -> dict[str, int]:
    for key in RELEVANCE_KEYS:
        value = record.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    rel: dict[str, int] = {}
    points = record.get("gold_points")
    if isinstance(points, list):
        for point in points:
            if not isinstance(point, dict):
                continue
            grade = point.get("relevance", point.get("rel"))
            if isinstance(grade, int) and not isinstance(grade, bool):
                for key in GOLD_ITEM_KEYS:
                    candidate = point.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        rel[candidate] = grade
                        break
    return rel


def _citations(record: Mapping) -> list[dict]:
    found: list[dict] = []
    top = record.get("citations")
    if isinstance(top, list):
        found.extend(item for item in top if isinstance(item, dict))
    answer = record.get("answer")
    if isinstance(answer, dict):
        nested = answer.get("citations")
        if isinstance(nested, list):
            found.extend(item for item in nested if isinstance(item, dict))
        points = answer.get("points")
        if isinstance(points, list):
            for point in points:
                if isinstance(point, dict) and isinstance(point.get("citations"), list):
                    found.extend(
                        item for item in point["citations"] if isinstance(item, dict)
                    )
    return found


def _evidence_pool(record: Mapping) -> dict[str, str]:
    pool: dict[str, str] = {}
    for key in EVIDENCE_KEYS:
        value = record.get(key)
        if isinstance(value, Mapping):
            for evidence_id, item in value.items():
                text = _evidence_text(item)
                if text is not None and isinstance(evidence_id, str):
                    pool[evidence_id] = text
        elif isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                evidence_id = item.get("id", item.get("evidence_id"))
                text = _evidence_text(item)
                if isinstance(evidence_id, str) and text is not None:
                    pool[evidence_id] = text
    return pool


def _evidence_text(item: object) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "source_text", "evidence_text"):
            value = item.get(key)
            if isinstance(value, str):
                return value
    return None


def _citations_with_sources(record: Mapping) -> list[dict]:
    pool = _evidence_pool(record)
    resolved: list[dict] = []
    for citation in _citations(record):
        entry = dict(citation)
        source = None
        for key in ("source_text", "evidence_text", "text"):
            value = citation.get(key)
            if isinstance(value, str) and value.strip():
                source = value
                break
        if source is None:
            evidence_id = citation.get("id")
            if isinstance(evidence_id, str):
                source = pool.get(evidence_id)
        if source is not None:
            entry["source_text"] = source
        resolved.append(entry)
    return resolved


def _records_from_payload(payload: object, source: Path) -> list[dict]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        for key in ("questions", "cases", "answers", "results", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        else:
            records = [payload]
    else:
        raise ScoreRunError(f"{source}: expected a JSON object or list")
    cleaned = [item for item in records if isinstance(item, dict)]
    if len(cleaned) != len(records):
        raise ScoreRunError(f"{source}: every record must be a JSON object")
    return cleaned


def _load_file_records(path: Path, fallback_id: str | None = None) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records: list[dict] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScoreRunError(
                    f"{path}:{line_number}: invalid JSON ({exc.msg})"
                ) from exc
            if not isinstance(row, dict):
                raise ScoreRunError(f"{path}:{line_number}: expected a JSON object")
            records.append(row)
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ScoreRunError(f"{path}: invalid JSON ({exc.msg})") from exc
        records = _records_from_payload(payload, path)

    if fallback_id and len(records) == 1:
        record = records[0]
        if _question_id(record) is None:
            record = dict(record)
            record["question_id"] = fallback_id
            records = [record]
    return records


def load_gold_records(gold_path: Path) -> list[dict]:
    """Load gold records from a JSONL/JSON file."""
    path = Path(gold_path)
    if not path.is_file():
        raise FileNotFoundError(f"gold file not found: {path}")
    return _load_file_records(path)


def load_run_records(run_dir: Path) -> list[dict]:
    """Load per-question run records from a directory or a single file."""
    path = Path(run_dir)
    if path.is_file():
        return _load_file_records(path)
    if not path.is_dir():
        raise FileNotFoundError(f"run directory not found: {path}")

    for name in PREFERRED_RUN_FILES:
        candidate = path / name
        if candidate.is_file():
            return _load_file_records(candidate)

    records: list[dict] = []
    for file in sorted(path.iterdir()):
        if not file.is_file() or file.suffix.lower() not in {".json", ".jsonl"}:
            continue
        if file.name in AGGREGATE_FILENAMES:
            continue
        records.extend(_load_file_records(file, fallback_id=file.stem))
    return records


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _retrieval_metrics(
    gold_ids: set[str],
    retrieved: list[str],
    rel: Mapping[str, int],
) -> dict[str, float]:
    return {
        "recall@3": recall_at_k(gold_ids, retrieved, 3),
        "recall@6": recall_at_k(gold_ids, retrieved, 6),
        "recall@10": recall_at_k(gold_ids, retrieved, 10),
        "all_gold@6": all_gold_at_k(gold_ids, retrieved, 6),
        "mrr@10": mrr_at_k(gold_ids, retrieved, 10),
        "ndcg@10": ndcg_at_k(gold_ids, retrieved, 10, rel=dict(rel)),
    }


def score_run(run_dir: Path, gold_path: Path) -> dict:
    """Score one run against gold and return JSON-serializable metrics."""
    gold_records = load_gold_records(Path(gold_path))
    run_records = load_run_records(Path(run_dir))

    gold_by_id: dict[str, dict] = {}
    for record in gold_records:
        question_id = _question_id(record)
        if question_id is None:
            raise ScoreRunError(f"gold record without a question id: {record}")
        gold_by_id[question_id] = record

    run_by_id: dict[str, dict] = {}
    for record in run_records:
        question_id = _question_id(record)
        if question_id is None:
            raise ScoreRunError(f"run record without a question id: {record}")
        run_by_id[question_id] = record

    missing_run_ids = [qid for qid in gold_by_id if qid not in run_by_id]
    extra_run_ids = [qid for qid in run_by_id if qid not in gold_by_id]

    per_question: list[dict] = []
    retrieval_columns: dict[str, list[float]] = {
        key: [] for key in RETRIEVAL_METRIC_KEYS
    }
    all_citations: list[dict] = []
    schema_runs: list[dict] = []
    n_retrieval = 0
    n_failed = 0

    for question_id, gold_record in gold_by_id.items():
        run_record = run_by_id.get(question_id)
        gold_ids = _gold_ids(gold_record)
        gold_set = set(gold_ids)
        retrieved = _retrieved_ids(run_record) if run_record is not None else []
        citations = (
            _citations_with_sources(run_record) if run_record is not None else []
        )
        rel = _relevance_map(gold_record)
        schema_ok = run_schema_ok(run_record) if run_record is not None else False
        status = (
            run_record.get("status", "ok") if run_record is not None else "missing_run"
        )
        if not schema_ok:
            n_failed += 1
        schema_runs.append(run_record if run_record is not None else {"status": "missing_run"})
        all_citations.extend(citations)

        scored = bool(gold_set)
        if scored:
            n_retrieval += 1
            question_metrics = (
                _retrieval_metrics(gold_set, retrieved, rel)
                if schema_ok
                else {key: 0.0 for key in RETRIEVAL_METRIC_KEYS}
            )
            for key, value in question_metrics.items():
                retrieval_columns[key].append(value)
        else:
            question_metrics = None

        per_question.append(
            {
                "question_id": question_id,
                "status": status,
                "schema_ok": schema_ok,
                "failed": not schema_ok,
                "retrieval_scored": scored,
                "gold_count": len(gold_ids),
                "retrieved_count": len(retrieved),
                "metrics": question_metrics,
                "n_citations": len(citations),
                "quote_integrity": (
                    quote_integrity(citations) if citations else None
                ),
            }
        )

    metrics: dict[str, float] = {
        key: _mean(values) for key, values in retrieval_columns.items()
    }
    metrics["quote_integrity"] = quote_integrity(all_citations)
    metrics["schema_success"] = (
        sum(1 for run in schema_runs if run_schema_ok(run)) / len(schema_runs)
        if schema_runs
        else 0.0
    )

    return {
        "version": METRICS_VERSION,
        "n": len(gold_by_id),
        "n_retrieval": n_retrieval,
        "n_unanswerable": len(gold_by_id) - n_retrieval,
        "n_failed": n_failed,
        "n_schema_ok": len(schema_runs) - n_failed,
        "n_citations": len(all_citations),
        "missing_run_ids": missing_run_ids,
        "extra_run_ids": extra_run_ids,
        "metrics": metrics,
        "per_question": per_question,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score one PaperLens Eval-lite run with deterministic metrics."
    )
    parser.add_argument("run_dir", type=Path, help="run directory or answers file")
    parser.add_argument("gold_path", type=Path, help="gold JSON/JSONL file")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional path to write the JSON result (default: stdout)",
    )
    args = parser.parse_args(argv)

    result = score_run(args.run_dir, args.gold_path)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
