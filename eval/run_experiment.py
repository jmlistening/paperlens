"""Experiment runner for retrieval-first PaperLens Eval-lite (Task E6).

The runner is deliberately dependency-injected. :class:`ExperimentDeps` only
requires ``run_system(system, question) -> dict``; tests pass a fake and the
command-line entry point builds :class:`RetrievalDeps`, which performs
deterministic BM25/dense/RRF retrieval against ``eval/benchmark_v2/index`` and
never calls an LLM.

Contracts
---------
* One JSON checkpoint per ``(system, question)`` is written to
  ``out_dir/<system>/<question_id>.json`` as soon as the question finishes.
  A re-run skips every checkpoint whose status is ``ok``/``failed``/``invalid``
  and whose ``question_id`` matches; corrupt or partial files are re-run.
  ``retry_failed=True`` re-runs failed/invalid checkpoints and keeps the
  attempt counter.
* A raised exception, a malformed system output or an explicit failure status
  becomes a scored 0 for that question. Failures stay in the denominator and
  are never silently dropped.
* ``budget`` caps recorded tokens across the run. The runner checks the budget
  before starting the next question; a question already in flight is allowed
  to finish, so the final total can exceed the cap by at most one question.
  When the budget stops the run, ``run.json`` records the stop and the
  remaining question IDs. Re-running resumes from the checkpoints.
* ``limit`` is a debugging cap on how many benchmark questions each system
  sees. A limit shorter than the benchmark finishes with status ``incomplete``
  and stops cleanly; re-running without the limit continues.
* Systems declared ``implemented: false`` in the matrix are skipped with an
  explicit ``not_implemented`` reason. They produce no numbers.

The module itself never reads ``data/``. Only :class:`RetrievalDeps` reads the
benchmark index directory, and only when the CLI actually runs a system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics_v2.metrics import (  # noqa: E402
    all_gold_at_k,
    mrr_at_k,
    ndcg_at_k,
    quote_integrity,
    recall_at_k,
)

EXPERIMENT_VERSION = "eval_experiment_e6"

SUCCESS_STATUSES = frozenset({"ok", "success", "answered", "partial", "insufficient"})
FAILURE_STATUSES = frozenset(
    {"failed", "error", "timeout", "schema_invalid", "invalid", "missing_run", "not_implemented"}
)
TERMINAL_STATUSES = frozenset({"ok", "failed", "invalid"})

RETRIEVAL_METRIC_KEYS = (
    "recall@3",
    "recall@6",
    "recall@10",
    "all_gold@6",
    "mrr@10",
    "ndcg@10",
)
METRIC_KEYS = RETRIEVAL_METRIC_KEYS + ("quote_integrity", "schema_success")

QUESTION_FILES = ("questions.jsonl", "questions.json")
QREL_FILES = ("qrels.jsonl", "qrels.json")
QUESTION_ROOT_KEYS = ("questions", "cases", "records")
QREL_ROOT_KEYS = ("qrels", "gold", "gold_records", "records")

QUESTION_ID_KEYS = ("question_id", "qid", "id")
GOLD_KEYS = (
    "gold_parent_ids",
    "gold_parent_chunk_ids",
    "gold_chunk_ids",
    "gold_chunks",
    "gold_parents",
)
GOLD_ITEM_KEYS = ("parent_chunk_id", "parent_id", "chunk_id", "id")
RELEVANCE_KEYS = ("rel", "relevance_grades", "grades")

# Keys that must never leak from the benchmark into a system call.
LEAKY_KEYS = GOLD_KEYS + ("gold_points", "rel", "relevance_grades", "grades")

MAX_ERROR_CHARS = 500


class ExperimentDeps:
    """Dependency-injection seam for the runner.

    Tests implement ``run_system`` directly. Real runs use
    :class:`RetrievalDeps` or a future answer-system adapter.
    """

    def run_system(self, system: str, question: dict) -> dict:
        raise NotImplementedError(
            "ExperimentDeps.run_system must be provided by the caller"
        )


# ---------------------------------------------------------------------------
# Loading benchmark inputs
# ---------------------------------------------------------------------------


def _first_existing(directory: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _question_id(record: Mapping) -> str | None:
    for key in QUESTION_ID_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _safe_component(question_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", question_id).strip("._")
    if not cleaned:
        cleaned = "q"
    if len(cleaned) > 100:
        digest = hashlib.sha256(question_id.encode("utf-8")).hexdigest()[:8]
        cleaned = f"{cleaned[:80]}_{digest}"
    return cleaned


def _load_records(path: Path, root_keys: Sequence[str]) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows: list[dict] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc.msg})") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
        return rows

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON ({exc.msg})") from exc
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        for key in root_keys:
            value = payload.get(key)
            if isinstance(value, list):
                rows = value
                break
        else:
            if _question_id(payload) is not None:
                rows = [payload]
            else:
                raise ValueError(f"{path}: expected a JSON list or one of {root_keys}")
    else:
        raise ValueError(f"{path}: expected a JSON list or object")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: every record must be a JSON object")
    return list(rows)


def load_benchmark(benchmark_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    """Load ``questions.jsonl``/``.json`` and ``qrels.jsonl``/``.json``.

    Question IDs must be unique and must map to distinct checkpoint filenames.
    Qrels records without a matching question are retained in the returned map
    so callers can report them, but they never create work.
    """
    directory = Path(benchmark_dir)
    questions_path = _first_existing(directory, QUESTION_FILES)
    qrels_path = _first_existing(directory, QREL_FILES)
    if questions_path is None:
        raise FileNotFoundError(
            f"{directory}: expected one of {', '.join(QUESTION_FILES)}"
        )
    if qrels_path is None:
        raise FileNotFoundError(f"{directory}: expected one of {', '.join(QREL_FILES)}")

    questions = _load_records(questions_path, QUESTION_ROOT_KEYS)
    if not questions:
        raise ValueError(f"{questions_path}: benchmark has no questions")

    by_id: dict[str, dict] = {}
    used_names: dict[str, str] = {}
    for question in questions:
        question_id = _question_id(question)
        if question_id is None:
            raise ValueError(f"{questions_path}: question without an id: {question}")
        if question_id in by_id:
            raise ValueError(f"{questions_path}: duplicate question id {question_id!r}")
        safe_name = _safe_component(question_id)
        if safe_name in used_names:
            raise ValueError(
                f"{questions_path}: question ids {used_names[safe_name]!r} and "
                f"{question_id!r} map to the same checkpoint filename {safe_name!r}"
            )
        used_names[safe_name] = question_id
        by_id[question_id] = question

    qrels_by_id: dict[str, dict] = {}
    for record in _load_records(qrels_path, QREL_ROOT_KEYS):
        question_id = _question_id(record)
        if question_id is None:
            raise ValueError(f"{qrels_path}: qrels record without an id: {record}")
        if question_id in qrels_by_id:
            raise ValueError(f"{qrels_path}: duplicate qrels id {question_id!r}")
        qrels_by_id[question_id] = record

    return questions, qrels_by_id


def _ids_from_value(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    ids: list[str] = []
    for item in value:
        if isinstance(item, str):
            if item.strip():
                ids.append(item.strip())
            continue
        if isinstance(item, dict):
            for key in GOLD_ITEM_KEYS:
                candidate = item.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    ids.append(candidate.strip())
                    break
    ordered: list[str] = []
    seen: set[str] = set()
    for item in ids:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _gold_ids(record: Mapping | None) -> list[str]:
    if not isinstance(record, Mapping):
        return []
    for key in GOLD_KEYS:
        if key in record and record[key] is not None:
            ids = _ids_from_value(record[key])
            if ids or record[key] == []:
                return ids
    ids = []
    points = record.get("gold_points")
    if isinstance(points, list):
        for point in points:
            if not isinstance(point, dict):
                continue
            for key in GOLD_ITEM_KEYS:
                candidate = point.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    ids.append(candidate.strip())
                    break
    return _ids_from_value(ids)


def _relevance_map(record: Mapping | None) -> dict[str, int]:
    if not isinstance(record, Mapping):
        return {}
    for key in RELEVANCE_KEYS:
        value = record.get(key)
        if isinstance(value, Mapping):
            return {
                str(item): grade
                for item, grade in value.items()
                if isinstance(grade, int) and not isinstance(grade, bool) and grade >= 0
            }
    grades: dict[str, int] = {}
    points = record.get("gold_points")
    if isinstance(points, list):
        for point in points:
            if not isinstance(point, dict):
                continue
            grade = point.get("relevance", point.get("rel"))
            if not isinstance(grade, int) or isinstance(grade, bool) or grade < 0:
                continue
            for key in GOLD_ITEM_KEYS:
                candidate = point.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    grades[candidate.strip()] = grade
                    break
    return grades


def _question_view(question: Mapping) -> dict:
    """Copy a question for the system, stripping every gold field."""
    return {key: value for key, value in question.items() if key not in LEAKY_KEYS}


# ---------------------------------------------------------------------------
# Matrix and budget handling
# ---------------------------------------------------------------------------


def load_matrix(path: Path) -> dict:
    """Load a YAML or JSON experiment matrix."""
    matrix_path = Path(path)
    text = matrix_path.read_text(encoding="utf-8")
    if matrix_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - PyYAML is pinned
            raise RuntimeError("PyYAML is required to read matrix.yaml") from exc
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{matrix_path}: matrix must be a JSON/YAML object")
    return payload


def _normalize_systems(matrix: Mapping) -> dict[str, dict]:
    if not isinstance(matrix, Mapping):
        raise TypeError("matrix must be a mapping")
    systems = matrix.get("systems")
    if not isinstance(systems, Mapping) or not systems:
        raise ValueError("matrix.systems must be a non-empty mapping")
    normalized: dict[str, dict] = {}
    for name, raw_config in systems.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("matrix system names must be non-empty strings")
        if raw_config is None:
            config: dict = {}
        elif isinstance(raw_config, Mapping):
            config = dict(raw_config)
        else:
            raise ValueError(f"matrix.systems.{name} must be a mapping")
        kind = config.get("kind", "retrieval")
        if kind not in {"retrieval", "answer"}:
            raise ValueError(f"matrix.systems.{name}.kind must be retrieval or answer")
        implemented = config.get("implemented", True)
        if not isinstance(implemented, bool):
            raise ValueError(f"matrix.systems.{name}.implemented must be a boolean")
        if "top_k" in config:
            top_k = config["top_k"]
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
                raise ValueError(f"matrix.systems.{name}.top_k must be an integer >= 0")
        config["kind"] = kind
        config["implemented"] = implemented
        normalized[name] = config
    return normalized


def _matrix_budget(matrix: Mapping) -> int | None:
    budget = matrix.get("budget")
    if budget is None:
        return None
    if isinstance(budget, bool):
        raise ValueError("matrix.budget must be an integer or mapping")
    if isinstance(budget, int):
        if budget < 0:
            raise ValueError("matrix.budget must be >= 0")
        return budget
    if isinstance(budget, Mapping):
        value = budget.get("max_tokens")
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("matrix.budget.max_tokens must be an integer >= 0")
        return value
    raise ValueError("matrix.budget must be an integer or mapping")


def _selected_systems(
    systems: Mapping[str, dict], requested: Sequence[str] | str | None
) -> list[str]:
    if requested is None:
        return list(systems)
    if isinstance(requested, str):
        requested = [item.strip() for item in requested.split(",") if item.strip()]
    requested_names = list(dict.fromkeys(str(item) for item in requested))
    unknown = [name for name in requested_names if name not in systems]
    if unknown:
        raise ValueError(
            f"unknown system(s) {unknown}; matrix declares {list(systems)}"
        )
    requested_set = set(requested_names)
    return [name for name in systems if name in requested_set]


# ---------------------------------------------------------------------------
# Checkpoints and scoring
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_error(error: object) -> str:
    text = " ".join(str(error).split())
    return text[:MAX_ERROR_CHARS]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_checkpoint(path: Path, question_id: str) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("question_id") != question_id:
        return None
    if payload.get("status") not in TERMINAL_STATUSES:
        return None
    if not isinstance(payload.get("metrics"), dict):
        return None
    return payload


def _extract_tokens(raw: object) -> int:
    if not isinstance(raw, Mapping):
        return 0

    def clean(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return max(0, value)
        if isinstance(value, float) and math.isfinite(value):
            return max(0, int(value))
        return None

    for key in ("tokens", "total_tokens"):
        value = clean(raw.get(key))
        if value is not None:
            return value
    tokens = raw.get("tokens")
    if isinstance(tokens, Mapping):
        for key in ("total", "total_tokens", "tokens"):
            value = clean(tokens.get(key))
            if value is not None:
                return value
    usage = raw.get("usage")
    if isinstance(usage, Mapping):
        for key in ("total_tokens", "tokens"):
            value = clean(usage.get(key))
            if value is not None:
                return value
        prompt = clean(usage.get("prompt_tokens"))
        completion = clean(usage.get("completion_tokens"))
        if prompt is not None and completion is not None:
            return prompt + completion
    return 0


def _normalize_ids(value: object) -> list[str] | None:
    if not isinstance(value, (list, tuple)):
        return None
    ids: list[str] = []
    seen: set[str] = set()
    for item in value:
        candidate: str | None = None
        if isinstance(item, str):
            if item.strip():
                candidate = item.strip()
        elif isinstance(item, dict):
            for key in ("parent_chunk_id", "parent_id", "chunk_id", "id"):
                entry = item.get(key)
                if isinstance(entry, str) and entry.strip():
                    candidate = entry.strip()
                    break
        if candidate is None:
            return None
        if candidate not in seen:
            seen.add(candidate)
            ids.append(candidate)
    return ids


def _retrieval_scores(
    gold: set[str], retrieved: list[str], rel: Mapping[str, int]
) -> dict[str, float]:
    return {
        "recall@3": recall_at_k(gold, retrieved, 3),
        "recall@6": recall_at_k(gold, retrieved, 6),
        "recall@10": recall_at_k(gold, retrieved, 10),
        "all_gold@6": all_gold_at_k(gold, retrieved, 6),
        "mrr@10": mrr_at_k(gold, retrieved, 10),
        "ndcg@10": ndcg_at_k(gold, retrieved, 10, rel=dict(rel)),
    }


def _metrics_for(
    *,
    ok: bool,
    gold: set[str],
    retrieved: list[str],
    rel: Mapping[str, int],
    citations: list[dict],
    citations_present: bool,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    if not gold:
        for key in RETRIEVAL_METRIC_KEYS:
            metrics[key] = None
    elif not ok:
        for key in RETRIEVAL_METRIC_KEYS:
            metrics[key] = 0.0
    else:
        metrics.update(_retrieval_scores(gold, retrieved, rel))
    if not citations_present:
        metrics["quote_integrity"] = None
    else:
        metrics["quote_integrity"] = quote_integrity(citations) if ok else 0.0
    metrics["schema_success"] = 1.0 if ok else 0.0
    return metrics


def _make_record(
    *,
    system: str,
    question_id: str,
    status: str,
    gold: set[str],
    retrieved: list[str],
    rel: Mapping[str, int],
    citations: list[dict],
    citations_present: bool,
    tokens: int,
    attempt: int,
    raw_status: object = None,
    error: str | None = None,
) -> dict:
    ok = status == "ok"
    return {
        "question_id": question_id,
        "system": system,
        "status": status,
        "ok": ok,
        "schema_ok": ok,
        "raw_status": raw_status,
        "error": error,
        "retrieved_parent_ids": retrieved,
        "citations": citations,
        "tokens": tokens,
        "retrieval_scored": bool(gold),
        "metrics": _metrics_for(
            ok=ok,
            gold=gold,
            retrieved=retrieved,
            rel=rel,
            citations=citations,
            citations_present=citations_present,
        ),
        "attempt": attempt,
        "finished_at": _now(),
    }


def _normalize_result(
    raw: object,
    *,
    system: str,
    question_id: str,
    kind: str,
    gold: set[str],
    rel: Mapping[str, int],
    attempt: int,
) -> dict:
    if not isinstance(raw, Mapping):
        return _make_record(
            system=system,
            question_id=question_id,
            status="invalid",
            gold=gold,
            retrieved=[],
            rel=rel,
            citations=[],
            citations_present=False,
            tokens=0,
            attempt=attempt,
            error="system output must be a JSON object",
        )

    raw_status = raw.get("status", "ok")
    tokens = _extract_tokens(raw)
    if not isinstance(raw_status, str):
        return _make_record(
            system=system,
            question_id=question_id,
            status="invalid",
            gold=gold,
            retrieved=[],
            rel=rel,
            citations=[],
            citations_present=False,
            tokens=tokens,
            attempt=attempt,
            error="system status must be a string",
        )

    if raw_status in FAILURE_STATUSES or raw.get("error"):
        error = raw.get("error") or f"system reported status {raw_status}"
        return _make_record(
            system=system,
            question_id=question_id,
            status="failed",
            gold=gold,
            retrieved=[],
            rel=rel,
            citations=[],
            citations_present=False,
            tokens=tokens,
            attempt=attempt,
            raw_status=raw_status,
            error=_clean_error(error),
        )

    if raw_status not in SUCCESS_STATUSES:
        return _make_record(
            system=system,
            question_id=question_id,
            status="invalid",
            gold=gold,
            retrieved=[],
            rel=rel,
            citations=[],
            citations_present=False,
            tokens=tokens,
            attempt=attempt,
            raw_status=raw_status,
            error=f"unknown system status {raw_status!r}",
        )

    retrieved_raw = raw.get("retrieved_parent_ids")
    retrieved: list[str] = []
    if kind == "retrieval" or retrieved_raw is not None:
        cleaned = _normalize_ids(retrieved_raw)
        if cleaned is None:
            return _make_record(
                system=system,
                question_id=question_id,
                status="invalid",
                gold=gold,
                retrieved=[],
                rel=rel,
                citations=[],
                citations_present=False,
                tokens=tokens,
                attempt=attempt,
                raw_status=raw_status,
                error=(
                    "missing retrieved_parent_ids"
                    if retrieved_raw is None
                    else "retrieved_parent_ids must be a list of non-empty ids"
                ),
            )
        retrieved = cleaned

    citations_raw = raw.get("citations")
    citations_present = isinstance(citations_raw, list)
    citations = (
        [item for item in citations_raw if isinstance(item, dict)]
        if citations_present
        else []
    )

    return _make_record(
        system=system,
        question_id=question_id,
        status="ok",
        gold=gold,
        retrieved=retrieved,
        rel=rel,
        citations=citations,
        citations_present=citations_present,
        tokens=tokens,
        attempt=attempt,
        raw_status=raw_status,
    )


def _run_one(
    deps: ExperimentDeps,
    *,
    system: str,
    question: Mapping,
    qrels_record: Mapping | None,
    kind: str,
    previous: dict | None,
) -> dict:
    question_id = _question_id(question)
    if question_id is None:  # load_benchmark already guarantees this
        raise ValueError("question without an id")
    gold = set(_gold_ids(qrels_record))
    rel = _relevance_map(qrels_record)
    attempt = int(previous.get("attempt", 0) or 0) + 1 if previous else 1
    try:
        raw = deps.run_system(system, _question_view(question))
    except Exception as exc:  # noqa: BLE001 - a system crash is a failed question
        return _make_record(
            system=system,
            question_id=question_id,
            status="failed",
            gold=gold,
            retrieved=[],
            rel=rel,
            citations=[],
            citations_present=False,
            tokens=0,
            attempt=attempt,
            error=_clean_error(f"{type(exc).__name__}: {exc}"),
        )
    return _normalize_result(
        raw,
        system=system,
        question_id=question_id,
        kind=kind,
        gold=gold,
        rel=rel,
        attempt=attempt,
    )


# ---------------------------------------------------------------------------
# Summaries and the runner
# ---------------------------------------------------------------------------


def _summarize(
    system: str,
    config: Mapping,
    records: list[dict],
    *,
    status: str,
    skipped_reason: str | None = None,
) -> dict:
    metric_samples: dict[str, list[float]] = {}
    for key in METRIC_KEYS:
        samples: list[float] = []
        for record in records:
            value = (record.get("metrics") or {}).get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                samples.append(float(value))
        if samples:
            metric_samples[key] = samples

    metrics = {
        key: sum(values) / len(values) for key, values in metric_samples.items()
    }
    failures = [
        record.get("question_id")
        for record in records
        if record.get("status") != "ok"
    ]
    return {
        "system": system,
        "kind": config.get("kind", "retrieval"),
        "status": status,
        "n": len(records),
        "n_failed": len(failures),
        "n_retrieval": sum(1 for record in records if record.get("retrieval_scored")),
        "tokens": sum(int(record.get("tokens", 0) or 0) for record in records),
        "failures": failures,
        "skipped_reason": skipped_reason,
        "metrics": metrics,
        "metric_samples": metric_samples,
        "per_question": [
            {
                "question_id": record.get("question_id"),
                "status": record.get("status"),
                "error": record.get("error"),
                "tokens": record.get("tokens", 0),
                "metrics": record.get("metrics", {}),
            }
            for record in records
        ],
    }


def _skipped_summary(system: str, config: Mapping, reason: str) -> dict:
    return _summarize(
        system,
        config,
        [],
        status="skipped",
        skipped_reason=reason,
    )


def _system_status(
    *,
    config: Mapping,
    target_ids: Sequence[str],
    records: list[dict],
    total_questions: int,
    stopped_here: bool,
    run_stopped: bool,
) -> str:
    if not config.get("implemented", True):
        return "skipped"
    attempted = {record.get("question_id") for record in records}
    complete = all(question_id in attempted for question_id in target_ids)
    if complete:
        if len(target_ids) < total_questions:
            return "incomplete"
        return "completed"
    if stopped_here:
        return "budget_exhausted"
    if not attempted:
        return "not_started" if run_stopped else "incomplete"
    return "partial"


def _remaining_ids(
    selected: Sequence[str],
    systems: Mapping[str, dict],
    target_ids: Sequence[str],
    out_dir: Path,
    safe_names: Mapping[str, str],
) -> list[str]:
    remaining: list[str] = []
    for question_id in target_ids:
        for system in selected:
            if not systems[system].get("implemented", True):
                continue
            path = out_dir / system / f"{safe_names[question_id]}.json"
            if _load_checkpoint(path, question_id) is None:
                remaining.append(question_id)
                break
    return remaining


def run_experiment(
    matrix: dict,
    benchmark_dir: Path,
    out_dir: Path,
    deps: ExperimentDeps,
    *,
    budget: int | None = None,
    systems: Sequence[str] | str | None = None,
    limit: int | None = None,
    retry_failed: bool = False,
) -> dict:
    """Run an Eval-lite matrix with per-question checkpointing.

    Returns an aggregate dict with ``status``, ``stop_reason``, ``budget``,
    ``systems`` (name -> summary) and ``results`` (summaries in matrix order).
    """
    if not isinstance(deps, ExperimentDeps) and not hasattr(deps, "run_system"):
        raise TypeError("deps must provide run_system(system, question)")

    normalized_systems = _normalize_systems(matrix)
    selected = _selected_systems(normalized_systems, systems)

    matrix_budget = _matrix_budget(matrix)
    max_tokens = matrix_budget if budget is None else budget
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0:
            raise ValueError("budget must be an integer >= 0")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be an integer >= 0")

    questions, qrels_by_id = load_benchmark(Path(benchmark_dir))
    benchmark_dir = Path(benchmark_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    question_ids = [_question_id(question) for question in questions]
    safe_names = {
        question_id: _safe_component(question_id)
        for question_id in question_ids
        if question_id is not None
    }
    target_questions = questions[:limit] if limit is not None else list(questions)
    target_ids = [_question_id(question) for question in target_questions]
    target_ids = [question_id for question_id in target_ids if question_id is not None]

    started_at = _now()
    tokens_used = 0
    run_stopped = False
    stop_reason: str | None = None
    results: list[dict] = []
    summaries: dict[str, dict] = {}

    for system in selected:
        config = normalized_systems[system]
        system_dir = out_dir / system

        if not config.get("implemented", True):
            summary = _skipped_summary(system, config, "not_implemented")
            _write_json(system_dir / "run.json", summary)
            summaries[system] = summary
            results.append(summary)
            continue

        records: list[dict] = []
        stopped_here = False
        for question in target_questions:
            question_id = _question_id(question)
            if question_id is None:
                continue
            path = system_dir / f"{safe_names[question_id]}.json"
            checkpoint = _load_checkpoint(path, question_id)
            previous: dict | None = None
            if checkpoint is not None:
                if retry_failed and checkpoint.get("status") != "ok":
                    previous = checkpoint
                    checkpoint = None
                else:
                    tokens_used += int(checkpoint.get("tokens", 0) or 0)
                    records.append(checkpoint)
                    continue
            if run_stopped:
                continue
            if max_tokens is not None and tokens_used >= max_tokens:
                run_stopped = True
                stopped_here = True
                stop_reason = "budget_exhausted"
                break

            record = _run_one(
                deps,
                system=system,
                question=question,
                qrels_record=qrels_by_id.get(question_id),
                kind=str(config.get("kind", "retrieval")),
                previous=previous,
            )
            _write_json(path, record)
            tokens_used += int(record.get("tokens", 0) or 0)
            records.append(record)

        status = _system_status(
            config=config,
            target_ids=target_ids,
            records=records,
            total_questions=len(questions),
            stopped_here=stopped_here,
            run_stopped=run_stopped,
        )
        summary = _summarize(system, config, records, status=status)
        _write_json(system_dir / "run.json", summary)
        summaries[system] = summary
        results.append(summary)

        if (
            not run_stopped
            and limit is not None
            and len(target_ids) < len(questions)
        ):
            stop_reason = "limit"

    if stop_reason == "budget_exhausted":
        final_status = "budget_exhausted"
    elif limit is not None and len(target_ids) < len(questions):
        final_status = "incomplete"
        stop_reason = stop_reason or "limit"
    else:
        final_status = "completed"

    remaining = _remaining_ids(
        selected, normalized_systems, target_ids, out_dir, safe_names
    )
    budget_state = {
        "max_tokens": max_tokens,
        "tokens_used": tokens_used,
        "stopped": stop_reason == "budget_exhausted",
        "stop_reason": stop_reason,
        "remaining_question_ids": remaining,
    }
    finished_at = _now()
    matrix_hash = hashlib.sha256(
        json.dumps(matrix, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()

    aggregate = {
        "version": EXPERIMENT_VERSION,
        "status": final_status,
        "stop_reason": stop_reason,
        "matrix_hash": matrix_hash,
        "benchmark_dir": str(benchmark_dir),
        "out_dir": str(out_dir),
        "n_questions": len(questions),
        "n_target": len(target_ids),
        "limit": limit,
        "systems_requested": selected,
        "qrels_without_questions": [
            question_id for question_id in qrels_by_id if question_id not in safe_names
        ],
        "budget": budget_state,
        "started_at": started_at,
        "finished_at": finished_at,
        "results": results,
        "systems": summaries,
    }
    _write_json(out_dir / "run.json", aggregate)
    return aggregate


# ---------------------------------------------------------------------------
# Default retrieval dependencies (deterministic, no LLM)
# ---------------------------------------------------------------------------


class RetrievalDeps(ExperimentDeps):
    """Deterministic BM25/dense/RRF systems over a prepared benchmark index.

    The index is loaded lazily so importing this module (and every offline
    test) never touches the filesystem. Answer/rerank systems are not
    implemented here: they need a model and must be added explicitly.
    """

    def __init__(
        self,
        benchmark_dir: Path,
        systems: Mapping[str, Mapping] | None = None,
        *,
        embedder: Any | None = None,
        rerank_fn: Any | None = None,
    ) -> None:
        self.benchmark_dir = Path(benchmark_dir)
        self.systems = dict(systems or {})
        self._rerank_fn = rerank_fn
        self._loaded = False
        self._index = None
        self._metadata: dict = {}
        self._clean: dict = {}
        self._embedder = embedder

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from retrieval_core import load_index  # heavy imports only on real runs

        index_dir = self.benchmark_dir / "index"
        self._index, self._metadata, _parents = load_index(index_dir)
        self._clean = json.loads(
            (index_dir / "corpus_clean.json").read_text(encoding="utf-8")
        )
        self._loaded = True

    def _ensure_embedder(self) -> Any:
        if self._embedder is not None:
            return self._embedder
        from sentence_transformers import SentenceTransformer

        self._ensure_loaded()
        self._embedder = SentenceTransformer(
            self._metadata["model_name"],
            device="cpu",
            cache_folder=str(PROJECT_ROOT / "data" / "models"),
            local_files_only=True,
        )
        return self._embedder

    def run_system(self, system: str, question: dict) -> dict:
        config = self.systems.get(system) or {}
        if (
            not config.get("implemented", True)
            or config.get("kind", "retrieval") != "retrieval"
            or (config.get("requires_llm", False) and self._rerank_fn is None)
        ):
            return {
                "status": "not_implemented",
                "error": f"{system} needs a model and is not implemented in E6",
            }
        mode = str(config.get("mode", "bm25"))
        top_k = int(config.get("top_k", 6))
        query_zh = str(question.get("query_zh") or question.get("question_zh") or "")
        query_en = str(question.get("query_en") or query_zh)
        query_field = str(config.get("query", "query_en"))
        if query_field not in {"query_zh", "query_en"}:
            raise ValueError(f"unsupported query field {query_field!r}")
        retrieval_query = query_zh if query_field == "query_zh" else query_en

        from retrieval_core import retrieve_bm25, retrieve_dense, rrf_fuse

        self._ensure_loaded()
        if mode == "bm25":
            hits = retrieve_bm25(self._clean, query_en, top_k)
        elif mode == "dense":
            hits = retrieve_dense(
                self._index,
                self._metadata,
                retrieval_query,
                top_k,
                embedder=self._ensure_embedder(),
            )
        elif mode == "hybrid":
            pool = max(int(config.get("pool", 50)), top_k)
            rerank_tokens = 0
            if config.get("rerank") == "llm":
                if self._rerank_fn is None:
                    return {
                        "status": "not_implemented",
                        "error": f"{system} requires a rerank_fn",
                    }
                pool = max(int(config.get("rerank_top_n", 20)), top_k)
            dense_hits = retrieve_dense(
                self._index,
                self._metadata,
                retrieval_query,
                pool,
                embedder=self._ensure_embedder(),
            )
            bm25_hits = retrieve_bm25(self._clean, query_en, pool)
            fused = rrf_fuse(
                dense_hits,
                bm25_hits,
                k=int(config.get("rrf_k", 60)),
                top_n=pool,
            )
            if config.get("rerank") == "llm":
                hits, rerank_tokens = self._rerank_fn(
                    retrieval_query, fused, top_k
                )
            else:
                hits = fused[:top_k]
        else:
            raise ValueError(f"unsupported retrieval mode {mode!r}")

        return {
            "status": "ok",
            "retrieved_parent_ids": [hit.parent_chunk_id for hit in hits],
            "tokens": rerank_tokens if mode == "hybrid" else 0,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a PaperLens Eval-lite system matrix with checkpointing."
    )
    parser.add_argument("--matrix", type=Path, required=True, help="matrix.yaml/json")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=PROJECT_ROOT / "eval" / "benchmark_v2",
        help="directory with questions/qrels and index/",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: eval/benchmark_v2/runs/<timestamp>)",
    )
    parser.add_argument(
        "--systems",
        default=None,
        help="comma-separated subset of matrix systems",
    )
    parser.add_argument("--budget", type=int, default=None, help="max recorded tokens")
    parser.add_argument("--limit", type=int, default=None, help="cap questions per system")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="re-run checkpointed failed/invalid questions",
    )
    parser.add_argument("--no-report", action="store_true", help="skip the Markdown report")
    parser.add_argument(
        "--llm-rerank",
        action="store_true",
        help="enable the DeepSeek rerank system (spends API tokens)",
    )
    args = parser.parse_args(argv)

    matrix = load_matrix(args.matrix)
    out_dir = args.out
    if out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "eval" / "benchmark_v2" / "runs" / stamp

    rerank_fn = None
    if args.llm_rerank:
        from eval.rerank import make_deepseek_rerank_fn

        rerank_fn = make_deepseek_rerank_fn()
    deps = RetrievalDeps(
        args.benchmark_dir, matrix.get("systems") or {}, rerank_fn=rerank_fn
    )
    result = run_experiment(
        matrix,
        args.benchmark_dir,
        out_dir,
        deps,
        budget=args.budget,
        systems=args.systems,
        limit=args.limit,
        retry_failed=args.retry_failed,
    )

    if args.no_report:
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "stop_reason": result["stop_reason"],
                    "budget": result["budget"],
                    "systems": {
                        name: {
                            "status": summary["status"],
                            "n": summary["n"],
                            "n_failed": summary["n_failed"],
                            "metrics": summary["metrics"],
                        }
                        for name, summary in result["systems"].items()
                    },
                    "out_dir": result["out_dir"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    from eval.analysis.make_report import make_report

    print(make_report(result["results"], Path(out_dir)))
    print(f"\nArtifacts: {Path(out_dir).resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
