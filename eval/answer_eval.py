"""Deterministic answer-level (citation-level) evaluation for Eval-lite.

The evaluator answers each benchmark question with the evidence supplied by an
existing retrieval run, then scores only properties that can be checked without
a judge:

* ``schema_success`` -- the response parsed as JSON and passed
  :func:`answer_question.validate_answer`.
* ``quote_integrity`` -- every citation quote is an exact,
  whitespace-normalized substring of its cited evidence (0 when the schema
  fails).
* ``cited_gold_parent_recall`` -- fraction of gold parent chunks that the
  answer actually cited.
* ``gold_point_citation_recall`` -- fraction of gold points whose quote appears
  in at least one *cited* evidence text.

This is **not** answer accuracy, correctness, completeness or faithfulness.
No LLM judge and no human review are involved. The Markdown and JSON outputs
state this explicitly.

Retrieval is never re-run here: the module consumes the per-question
``<question_id>.json`` files written by :mod:`eval.run_experiment`, maps their
``retrieved_parent_ids`` to chunks in ``index/corpus_clean.json``, and builds
the same ``{"question": ..., "evidence": [...]}`` payload used by
``answer_question.main``.

Everything is dependency-injected through ``answer_fn(messages) -> str | dict``
so tests run fully offline. :func:`make_deepseek_answer_fn` is the only code
that reads ``.env`` or touches the network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from answer_question import PROMPT, validate_answer  # noqa: E402
from answer_schema import normalize_answer_schema  # noqa: E402
from eval.analysis.make_report import (  # noqa: E402
    CONFIDENCE,
    DEFAULT_RESAMPLES,
    bootstrap_ci,
)
from eval.run_experiment import load_benchmark  # noqa: E402

ANSWER_EVAL_VERSION = "answer_eval_v1"

DEFAULT_BENCHMARK_DIR = PROJECT_ROOT / "eval" / "benchmark_v2" / "eval120"
DEFAULT_RETRIEVAL_RUN = PROJECT_ROOT / "eval" / "benchmark_v2" / "runs" / "eval120_rerank"
DEFAULT_RETRIEVAL_SYSTEM = "hybrid_rrf_rerank_top6"

METRIC_KEYS = (
    "schema_success",
    "quote_integrity",
    "cited_gold_parent_recall",
    "gold_point_citation_recall",
)
METRIC_LABELS = {
    "schema_success": "Schema success",
    "quote_integrity": "Quote integrity",
    "cited_gold_parent_recall": "Cited gold parent recall",
    "gold_point_citation_recall": "Gold point citation recall",
}

ANSWER_STATUSES = ("answered", "partial", "insufficient", "invalid")
TERMINAL_STATUSES = frozenset({"ok", "failed"})
MAX_ERROR_CHARS = 500

DISCLAIMER = (
    "Citation-level evaluation only: this measures JSON schema validity, exact "
    "quote copying, and whether the answer cited the gold parent chunks and "
    "gold-point quotes. It is NOT answer accuracy, correctness, completeness, "
    "or faithfulness. There is no LLM judge and no human review."
)

AnswerFn = Callable[[list[dict]], "str | dict"]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_error(error: object) -> str:
    return " ".join(str(error).split())[:MAX_ERROR_CHARS]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe_component(question_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(question_id)).strip("._")
    if not cleaned:
        cleaned = "q"
    if len(cleaned) > 100:
        digest = hashlib.sha256(str(question_id).encode("utf-8")).hexdigest()[:8]
        cleaned = f"{cleaned[:80]}_{digest}"
    return cleaned


def normalize_text(value: object) -> str:
    """Whitespace-only normalization, matching ``answer_question.validate_answer``."""
    return " ".join(str(value).split())


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    cleaned = sorted(float(value) for value in values if _is_number(value))
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    position = max(0.0, min(1.0, quantile)) * (len(cleaned) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return cleaned[lower]
    fraction = position - lower
    return cleaned[lower] * (1.0 - fraction) + cleaned[upper] * fraction


def _seed_for(metric: str) -> int:
    digest = hashlib.sha256(f"answer_eval\x1f{metric}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


# ---------------------------------------------------------------------------
# Inputs: corpus, question, retrieval run
# ---------------------------------------------------------------------------


def load_corpus(corpus_path: Path) -> dict[str, dict]:
    """Load ``corpus_clean.json`` and index chunks by parent ``chunk_id``."""
    path = Path(corpus_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks") if isinstance(payload, Mapping) else None
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"{path}: expected a non-empty 'chunks' list")

    by_id: dict[str, dict] = {}
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            raise ValueError(f"{path}: every chunk must be an object")
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"{path}: chunk without a chunk_id")
        if chunk_id in by_id:
            raise ValueError(f"{path}: duplicate chunk_id {chunk_id!r}")
        if not isinstance(chunk.get("text"), str):
            raise ValueError(f"{path}: chunk {chunk_id!r} has no text")
        by_id[chunk_id] = dict(chunk)
    return by_id


def _looks_like_run_dir(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    for path in directory.glob("*.json"):
        if path.name in {"run.json", "results.json", "aggregate.json"}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "retrieved_parent_ids" in payload:
            return True
    return False


def resolve_retrieval_run(
    retrieval_run: Path, retrieval_system: str | None
) -> tuple[Path, Path]:
    """Resolve ``--retrieval-run`` to ``(system_dir, run_root)``.

    ``retrieval_run`` may be either the run root (containing one directory per
    system, e.g. ``runs/eval120_rerank``) or a single system directory itself
    (e.g. ``.../hybrid_rrf_rerank_top6``). ``run_root`` is used only to pick the
    default output directory.
    """
    base = Path(retrieval_run)
    system = str(retrieval_system).strip() if retrieval_system else ""
    if system:
        candidate = base / system
        if _looks_like_run_dir(candidate):
            return candidate, base
    if _looks_like_run_dir(base):
        return base, base.parent
    if system:
        return base / system, base
    return base, base.parent


def _question_text(question: Mapping) -> str:
    for key in ("question_zh", "query_zh", "question", "query_en", "query"):
        value = question.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("question has no non-empty question text")


def _gold_parent_ids(qrels_record: Mapping | None) -> list[str]:
    if not isinstance(qrels_record, Mapping):
        return []
    value = qrels_record.get("gold_parent_ids")
    if not isinstance(value, (list, tuple)):
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, str) and item.strip():
            item = item.strip()
            if item not in seen:
                seen.add(item)
                ordered.append(item)
    return ordered


def _gold_points(question: Mapping) -> list[dict]:
    value = question.get("gold_points")
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def load_retrieval_record(
    run_dir: Path, question_id: str
) -> tuple[dict, list[str], str, str | None]:
    """Return ``(record, retrieved_parent_ids, status, error)``."""
    path = Path(run_dir) / f"{_safe_component(question_id)}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing retrieval record {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: retrieval record must be a JSON object")
    stored_id = payload.get("question_id")
    if stored_id is not None and stored_id != question_id:
        raise ValueError(
            f"{path}: question_id {stored_id!r} does not match {question_id!r}"
        )

    raw_ids = payload.get("retrieved_parent_ids")
    if raw_ids is None:
        raw_ids = []
    if not isinstance(raw_ids, (list, tuple)):
        raise ValueError(f"{path}: retrieved_parent_ids must be a list")

    retrieved: list[str] = []
    seen: set[str] = set()
    for item in raw_ids:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"{path}: retrieved_parent_ids must contain non-empty strings"
            )
        item = item.strip()
        if item not in seen:
            seen.add(item)
            retrieved.append(item)

    status = payload.get("status", "ok")
    if not isinstance(status, str):
        raise ValueError(f"{path}: status must be a string")
    error = payload.get("error")
    if error is not None:
        error = _clean_error(error)
    return payload, retrieved, status, error


def build_evidence(
    retrieved_parent_ids: Sequence[str], chunks: Mapping[str, Mapping]
) -> dict[str, dict]:
    """Map parent IDs to evidence units with ids ``E1..En`` in retrieval order."""
    missing = [
        parent_id for parent_id in retrieved_parent_ids if parent_id not in chunks
    ]
    if missing:
        raise ValueError(
            "retrieved parent id(s) missing from corpus_clean.json: "
            + ", ".join(missing[:5])
        )

    evidence: dict[str, dict] = {}
    for number, parent_id in enumerate(retrieved_parent_ids, start=1):
        chunk = chunks[parent_id]
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"chunk {parent_id!r} has empty text")
        evidence[f"E{number}"] = {
            "id": f"E{number}",
            "parent_chunk_id": parent_id,
            "title": str(chunk.get("title") or ""),
            "page_number": chunk.get("page_number"),
            "text": text,
        }
    return evidence


def build_payload(question_text: str, evidence: Mapping[str, Mapping]) -> dict:
    """Exactly the payload shape used by ``answer_question.main``."""
    return {
        "question": question_text,
        "evidence": [
            {
                "id": unit["id"],
                "title": unit["title"],
                "text": unit["text"],
            }
            for unit in evidence.values()
        ],
    }


# ---------------------------------------------------------------------------
# Scoring one answer
# ---------------------------------------------------------------------------


def _parse_answer(raw: object) -> tuple[dict | None, str | None]:
    if isinstance(raw, Mapping):
        return dict(raw), None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc.msg}"
        if not isinstance(parsed, dict):
            return None, "answer JSON must be an object"
        return parsed, None
    return None, "answer_fn must return a JSON object or a JSON string"


def _empty_metrics(gold_parent_count: int, gold_point_count: int) -> dict:
    return {
        "schema_success": 0.0,
        "quote_integrity": 0.0,
        "cited_gold_parent_recall": 0.0 if gold_parent_count else None,
        "gold_point_citation_recall": 0.0 if gold_point_count else None,
    }


def evaluate_answer(
    raw: object,
    evidence: Mapping[str, Mapping],
    gold_parent_ids: Sequence[str],
    gold_points: Sequence[Mapping],
) -> dict:
    """Score one raw answer against the evidence and gold data.

    Returns a dict with ``metrics``, ``answer_status``, the normalized answer
    (or ``None``), citation metadata and a validation error, if any.
    """
    gold_ids = list(dict.fromkeys(str(item) for item in gold_parent_ids if str(item).strip()))
    points = list(gold_points)
    parse_error: str | None = None
    answer: dict | None = None
    normalizations: list[dict] = []
    parsed, parse_error = _parse_answer(raw)

    if parsed is not None:
        try:
            normalized, changes = normalize_answer_schema(parsed)
            validate_answer(normalized, evidence)
        except (TypeError, ValueError) as exc:
            parse_error = _clean_error(exc)
        else:
            answer = normalized
            normalizations = list(changes)

    schema_success = answer is not None
    metrics = {
        "schema_success": 1.0 if schema_success else 0.0,
        "quote_integrity": 0.0,
        "cited_gold_parent_recall": None,
        "gold_point_citation_recall": None,
    }

    cited_evidence_ids: list[str] = []
    cited_parent_ids: list[str] = []
    quotes_ok = True

    if schema_success:
        assert answer is not None
        seen_evidence: set[str] = set()
        seen_parents: set[str] = set()
        for point in answer.get("points") or []:
            if not isinstance(point, Mapping):
                continue
            for citation in point.get("citations") or []:
                if not isinstance(citation, Mapping):
                    continue
                evidence_id = citation.get("id")
                quote = citation.get("quote")
                if evidence_id not in evidence:
                    quotes_ok = False
                    continue
                if evidence_id not in seen_evidence:
                    seen_evidence.add(str(evidence_id))
                    cited_evidence_ids.append(str(evidence_id))
                parent_id = str(evidence[evidence_id]["parent_chunk_id"])
                if parent_id not in seen_parents:
                    seen_parents.add(parent_id)
                    cited_parent_ids.append(parent_id)
                if not isinstance(quote, str) or normalize_text(quote) not in normalize_text(
                    evidence[evidence_id]["text"]
                ):
                    quotes_ok = False
        metrics["quote_integrity"] = 1.0 if quotes_ok else 0.0

    if gold_ids:
        hits = len(set(gold_ids).intersection(cited_parent_ids))
        metrics["cited_gold_parent_recall"] = hits / len(gold_ids) if schema_success else 0.0

    if points:
        if schema_success:
            cited_texts = [
                normalize_text(evidence[evidence_id]["text"])
                for evidence_id in cited_evidence_ids
            ]
            matched = 0
            for point in points:
                quote = point.get("quote")
                normalized_quote = normalize_text(quote) if quote is not None else ""
                if normalized_quote and any(
                    normalized_quote in text for text in cited_texts
                ):
                    matched += 1
            metrics["gold_point_citation_recall"] = matched / len(points)
        else:
            metrics["gold_point_citation_recall"] = 0.0

    answer_status = (
        str(answer.get("status")) if schema_success and answer else "invalid"
    )
    if answer_status not in ("answered", "partial", "insufficient"):
        answer_status = "invalid"

    return {
        "schema_success": schema_success,
        "answer_status": answer_status,
        "answer": answer,
        "schema_normalizations": normalizations,
        "error": parse_error,
        "metrics": metrics,
        "cited_evidence_ids": cited_evidence_ids,
        "cited_parent_ids": cited_parent_ids,
    }


# ---------------------------------------------------------------------------
# Running one question and the whole benchmark
# ---------------------------------------------------------------------------


def _usage_tokens(usage: object) -> int:
    if not isinstance(usage, Mapping):
        return 0

    def clean(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return max(0, value)
        if isinstance(value, float) and math.isfinite(value):
            return max(0, int(value))
        return None

    for key in ("total_tokens", "tokens"):
        value = clean(usage.get(key))
        if value is not None:
            return value
    prompt = clean(usage.get("prompt_tokens"))
    completion = clean(usage.get("completion_tokens"))
    if prompt is not None and completion is not None:
        return prompt + completion
    return 0


def _safe_usage(usage: object) -> dict | None:
    return dict(usage) if isinstance(usage, Mapping) else None


def _safe_raw(value: object) -> object | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            return None
        return dict(value)
    return None


def _model_name(answer_fn: object) -> str | None:
    value = getattr(answer_fn, "model", None)
    return str(value) if value else None


def _evidence_metadata(evidence: Mapping[str, Mapping]) -> list[dict]:
    return [
        {
            "id": unit["id"],
            "parent_chunk_id": unit["parent_chunk_id"],
            "title": unit["title"],
            "page_number": unit["page_number"],
        }
        for unit in evidence.values()
    ]


def _failed_record(
    *,
    question_id: str,
    error: str,
    gold_parent_ids: Sequence[str],
    gold_points: Sequence[Mapping],
    retrieved_parent_ids: Sequence[str] = (),
    evidence: Mapping[str, Mapping] | None = None,
    raw_response: object | None = None,
    latency_ms: float = 0.0,
    tokens: int = 0,
    usage: object = None,
    model: str | None = None,
    attempt: int = 1,
) -> dict:
    return {
        "question_id": question_id,
        "status": "failed",
        "answer_status": "invalid",
        "error": _clean_error(error),
        "model": model,
        "raw_response": raw_response,
        "usage": _safe_usage(usage),
        "tokens": tokens,
        "latency_ms": round(float(latency_ms), 3),
        "retrieved_parent_ids": list(retrieved_parent_ids),
        "evidence": _evidence_metadata(evidence or {}),
        "cited_evidence_ids": [],
        "cited_parent_ids": [],
        "gold_parent_ids": list(gold_parent_ids),
        "gold_point_count": len(gold_points),
        "metrics": _empty_metrics(len(gold_parent_ids), len(gold_points)),
        "attempt": attempt,
        "finished_at": _now(),
    }


def _run_question(
    *,
    question: Mapping,
    qrels_record: Mapping | None,
    run_dir: Path,
    chunks: Mapping[str, Mapping],
    answer_fn: AnswerFn,
    attempt: int = 1,
) -> dict:
    question_id = str(question.get("question_id") or question.get("qid") or question.get("id") or "").strip()
    if not question_id:
        raise ValueError("question without an id")
    gold_ids = _gold_parent_ids(qrels_record)
    gold_points = _gold_points(question)
    model = _model_name(answer_fn)

    try:
        _record, retrieved, retrieval_status, retrieval_error = load_retrieval_record(
            run_dir, question_id
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _failed_record(
            question_id=question_id,
            error=f"retrieval record unavailable: {type(exc).__name__}: {exc}",
            gold_parent_ids=gold_ids,
            gold_points=gold_points,
            model=model,
            attempt=attempt,
        )

    if retrieval_status != "ok" or retrieval_error:
        detail = retrieval_error or f"status {retrieval_status}"
        return _failed_record(
            question_id=question_id,
            error=f"retrieval run unavailable for {question_id}: {detail}",
            gold_parent_ids=gold_ids,
            gold_points=gold_points,
            retrieved_parent_ids=retrieved,
            model=model,
            attempt=attempt,
        )

    try:
        evidence = build_evidence(retrieved, chunks)
        if not evidence:
            raise ValueError("retrieval record returned no evidence")
        payload = build_payload(_question_text(question), evidence)
    except ValueError as exc:
        return _failed_record(
            question_id=question_id,
            error=f"evidence build failed: {exc}",
            gold_parent_ids=gold_ids,
            gold_points=gold_points,
            retrieved_parent_ids=retrieved,
            model=model,
            attempt=attempt,
        )

    messages = [
        {"role": "system", "content": PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]

    started = time.perf_counter()
    try:
        raw = answer_fn(messages)
    except Exception as exc:  # noqa: BLE001 - a model crash is a failed question
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = getattr(answer_fn, "last_usage", None)
        return _failed_record(
            question_id=question_id,
            error=f"{type(exc).__name__}: {exc}",
            gold_parent_ids=gold_ids,
            gold_points=gold_points,
            retrieved_parent_ids=retrieved,
            evidence=evidence,
            latency_ms=latency_ms,
            tokens=_usage_tokens(usage),
            usage=usage,
            model=model,
            attempt=attempt,
        )

    latency_ms = (time.perf_counter() - started) * 1000.0
    usage = getattr(answer_fn, "last_usage", None)
    tokens = _usage_tokens(usage)
    raw_response = _safe_raw(raw)
    evaluation = evaluate_answer(raw, evidence, gold_ids, gold_points)

    error = evaluation["error"]
    if raw_response is None:
        error = error or "answer_fn returned an unsupported value"
    return {
        "question_id": question_id,
        "status": "ok",
        "answer_status": evaluation["answer_status"],
        "error": error,
        "model": model,
        "raw_response": raw_response,
        "usage": _safe_usage(usage),
        "tokens": tokens,
        "latency_ms": round(latency_ms, 3),
        "retrieved_parent_ids": list(retrieved),
        "evidence": _evidence_metadata(evidence),
        "cited_evidence_ids": evaluation["cited_evidence_ids"],
        "cited_parent_ids": evaluation["cited_parent_ids"],
        "gold_parent_ids": list(gold_ids),
        "gold_point_count": len(gold_points),
        "metrics": evaluation["metrics"],
        "attempt": attempt,
        "finished_at": _now(),
    }


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


def _aggregate(
    records: Sequence[Mapping],
    *,
    resamples: int,
    run_dir: Path,
    run_root: Path,
    retrieval_system: str,
    benchmark_dir: Path,
    corpus_path: Path,
    answer_model: str | None,
) -> dict:
    samples: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
    for record in records:
        metrics = record.get("metrics") or {}
        for key in METRIC_KEYS:
            value = metrics.get(key)
            if _is_number(value):
                samples[key].append(float(value))

    metrics_payload: dict[str, dict] = {}
    for key in METRIC_KEYS:
        interval = bootstrap_ci(samples[key], resamples=resamples, seed=_seed_for(key))
        if interval is not None:
            interval = dict(interval)
            interval["label"] = METRIC_LABELS[key]
            metrics_payload[key] = interval

    status_counts = {status: 0 for status in ANSWER_STATUSES}
    for record in records:
        status = str(record.get("answer_status") or "invalid")
        status_counts[status] = status_counts.get(status, 0) + 1

    failures = [
        str(record.get("question_id"))
        for record in records
        if record.get("status") != "ok"
        or (record.get("metrics") or {}).get("schema_success") != 1.0
    ]

    latencies = [
        float(record["latency_ms"])
        for record in records
        if _is_number(record.get("latency_ms"))
    ]
    token_values = [
        max(0, int(record.get("tokens", 0) or 0)) for record in records
    ]
    n = len(records)
    latency_mean = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "version": ANSWER_EVAL_VERSION,
        "generated_by": "eval/answer_eval.py",
        "created_at": _now(),
        "retrieval_run": str(run_dir),
        "retrieval_run_root": str(run_root),
        "retrieval_system": retrieval_system,
        "benchmark_dir": str(benchmark_dir),
        "corpus_path": str(corpus_path),
        "answer_model": answer_model,
        "confidence": CONFIDENCE,
        "bootstrap_resamples": resamples,
        "bootstrap_seed_policy": "sha256('answer_eval\\x1f<metric>')[:8]; deterministic",
        "disclaimer": DISCLAIMER,
        "n": n,
        "n_failed": len(failures),
        "failures": failures,
        "status_counts": status_counts,
        "metrics": metrics_payload,
        "tokens": {
            "n": n,
            "total": sum(token_values),
            "mean": (sum(token_values) / n) if n else 0.0,
        },
        "latency_ms": {
            "n": len(latencies),
            "mean": latency_mean,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "per_question": [
            {
                "question_id": record.get("question_id"),
                "status": record.get("status"),
                "answer_status": record.get("answer_status"),
                "error": record.get("error"),
                "tokens": record.get("tokens", 0),
                "latency_ms": record.get("latency_ms", 0.0),
                "metrics": record.get("metrics", {}),
            }
            for record in records
        ],
        "notes": [
            "Failed or schema-invalid answers stay in the denominator and are scored 0.",
            "Empty-gold questions are excluded from ratio metrics (n is smaller) but not from schema_success or status counts.",
            "Bootstrap intervals are descriptive: the benchmark is a fixed set, not a random sample.",
        ],
    }


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_interval(interval: Mapping | None) -> str:
    if not isinstance(interval, Mapping):
        return "n/a"
    return (
        f"{float(interval['mean']):.3f} "
        f"[{float(interval['ci_low']):.3f}, {float(interval['ci_high']):.3f}] "
        f"(n={int(interval['n'])})"
    )


def _format_metric_value(value: object) -> str:
    return f"{float(value):.3f}" if _is_number(value) else "n/a"


def _render_markdown(aggregate: Mapping) -> str:
    lines = [
        "# PaperLens Answer-Level Evaluation (citation-level)",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "## Configuration",
        "",
        f"- Retrieval run: `{aggregate['retrieval_run']}`",
        f"- Retrieval system: `{aggregate['retrieval_system']}`",
        f"- Benchmark: `{aggregate['benchmark_dir']}`",
        f"- Corpus: `{aggregate['corpus_path']}`",
        f"- Answer model: `{aggregate.get('answer_model') or 'injected answer_fn'}`",
        f"- Questions: {aggregate['n']} (failed or invalid: {aggregate['n_failed']})",
        f"- Bootstrap: {aggregate['bootstrap_resamples']} percentile resamples, "
        f"{float(aggregate['confidence']):.0%} interval, deterministic seed",
        "",
        "## Metrics",
        "",
        "Higher is better. A schema-invalid answer contributes 0 to every metric. "
        "Empty-gold questions are excluded from ratio metrics; their `n` is shown.",
        "",
        "| Metric | Mean [95% CI] |",
        "|---|---|",
    ]
    for key in METRIC_KEYS:
        interval = (aggregate.get("metrics") or {}).get(key)
        lines.append(
            f"| {METRIC_LABELS[key]} | {_format_interval(interval)} |"
        )

    lines.extend(
        [
            "",
            "## Answer status counts",
            "",
            "| Status | Count |",
            "|---|---|",
        ]
    )
    for status, count in (aggregate.get("status_counts") or {}).items():
        lines.append(f"| {_md_cell(status)} | {count} |")

    failures = aggregate.get("failures") or []
    lines.extend(["", "## Failures", ""])
    if failures:
        lines.append(
            "Questions with a failed model call, a failed/unavailable retrieval "
            "record, or failed answer validation:"
        )
        lines.append("")
        lines.extend(f"- `{_md_cell(question_id)}`" for question_id in failures)
    else:
        lines.append("None.")

    tokens = aggregate.get("tokens") or {}
    latency = aggregate.get("latency_ms") or {}
    lines.extend(
        [
            "",
            "## Cost and latency",
            "",
            f"- Tokens: total {tokens.get('total', 0)}, mean {float(tokens.get('mean', 0.0)):.1f}",
            f"- Latency (ms): mean {float(latency.get('mean', 0.0)):.1f}, "
            f"p50 {float(latency.get('p50') or 0.0):.1f}, "
            f"p95 {float(latency.get('p95') or 0.0):.1f}",
            "",
            "## Per question",
            "",
            "| Question | Pipeline | Answer | Schema | Quote | Cited gold | Gold points | Tokens | Latency ms |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for item in aggregate.get("per_question") or []:
        metrics = item.get("metrics") or {}
        lines.append(
            "| {q} | {pipe} | {answer} | {schema} | {quote} | {cited} | {points} | "
            "{tokens} | {latency} |".format(
                q=_md_cell(item.get("question_id")),
                pipe=_md_cell(item.get("status")),
                answer=_md_cell(item.get("answer_status")),
                schema=_format_metric_value(metrics.get("schema_success")),
                quote=_format_metric_value(metrics.get("quote_integrity")),
                cited=_format_metric_value(metrics.get("cited_gold_parent_recall")),
                points=_format_metric_value(metrics.get("gold_point_citation_recall")),
                tokens=item.get("tokens", 0),
                latency=f"{float(item.get('latency_ms') or 0.0):.1f}",
            )
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This is citation-level scoring, not answer accuracy or faithfulness.",
            "- Schema success means the JSON passed structural and quote validation; it does not mean the explanation is correct.",
            "- `gold_point_citation_recall` only checks that the gold quote was copied inside a cited evidence text; it does not check that the answer used or interpreted that quote.",
            "- There is no LLM judge, no human review and no claim-level entailment metric.",
            "- Bootstrap intervals are descriptive because the benchmark is a fixed set, not a random sample.",
            "",
        ]
    )
    return "\n".join(lines)


def run_answer_eval(
    benchmark_dir: Path,
    retrieval_run: Path,
    out_dir: Path,
    answer_fn: AnswerFn,
    *,
    retrieval_system: str | None = None,
    limit: int | None = None,
    resume: bool = True,
    resamples: int = DEFAULT_RESAMPLES,
    corpus_path: Path | None = None,
) -> dict:
    """Score every (or the first ``limit``) benchmark question and write outputs.

    Writes one checkpoint per question to ``out_dir/<question_id>.json`` plus
    ``out_dir/aggregate.json`` and ``out_dir/report.md``. With ``resume=True``
    (the default), terminal checkpoints are reused and no model call is made
    for them.
    """
    if not callable(answer_fn):
        raise TypeError("answer_fn must be callable")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
    ):
        raise ValueError("limit must be an integer >= 0")
    if not isinstance(resume, bool):
        raise ValueError("resume must be a boolean")
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples <= 0:
        raise ValueError("resamples must be a positive integer")

    benchmark_dir = Path(benchmark_dir)
    out_dir = Path(out_dir)
    run_dir, run_root = resolve_retrieval_run(retrieval_run, retrieval_system)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"retrieval run directory not found: {run_dir}")
    if out_dir.resolve() == run_dir.resolve():
        raise ValueError(
            "out_dir must differ from the retrieval run directory; "
            "answer checkpoints would overwrite the retrieval records"
        )

    actual_corpus = (
        Path(corpus_path)
        if corpus_path is not None
        else benchmark_dir / "index" / "corpus_clean.json"
    )
    chunks = load_corpus(actual_corpus)
    questions, qrels_by_id = load_benchmark(benchmark_dir)

    target_questions = questions[:limit] if limit is not None else list(questions)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for question in target_questions:
        question_id = str(
            question.get("question_id")
            or question.get("qid")
            or question.get("id")
            or ""
        ).strip()
        if not question_id:
            raise ValueError(f"question without an id: {question!r}")
        path = out_dir / f"{_safe_component(question_id)}.json"
        if resume:
            checkpoint = _load_checkpoint(path, question_id)
            if checkpoint is not None:
                records.append(checkpoint)
                continue
        record = _run_question(
            question=question,
            qrels_record=qrels_by_id.get(question_id),
            run_dir=run_dir,
            chunks=chunks,
            answer_fn=answer_fn,
            attempt=1,
        )
        _write_json(path, record)
        records.append(record)

    system_label = str(retrieval_system).strip() if retrieval_system else run_dir.name
    aggregate = _aggregate(
        records,
        resamples=resamples,
        run_dir=run_dir,
        run_root=run_root,
        retrieval_system=system_label,
        benchmark_dir=benchmark_dir,
        corpus_path=actual_corpus,
        answer_model=_model_name(answer_fn),
    )
    _write_json(out_dir / "aggregate.json", aggregate)
    (out_dir / "report.md").write_text(
        _render_markdown(aggregate), encoding="utf-8"
    )
    return aggregate


# ---------------------------------------------------------------------------
# Default DeepSeek-backed answer function
# ---------------------------------------------------------------------------


def make_deepseek_answer_fn(
    *,
    model: str | None = None,
    env: Mapping[str, str] | None = None,
) -> AnswerFn:
    """Build a DeepSeek answer function. Configuration is read lazily.

    Reads ``.env`` (UTF-8 with optional BOM) unless ``env`` is supplied. The
    returned function records the last usage on ``answer_fn.last_usage`` and
    the resolved model on ``answer_fn.model``.
    """
    from dotenv import dotenv_values
    from openai import OpenAI

    values = (
        dict(env)
        if env is not None
        else {
            key: str(value)
            for key, value in dotenv_values(
                PROJECT_ROOT / ".env", encoding="utf-8-sig"
            ).items()
            if value is not None
        }
    )
    api_key = (values.get("DEEPSEEK_API_KEY") or "").strip()
    base_url = (values.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").strip()
    resolved_model = (model or values.get("DEEPSEEK_MODEL") or "").strip()
    if not api_key or not resolved_model:
        raise ValueError("DEEPSEEK_API_KEY and DEEPSEEK_MODEL are required")

    client = OpenAI(
        api_key=api_key, base_url=base_url, timeout=120, max_retries=0
    )

    def answer_fn(messages: list[dict]) -> str:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
            max_tokens=4096,
        )
        answer_fn.last_usage = (  # type: ignore[attr-defined]
            response.usage.model_dump() if response.usage else None
        )
        if not response.choices:
            raise ValueError("answer model returned no choices")
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("answer model returned empty content")
        return content

    answer_fn.last_usage = None  # type: ignore[attr-defined]
    answer_fn.model = resolved_model  # type: ignore[attr-defined]
    return answer_fn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic citation-level answer evaluation over a precomputed "
            "retrieval run (no judge, no human review)."
        )
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help="directory with questions/qrels and index/corpus_clean.json",
    )
    parser.add_argument(
        "--retrieval-run",
        type=Path,
        default=DEFAULT_RETRIEVAL_RUN,
        help="retrieval run root or a single system directory",
    )
    parser.add_argument(
        "--retrieval-system",
        default=DEFAULT_RETRIEVAL_SYSTEM,
        help="system subdirectory under --retrieval-run",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: <run root>/answer_eval/<system>)",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap questions")
    parser.add_argument(
        "--answer-model",
        default=None,
        help="override DEEPSEEK_MODEL from .env",
    )
    args = parser.parse_args(argv)

    run_dir, run_root = resolve_retrieval_run(
        args.retrieval_run, args.retrieval_system
    )
    out_dir = args.out
    if out_dir is None:
        out_dir = (
            run_root
            / "answer_eval"
            / (args.retrieval_system or run_dir.name)
        )

    answer_fn = make_deepseek_answer_fn(model=args.answer_model)
    aggregate = run_answer_eval(
        args.benchmark_dir,
        args.retrieval_run,
        out_dir,
        answer_fn,
        retrieval_system=args.retrieval_system,
        limit=args.limit,
    )

    summary = {
        "status": "completed",
        "n": aggregate["n"],
        "n_failed": aggregate["n_failed"],
        "failures": aggregate["failures"],
        "status_counts": aggregate["status_counts"],
        "metrics": {
            key: {
                "mean": interval["mean"],
                "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"],
                "n": interval["n"],
            }
            for key, interval in aggregate["metrics"].items()
        },
        "tokens": aggregate["tokens"],
        "latency_ms": aggregate["latency_ms"],
        "out_dir": str(Path(out_dir).resolve()),
        "disclaimer": DISCLAIMER,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
