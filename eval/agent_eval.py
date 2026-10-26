"""A/B evaluation harness: fixed RAG control vs the bounded retrieval agent.

The harness answers every benchmark question twice (or three times) under the
same evidence budget (6 units) and the same answer prompt
(``answer_question.PROMPT``):

* ``fixed_rag`` -- the control. It consumes the precomputed rerank retrieval
  run exactly as :mod:`eval.answer_eval` does (``retrieved_parent_ids`` mapped
  through ``index/corpus_clean.json``) and makes one answer call.
* ``agent_single`` -- one coverage-decision step over dense top-20, at most six
  selected evidence units, then one answer call. At most 2 model calls.
* ``agent_multi`` -- up to 3 coverage-decision rounds / 4 model calls with the
  dense and BM25 tools, at most six selected evidence units, then one answer
  call.

Scoring uses :func:`eval.answer_eval.evaluate_answer` unchanged. Gold data is
only read *after* the answer is generated; it is never placed in a prompt.
Retrieval is deterministic for the agent (local tools), and the fixed side uses
the already computed retrieval run. No answer here is re-retrieved for the
fixed control.

The module is fully dependency-injected: ``answer_fn(messages)`` and, for the
agent systems, ``decision_backend(prompt, payload, max_tokens)`` and
``tools.search(...)`` can be faked. The only code that reads ``.env`` or touches
the network is :func:`make_deepseek_answer_fn` /
:func:`make_deepseek_decision_backend`, which the tests never call.

Definitions used in the output
-----------------------------
``tool_calls``
    Number of successful retrieval-tool invocations (the initial dense search
    plus any supplemental searches). ``fixed_rag`` is 0 because its retrieval is
    precomputed.
``retrieval_rounds``
    Number of agent coverage-decision rounds actually completed. This is the
    "round" in the 1-round / 3-round budget. ``fixed_rag`` is 0.
``tokens``
    Total prompt+completion tokens across decision and answer calls, from the
    backend's ``usage`` / ``answer_fn.last_usage`` (0 when unavailable).
``latency_ms``
    Wall time for the whole question. ``answer_latency_ms`` and
    ``retrieval_latency_ms`` are reported separately so the fixed control's
    answer-call latency can be compared with the agent's answer call rather
    than with precomputed retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_policy import DECISION_PROMPT, normalized_query, validate_decision  # noqa: E402
from answer_question import PROMPT as ANSWER_PROMPT  # noqa: E402
from answer_schema import VERSION as ANSWER_SCHEMA_VERSION  # noqa: E402
from eval.answer_eval import (  # noqa: E402
    ANSWER_EVAL_VERSION,
    DEFAULT_BENCHMARK_DIR,
    DEFAULT_RETRIEVAL_RUN,
    DEFAULT_RETRIEVAL_SYSTEM,
    METRIC_KEYS,
    METRIC_LABELS,
    build_evidence,
    build_payload,
    evaluate_answer,
    load_corpus,
    load_retrieval_record,
    resolve_retrieval_run,
)
from eval.run_experiment import load_benchmark  # noqa: E402

AGENT_EVAL_VERSION = "agent_eval_v1"
SYSTEMS = ("fixed_rag", "agent_single", "agent_multi")
AGENT_SYSTEMS = ("agent_single", "agent_multi")
EVIDENCE_BUDGET = 6
DEFAULT_TOP_K = 20
SUPPLEMENT_TOP_K = 12
MAX_POOL = 32
DECISION_MAX_TOKENS = 1600
ANSWER_MAX_TOKENS = 4096
TERMINAL_STATUSES = frozenset({"ok", "failed"})
MAX_ERROR_CHARS = 500
MAX_PAYLOAD_CHARS = 70000  # mirrors run_agent.MAX_PAYLOAD_CHARS

SYSTEM_CONFIG = {
    "agent_single": {"max_rounds": 1, "max_calls": 2},
    "agent_multi": {"max_rounds": 3, "max_calls": 4},
}

DISCLAIMER = (
    "Citation-level evaluation only: schema_success, quote_integrity, "
    "cited_gold_parent_recall and gold_point_citation_recall measure JSON "
    "validity, exact quote copying and whether gold parents/quotes were "
    "present in cited evidence. They are NOT answer accuracy, correctness, "
    "completeness or faithfulness. No LLM judge and no human review are used. "
    "Harness-generated abstentions (no answer-model call) are excluded from "
    "schema/quote means. fixed_rag consumes a precomputed retrieval run; agent "
    "systems retrieve locally. Both use the same six-unit evidence budget and "
    "answer prompt."
)

AnswerFn = Callable[[list[dict]], "str | dict"]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_error(value: object) -> str:
    return " ".join(str(value).split())[:MAX_ERROR_CHARS]


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


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


def _reset_usage(target: object) -> None:
    """Clear a backend's ``last_usage`` slot before a new call.

    ``last_usage`` describes one call, not a backend lifetime. Without this
    reset, a call that raises before the provider returns usage would inherit
    the previous question's token count.
    """
    try:
        setattr(target, "last_usage", None)
    except (AttributeError, TypeError):
        pass


def _messages_too_large(messages: Sequence[Mapping]) -> bool:
    return sum(len(str(message.get("content") or "")) for message in messages) > MAX_PAYLOAD_CHARS


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    cleaned = sorted(float(value) for value in values if _is_number(value))
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    position = max(0.0, min(1.0, quantile)) * (len(cleaned) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return cleaned[lower]
    fraction = position - lower
    return cleaned[lower] * (1.0 - fraction) + cleaned[upper] * fraction


def _mean(values: Sequence[float]) -> float | None:
    cleaned = [float(value) for value in values if _is_number(value)]
    return sum(cleaned) / len(cleaned) if cleaned else None


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


def _sha256_file(path: Path) -> str | None:
    try:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
        return hasher.hexdigest()
    except OSError:
        return None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_decision_sha256() -> str:
    try:
        source = inspect.getsource(validate_decision)
    except (OSError, TypeError):
        source = repr(validate_decision.__code__.co_code)
    return _sha256_text(source)


VALIDATE_DECISION_SHA256 = _validate_decision_sha256()


def _benchmark_fingerprint(benchmark_dir: Path) -> dict:
    """Hash the frozen benchmark's question/qrels/index identity.

    The index manifest is preferred because it is small and pins the corpus,
    vector store and index hashes. A synthetic test benchmark without a
    manifest falls back to hashing ``corpus_clean.json`` directly.
    """
    index_dir = benchmark_dir / "index"
    files = {
        "questions_sha256": _sha256_file(benchmark_dir / "questions.jsonl"),
        "qrels_sha256": _sha256_file(benchmark_dir / "qrels.jsonl"),
        "index_manifest_sha256": _sha256_file(index_dir / "manifest.json"),
        # Hash the actual files too: a stale manifest must not hide a changed
        # corpus or vector store.
        "corpus_clean_sha256": _sha256_file(index_dir / "corpus_clean.json"),
        "vector_store_sha256": _sha256_file(index_dir / "vector_store.npz"),
    }
    material = {"version": AGENT_EVAL_VERSION, **files}
    files["fingerprint"] = hashlib.sha256(
        json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return files


def _agent_index_files(rag_dir: Path) -> dict:
    return {
        "corpus_clean_sha256": _sha256_file(rag_dir / "corpus_clean.json"),
        "vector_store_sha256": _sha256_file(rag_dir / "vector_store.npz"),
        "manifest_sha256": _sha256_file(rag_dir / "manifest.json"),
    }


def _validate_agent_index(benchmark_dir: Path, rag_dir: Path) -> None:
    """Refuse an agent index whose clean corpus differs from the benchmark's.

    Fixed evidence and qrels gold parent IDs are built from
    ``<benchmark-dir>/index``. If ``--rag-dir`` points at another corpus, the
    agent's cited parent IDs cannot be compared with the same gold set; fail
    instead of producing a plausible-looking but invalid A/B.
    """
    expected = _sha256_file(benchmark_dir / "index" / "corpus_clean.json")
    actual = _sha256_file(rag_dir / "corpus_clean.json")
    if expected and actual and expected != actual:
        raise ValueError(
            "--rag-dir corpus_clean.json does not match <benchmark-dir>/index; "
            "fixed evidence and agent evidence would use different corpora"
        )


def _run_signature(
    *,
    system: str,
    question: Mapping,
    qrels_record: Mapping | None,
    answer_model: str | None,
    decision_model: str | None,
    retrieval_label: str | None,
    retrieval_record_sha256: str | None,
    tools_corpus_sha256: str | None,
    agent_index: Mapping | None,
    benchmark_fingerprint: str,
    paper: str | None,
) -> str:
    """Stable signature for one (system, question) checkpoint.

    A resumed checkpoint is reused only when every input that can change its
    answer or metrics matches: models, prompts, benchmark/index, gold data,
    question text, retrieval record, evidence budget and call limits.
    """
    material = {
        "signature_version": "agent_eval_signature_v1",
        "eval_version": AGENT_EVAL_VERSION,
        "system": system,
        "question_id": _question_id(question),
        "question_text": _question_text(question),
        "gold_parent_ids": _gold_parent_ids(qrels_record),
        "gold_points": _gold_points(question),
        "answer_model": answer_model,
        "decision_model": decision_model if system in AGENT_SYSTEMS else None,
        "retrieval_label": retrieval_label if system == "fixed_rag" else None,
        "retrieval_record_sha256": (
            retrieval_record_sha256 if system == "fixed_rag" else None
        ),
        "tools_corpus_sha256": (
            tools_corpus_sha256 if system in AGENT_SYSTEMS else None
        ),
        "agent_index": (
            dict(agent_index or {}) if system in AGENT_SYSTEMS else None
        ),
        "retrieval_config": (
            {
                "initial_top_k": DEFAULT_TOP_K,
                "supplement_top_k": SUPPLEMENT_TOP_K,
                "max_pool": MAX_POOL,
                "max_payload_chars": MAX_PAYLOAD_CHARS,
                "initial_query": "question_zh",
            }
            if system in AGENT_SYSTEMS
            else None
        ),
        "paper": paper,
        "evidence_budget": EVIDENCE_BUDGET,
        "answer_prompt_sha256": _sha256_text(ANSWER_PROMPT),
        "decision_prompt_sha256": (
            _sha256_text(DECISION_PROMPT) if system in AGENT_SYSTEMS else None
        ),
        "answer_eval_version": ANSWER_EVAL_VERSION,
        "answer_schema_version": ANSWER_SCHEMA_VERSION,
        "validate_decision_sha256": VALIDATE_DECISION_SHA256,
        "system_config": SYSTEM_CONFIG.get(system),
        "answer_max_tokens": ANSWER_MAX_TOKENS,
        "decision_max_tokens": (
            DECISION_MAX_TOKENS if system in AGENT_SYSTEMS else None
        ),
        "benchmark_fingerprint": benchmark_fingerprint,
    }
    return hashlib.sha256(
        json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _question_id(question: Mapping) -> str:
    value = (
        question.get("question_id")
        or question.get("qid")
        or question.get("id")
        or ""
    )
    value = str(value).strip()
    if not value:
        raise ValueError("question without an id")
    return value


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
    return list(dict.fromkeys(
        item.strip() for item in value
        if isinstance(item, str) and item.strip()
    ))


def _gold_points(question: Mapping) -> list[dict]:
    value = question.get("gold_points")
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


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


def _empty_metrics(gold_parent_count: int, gold_point_count: int) -> dict:
    return {
        "schema_success": 0.0,
        "quote_integrity": 0.0,
        "cited_gold_parent_recall": 0.0 if gold_parent_count else None,
        "gold_point_citation_recall": 0.0 if gold_point_count else None,
    }


def _as_harness_abstention(evaluation: Mapping) -> dict:
    """Mark a harness-synthesized abstention (no answer-model call).

    The agent may validly finish with no selected evidence. In that case the
    harness writes a minimal ``insufficient`` answer itself; scoring that as a
    schema/quote success would inflate the agent relative to fixed_rag, which
    always calls the answer model. Those two metrics become null (excluded
    from their means); status and citation metrics remain in the denominator.
    """
    marked = dict(evaluation)
    marked["metrics"] = {
        **dict(evaluation.get("metrics") or {}),
        "schema_success": None,
        "quote_integrity": None,
    }
    return marked


def _normalize_systems(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise ValueError("systems must be a comma-separated string or a sequence")
    if not items:
        raise ValueError("at least one system is required")
    unknown = [item for item in items if item not in SYSTEMS]
    if unknown:
        raise ValueError(
            f"unknown system(s): {', '.join(unknown)}; expected one of "
            f"{', '.join(SYSTEMS)}"
        )
    return list(dict.fromkeys(items))


# ---------------------------------------------------------------------------
# Shared answer construction
# ---------------------------------------------------------------------------


def _answer_messages(question_text: str, evidence: Mapping[str, Mapping]) -> list[dict]:
    """Exactly ``answer_question.main``'s messages, for fixed and agent alike."""
    return [
        {"role": "system", "content": ANSWER_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                build_payload(question_text, evidence), ensure_ascii=False
            ),
        },
    ]


def _evidence_metadata(evidence: Mapping[str, Mapping]) -> list[dict]:
    return [
        {
            "id": unit["id"],
            "parent_chunk_id": unit["parent_chunk_id"],
            "title": unit.get("title") or "",
            "page_number": unit.get("page_number"),
        }
        for unit in evidence.values()
    ]


def _ok_record(
    *,
    question_id: str,
    system: str,
    raw_response: object | None,
    evaluation: Mapping,
    evidence: Mapping[str, Mapping],
    gold_parent_ids: Sequence[str],
    gold_points: Sequence[Mapping],
    tokens: int,
    latency_ms: float,
    answer_latency_ms: float | None,
    retrieval_latency_ms: float | None,
    tool_calls: int,
    retrieval_rounds: int,
    stop_reason: str,
    model: str | None,
    decision_model: str | None,
    usage: object = None,
    selected_ids: Sequence[str] = (),
    candidate_ids: Sequence[str] = (),
    trace: Sequence[Mapping] = (),
    attempt: int = 1,
    answer_source: str | None = "model",
) -> dict:
    return {
        "question_id": question_id,
        "system": system,
        "status": "ok",
        "answer_status": evaluation["answer_status"],
        "answer_source": answer_source,
        "error": evaluation.get("error"),
        "model": model,
        "decision_model": decision_model,
        "raw_response": raw_response,
        "usage": _safe_usage(usage),
        "tokens": int(tokens),
        "latency_ms": round(float(latency_ms), 3),
        "answer_latency_ms": (
            round(float(answer_latency_ms), 3)
            if _is_number(answer_latency_ms)
            else None
        ),
        "retrieval_latency_ms": (
            round(float(retrieval_latency_ms), 3)
            if _is_number(retrieval_latency_ms)
            else None
        ),
        "tool_calls": int(tool_calls),
        "retrieval_rounds": int(retrieval_rounds),
        "stop_reason": stop_reason,
        "retrieved_parent_ids": [
            str(unit["parent_chunk_id"]) for unit in evidence.values()
        ],
        "selected_evidence_ids": list(selected_ids),
        "candidate_ids": list(candidate_ids),
        "evidence": _evidence_metadata(evidence),
        "cited_evidence_ids": evaluation["cited_evidence_ids"],
        "cited_parent_ids": evaluation["cited_parent_ids"],
        "gold_parent_ids": list(gold_parent_ids),
        "gold_point_count": len(gold_points),
        "metrics": evaluation["metrics"],
        "trace": [dict(item) for item in trace],
        "attempt": attempt,
        "finished_at": _now(),
    }


def _failed_record(
    *,
    question_id: str,
    system: str,
    error: str,
    gold_parent_ids: Sequence[str],
    gold_points: Sequence[Mapping],
    stop_reason: str,
    tokens: int = 0,
    latency_ms: float = 0.0,
    answer_latency_ms: float | None = None,
    retrieval_latency_ms: float | None = None,
    tool_calls: int = 0,
    retrieval_rounds: int = 0,
    model: str | None = None,
    decision_model: str | None = None,
    selected_ids: Sequence[str] = (),
    candidate_ids: Sequence[str] = (),
    trace: Sequence[Mapping] = (),
    attempt: int = 1,
    answer_source: str | None = None,
) -> dict:
    return {
        "question_id": question_id,
        "system": system,
        "status": "failed",
        "answer_status": "invalid",
        "answer_source": answer_source,
        "error": _clean_error(error),
        "model": model,
        "decision_model": decision_model,
        "raw_response": None,
        "usage": None,
        "tokens": int(tokens),
        "latency_ms": round(float(latency_ms), 3),
        "answer_latency_ms": (
            round(float(answer_latency_ms), 3)
            if _is_number(answer_latency_ms)
            else None
        ),
        "retrieval_latency_ms": (
            round(float(retrieval_latency_ms), 3)
            if _is_number(retrieval_latency_ms)
            else None
        ),
        "tool_calls": int(tool_calls),
        "retrieval_rounds": int(retrieval_rounds),
        "stop_reason": stop_reason,
        "retrieved_parent_ids": [],
        "selected_evidence_ids": list(selected_ids),
        "candidate_ids": list(candidate_ids),
        "evidence": [],
        "cited_evidence_ids": [],
        "cited_parent_ids": [],
        "gold_parent_ids": list(gold_parent_ids),
        "gold_point_count": len(gold_points),
        "metrics": _empty_metrics(len(gold_parent_ids), len(gold_points)),
        "trace": [dict(item) for item in trace],
        "attempt": attempt,
        "finished_at": _now(),
    }


# ---------------------------------------------------------------------------
# fixed_rag control
# ---------------------------------------------------------------------------


def _run_fixed(
    *,
    question: Mapping,
    qrels_record: Mapping | None,
    run_dir: Path,
    chunks: Mapping[str, Mapping],
    answer_fn: AnswerFn,
    attempt: int = 1,
) -> dict:
    question_id = _question_id(question)
    gold_ids = _gold_parent_ids(qrels_record)
    gold_points = _gold_points(question)
    model = getattr(answer_fn, "model", None)
    model = str(model) if model else None

    try:
        _record, retrieved, retrieval_status, retrieval_error = load_retrieval_record(
            run_dir, question_id
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _failed_record(
            question_id=question_id,
            system="fixed_rag",
            error=f"retrieval record unavailable: {type(exc).__name__}: {exc}",
            gold_parent_ids=gold_ids,
            gold_points=gold_points,
            stop_reason="retrieval_unavailable",
            retrieval_latency_ms=0.0,
            model=model,
            attempt=attempt,
        )

    if retrieval_status != "ok" or retrieval_error:
        detail = retrieval_error or f"status {retrieval_status}"
        return _failed_record(
            question_id=question_id,
            system="fixed_rag",
            error=f"retrieval run unavailable for {question_id}: {detail}",
            gold_parent_ids=gold_ids,
            gold_points=gold_points,
            stop_reason="retrieval_unavailable",
            retrieval_latency_ms=0.0,
            model=model,
            attempt=attempt,
        )

    retrieved = list(retrieved)[:EVIDENCE_BUDGET]
    try:
        evidence = build_evidence(retrieved, chunks)
        if not evidence:
            raise ValueError("retrieval record returned no evidence")
        messages = _answer_messages(_question_text(question), evidence)
    except ValueError as exc:
        return _failed_record(
            question_id=question_id,
            system="fixed_rag",
            error=f"evidence build failed: {exc}",
            gold_parent_ids=gold_ids,
            gold_points=gold_points,
            stop_reason="evidence_build_failed",
            retrieval_latency_ms=0.0,
            model=model,
            attempt=attempt,
        )

    if _messages_too_large(messages):
        return _failed_record(
            question_id=question_id,
            system="fixed_rag",
            error=f"answer payload exceeds {MAX_PAYLOAD_CHARS} characters; no call made",
            gold_parent_ids=gold_ids,
            gold_points=gold_points,
            stop_reason="payload_too_large",
            retrieval_latency_ms=0.0,
            model=model,
            attempt=attempt,
        )
    _reset_usage(answer_fn)
    started = time.perf_counter()
    try:
        raw = answer_fn(messages)
    except Exception as exc:  # noqa: BLE001 - a model crash is a failed question
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = getattr(answer_fn, "last_usage", None)
        return _failed_record(
            question_id=question_id,
            system="fixed_rag",
            error=f"{type(exc).__name__}: {exc}",
            gold_parent_ids=gold_ids,
            gold_points=gold_points,
            stop_reason="answer_error",
            tokens=_usage_tokens(usage),
            latency_ms=latency_ms,
            answer_latency_ms=latency_ms,
            retrieval_latency_ms=0.0,
            model=model,
            attempt=attempt,
        )

    latency_ms = (time.perf_counter() - started) * 1000.0
    usage = getattr(answer_fn, "last_usage", None)
    evaluation = evaluate_answer(raw, evidence, gold_ids, gold_points)
    return _ok_record(
        question_id=question_id,
        system="fixed_rag",
        raw_response=_safe_raw(raw),
        evaluation=evaluation,
        evidence=evidence,
        gold_parent_ids=gold_ids,
        gold_points=gold_points,
        tokens=_usage_tokens(usage),
        latency_ms=latency_ms,
        answer_latency_ms=latency_ms,
        retrieval_latency_ms=0.0,
        tool_calls=0,
        retrieval_rounds=0,
        stop_reason="precomputed_retrieval",
        model=model,
        decision_model=None,
        usage=usage,
        attempt=attempt,
    )


# ---------------------------------------------------------------------------
# Bounded agent systems
# ---------------------------------------------------------------------------


def _call_decision(backend: Callable, payload: Mapping) -> tuple[dict, int, str | None]:
    """Call an injected decision backend and parse its JSON decision."""
    _reset_usage(backend)
    try:
        payload_chars = len(DECISION_PROMPT) + len(
            json.dumps(payload, ensure_ascii=False)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"decision payload is not JSON-serializable: {exc}") from None
    if payload_chars > MAX_PAYLOAD_CHARS:
        raise ValueError(
            f"decision payload exceeds {MAX_PAYLOAD_CHARS} characters; no call made"
        )
    raw = backend(DECISION_PROMPT, payload, DECISION_MAX_TOKENS)
    usage = getattr(backend, "last_usage", None)
    if isinstance(raw, Mapping):
        raw_usage = raw.get("usage")
        if isinstance(raw_usage, Mapping):
            usage = raw_usage
        # Preserve this call's usage even if a later validation raises.
        try:
            setattr(backend, "last_usage", usage)
        except (AttributeError, TypeError):
            pass
    if not isinstance(raw, Mapping):
        raise ValueError("decision backend must return an object")
    if raw.get("finish_reason") != "stop":
        raise ValueError("decision output did not finish; no automatic retry")
    content = raw.get("content")
    if isinstance(content, str):
        value = json.loads(content)
    elif isinstance(content, Mapping):
        value = dict(content)
    else:
        raise ValueError("decision backend returned no JSON content")
    if not isinstance(value, Mapping):
        raise ValueError("decision content must be a JSON object")
    model = raw.get("model") or getattr(backend, "model", None)
    return dict(value), _usage_tokens(usage), str(model) if model else None


def _check_tools(tools: object) -> None:
    """Run the tool's index-change check when it exposes one."""
    checker = getattr(tools, "check_current", None)
    if callable(checker):
        checker()


def _register_hits(
    hits: Sequence[Mapping],
    tools: object,
    registry: dict[str, dict],
    chunk_ids: dict[str, str],
    paper: str | None = None,
) -> tuple[list[str], int]:
    """Validate tool hits against the index and assign stable ``C<n>`` ids."""
    canonical_map = getattr(tools, "canonical", None)
    keys: list[str] = []
    new = 0
    for hit in hits:
        if not isinstance(hit, Mapping):
            raise ValueError("tool returned a non-object hit")
        chunk_id = hit.get("chunk_id")
        if not isinstance(chunk_id, str):
            raise ValueError("tool result has no string chunk_id")
        canonical = canonical_map.get(chunk_id) if isinstance(canonical_map, Mapping) else None
        if canonical is None:
            raise ValueError("tool result is not in the current index")
        if any(hit.get(key) != value for key, value in canonical.items()):
            raise ValueError("tool result differs from the current index")
        if paper and canonical.get("arxiv_id") != paper:
            raise ValueError("tool result is outside the paper filter")
        if chunk_id not in chunk_ids:
            evidence_id = f"C{len(registry) + 1}"
            chunk_ids[chunk_id] = evidence_id
            registry[evidence_id] = dict(canonical)
            new += 1
        keys.append(chunk_ids[chunk_id])
    return keys, new


def _decision_payload(
    question_text: str,
    paper: str | None,
    can_search: bool,
    trace: Sequence[Mapping],
    pool: Sequence[str],
    registry: Mapping[str, Mapping],
) -> dict:
    """The same candidate payload the production agent sends. No gold data."""
    return {
        "question": question_text,
        "paper_filter": paper,
        "supplementary_search_allowed": bool(can_search),
        "history": [
            dict(item) for item in trace
            if item.get("kind") in ("retrieval", "decision")
        ],
        "candidates": [
            {
                "id": key,
                "title": registry[key].get("title") or "",
                "arxiv_id": registry[key].get("arxiv_id"),
                "page": registry[key].get("page_number"),
                "text": registry[key]["text"],
            }
            for key in pool
        ],
    }


def _selected_evidence(selected_ids: Sequence[str], registry: Mapping[str, Mapping]) -> dict[str, dict]:
    evidence: dict[str, dict] = {}
    for number, key in enumerate(list(selected_ids)[:EVIDENCE_BUDGET], start=1):
        unit = registry[key]
        evidence[f"E{number}"] = {
            "id": f"E{number}",
            "parent_chunk_id": unit.get("parent_chunk_id") or unit["chunk_id"],
            "title": unit.get("title") or "",
            "page_number": unit.get("page_number"),
            "text": unit["text"],
        }
    return evidence


def _run_agent(
    *,
    question: Mapping,
    qrels_record: Mapping | None,
    system: str,
    tools: object,
    backend: Callable,
    answer_fn: AnswerFn,
    max_rounds: int,
    max_calls: int,
    paper: str | None = None,
    attempt: int = 1,
) -> dict:
    question_id = _question_id(question)
    question_text = _question_text(question)
    gold_ids = _gold_parent_ids(qrels_record)
    gold_points = _gold_points(question)
    answer_model = getattr(answer_fn, "model", None)
    answer_model = str(answer_model) if answer_model else None
    started_all = time.perf_counter()
    retrieval_ms = 0.0
    answer_ms = 0.0
    answer_called = False
    tokens = 0
    tool_calls = 0
    decision_rounds = 0
    decision_model: str | None = None
    trace: list[dict] = []

    registry: dict[str, dict] = {}
    chunk_ids: dict[str, str] = {}
    pool: list[str] = []
    history: set[tuple[str, str]] = set()
    decision: dict | None = None
    stop_reason: str | None = None

    def fail(error: str, reason: str) -> dict:
        latency = (time.perf_counter() - started_all) * 1000.0
        return _failed_record(
            question_id=question_id,
            system=system,
            error=error,
            gold_parent_ids=gold_ids,
            gold_points=gold_points,
            stop_reason=reason,
            tokens=tokens,
            latency_ms=latency,
            answer_latency_ms=answer_ms if answer_called else None,
            retrieval_latency_ms=retrieval_ms,
            tool_calls=tool_calls,
            retrieval_rounds=decision_rounds,
            model=answer_model,
            decision_model=decision_model,
            selected_ids=list(decision.get("selected_ids") or []) if decision else [],
            candidate_ids=list(chunk_ids.values()),
            trace=trace,
            attempt=attempt,
        )

    # Round 1 retrieval is deterministic dense top-20.
    retrieval_started = time.perf_counter()
    try:
        hits = tools.search("search_dense", question_text, DEFAULT_TOP_K)
    except Exception as exc:  # noqa: BLE001 - a tool failure is a stopped run
        retrieval_ms += (time.perf_counter() - retrieval_started) * 1000.0
        try:
            _check_tools(tools)
        except Exception as check_exc:  # noqa: BLE001
            return fail(
                f"index changed: {type(check_exc).__name__}: {check_exc}",
                "index_changed",
            )
        return fail(f"{type(exc).__name__}: {exc}", "tool_error")
    retrieval_ms += (time.perf_counter() - retrieval_started) * 1000.0
    tool_calls += 1
    try:
        keys, new = _register_hits(hits, tools, registry, chunk_ids, paper=paper)
        pool = list(dict.fromkeys(keys))[:MAX_POOL]
    except ValueError as exc:
        return fail(f"tool result rejected: {exc}", "tool_error")
    trace.append({
        "kind": "retrieval", "round": 1, "tool": "search_dense",
        "query": question_text, "candidate_count": len(keys),
        "new_count": new,
        "sources": [
            {
                "id": key,
                "paper": registry[key].get("arxiv_id"),
                "page": registry[key].get("page_number"),
            }
            for key in keys
        ],
    })
    history.add(("search_dense", normalized_query(question_text)))

    if not pool:
        stop_reason = "empty_retrieval"
        raw = {
            "status": "insufficient",
            "points": [],
            "limitations": ["未检索到可用的本地原文证据。"],
        }
        evaluation = _as_harness_abstention(
            evaluate_answer(raw, {}, gold_ids, gold_points)
        )
        latency = (time.perf_counter() - started_all) * 1000.0
        return _ok_record(
            question_id=question_id,
            system=system,
            raw_response=_safe_raw(raw),
            evaluation=evaluation,
            evidence={},
            answer_source="harness_abstention",
            gold_parent_ids=gold_ids,
            gold_points=gold_points,
            tokens=tokens,
            latency_ms=latency,
            answer_latency_ms=answer_ms if answer_called else None,
            retrieval_latency_ms=retrieval_ms,
            tool_calls=tool_calls,
            retrieval_rounds=decision_rounds,
            stop_reason=stop_reason,
            model=answer_model,
            decision_model=decision_model,
            candidate_ids=list(chunk_ids.values()),
            trace=trace,
            attempt=attempt,
        )

    for step in range(1, max_rounds + 1):
        # Reserve the final answer call.
        if decision_rounds >= max_calls - 1:
            stop_reason = "call_budget"
            break
        can_search = step < max_rounds and decision_rounds + 2 < max_calls
        payload = _decision_payload(
            question_text, paper, can_search, trace, pool, registry
        )
        try:
            _check_tools(tools)
        except Exception as exc:  # noqa: BLE001
            return fail(f"index changed: {type(exc).__name__}: {exc}", "index_changed")
        try:
            value, call_tokens, call_model = _call_decision(backend, payload)
        except Exception as exc:  # noqa: BLE001
            tokens += _usage_tokens(getattr(backend, "last_usage", None))
            return fail(f"{type(exc).__name__}: {exc}", "decision_error")
        tokens += call_tokens
        decision_model = call_model or decision_model
        try:
            decision = validate_decision(value, set(pool))
        except ValueError as exc:
            return fail(f"decision rejected: {exc}", "invalid_decision")
        decision_rounds += 1
        trace.append({
            "kind": "decision",
            "round": step,
            "assessment": decision["assessment"],
            "selected_ids": list(decision["selected_ids"]),
            "action": decision["action"],
            "query": decision["query"],
            "missing": list(decision["missing"]),
            "reason": decision["reason"],
        })

        if decision["action"] == "finish":
            stop_reason = (
                "sufficient" if decision["assessment"] == "sufficient"
                else "model_stopped_with_gaps"
            )
            break
        if not can_search:
            stop_reason = "round_budget" if step >= max_rounds else "call_budget"
            break
        key = (decision["action"], normalized_query(decision["query"]))
        if key in history:
            stop_reason = "repeated_query"
            break

        retrieval_started = time.perf_counter()
        try:
            hits = tools.search(
                decision["action"], decision["query"], SUPPLEMENT_TOP_K
            )
        except Exception as exc:  # noqa: BLE001 - keep prior selected evidence
            retrieval_ms += (time.perf_counter() - retrieval_started) * 1000.0
            try:
                _check_tools(tools)
            except Exception as check_exc:  # noqa: BLE001
                return fail(
                    f"index changed: {type(check_exc).__name__}: {check_exc}",
                    "index_changed",
                )
            stop_reason = "tool_error"
            trace.append({
                "kind": "tool_error", "round": step + 1,
                "tool": decision["action"], "error": _clean_error(
                    f"{type(exc).__name__}: {exc}"
                ),
            })
            break
        retrieval_ms += (time.perf_counter() - retrieval_started) * 1000.0
        tool_calls += 1
        history.add(key)
        try:
            keys, new = _register_hits(hits, tools, registry, chunk_ids, paper=paper)
        except ValueError as exc:
            return fail(f"tool result rejected: {exc}", "tool_error")
        pool = list(dict.fromkeys([
            *decision["selected_ids"], *keys, *pool,
        ]))[:MAX_POOL]
        trace.append({
            "kind": "retrieval", "round": step + 1, "tool": decision["action"],
            "query": decision["query"], "candidate_count": len(keys),
            "new_count": new,
            "sources": [
                {
                    "id": key,
                    "paper": registry[key].get("arxiv_id"),
                    "page": registry[key].get("page_number"),
                }
                for key in keys
            ],
        })
        if not new:
            stop_reason = "no_new_evidence"
            break

    if stop_reason is None:
        stop_reason = "round_budget"

    selected = list(decision.get("selected_ids") or []) if decision else []
    selected = list(dict.fromkeys(selected))[:EVIDENCE_BUDGET]
    evidence = _selected_evidence(selected, registry)
    answer_source = "model"

    if evidence:
        messages = _answer_messages(question_text, evidence)
        if _messages_too_large(messages):
            return fail(
                f"answer payload exceeds {MAX_PAYLOAD_CHARS} characters; no call made",
                "payload_too_large",
            )
        try:
            _check_tools(tools)
        except Exception as exc:  # noqa: BLE001
            return fail(f"index changed: {type(exc).__name__}: {exc}", "index_changed")
        answer_called = True
        _reset_usage(answer_fn)
        answer_started = time.perf_counter()
        try:
            raw = answer_fn(messages)
        except Exception as exc:  # noqa: BLE001
            answer_ms += (time.perf_counter() - answer_started) * 1000.0
            tokens += _usage_tokens(getattr(answer_fn, "last_usage", None))
            return fail(f"{type(exc).__name__}: {exc}", "answer_error")
        answer_ms += (time.perf_counter() - answer_started) * 1000.0
        usage = getattr(answer_fn, "last_usage", None)
        tokens += _usage_tokens(usage)
        try:
            _check_tools(tools)
        except Exception as exc:  # noqa: BLE001
            return fail(f"index changed: {type(exc).__name__}: {exc}", "index_changed")
    else:
        usage = None
        answer_source = "harness_abstention"
        limitations = (
            list(decision.get("missing") or []) if decision
            else ["没有选中的证据。"]
        )
        raw = {
            "status": "insufficient",
            "points": [],
            "limitations": limitations or ["没有选中的证据。"],
        }

    evaluation = evaluate_answer(raw, evidence, gold_ids, gold_points)
    if answer_source == "harness_abstention":
        evaluation = _as_harness_abstention(evaluation)
    latency = (time.perf_counter() - started_all) * 1000.0
    return _ok_record(
        question_id=question_id,
        system=system,
        raw_response=_safe_raw(raw),
        evaluation=evaluation,
        evidence=evidence,
        gold_parent_ids=gold_ids,
        gold_points=gold_points,
        tokens=tokens,
        latency_ms=latency,
        answer_latency_ms=answer_ms if answer_called else None,
        retrieval_latency_ms=retrieval_ms,
        tool_calls=tool_calls,
        retrieval_rounds=decision_rounds,
        stop_reason=stop_reason,
        model=answer_model,
        decision_model=decision_model,
        usage=usage,
        selected_ids=selected,
        candidate_ids=list(chunk_ids.values()),
        trace=trace,
        attempt=attempt,
        answer_source=answer_source,
    )


# ---------------------------------------------------------------------------
# Aggregation and reports
# ---------------------------------------------------------------------------


def _metric_summary(records: Sequence[Mapping]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for key in METRIC_KEYS:
        values = [
            float(record["metrics"][key])
            for record in records
            if isinstance(record.get("metrics"), Mapping)
            and _is_number(record["metrics"].get(key))
        ]
        summary[key] = {
            "mean": (sum(values) / len(values)) if values else None,
            "n": len(values),
            "label": METRIC_LABELS[key],
        }
    return summary


def _is_schema_failure(record: Mapping) -> bool:
    value = (record.get("metrics") or {}).get("schema_success")
    return _is_number(value) and float(value) != 1.0


def _aggregate(
    records: Sequence[Mapping],
    *,
    system: str,
    benchmark_dir: Path,
    retrieval_run: Path | None,
    retrieval_system: str | None,
    answer_model: str | None,
    benchmark_fingerprint: Mapping | None = None,
    agent_index: Mapping | None = None,
) -> dict:
    failures = [
        str(record.get("question_id"))
        for record in records
        if record.get("status") != "ok" or _is_schema_failure(record)
    ]
    status_counts = {"answered": 0, "partial": 0, "insufficient": 0, "invalid": 0}
    stop_reason_counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("answer_status") or "invalid")
        status_counts[status] = status_counts.get(status, 0) + 1
        reason = str(record.get("stop_reason") or "unknown")
        stop_reason_counts[reason] = stop_reason_counts.get(reason, 0) + 1

    tool_calls = [int(record.get("tool_calls", 0) or 0) for record in records]
    rounds = [int(record.get("retrieval_rounds", 0) or 0) for record in records]
    tokens = [max(0, int(record.get("tokens", 0) or 0)) for record in records]
    latencies = [
        float(record["latency_ms"]) for record in records
        if _is_number(record.get("latency_ms"))
    ]
    answer_latencies = [
        float(record["answer_latency_ms"]) for record in records
        if _is_number(record.get("answer_latency_ms"))
    ]
    retrieval_latencies = [
        float(record["retrieval_latency_ms"]) for record in records
        if _is_number(record.get("retrieval_latency_ms"))
    ]
    n = len(records)

    return {
        "version": AGENT_EVAL_VERSION,
        "generated_by": "eval/agent_eval.py",
        "created_at": _now(),
        "system": system,
        "benchmark_dir": str(benchmark_dir),
        "retrieval_run": str(retrieval_run) if retrieval_run is not None else None,
        "retrieval_system": retrieval_system,
        "answer_model": answer_model,
        "benchmark_fingerprint": (benchmark_fingerprint or {}).get("fingerprint"),
        "benchmark_files": dict(benchmark_fingerprint or {}),
        "agent_index": dict(agent_index) if agent_index is not None else None,
        "disclaimer": DISCLAIMER,
        "n": n,
        "n_failed": len(failures),
        "failures": failures,
        "status_counts": status_counts,
        "stop_reason_counts": stop_reason_counts,
        "metrics": _metric_summary(records),
        "tool_calls": {
            "total": sum(tool_calls),
            "mean": (sum(tool_calls) / n) if n else 0.0,
            "max": max(tool_calls) if tool_calls else 0,
        },
        "retrieval_rounds": {
            "total": sum(rounds),
            "mean": (sum(rounds) / n) if n else 0.0,
            "max": max(rounds) if rounds else 0,
        },
        "tokens": {
            "total": sum(tokens),
            "mean": (sum(tokens) / n) if n else 0.0,
        },
        "latency_ms": {
            "n": len(latencies),
            "mean": _mean(latencies) or 0.0,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "answer_latency_ms": {
            "n": len(answer_latencies),
            "mean": _mean(answer_latencies) or 0.0,
            "p50": _percentile(answer_latencies, 0.50),
            "p95": _percentile(answer_latencies, 0.95),
        },
        "retrieval_latency_ms": {
            "n": len(retrieval_latencies),
            "mean": _mean(retrieval_latencies) or 0.0,
            "p50": _percentile(retrieval_latencies, 0.50),
            "p95": _percentile(retrieval_latencies, 0.95),
        },
        "per_question": [
            {
                "question_id": record.get("question_id"),
                "status": record.get("status"),
                "answer_status": record.get("answer_status"),
                "answer_source": record.get("answer_source"),
                "stop_reason": record.get("stop_reason"),
                "error": record.get("error"),
                "tool_calls": record.get("tool_calls", 0),
                "retrieval_rounds": record.get("retrieval_rounds", 0),
                "tokens": record.get("tokens", 0),
                "latency_ms": record.get("latency_ms", 0.0),
                "answer_latency_ms": record.get("answer_latency_ms", 0.0),
                "retrieval_latency_ms": record.get("retrieval_latency_ms", 0.0),
                "metrics": record.get("metrics", {}),
            }
            for record in records
        ],
        "notes": [
            "Failed or schema-invalid answers stay in the denominator and score 0.",
            "Empty-gold questions are excluded only from ratio metrics; schema_success keeps them.",
            "tool_calls counts successful search-tool invocations; fixed_rag is 0 (precomputed).",
            "retrieval_rounds counts completed agent coverage-decision rounds; fixed_rag is 0.",
            "latency_ms is whole-question wall time; answer_latency_ms isolates the answer call.",
        ],
    }


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_metric(value: object) -> str:
    return f"{float(value):.3f}" if _is_number(value) else "n/a"


def _format_mean(metric: Mapping | None) -> str:
    if not isinstance(metric, Mapping) or not _is_number(metric.get("mean")):
        return "n/a"
    return f"{float(metric['mean']):.3f} (n={int(metric.get('n', 0))})"


def _render_system_report(aggregate: Mapping) -> str:
    metrics = aggregate.get("metrics") or {}
    lines = [
        f"# PaperLens Agent A/B — {aggregate['system']}",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "## Configuration",
        "",
        f"- System: `{aggregate['system']}`",
        f"- Benchmark: `{aggregate['benchmark_dir']}`",
        f"- Retrieval run: `{aggregate.get('retrieval_run') or 'n/a'}`",
        f"- Retrieval system: `{aggregate.get('retrieval_system') or 'n/a'}`",
        f"- Benchmark fingerprint: `{aggregate.get('benchmark_fingerprint') or 'n/a'}`",
        f"- Agent index: `{((aggregate.get('agent_index') or {}).get('dir') or ('injected tools' if (aggregate.get('agent_index') or {}).get('injected_tools') else 'n/a'))}`",
        f"- Answer model: `{aggregate.get('answer_model') or 'injected answer_fn'}`",
        f"- Questions: {aggregate['n']} (failed or invalid: {aggregate['n_failed']})",
        f"- Evidence budget: {EVIDENCE_BUDGET}",
        "",
        "## Metrics",
        "",
        "| Metric | Mean (n) |",
        "|---|---|",
    ]
    for key in METRIC_KEYS:
        lines.append(f"| {METRIC_LABELS[key]} | {_format_mean(metrics.get(key))} |")

    lines.extend([
        "",
        "## Behavior",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Tool calls (total / mean) | {aggregate['tool_calls']['total']} / "
        f"{float(aggregate['tool_calls']['mean']):.2f} |",
        f"| Retrieval rounds (total / mean) | {aggregate['retrieval_rounds']['total']} / "
        f"{float(aggregate['retrieval_rounds']['mean']):.2f} |",
        f"| Tokens (total / mean) | {aggregate['tokens']['total']} / "
        f"{float(aggregate['tokens']['mean']):.1f} |",
        f"| Latency ms (mean / p50 / p95) | "
        f"{float(aggregate['latency_ms']['mean']):.1f} / "
        f"{_format_metric(aggregate['latency_ms']['p50'])} / "
        f"{_format_metric(aggregate['latency_ms']['p95'])} |",
        f"| Answer latency ms (mean, n) | "
        f"{float(aggregate['answer_latency_ms']['mean']):.1f} "
        f"(n={aggregate['answer_latency_ms']['n']}) |",
        f"| Retrieval latency ms (mean, n) | "
        f"{float(aggregate['retrieval_latency_ms']['mean']):.1f} "
        f"(n={aggregate['retrieval_latency_ms']['n']}) |",
        "",
        "## Answer status",
        "",
        "| Status | Count |",
        "|---|---|",
    ])
    for status, count in (aggregate.get("status_counts") or {}).items():
        lines.append(f"| {_md_cell(status)} | {count} |")
    lines.extend(["", "## Stop reasons", "", "| Reason | Count |", "|---|---|"])
    for reason, count in sorted((aggregate.get("stop_reason_counts") or {}).items()):
        lines.append(f"| {_md_cell(reason)} | {count} |")

    failures = aggregate.get("failures") or []
    lines.extend(["", "## Failures", ""])
    if failures:
        lines.extend(f"- `{_md_cell(item)}`" for item in failures)
    else:
        lines.append("None.")

    lines.extend([
        "",
        "## Per question",
        "",
        "| Question | Pipeline | Answer | Stop reason | Schema | Quote | "
        "Cited gold | Gold points | Tools | Rounds | Tokens | Latency ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ])
    for item in aggregate.get("per_question") or []:
        item_metrics = item.get("metrics") or {}
        lines.append(
            "| {q} | {pipe} | {answer} | {stop} | {schema} | {quote} | {cited} | "
            "{points} | {tools} | {rounds} | {tokens} | {latency} |".format(
                q=_md_cell(item.get("question_id")),
                pipe=_md_cell(item.get("status")),
                answer=_md_cell(item.get("answer_status")),
                stop=_md_cell(item.get("stop_reason")),
                schema=_format_metric(item_metrics.get("schema_success")),
                quote=_format_metric(item_metrics.get("quote_integrity")),
                cited=_format_metric(item_metrics.get("cited_gold_parent_recall")),
                points=_format_metric(item_metrics.get("gold_point_citation_recall")),
                tools=item.get("tool_calls", 0),
                rounds=item.get("retrieval_rounds", 0),
                tokens=item.get("tokens", 0),
                latency=f"{float(item.get('latency_ms') or 0.0):.1f}",
            )
        )

    lines.extend([
        "",
        "## Limitations",
        "",
        "- Citation-level scoring only; not answer accuracy or faithfulness.",
        "- fixed_rag retrieval comes from a precomputed run, so its retrieval latency is 0 and its tool calls are 0; fixed `latency_ms` is answer-call time, while agent `latency_ms` is whole-question.",
        "- Per-stage answer/retrieval latency means use only records where that stage ran; the report shows n.",
        "- Agent retrieval returns indexed child passages; fixed evidence uses parent chunks. Text granularity differs by pipeline.",
        "- Agent answers are scored exactly as returned; production's coverage-demotion/limitation merging is not applied, so fixed and agent share one scoring path.",
        "- `quote_integrity` is not independent of `schema_success`: the shared validator already rejects non-substring quotes, so the two agree by construction.",
        "- A per-question failure (invalid decision, answer schema failure) scores 0 and stays in the denominator.",
        "- Harness-generated abstentions (no selected evidence, no answer-model call) are marked `answer_source=harness_abstention`; their schema/quote values are null and excluded from those means, while status and citation metrics remain.",
        "- The fixed control's precomputed run used the benchmark `query_en`; agent searches use `question_zh`, matching the production agent. Query language and retrieval method differ by design.",
        "- `tokens` counts decision + answer calls only; the fixed control's precomputed rerank cost is excluded, so this is not an end-to-end cost comparison.",
        "- Agent answer calls use `answer_question.PROMPT` only; production's `ANSWER_EXTRA`/coverage payload is not applied.",
        "- `gold_point_citation_recall` is bounded by qrels coverage: 13/145 v2 gold points do not appear in the frozen corpus.",
        "- No paired bootstrap CI or significance test is computed; n is descriptive.",
        "- Error handling is eval-specific: a failed question is scored 0 and the run continues; a supplemental tool error keeps prior evidence and still answers, whereas production aborts the run.",
        "",
    ])
    return "\n".join(lines)


def _render_combined_report(summary: Mapping) -> str:
    lines = [
        "# PaperLens Agent A/B Evaluation",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "## Configuration",
        "",
        f"- Benchmark: `{summary['benchmark_dir']}`",
        f"- Retrieval run: `{summary.get('retrieval_run') or 'n/a'}`",
        f"- Retrieval system: `{summary.get('retrieval_system') or 'n/a'}`",
        f"- Benchmark fingerprint: `{summary.get('benchmark_fingerprint') or 'n/a'}`",
        f"- Agent index: `{((summary.get('agent_index') or {}).get('dir') or ('injected tools' if (summary.get('agent_index') or {}).get('injected_tools') else 'n/a'))}`",
        f"- Evidence budget: {EVIDENCE_BUDGET}",
        f"- Systems: {', '.join(summary['systems'])}",
        "",
        "## Comparison",
        "",
        "| System | n | Failed | Schema | Quote | Cited gold | Gold points | "
        "Tool calls | Rounds | Tokens | Latency ms (fixed=answer-only) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, aggregate in summary["systems"].items():
        metrics = aggregate["metrics"]
        lines.append(
            "| {name} | {n} | {failed} | {schema} | {quote} | {cited} | {points} | "
            "{tools} | {rounds} | {tokens} | {latency} |".format(
                name=_md_cell(name),
                n=aggregate["n"],
                failed=aggregate["n_failed"],
                schema=_format_mean(metrics["schema_success"]),
                quote=_format_mean(metrics["quote_integrity"]),
                cited=_format_mean(metrics["cited_gold_parent_recall"]),
                points=_format_mean(metrics["gold_point_citation_recall"]),
                tools=aggregate["tool_calls"]["mean"],
                rounds=aggregate["retrieval_rounds"]["mean"],
                tokens=aggregate["tokens"]["mean"],
                latency=f"{float(aggregate['latency_ms']['mean']):.1f}",
            )
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- All systems receive the same evidence budget and the same answer prompt.",
        "- Gold evidence never enters a decision or answer prompt; it is used only for scoring.",
        "- fixed_rag consumes the precomputed retrieval run and makes one answer call per question.",
        "- agent_single allows one decision round and 2 model calls; agent_multi allows 3 rounds and 4 calls.",
        "- Empty-gold questions are excluded from ratio metrics; failures remain in the denominator.",
        "- `quote_integrity` is not independent of `schema_success`; the shared validator enforces the same exact-substring rule.",
        "- Harness-generated abstentions are excluded from schema/quote means (marked `answer_source=harness_abstention`).",
        "- The fixed run used the benchmark `query_en`; agent searches use `question_zh` (production behavior), so query language and retrieval method differ.",
        "- `latency_ms` is whole-question time for agents; for fixed_rag it is answer-call time because retrieval is precomputed.",
        "- Per-stage answer/retrieval latency means use only records where that stage ran (`null` otherwise); the report shows n.",
        "- `tokens` excludes the fixed control's precomputed rerank cost, so it is not an end-to-end cost comparison.",
        "- Agent answer calls use `answer_question.PROMPT` only; production's `ANSWER_EXTRA`/coverage payload is not applied.",
        "- `gold_point_citation_recall` is bounded by qrels coverage (13/145 v2 gold points are unmatched).",
        "- No paired bootstrap CI or significance test is computed; n is descriptive.",
        "- A supplemental tool error keeps prior evidence and still answers, whereas production aborts the run.",
        "- These metrics are citation-level only; see the disclaimer above.",
        "",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _load_checkpoint(
    path: Path, question_id: str, system: str, signature: str
) -> dict | None:
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
    if payload.get("system") not in (None, system):
        return None
    if payload.get("run_signature") != signature:
        return None
    if payload.get("status") not in TERMINAL_STATUSES:
        return None
    if not isinstance(payload.get("metrics"), dict):
        return None
    return payload


def run_ab_eval(
    benchmark_dir: Path,
    retrieval_run: Path,
    out_dir: Path,
    *,
    answer_fn: AnswerFn,
    systems: str | Sequence[str] = SYSTEMS,
    decision_backend: Callable | None = None,
    tools: object | None = None,
    rag_dir: Path | None = None,
    retrieval_system: str | None = None,
    limit: int | None = None,
    resume: bool = True,
    paper: str | None = None,
    question_ids: Sequence[str] | None = None,
) -> dict:
    """Run the requested systems and write checkpoints, aggregates and reports.

    ``answer_fn(messages) -> str | dict`` is required. Agent systems also need a
    ``decision_backend(prompt, payload, max_tokens)``. Tests inject ``tools``;
    otherwise the frozen index at ``rag_dir`` (default ``<benchmark>/index``) is
    loaded through :class:`agent_tools.LocalTools` with ``verify_sources=False``.

    ``question_ids`` optionally restricts the run to specific benchmark IDs
    (useful for a small pilot that avoids known failed retrieval records);
    ``limit`` is applied after that filter and preserves benchmark order.
    """
    if not callable(answer_fn):
        raise TypeError("answer_fn must be callable")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
    ):
        raise ValueError("limit must be an integer >= 0")
    if not isinstance(resume, bool):
        raise ValueError("resume must be a boolean")

    systems = _normalize_systems(systems)
    benchmark_dir = Path(benchmark_dir)
    out_dir = Path(out_dir)
    uses_fixed = "fixed_rag" in systems
    uses_agent = any(system in AGENT_SYSTEMS for system in systems)

    run_dir: Path | None = None
    run_root: Path | None = None
    chunks: Mapping[str, Mapping] | None = None
    if uses_fixed:
        run_dir, run_root = resolve_retrieval_run(retrieval_run, retrieval_system)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"retrieval run directory not found: {run_dir}")
        chunks = load_corpus(benchmark_dir / "index" / "corpus_clean.json")

    actual_rag_dir: Path | None = None
    injected_tools = tools is not None
    if uses_agent:
        if decision_backend is None or not callable(decision_backend):
            raise ValueError("decision_backend is required for agent systems")
        if tools is None:
            from agent_tools import LocalTools

            actual_rag_dir = (
                Path(rag_dir) if rag_dir is not None else benchmark_dir / "index"
            )
            _validate_agent_index(benchmark_dir, actual_rag_dir)
            tools = LocalTools(
                PROJECT_ROOT,
                paper=paper,
                rag_dir=actual_rag_dir,
                verify_sources=False,
            )

    questions, qrels_by_id = load_benchmark(benchmark_dir)
    benchmark_fingerprint = _benchmark_fingerprint(benchmark_dir)
    answer_model_name = getattr(answer_fn, "model", None)
    answer_model_name = str(answer_model_name) if answer_model_name else None
    decision_model_name = (
        getattr(decision_backend, "model", None)
        if decision_backend is not None
        else None
    )
    decision_model_name = (
        str(decision_model_name) if decision_model_name else None
    )
    tools_metadata = getattr(tools, "metadata", None) if uses_agent else None
    tools_corpus_sha256 = (
        str(tools_metadata.get("corpus_sha256"))
        if isinstance(tools_metadata, Mapping)
        and tools_metadata.get("corpus_sha256")
        else None
    )
    agent_index: dict | None = None
    if uses_agent:
        agent_index = {
            "injected_tools": injected_tools,
            "dir": str(actual_rag_dir) if actual_rag_dir is not None else None,
            "tools_corpus_sha256": tools_corpus_sha256,
            "model_name": (
                tools_metadata.get("model_name")
                if isinstance(tools_metadata, Mapping)
                else None
            ),
            "model_revision": (
                tools_metadata.get("model_revision")
                if isinstance(tools_metadata, Mapping)
                else None
            ),
            "query_prefix": (
                tools_metadata.get("query_prefix")
                if isinstance(tools_metadata, Mapping)
                else None
            ),
        }
        if actual_rag_dir is not None:
            agent_index.update(_agent_index_files(actual_rag_dir))
    retrieval_label = run_dir.name if run_dir is not None else None
    target_questions = list(questions)
    if question_ids is not None:
        if isinstance(question_ids, str):
            wanted = [item.strip() for item in question_ids.split(",") if item.strip()]
        elif isinstance(question_ids, (list, tuple)):
            wanted = [str(item).strip() for item in question_ids if str(item).strip()]
        else:
            raise ValueError("question_ids must be a comma-separated string or a sequence")
        if not wanted:
            raise ValueError("question_ids must not be empty")
        available = {_question_id(question) for question in questions}
        unknown = [item for item in wanted if item not in available]
        if unknown:
            raise ValueError(
                "unknown question id(s): " + ", ".join(unknown[:10])
            )
        wanted_set = set(wanted)
        target_questions = [
            question for question in questions
            if _question_id(question) in wanted_set
        ]
    if limit is not None:
        target_questions = target_questions[:limit]

    missing_qrels = [
        _question_id(question)
        for question in target_questions
        if _question_id(question) not in qrels_by_id
    ]
    if missing_qrels:
        raise ValueError(
            "missing qrels record for question(s): "
            + ", ".join(missing_qrels[:10])
            + "; use an explicit empty gold_parent_ids record for empty-gold questions"
        )

    guard_dirs: list[Path] = []
    if uses_fixed and run_dir is not None:
        guard_dirs = [run_dir, run_root] if run_root is not None else [run_dir]
    else:
        try:
            candidate_dir, candidate_root = resolve_retrieval_run(
                retrieval_run, retrieval_system
            )
        except Exception:  # noqa: BLE001 - agent-only runs may not have a run dir
            candidate_dir = candidate_root = None
        if candidate_dir is not None and candidate_dir.is_dir():
            guard_dirs = (
                [candidate_dir, candidate_root]
                if candidate_root is not None
                else [candidate_dir]
            )
    for guard in guard_dirs:
        if guard is not None and out_dir.resolve() == Path(guard).resolve():
            raise ValueError(
                "out_dir must differ from the retrieval run root and system "
                "directory; choose a new empty output directory"
            )

    out_dir.mkdir(parents=True, exist_ok=True)

    aggregates: dict[str, dict] = {}
    for system in systems:
        system_dir = out_dir / system
        if uses_fixed and system == "fixed_rag":
            assert run_dir is not None
            if system_dir.resolve() == run_dir.resolve():
                raise ValueError(
                    "fixed_rag output must differ from the retrieval run directory; "
                    "checkpoints would overwrite retrieval records"
                )
        system_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict] = []
        for question in target_questions:
            question_id = _question_id(question)
            path = system_dir / f"{_safe_component(question_id)}.json"
            retrieval_record_sha256 = (
                _sha256_file(run_dir / f"{_safe_component(question_id)}.json")
                if system == "fixed_rag" and run_dir is not None
                else None
            )
            signature = _run_signature(
                system=system,
                question=question,
                qrels_record=qrels_by_id.get(question_id),
                answer_model=answer_model_name,
                decision_model=decision_model_name,
                retrieval_label=retrieval_label,
                retrieval_record_sha256=retrieval_record_sha256,
                tools_corpus_sha256=tools_corpus_sha256,
                agent_index=agent_index,
                benchmark_fingerprint=benchmark_fingerprint["fingerprint"],
                paper=paper,
            )
            if resume:
                checkpoint = _load_checkpoint(
                    path, question_id, system, signature
                )
                if checkpoint is not None:
                    records.append(checkpoint)
                    continue

            if system == "fixed_rag":
                assert run_dir is not None and chunks is not None
                record = _run_fixed(
                    question=question,
                    qrels_record=qrels_by_id.get(question_id),
                    run_dir=run_dir,
                    chunks=chunks,
                    answer_fn=answer_fn,
                )
            else:
                config = SYSTEM_CONFIG[system]
                record = _run_agent(
                    question=question,
                    qrels_record=qrels_by_id.get(question_id),
                    system=system,
                    tools=tools,
                    backend=decision_backend,
                    answer_fn=answer_fn,
                    max_rounds=config["max_rounds"],
                    max_calls=config["max_calls"],
                    paper=paper,
                )
            record["run_signature"] = signature
            _write_json(path, record)
            records.append(record)

        aggregate = _aggregate(
            records,
            system=system,
            benchmark_dir=benchmark_dir,
            retrieval_run=run_dir if system == "fixed_rag" else None,
            retrieval_system=(
                retrieval_label if system == "fixed_rag" else "local_tools"
            ),
            answer_model=answer_model_name,
            benchmark_fingerprint=benchmark_fingerprint,
            agent_index=agent_index,
        )
        _write_json(system_dir / "aggregate.json", aggregate)
        (system_dir / "report.md").write_text(
            _render_system_report(aggregate), encoding="utf-8"
        )
        aggregates[system] = aggregate

    summary = {
        "version": AGENT_EVAL_VERSION,
        "generated_by": "eval/agent_eval.py",
        "created_at": _now(),
        "benchmark_dir": str(benchmark_dir),
        "retrieval_run": str(run_root) if run_root is not None else None,
        "retrieval_system": retrieval_label if uses_fixed else "local_tools",
        "evidence_budget": EVIDENCE_BUDGET,
        "benchmark_fingerprint": benchmark_fingerprint["fingerprint"],
        "benchmark_files": benchmark_fingerprint,
        "agent_index": agent_index,
        "disclaimer": DISCLAIMER,
        "systems": aggregates,
    }
    _write_json(out_dir / "summary.json", summary)
    (out_dir / "report.md").write_text(
        _render_combined_report(summary), encoding="utf-8"
    )
    return summary


# ---------------------------------------------------------------------------
# Default DeepSeek backends (the only network code in this module)
# ---------------------------------------------------------------------------


def _deepseek_values(model: str | None, env: Mapping[str, str] | None) -> tuple[str, str, str]:
    from dotenv import dotenv_values

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
    return api_key, base_url, resolved_model


def make_deepseek_answer_fn(
    *,
    model: str | None = None,
    env: Mapping[str, str] | None = None,
) -> AnswerFn:
    """DeepSeek answer function with ``answer_fn.last_usage`` / ``.model``."""
    from openai import OpenAI

    api_key, base_url, resolved_model = _deepseek_values(model, env)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120, max_retries=0)

    def answer_fn(messages: list[dict]) -> str:
        try:
            response = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
                max_tokens=ANSWER_MAX_TOKENS,
            )
        except Exception as exc:
            # Match production: never persist provider exception text.
            raise RuntimeError(
                f"模型请求未完成（{type(exc).__name__}）；没有自动重试。"
            ) from None
        answer_fn.last_usage = (  # type: ignore[attr-defined]
            response.usage.model_dump() if response.usage else None
        )
        if not response.choices:
            raise ValueError("answer model returned no choices")
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise ValueError(
                f"answer model did not finish: {choice.finish_reason}"
            )
        content = choice.message.content
        if not content or not content.strip():
            raise ValueError("answer model returned empty content")
        return content

    answer_fn.last_usage = None  # type: ignore[attr-defined]
    answer_fn.model = resolved_model  # type: ignore[attr-defined]
    return answer_fn


def make_deepseek_decision_backend(
    *,
    model: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Callable:
    """DeepSeek decision backend matching ``run_agent.LiveBackend``'s shape."""
    from openai import OpenAI

    api_key, base_url, resolved_model = _deepseek_values(model, env)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120, max_retries=0)

    def backend(prompt: str, payload: Mapping, max_tokens: int) -> dict:
        try:
            response = client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
                max_tokens=max_tokens,
            )
        except Exception as exc:
            # Match production: never persist provider exception text.
            raise RuntimeError(
                f"模型请求未完成（{type(exc).__name__}）；没有自动重试。"
            ) from None
        if not response.choices:
            raise ValueError("decision model returned no choices")
        choice = response.choices[0]
        return {
            "model": response.model,
            "finish_reason": choice.finish_reason,
            "content": choice.message.content,
            "usage": response.usage.model_dump() if response.usage else None,
        }

    backend.model = resolved_model  # type: ignore[attr-defined]
    backend.simulated = False  # type: ignore[attr-defined]
    return backend


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "A/B evaluation harness: fixed RAG control vs the bounded agent. "
            "Citation-level metrics only; no judge."
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
        help="precomputed retrieval run root or a single system directory",
    )
    parser.add_argument(
        "--retrieval-system",
        default=DEFAULT_RETRIEVAL_SYSTEM,
        help="system subdirectory under --retrieval-run (fixed_rag control)",
    )
    parser.add_argument(
        "--systems",
        default=",".join(SYSTEMS),
        help="comma-separated subset of fixed_rag,agent_single,agent_multi",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap questions")
    parser.add_argument(
        "--question-ids",
        default=None,
        help=(
            "optional comma-separated benchmark IDs (e.g. q001,q002); "
            "applied before --limit"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "output directory (default: "
            "<run root>/agent_eval/<retrieval-system-or-local_tools>, with one "
            "subfolder per system; must differ from the retrieval run root/system)"
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse terminal checkpoints whose run signature matches (default: true)",
    )
    parser.add_argument(
        "--answer-model",
        default=None,
        help="override DEEPSEEK_MODEL from .env for answer and decisions",
    )
    parser.add_argument(
        "--rag-dir",
        type=Path,
        default=None,
        help="frozen index directory for agent systems (default: <benchmark>/index)",
    )
    args = parser.parse_args(argv)

    systems = _normalize_systems(args.systems)
    try:
        run_dir, run_root = resolve_retrieval_run(
            args.retrieval_run, args.retrieval_system
        )
    except Exception:
        run_dir, run_root = args.retrieval_run, args.retrieval_run
    out_dir = args.out
    if out_dir is None:
        label = (
            Path(run_dir).name
            if "fixed_rag" in systems
            else "local_tools"
        )
        out_dir = Path(run_root) / "agent_eval" / label

    answer_fn = make_deepseek_answer_fn(model=args.answer_model)
    decision_backend = (
        make_deepseek_decision_backend(model=args.answer_model)
        if any(system in AGENT_SYSTEMS for system in systems)
        else None
    )

    summary = run_ab_eval(
        args.benchmark_dir,
        args.retrieval_run,
        out_dir,
        answer_fn=answer_fn,
        systems=systems,
        decision_backend=decision_backend,
        rag_dir=args.rag_dir,
        retrieval_system=args.retrieval_system,
        limit=args.limit,
        question_ids=args.question_ids,
        resume=args.resume,
    )

    compact = {
        "status": "completed",
        "out_dir": str(Path(out_dir).resolve()),
        "limit": args.limit,
        "systems": {
            name: {
                "n": aggregate["n"],
                "n_failed": aggregate["n_failed"],
                "status_counts": aggregate["status_counts"],
                "metrics": {
                    key: {
                        "mean": aggregate["metrics"][key]["mean"],
                        "n": aggregate["metrics"][key]["n"],
                    }
                    for key in METRIC_KEYS
                },
                "tool_calls_mean": aggregate["tool_calls"]["mean"],
                "retrieval_rounds_mean": aggregate["retrieval_rounds"]["mean"],
                "tokens": aggregate["tokens"],
                "latency_ms": aggregate["latency_ms"],
            }
            for name, aggregate in summary["systems"].items()
        },
        "disclaimer": DISCLAIMER,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
