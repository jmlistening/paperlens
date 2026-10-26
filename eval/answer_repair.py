"""One-shot, opt-in repair pass for answer-eval checkpoints.

The deterministic evaluator in :mod:`eval.answer_eval` writes one checkpoint
per question. Some checkpoints are ``answer_status == "invalid"`` because the
model produced malformed JSON, a quote that is not an exact substring, or a
quote outside the 10-800 character window; others are ``status == "failed"``
because the model call itself failed (network error, empty response).

This module re-asks the model **once** for exactly those checkpoints. It is
deliberately conservative:

* Only checkpoints with ``answer_status == "invalid"`` or ``status ==
  "failed"`` are selected. A valid checkpoint is never touched.
* A checkpoint without evidence (for example a retrieval/rerank failure) is
  reported as *skipped*, never repaired: there is nothing to cite.
* The original payload is rebuilt exactly from the checkpoint's evidence
  metadata (``resource_id``/``parent_chunk_id`` -> corpus text) and the
  benchmark's ``question_zh``. Gold data is never sent to the model.
* For a network-only failure (``raw_response`` is ``None`` but evidence
  exists) the original messages are retried once; otherwise a repair user
  message carrying the previous error and previous raw response is appended.
* At most one model call is made per selected checkpoint per invocation.
* A successful repair replaces only that checkpoint. A failed repair keeps the
  original error/raw response and records a ``repaired_attempt`` entry, so the
  failure stays visible.

No retrieval is re-run, no paid API is touched by the tests, and nothing under
``data/``, the benchmark questions/qrels or the retrieval run is written.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from answer_question import PROMPT  # noqa: E402
from eval.answer_eval import (  # noqa: E402
    _clean_error,
    _model_name,
    _now,
    _safe_component,
    _safe_raw,
    _safe_usage,
    _usage_tokens,
    _write_json,
    evaluate_answer,
    load_corpus,
)
from eval.run_experiment import load_benchmark  # noqa: E402

REPAIR_VERSION = "answer_repair_v1"

DEFAULT_EVAL_DIR = (
    PROJECT_ROOT / "eval" / "benchmark_v2" / "runs" / "answer_eval120"
)
DEFAULT_BENCHMARK_DIR = PROJECT_ROOT / "eval" / "benchmark_v2" / "eval120"
DEFAULT_MAX_TOKENS = 8192

ANSWER_STATUSES = ("answered", "partial", "insufficient", "invalid")
_VALID_ANSWER_STATUSES = frozenset({"answered", "partial", "insufficient"})
_REPORT_NAMES = frozenset({"aggregate.json", "repair_report.json"})
_NON_CHECKPOINT_NAMES = _REPORT_NAMES | frozenset(
    {"report.json", "run.json", "results.json"}
)

AnswerFn = Callable[[list[dict]], "str | dict"]


class EvidenceUnavailable(ValueError):
    """Raised when a checkpoint's evidence cannot be rebuilt from the corpus."""


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _question_id(question: Mapping) -> str:
    for key in ("question_id", "qid", "id"):
        value = question.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _gold_parent_ids(checkpoint: Mapping, qrels_record: Mapping | None) -> list[str]:
    value = checkpoint.get("gold_parent_ids")
    if isinstance(value, (list, tuple)):
        ordered: list[str] = []
        seen: set[str] = set()
        for item in value:
            if isinstance(item, str) and item.strip() and item.strip() not in seen:
                seen.add(item.strip())
                ordered.append(item.strip())
        if ordered:
            return ordered
    if isinstance(qrels_record, Mapping):
        value = qrels_record.get("gold_parent_ids")
        if isinstance(value, (list, tuple)):
            return [
                item.strip()
                for item in value
                if isinstance(item, str) and item.strip()
            ]
    return []


def _gold_points(question: Mapping) -> list[dict]:
    value = question.get("gold_points")
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _parse_ids(value: str | None) -> list[str] | None:
    if value is None:
        return None
    ordered: list[str] = []
    seen: set[str] = set()
    for item in str(value).split(","):
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _previous_errors(checkpoint: Mapping) -> list[str]:
    errors: list[str] = []
    existing = checkpoint.get("previous_errors")
    if isinstance(existing, (list, tuple)):
        for item in existing:
            if isinstance(item, str) and item.strip():
                errors.append(item)
    error = checkpoint.get("error")
    if isinstance(error, str) and error.strip() and error not in errors:
        errors.append(error)
    return errors


def _as_number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _non_negative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and value == value:
        return max(0, int(value))
    return 0


# ---------------------------------------------------------------------------
# Rebuilding the original payload
# ---------------------------------------------------------------------------


def rebuild_evidence(
    checkpoint: Mapping, chunks: Mapping[str, Mapping]
) -> dict[str, dict]:
    """Rebuild ``{E1: unit, ...}`` from checkpoint evidence metadata + corpus.

    The checkpoint stores ``id``/``parent_chunk_id``/``title``/``page_number``
    per evidence unit but not its text. The text is looked up read-only in the
    benchmark's ``corpus_clean.json``. The order is preserved so the rebuilt
    payload is byte-for-byte equivalent to the one the original run used.
    """
    raw_units = checkpoint.get("evidence")
    if not isinstance(raw_units, list) or not raw_units:
        raise EvidenceUnavailable("no evidence recorded in checkpoint")

    evidence: dict[str, dict] = {}
    for number, unit in enumerate(raw_units, start=1):
        if not isinstance(unit, Mapping):
            raise EvidenceUnavailable(f"evidence unit #{number} is not an object")
        evidence_id = unit.get("id")
        parent_id = unit.get("parent_chunk_id")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise EvidenceUnavailable(f"evidence unit #{number} has no id")
        if not isinstance(parent_id, str) or not parent_id.strip():
            raise EvidenceUnavailable(
                f"evidence {evidence_id!r} has no parent_chunk_id"
            )
        parent_id = parent_id.strip()
        chunk = chunks.get(parent_id)
        if not isinstance(chunk, Mapping):
            raise EvidenceUnavailable(
                f"evidence {evidence_id!r}: chunk {parent_id!r} not in corpus"
            )
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise EvidenceUnavailable(
                f"evidence {evidence_id!r}: chunk {parent_id!r} has empty text"
            )
        title = unit.get("title")
        if not isinstance(title, str) or not title.strip():
            title = str(chunk.get("title") or "")
        page_number = unit.get("page_number")
        if page_number is None:
            page_number = chunk.get("page_number")
        if evidence_id in evidence:
            raise EvidenceUnavailable(f"duplicate evidence id {evidence_id!r}")
        evidence[evidence_id] = {
            "id": evidence_id,
            "parent_chunk_id": parent_id,
            "title": title,
            "page_number": page_number,
            "text": text,
        }
    return evidence


def build_payload(question_text: str, evidence: Mapping[str, Mapping]) -> dict:
    """The exact payload shape used by the original answer run."""
    return {
        "question": question_text,
        "evidence": [
            {"id": unit["id"], "title": unit["title"], "text": unit["text"]}
            for unit in evidence.values()
        ],
    }


def _previous_raw_text(raw: object) -> str:
    if raw is None:
        return "（无原始输出）"
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(raw)


def build_repair_message(error: object, raw_response: object) -> str:
    """Repair user message with the previous error and previous raw response."""
    return (
        "上一次回答未通过程序校验，未生成有效答案。\n"
        f"校验错误：{_clean_error(error) or '未知错误'}\n"
        "上一次的原始回答如下（仅作为修复参考，不要照抄错误部分）：\n"
        f"{_previous_raw_text(raw_response)}\n\n"
        "请修复上面的问题，重新输出一个紧凑、完整的 JSON 对象，只包含 "
        "status、points、limitations 三个字段，不要输出 JSON 之外的任何文字。\n"
        "要求：只使用已提供的 evidence；每个 citation 的 id 必须存在于 "
        "evidence 中；quote 必须是该证据中连续、完整的原文，长度 10～800 "
        "字符，不得翻译、改写、拼接或使用省略号；status 与 points 数量必须"
        "自洽（insufficient 时 points 必须为空）；partial 或 insufficient 时 "
        "limitations 要说明缺少的证据。"
    )


def _build_messages(
    question_text: str, evidence: Mapping[str, Mapping], checkpoint: Mapping
) -> tuple[list[dict], str]:
    """Return ``(messages, mode)`` where mode is ``repair`` or ``retry_original``."""
    payload = build_payload(question_text, evidence)
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    if checkpoint.get("raw_response") is None:
        # Network-only failure: retry the original messages once, unchanged.
        return messages, "retry_original"
    messages.append(
        {
            "role": "user",
            "content": build_repair_message(
                checkpoint.get("error"), checkpoint.get("raw_response")
            ),
        }
    )
    return messages, "repair"


# ---------------------------------------------------------------------------
# Checkpoint discovery / selection
# ---------------------------------------------------------------------------


def _scan_checkpoints(eval_dir: Path) -> tuple[list[tuple[Path, dict]], list[dict]]:
    """Return ``(checkpoints, problems)`` scanning ``eval_dir/*.json``."""
    checkpoints: list[tuple[Path, dict]] = []
    problems: list[dict] = []
    seen: set[str] = set()
    for path in sorted(Path(eval_dir).glob("*.json")):
        if path.name in _NON_CHECKPOINT_NAMES:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(
                {
                    "question_id": path.stem,
                    "reason": f"unreadable checkpoint: {type(exc).__name__}: {exc}",
                }
            )
            continue
        if not isinstance(payload, dict):
            problems.append(
                {"question_id": path.stem, "reason": "checkpoint is not an object"}
            )
            continue
        qid = payload.get("question_id")
        if not isinstance(qid, str) or not qid.strip():
            # aggregate.json and other non-checkpoint JSON files live here.
            continue
        qid = qid.strip()
        if qid in seen:
            problems.append(
                {"question_id": qid, "reason": f"duplicate checkpoint; ignoring {path.name}"}
            )
            continue
        seen.add(qid)
        checkpoints.append((path, payload))
    checkpoints.sort(key=lambda item: str(item[1].get("question_id")))
    return checkpoints, problems


def _select_checkpoints(
    checkpoints: Sequence[tuple[Path, dict]], ids: Sequence[str] | None
) -> tuple[list[tuple[Path, dict]], list[dict], list[dict]]:
    """Return ``(selected, already_valid, missing)``.

    Selection is exactly ``answer_status == "invalid"`` or
    ``status == "failed"``. ``missing`` covers requested ids with no
    checkpoint; ``already_valid`` covers requested/visible valid checkpoints.
    """
    by_id = {
        str(payload.get("question_id")): (path, payload)
        for path, payload in checkpoints
    }
    if ids is not None:
        ordered: list[tuple[Path, dict]] = []
        missing: list[dict] = []
        already_valid: list[dict] = []
        for qid in ids:
            item = by_id.get(qid)
            if item is None:
                missing.append({"question_id": qid, "reason": "checkpoint not found"})
                continue
            path, payload = item
            if _is_repair_candidate(payload):
                ordered.append((path, payload))
            else:
                already_valid.append(
                    {
                        "question_id": qid,
                        "answer_status": payload.get("answer_status"),
                        "reason": "already valid; not repaired",
                    }
                )
        return ordered, already_valid, missing

    selected: list[tuple[Path, dict]] = []
    already_valid = []
    for path, payload in checkpoints:
        if _is_repair_candidate(payload):
            selected.append((path, payload))
        else:
            already_valid.append(
                {
                    "question_id": payload.get("question_id"),
                    "answer_status": payload.get("answer_status"),
                    "reason": "already valid; not repaired",
                }
            )
    return selected, already_valid, []


def _is_repair_candidate(payload: Mapping) -> bool:
    return (
        payload.get("answer_status") == "invalid"
        or payload.get("status") == "failed"
    )


# ---------------------------------------------------------------------------
# Repairing one checkpoint
# ---------------------------------------------------------------------------


def _repair_success_record(
    checkpoint: Mapping,
    *,
    evaluation: Mapping,
    raw_response: object,
    usage: object,
    new_tokens: int,
    new_latency_ms: float,
    model: str | None,
    started_at: str,
) -> dict:
    record = dict(checkpoint)
    previous_errors = _previous_errors(checkpoint)
    old_tokens = _non_negative_int(checkpoint.get("tokens"))
    old_latency = _as_number(checkpoint.get("latency_ms"))
    old_attempt = _non_negative_int(checkpoint.get("attempt")) or 1

    record.update(
        {
            "status": "ok",
            "answer_status": evaluation["answer_status"],
            "error": evaluation.get("error"),
            "metrics": dict(evaluation["metrics"]),
            "cited_evidence_ids": list(evaluation["cited_evidence_ids"]),
            "cited_parent_ids": list(evaluation["cited_parent_ids"]),
            "raw_response": _safe_raw(raw_response),
            "usage": _safe_usage(usage),
            "tokens": old_tokens + _non_negative_int(new_tokens),
            "latency_ms": round(old_latency + float(new_latency_ms), 3),
            "attempt": old_attempt + 1,
            "previous_errors": previous_errors,
            "repaired_at": started_at,
        }
    )
    if model:
        record["model"] = model
    record["repair"] = {
        "mode": "repair" if checkpoint.get("raw_response") is not None else "retry_original",
        "repaired_at": started_at,
        "model": model,
        "previous_error": checkpoint.get("error"),
        "previous_raw_response": _safe_raw(checkpoint.get("raw_response")),
        "previous_usage": _safe_usage(checkpoint.get("usage")),
        "previous_tokens": old_tokens,
        "previous_latency_ms": round(old_latency, 3),
        "attempt": old_attempt + 1,
    }
    return record


def _repair_failure_record(
    checkpoint: Mapping,
    *,
    evaluation: Mapping | None,
    error: str,
    raw_response: object,
    usage: object,
    new_tokens: int,
    new_latency_ms: float,
    model: str | None,
    mode: str,
    started_at: str,
) -> dict:
    """Keep the original record intact; add a visible ``repaired_attempt``."""
    record = dict(checkpoint)
    old_attempt = _non_negative_int(checkpoint.get("attempt")) or 1
    answer_status = (
        str(evaluation["answer_status"])
        if isinstance(evaluation, Mapping)
        else "invalid"
    )
    if answer_status not in ANSWER_STATUSES:
        answer_status = "invalid"
    record["repaired_attempt"] = {
        "attempt": old_attempt + 1,
        "repaired_at": started_at,
        "mode": mode,
        "model": model,
        "answer_status": answer_status,
        "error": _clean_error(error),
        "metrics": (
            dict(evaluation["metrics"])
            if isinstance(evaluation, Mapping)
            else None
        ),
        "raw_response": _safe_raw(raw_response),
        "usage": _safe_usage(usage),
        "tokens": _non_negative_int(new_tokens),
        "latency_ms": round(float(new_latency_ms), 3),
    }
    return record


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------


def run_repair(
    eval_dir: Path,
    benchmark_dir: Path,
    *,
    answer_fn: AnswerFn | None = None,
    ids: Sequence[str] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    dry_run: bool = False,
    out_dir: Path | None = None,
) -> dict:
    """Run the one-shot repair pass and return the report.

    ``answer_fn`` is required unless ``dry_run`` is true; a dry run classifies
    candidates and never calls the model or writes anything.
    """
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if not isinstance(dry_run, bool):
        raise ValueError("dry_run must be a boolean")
    if not dry_run and not callable(answer_fn):
        raise TypeError("answer_fn must be callable unless dry_run is true")

    eval_dir = Path(eval_dir)
    benchmark_dir = Path(benchmark_dir)
    out_dir = Path(out_dir) if out_dir is not None else eval_dir
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"eval directory not found: {eval_dir}")

    questions, qrels_by_id = load_benchmark(benchmark_dir)
    questions_by_id = {_question_id(question): question for question in questions}
    chunks = load_corpus(benchmark_dir / "index" / "corpus_clean.json")

    checkpoints, scan_problems = _scan_checkpoints(eval_dir)
    selected, already_valid, missing = _select_checkpoints(checkpoints, ids)

    created_at = _now()
    repaired: list[dict] = []
    skipped: list[dict] = list(scan_problems) + list(missing)
    still_invalid: list[dict] = []
    would_repair: list[dict] = []

    for path, checkpoint in selected:
        qid = str(checkpoint.get("question_id"))
        question = questions_by_id.get(qid)
        if question is None:
            skipped.append(
                {"question_id": qid, "reason": "question not found in benchmark"}
            )
            continue

        try:
            evidence = rebuild_evidence(checkpoint, chunks)
        except EvidenceUnavailable as exc:
            skipped.append(
                {
                    "question_id": qid,
                    "reason": f"no usable evidence: {_clean_error(exc)}",
                }
            )
            continue

        if dry_run:
            would_repair.append(
                {
                    "question_id": qid,
                    "answer_status": checkpoint.get("answer_status"),
                    "status": checkpoint.get("status"),
                    "previous_error": checkpoint.get("error"),
                }
            )
            continue

        question_text = _question_text(question)
        messages, mode = _build_messages(question_text, evidence, checkpoint)
        gold_ids = _gold_parent_ids(checkpoint, qrels_by_id.get(qid))
        gold_points = _gold_points(question)

        started_at = _now()
        call_started = time.perf_counter()
        try:
            raw = answer_fn(messages)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 - a model crash is a failed repair
            latency_ms = (time.perf_counter() - call_started) * 1000.0
            usage = getattr(answer_fn, "last_usage", None)
            error = f"{type(exc).__name__}: {exc}"
            record = _repair_failure_record(
                checkpoint,
                evaluation=None,
                error=error,
                raw_response=None,
                usage=usage,
                new_tokens=_usage_tokens(usage),
                new_latency_ms=latency_ms,
                model=_model_name(answer_fn),
                mode=mode,
                started_at=started_at,
            )
            _write_json(path, record)
            still_invalid.append(
                {
                    "question_id": qid,
                    "reason": error,
                    "original_error": checkpoint.get("error"),
                    "answer_status": "invalid",
                }
            )
            continue

        latency_ms = (time.perf_counter() - call_started) * 1000.0
        usage = getattr(answer_fn, "last_usage", None)
        new_tokens = _usage_tokens(usage)
        raw_response = _safe_raw(raw)
        evaluation = evaluate_answer(raw, evidence, gold_ids, gold_points)
        answer_status = str(evaluation["answer_status"])
        model = _model_name(answer_fn) or (
            str(checkpoint.get("model")) if checkpoint.get("model") else None
        )

        if answer_status in _VALID_ANSWER_STATUSES:
            record = _repair_success_record(
                checkpoint,
                evaluation=evaluation,
                raw_response=raw_response,
                usage=usage,
                new_tokens=new_tokens,
                new_latency_ms=latency_ms,
                model=model,
                started_at=started_at,
            )
            _write_json(path, record)
            repaired.append(
                {
                    "question_id": qid,
                    "answer_status": answer_status,
                    "previous_error": checkpoint.get("error"),
                    "attempt": record["attempt"],
                    "tokens_before": _non_negative_int(checkpoint.get("tokens")),
                    "tokens_after": record["tokens"],
                    "latency_ms_before": round(
                        _as_number(checkpoint.get("latency_ms")), 3
                    ),
                    "latency_ms_after": record["latency_ms"],
                }
            )
            continue

        error = evaluation.get("error") or "repair response did not pass validation"
        record = _repair_failure_record(
            checkpoint,
            evaluation=evaluation,
            error=str(error),
            raw_response=raw_response,
            usage=usage,
            new_tokens=new_tokens,
            new_latency_ms=latency_ms,
            model=model,
            mode=mode,
            started_at=started_at,
        )
        _write_json(path, record)
        still_invalid.append(
            {
                "question_id": qid,
                "reason": _clean_error(error),
                "original_error": checkpoint.get("error"),
                "answer_status": answer_status,
            }
        )

    report = {
        "version": REPAIR_VERSION,
        "generated_by": "eval/answer_repair.py",
        "created_at": created_at,
        "dry_run": bool(dry_run),
        "eval_dir": str(eval_dir),
        "benchmark_dir": str(benchmark_dir),
        "out_dir": str(out_dir),
        "answer_model": _model_name(answer_fn) if answer_fn is not None else None,
        "max_tokens": max_tokens,
        "requested_ids": list(ids) if ids is not None else None,
        "selected_ids": [str(payload.get("question_id")) for _, payload in selected],
        "repaired": repaired,
        "skipped": skipped,
        "still_invalid": still_invalid,
        "already_valid": already_valid,
        "would_repair": would_repair,
        "counts": {
            "selected": len(selected),
            "repaired": len(repaired),
            "skipped": len(skipped),
            "still_invalid": len(still_invalid),
            "already_valid": len(already_valid),
            "would_repair": len(would_repair),
        },
    }

    if not dry_run:
        _write_json(out_dir / "repair_report.json", report)
        (out_dir / "repair_report.md").write_text(
            render_report_markdown(report), encoding="utf-8"
        )
    return report


def _question_text(question: Mapping) -> str:
    for key in ("question_zh", "query_zh", "question", "query_en", "query"):
        value = question.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("benchmark question has no non-empty question text")


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _cell(value: object) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace(
        "\n", " "
    )


def render_report_markdown(report: Mapping) -> str:
    counts = report.get("counts") or {}
    lines = [
        "# Answer Repair Report (one-shot, opt-in)",
        "",
        f"- Generated at: `{report.get('created_at')}`",
        f"- Eval directory: `{report.get('eval_dir')}`",
        f"- Benchmark: `{report.get('benchmark_dir')}`",
        f"- Answer model: `{report.get('answer_model') or 'dry run (not called)'}`",
        f"- Dry run: {report.get('dry_run')}",
        f"- Selected: {counts.get('selected', 0)}; "
        f"repaired: {counts.get('repaired', 0)}; "
        f"skipped: {counts.get('skipped', 0)}; "
        f"still invalid: {counts.get('still_invalid', 0)}",
        "",
        "This pass only repairs checkpoints that were `answer_status == \"invalid\"` "
        "or `status == \"failed\"`. It is citation/schema-level repair, not a "
        "correctness or faithfulness judgement.",
        "",
        "## Repaired",
        "",
    ]
    repaired = report.get("repaired") or []
    if repaired:
        lines.extend(
            [
                "| Question | Answer status | Previous error | Attempt | Tokens (before -> after) |",
                "|---|---|---|---|---|",
            ]
        )
        for item in repaired:
            lines.append(
                "| {q} | {status} | {error} | {attempt} | {before} -> {after} |".format(
                    q=_cell(item.get("question_id")),
                    status=_cell(item.get("answer_status")),
                    error=_cell(item.get("previous_error")),
                    attempt=item.get("attempt"),
                    before=item.get("tokens_before"),
                    after=item.get("tokens_after"),
                )
            )
    else:
        lines.append("None.")

    lines.extend(["", "## Skipped (not repaired)", ""])
    skipped = report.get("skipped") or []
    if skipped:
        lines.extend(["| Question | Reason |", "|---|---|"])
        for item in skipped:
            lines.append(
                f"| {_cell(item.get('question_id'))} | {_cell(item.get('reason'))} |"
            )
    else:
        lines.append("None.")

    lines.extend(["", "## Still invalid", ""])
    still_invalid = report.get("still_invalid") or []
    if still_invalid:
        lines.extend(
            ["| Question | Original error | Repair error |", "|---|---|---|"]
        )
        for item in still_invalid:
            lines.append(
                "| {q} | {orig} | {repair} |".format(
                    q=_cell(item.get("question_id")),
                    orig=_cell(item.get("original_error")),
                    repair=_cell(item.get("reason")),
                )
            )
    else:
        lines.append("None.")

    already_valid = report.get("already_valid") or []
    if already_valid:
        lines.extend(
            [
                "",
                "## Already valid (no-op)",
                "",
                "| Question | Answer status |",
                "|---|---|",
            ]
        )
        for item in already_valid:
            lines.append(
                f"| {_cell(item.get('question_id'))} | "
                f"{_cell(item.get('answer_status'))} |"
            )

    would_repair = report.get("would_repair") or []
    if would_repair:
        lines.extend(
            [
                "",
                "## Would repair (dry run)",
                "",
                "| Question | Status | Answer status | Previous error |",
                "|---|---|---|---|",
            ]
        )
        for item in would_repair:
            lines.append(
                "| {q} | {status} | {answer} | {error} |".format(
                    q=_cell(item.get("question_id")),
                    status=_cell(item.get("status")),
                    answer=_cell(item.get("answer_status")),
                    error=_cell(item.get("previous_error")),
                )
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- A checkpoint with no usable evidence is skipped and never repaired.",
            "- A failed repair keeps the original error/raw response and records "
            "`repaired_attempt` in the checkpoint.",
            "- A successful repair overwrites only that checkpoint; token and "
            "latency totals are the sum of the original and repair calls.",
            "- `data/`, benchmark questions/qrels and the retrieval run are never written.",
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default DeepSeek-backed answer function (max_tokens configurable)
# ---------------------------------------------------------------------------


def make_repair_answer_fn(
    *,
    model: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    env: Mapping[str, str] | None = None,
) -> AnswerFn:
    """Build a DeepSeek answer function with a configurable ``max_tokens``.

    Mirrors :func:`eval.answer_eval.make_deepseek_answer_fn` but exposes the
    completion budget required by ``--max-tokens 8192``. Configuration is read
    lazily from ``.env`` unless ``env`` is supplied.
    """
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")

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
            max_tokens=max_tokens,
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
            "One-shot, opt-in repair pass for invalid/failed answer-eval "
            "checkpoints. Makes at most one model call per selected question."
        )
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
        help="directory holding answer checkpoints and receiving the report",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help="directory with questions/qrels and index/corpus_clean.json",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="optional comma-separated question ids to repair",
    )
    parser.add_argument(
        "--answer-model",
        default=None,
        help="override DEEPSEEK_MODEL from .env",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"completion budget for the repair call (default {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="classify candidates without calling the model or writing anything",
    )
    args = parser.parse_args(argv)

    answer_fn = None
    if not args.dry_run:
        answer_fn = make_repair_answer_fn(
            model=args.answer_model, max_tokens=args.max_tokens
        )

    report = run_repair(
        args.eval_dir,
        args.benchmark_dir,
        answer_fn=answer_fn,
        ids=_parse_ids(args.ids),
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
    )

    summary = {
        "status": "dry_run" if args.dry_run else "completed",
        "eval_dir": report["eval_dir"],
        "counts": report["counts"],
        "repaired": [item["question_id"] for item in report["repaired"]],
        "skipped": [
            {"question_id": item["question_id"], "reason": item.get("reason")}
            for item in report["skipped"]
        ],
        "still_invalid": [
            {"question_id": item["question_id"], "reason": item.get("reason")}
            for item in report["still_invalid"]
        ],
        "would_repair": [item["question_id"] for item in report["would_repair"]],
        "report_json": None
        if args.dry_run
        else str(Path(args.eval_dir) / "repair_report.json"),
        "report_md": None
        if args.dry_run
        else str(Path(args.eval_dir) / "repair_report.md"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
