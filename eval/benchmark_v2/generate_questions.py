"""Task E5b: semi-automatic question generation for the held-out benchmark.

The generator is intentionally dependency-light and injectable:

* ``generate_candidate`` turns one paper plus its extracted pages into one draft
  candidate by calling a supplied ``chat_fn(messages) -> str``.
* ``generate_for_split`` reads ``splits.json``, generates candidates for the
  test papers, validates them with :mod:`eval.benchmark_v2.validate_questions`,
  writes JSONL, and records model/prompt/usage metadata.
* ``default_chat_fn`` is the only place that knows about DeepSeek. It is never
  invoked on import and tests always inject a fake callable.

Prompts use pages 2-8 where available (page 1 is mostly title/abstract), capped
at roughly 12,000 characters, so gold quotes come from the paper body.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.benchmark_v2.validate_questions import (  # noqa: E402
    extract_abstract,
    load_pages,
    page_number,
    page_text,
    validate_candidate,
    validate_question_set,
)

DEFAULT_SPLITS = PROJECT_ROOT / "eval" / "benchmark_v2" / "splits.draft.json"
DEFAULT_OUT = PROJECT_ROOT / "eval" / "benchmark_v2" / "questions.draft.jsonl"
DEFAULT_USAGE_NAME = "generation_usage.json"
CONTEXT_PAGE_MIN = 2
CONTEXT_PAGE_MAX = 8
MAX_CONTEXT_CHARS = 12000
DEFAULT_MAX_TOKENS = 2048
SCHEMA_VERSION = 1
MAX_REPAIR_ATTEMPTS = 2

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

SYSTEM_PROMPT = """你是 PaperLens 科学论文检索评测的出题助手。
只输出一个 JSON 对象，不要输出 Markdown、解释或代码块围栏。

JSON 字段：
- question_zh: 中文问题，必须能由下面给出的论文正文回答
- query_en: 英文检索查询，用于 BM25/dense 检索
- type: factual | numeric | mechanism | multi_hop | partial | unanswerable
- answerability: answerable | partial | unanswerable
- gold_points: 数组，每项含 requirement、quote、pdf_page、section
- missing_aspects: 字符串数组；partial/unanswerable 题说明缺失方面
- source_title: 论文标题

硬性要求：
1. quote 必须是下面页面文本的逐字子串（只允许空白差异），并给出正确的
   pdf_page。不要改写、翻译或拼接不同位置的句子。
2. quote 不能只来自摘要；优先引用方法、实验、结果、讨论等正文段落。
3. numeric 题的 quote 必须同时包含数字和单位（如 40%、8 tokens、12.5 points）。
4. fact/mechanism/multi_hop 题给出 1-4 个 gold_points；partial 题也给出已覆盖的
   gold_points；unanswerable 题 gold_points 为空，并在 missing_aspects 说明缺什么。
5. 不要出与论文无关或依赖外部知识的问题；不要重复题目。
6. multi_hop 题应需要组合两个以上段落或两篇论文的证据。"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prompt_sha256(messages: Sequence[Mapping]) -> str:
    payload = json.dumps(list(messages), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _refuse_data_write(path: Path) -> None:
    """Refuse writes under the read-only project data directory."""
    resolved = Path(path).resolve()
    data_root = (PROJECT_ROOT / "data").resolve()
    if resolved == data_root or data_root in resolved.parents:
        raise ValueError(f"Refusing to write generated questions under data/: {path}")


# ---------------------------------------------------------------------------
# Context selection and prompt
# ---------------------------------------------------------------------------


def select_context_pages(pages: Sequence[Mapping]) -> tuple[list[Mapping], str]:
    """Select body pages (2-8 when present) and render them under a char cap."""
    ordered = sorted(
        (page for page in pages if isinstance(page, Mapping)),
        key=lambda page: page_number(page),
    )
    in_range = [
        page
        for page in ordered
        if CONTEXT_PAGE_MIN <= page_number(page) <= CONTEXT_PAGE_MAX
    ]
    context = in_range if in_range else ordered

    parts: list[str] = []
    selected: list[Mapping] = []
    total = 0
    for page in context:
        number = page_number(page)
        body = page_text(page)
        header = f"[page {number}]\n"
        separator = "\n\n" if parts else ""
        budget = MAX_CONTEXT_CHARS - total - len(separator) - len(header)
        if budget <= 0:
            break
        body_part = body[:budget]
        parts.append(header + body_part)
        selected.append(page)
        total += len(separator) + len(header) + len(body_part)
        if len(body_part) < len(body):
            break

    return selected, "\n\n".join(parts)


def build_messages(
    paper: Mapping,
    pages: Sequence[Mapping],
    extra_instruction: str = "",
) -> tuple[list[dict], str]:
    """Build the chat messages plus the prompt hash for one paper."""
    context_pages, context_text = select_context_pages(pages)
    payload = {
        "paper_id": paper.get("arxiv_id") or paper.get("paper_id") or "",
        "title": paper.get("title") or "",
        "year": paper.get("year"),
        "categories": paper.get("categories") or [],
        "context_page_numbers": [page_number(page) for page in context_pages],
        "pages": context_text,
    }
    system_prompt = SYSTEM_PROMPT
    if extra_instruction:
        system_prompt += "\n\n" + extra_instruction.strip()
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, indent=1),
        },
    ]
    return messages, _prompt_sha256(messages)


def _parse_json_response(text: str) -> dict:
    """Parse a model reply that should contain one JSON question object."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("chat_fn returned an empty response")

    cleaned = _FENCE.sub("", text.strip())
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("chat_fn response does not contain a JSON object")
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"chat_fn response is not valid JSON: {exc}") from exc

    if isinstance(payload, list):
        if not payload:
            raise ValueError("chat_fn response contains an empty JSON array")
        payload = payload[0]
    if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        if not payload["questions"]:
            raise ValueError("chat_fn response contains no questions")
        payload = payload["questions"][0]
    if not isinstance(payload, dict):
        raise ValueError("chat_fn response must decode to a JSON object")
    return payload


def _normalize_gold_points(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    points: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        points.append(
            {
                "requirement": item.get("requirement", ""),
                "quote": item.get("quote", ""),
                "pdf_page": item.get("pdf_page"),
                "section": item.get("section", ""),
            }
        )
    return points


TRANSLATE_SYSTEM_PROMPT = """你是检索查询翻译器。只输出 JSON 对象：
{"query_en": "..."}。
把用户给出的中文问题翻译成简洁的英文检索查询。
只使用问题中出现的概念；不要添加论文页面里的术语、数字或答案。
不要回答问题，不要输出解释。"""


def translate_query(
    question_zh: str,
    chat_fn: Callable[[list[dict]], str],
    *,
    model: str,
) -> str:
    """Translate a question to English without seeing the paper pages."""
    messages = [
        {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {"question_zh": question_zh}, ensure_ascii=False
            ),
        },
    ]
    payload = _parse_json_response(chat_fn(messages))
    query = payload.get("query_en")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("translation did not return query_en")
    return query.strip()


def _candidate_from_payload(
    payload: Mapping,
    paper: Mapping,
    prompt_sha: str,
    model: str,
) -> dict:
    paper_id = str(
        paper.get("arxiv_id") or paper.get("paper_id") or ""
    ).strip()
    raw_paper_ids = payload.get("paper_ids")
    if isinstance(raw_paper_ids, list) and raw_paper_ids:
        paper_ids = [str(item) for item in raw_paper_ids if str(item).strip()]
    else:
        paper_ids = [paper_id] if paper_id else []

    question_zh = payload.get("question_zh")
    query_en = payload.get("query_en")
    question_type = payload.get("type")
    answerability = payload.get("answerability")
    missing_aspects = payload.get("missing_aspects")

    return {
        "question_id": str(payload.get("question_id") or "").strip(),
        "paper_ids": paper_ids,
        "question_zh": question_zh if isinstance(question_zh, str) else "",
        "query_en": query_en if isinstance(query_en, str) else "",
        "type": question_type if isinstance(question_type, str) else "",
        "answerability": (
            answerability if isinstance(answerability, str) else ""
        ),
        "gold_points": _normalize_gold_points(payload.get("gold_points")),
        "missing_aspects": (
            [str(item) for item in missing_aspects]
            if isinstance(missing_aspects, list)
            else []
        ),
        "source_title": str(
            payload.get("source_title") or paper.get("title") or ""
        ),
        "generator": {
            "model": model,
            "prompt_sha256": prompt_sha,
            "created_at": _utc_now(),
        },
        "review": {"status": "pending", "reviewer": None, "timestamp": None},
    }


def generate_candidate(
    paper: dict,
    pages: list[dict],
    chat_fn: Callable[[list[dict]], str],
    *,
    model: str,
    extra_instruction: str = "",
) -> dict:
    """Generate one draft candidate for a paper using an injected chat function."""
    if not isinstance(paper, dict):
        raise TypeError("paper must be a dict")
    if not callable(chat_fn):
        raise TypeError("chat_fn must be callable")

    page_list = list(pages or [])
    messages, prompt_sha = build_messages(paper, page_list, extra_instruction)
    raw_text = chat_fn(messages)
    payload = _parse_json_response(raw_text)
    return _candidate_from_payload(payload, paper, prompt_sha, model)


def repair_candidate(
    candidate: Mapping,
    problems: Sequence[str],
    messages: Sequence[Mapping],
    paper: Mapping,
    chat_fn: Callable[[list[dict]], str],
    *,
    model: str,
) -> dict:
    """Ask the model to fix a candidate that failed deterministic validation."""
    repair_prompt = (
        "你上一条输出未通过程序校验，错误如下：\n- "
        + "\n- ".join(str(problem) for problem in problems)
        + "\n\n上一次输出：\n"
        + json.dumps(candidate, ensure_ascii=False, indent=2)
        + "\n\n请重新输出一个修正后的 JSON 对象（不要 Markdown）。要求：\n"
        "1. 每个 quote 必须从下面页面文本中逐字复制，只允许空白差异；"
        "不能改写、翻译、拼接或使用省略号。\n"
        "2. quote 的 pdf_page 必须对应它实际出现的页面。\n"
        "3. 如果 type 是 numeric，quote 必须同时包含数字和单位。\n"
        "4. 不要引用摘要；只引用正文方法/实验/结果/讨论。\n"
        "5. 不要重复已有题目；如果无法修正，返回一个不同但同样可回答的问题。\n"
    )
    repair_messages = list(messages) + [{"role": "user", "content": repair_prompt}]
    prompt_sha = _prompt_sha256(repair_messages)
    raw_text = chat_fn(repair_messages)
    payload = _parse_json_response(raw_text)
    return _candidate_from_payload(payload, paper, prompt_sha, model)


# ---------------------------------------------------------------------------
# Usage bookkeeping and split generation
# ---------------------------------------------------------------------------


def _usage_from_chat_fn(chat_fn: object) -> dict | None:
    usage = getattr(chat_fn, "last_usage", None)
    if usage is None:
        usage = getattr(chat_fn, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        return None

    normalized: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            normalized[key] = value
    if "total_tokens" not in normalized:
        prompt = normalized.get("prompt_tokens")
        completion = normalized.get("completion_tokens")
        if prompt is not None and completion is not None:
            normalized["total_tokens"] = prompt + completion
    return normalized or None


def _call_count(chat_fn: object) -> int | None:
    calls = getattr(chat_fn, "calls", None)
    if isinstance(calls, list):
        return len(calls)
    return None


def _resolve_prepared_path(paper: Mapping, root: Path) -> Path:
    for key in ("prepared_path", "prepared_pages", "pages_path"):
        value = paper.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value)
        if path.is_absolute():
            return path
        candidate = root / path
        if candidate.exists():
            return candidate
        return path
    raise FileNotFoundError(
        f"paper {paper.get('arxiv_id')!r} has no prepared_path"
    )


def _paper_id(paper: Mapping) -> str:
    return str(paper.get("arxiv_id") or paper.get("paper_id") or "")


def generate_for_split(
    splits_path: Path,
    out_path: Path,
    chat_fn: Callable[[list[dict]], str],
    *,
    limit: int | None = None,
    model: str | None = None,
    usage_path: Path | None = None,
    project_root: Path | None = None,
    separate_query_translation: bool = False,
    extra_instruction: str = "",
) -> dict:
    """Generate, validate, and write draft questions for a split's test papers."""
    splits_path = Path(splits_path)
    out_path = Path(out_path)
    usage_path = Path(usage_path) if usage_path else out_path.parent / DEFAULT_USAGE_NAME
    _refuse_data_write(out_path)
    _refuse_data_write(usage_path)

    if not callable(chat_fn):
        raise TypeError("chat_fn must be callable")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer or None")

    root = Path(project_root) if project_root else PROJECT_ROOT
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    papers = splits.get("test") if isinstance(splits, dict) else splits
    if not isinstance(papers, list):
        raise ValueError("splits file must contain a 'test' list")
    if limit is not None:
        papers = papers[:limit]

    resolved_model = (
        model
        or getattr(chat_fn, "model", None)
        or "unknown"
    )

    accepted: list[dict] = []
    existing_texts: list[str] = []
    errors: list[dict] = []
    rejected_candidates: list[dict] = []
    usage_records: list[dict] = []
    pages_by_paper: dict[str, list[dict]] = {}
    abstracts_by_paper: dict[str, str] = {}
    rejected = 0
    skipped = 0
    processed = 0

    for paper in papers:
        processed += 1
        if not isinstance(paper, dict):
            skipped += 1
            errors.append(
                {"paper_id": None, "errors": ["split entry is not an object"]}
            )
            continue

        paper_id = _paper_id(paper)
        try:
            page_path = _resolve_prepared_path(paper, root)
            pages = load_pages(page_path)
        except (FileNotFoundError, ValueError, OSError) as exc:
            skipped += 1
            errors.append({"paper_id": paper_id, "errors": [str(exc)]})
            continue
        if not pages:
            skipped += 1
            errors.append({"paper_id": paper_id, "errors": ["no pages found"]})
            continue

        abstract = paper.get("abstract") or extract_abstract(pages)
        question_id = f"q{processed:03d}"
        messages, _ = build_messages(paper, pages, extra_instruction)

        try:
            candidate = generate_candidate(
                paper,
                pages,
                chat_fn,
                model=resolved_model,
                extra_instruction=extra_instruction,
            )
        except Exception as exc:  # noqa: BLE001 - recorded as a rejected draft
            rejected += 1
            errors.append(
                {
                    "question_id": question_id,
                    "paper_id": paper_id,
                    "errors": [f"generation failed: {exc}"],
                }
            )
            usage_records.append(
                {
                    "question_id": question_id,
                    "paper_id": paper_id,
                    "attempt": 1,
                    "prompt_sha256": None,
                    "usage": _usage_from_chat_fn(chat_fn),
                }
            )
            continue

        candidate["question_id"] = question_id
        problems = validate_candidate(
            candidate,
            pages,
            abstract,
            existing_question_texts=existing_texts,
            strict_schema=True,
        )
        usage_records.append(
            {
                "question_id": question_id,
                "paper_id": paper_id,
                "attempt": 1,
                "prompt_sha256": candidate["generator"]["prompt_sha256"],
                "usage": _usage_from_chat_fn(chat_fn),
            }
        )

        repair_attempt = 0
        while problems and repair_attempt < MAX_REPAIR_ATTEMPTS:
            repair_attempt += 1
            try:
                candidate = repair_candidate(
                    candidate,
                    problems,
                    messages,
                    paper,
                    chat_fn,
                    model=resolved_model,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "question_id": question_id,
                        "paper_id": paper_id,
                        "errors": [f"repair {repair_attempt} failed: {exc}"],
                    }
                )
                usage_records.append(
                    {
                        "question_id": question_id,
                        "paper_id": paper_id,
                        "attempt": repair_attempt + 1,
                        "prompt_sha256": None,
                        "usage": _usage_from_chat_fn(chat_fn),
                    }
                )
                break
            candidate["question_id"] = question_id
            usage_records.append(
                {
                    "question_id": question_id,
                    "paper_id": paper_id,
                    "attempt": repair_attempt + 1,
                    "prompt_sha256": candidate["generator"]["prompt_sha256"],
                    "usage": _usage_from_chat_fn(chat_fn),
                }
            )
            problems = validate_candidate(
                candidate,
                pages,
                abstract,
                existing_question_texts=existing_texts,
                strict_schema=True,
            )

        if not problems and separate_query_translation:
            try:
                translated = translate_query(
                    candidate["question_zh"], chat_fn, model=resolved_model
                )
            except Exception as exc:  # noqa: BLE001
                problems = [f"query translation failed: {exc}"]
                usage_records.append(
                    {
                        "question_id": question_id,
                        "paper_id": paper_id,
                        "attempt": repair_attempt + 2,
                        "prompt_sha256": None,
                        "usage": _usage_from_chat_fn(chat_fn),
                    }
                )
            else:
                candidate["query_en"] = translated
                usage_records.append(
                    {
                        "question_id": question_id,
                        "paper_id": paper_id,
                        "attempt": repair_attempt + 2,
                        "prompt_sha256": None,
                        "usage": _usage_from_chat_fn(chat_fn),
                    }
                )
                problems = validate_candidate(
                    candidate,
                    pages,
                    abstract,
                    existing_question_texts=existing_texts,
                    strict_schema=True,
                )

        if problems:
            rejected += 1
            errors.append(
                {
                    "question_id": question_id,
                    "paper_id": paper_id,
                    "errors": problems,
                }
            )
            rejected_candidates.append(
                {
                    "question_id": question_id,
                    "paper_id": paper_id,
                    "candidate": candidate,
                    "errors": problems,
                }
            )
            continue

        accepted.append(candidate)
        existing_texts.append(candidate["question_zh"])
        pages_by_paper[paper_id] = pages
        abstracts_by_paper[paper_id] = abstract

    jsonl = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in accepted
    )
    _atomic_write_text(out_path, jsonl)
    rejected_path = out_path.with_name(out_path.stem + ".rejected.jsonl")
    if rejected_candidates:
        _atomic_write_text(
            rejected_path,
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in rejected_candidates
            ),
        )

    set_report = validate_question_set(
        accepted, pages_by_paper, abstracts_by_paper
    )
    for item in set_report["errors"]:
        errors.append({"question_id": None, "errors": [f"set: {item}"]})

    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "questions_with_usage": 0,
    }
    for record in usage_records:
        usage = record.get("usage")
        if not isinstance(usage, dict):
            continue
        totals["questions_with_usage"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value

    calls = _call_count(chat_fn)
    usage_payload = {
        "schema_version": SCHEMA_VERSION,
        "model": resolved_model,
        "created_at": _utc_now(),
        "calls": calls if calls is not None else len(usage_records),
        "question_count": len(accepted),
        "totals": totals,
        "questions": usage_records,
    }
    _atomic_write_text(
        usage_path,
        json.dumps(usage_payload, ensure_ascii=False, indent=2) + "\n",
    )

    return {
        "splits_path": str(splits_path),
        "out_path": str(out_path),
        "rejected_path": str(rejected_path) if rejected_candidates else None,
        "usage_path": str(usage_path),
        "model": resolved_model,
        "requested": len(papers),
        "processed": processed,
        "generated": len(accepted),
        "rejected": rejected,
        "skipped": skipped,
        "question_ids": [row["question_id"] for row in accepted],
        "errors": errors,
        "usage": {
            "calls": usage_payload["calls"],
            "totals": totals,
        },
        "set_validation": set_report,
        "ok": not errors and set_report["ok"],
    }


# ---------------------------------------------------------------------------
# Default DeepSeek chat function (never called by tests)
# ---------------------------------------------------------------------------


def make_deepseek_chat_fn(
    *,
    model: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Callable[[list[dict]], str]:
    """Build a DeepSeek-backed ``chat_fn``; configuration is read lazily."""
    if env is None:
        try:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env")
        except ImportError:
            pass
        env = os.environ

    api_key = str(env.get("DEEPSEEK_API_KEY", "")).strip()
    resolved_model = str(
        model or env.get("DEEPSEEK_MODEL", "")
    ).strip()
    base_url = str(
        env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    ).strip() or "https://api.deepseek.com"
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not configured in .env")
    if not resolved_model:
        raise ValueError("DEEPSEEK_MODEL is not configured in .env")

    def chat_fn(messages: list[dict]) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=120.0,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        chat_fn.last_usage = (  # type: ignore[attr-defined]
            response.usage.model_dump() if response.usage else None
        )
        chat_fn.model = response.model or resolved_model  # type: ignore[attr-defined]
        choice = response.choices[0]
        if not choice.message.content:
            raise ValueError("DeepSeek returned an empty response")
        return choice.message.content

    chat_fn.model = resolved_model  # type: ignore[attr-defined]
    chat_fn.last_usage = None  # type: ignore[attr-defined]
    return chat_fn


def default_chat_fn() -> Callable[[list[dict]], str]:
    """Convenience alias for the DeepSeek-backed default chat function."""
    return make_deepseek_chat_fn()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate draft benchmark questions for PaperLens"
    )
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="number of test papers to process (pilot uses 3)",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--usage", default=None)
    parser.add_argument(
        "--translate-query",
        action="store_true",
        help="translate question_zh to query_en in a separate page-blind call",
    )
    parser.add_argument(
        "--variant",
        action="store_true",
        help="ask for a second, different-angle question per paper",
    )
    args = parser.parse_args(argv)

    variant_instruction = (
        "这是同一篇论文的第二道题。请选择与定义/总体介绍不同的角度："
        "优先机制、方法对比、数值/表格、限制条件、失败模式或证据边界；"
        "不要重复常见的“本文提出什么/主要贡献是什么”问题。"
        if args.variant
        else ""
    )

    chat_fn = make_deepseek_chat_fn(model=args.model)
    summary = generate_for_split(
        Path(args.splits),
        Path(args.out),
        chat_fn,
        limit=args.limit,
        model=args.model,
        usage_path=Path(args.usage) if args.usage else None,
        separate_query_translation=args.translate_query,
        extra_instruction=variant_instruction,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
