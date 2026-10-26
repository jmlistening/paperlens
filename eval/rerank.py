"""LLM reranking for retrieval experiments (hybrid top-20 -> top-6)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval_core import Hit  # noqa: E402
from rerank_evidence import PROMPT, validate_ranking  # noqa: E402

ChatFn = Callable[[list[dict]], str]
RerankFn = Callable[[str, list[Hit], int], tuple[list[Hit], int]]


def _usage_tokens(chat_fn: Any) -> int:
    usage = getattr(chat_fn, "last_usage", None)
    if isinstance(usage, dict):
        value = usage.get("total_tokens")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def rerank_hits(
    question: str,
    hits: list[Hit],
    chat_fn: ChatFn,
    *,
    top_k: int = 6,
) -> tuple[list[Hit], int]:
    """Rank the given candidates with the shared evidence-rerank prompt.

    All candidates are graded 0-3; the result is ordered by grade (the prompt's
    validator keeps the model's order within a grade). The caller receives the
    first ``top_k`` candidates. No gold evidence is passed to the model.
    """
    if top_k <= 0:
        return [], 0
    if not hits:
        return [], 0
    candidates = [
        {
            "id": f"E{index}",
            "paper_id": hit.paper_id,
            "page": hit.page_number,
            "text": hit.text,
        }
        for index, hit in enumerate(hits, start=1)
    ]
    messages = [
        {"role": "system", "content": PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {"question": question, "candidates": candidates},
                ensure_ascii=False,
            ),
        },
    ]
    raw = chat_fn(messages)
    data = raw if isinstance(raw, dict) else json.loads(raw)
    ranking = validate_ranking(data, {candidate["id"] for candidate in candidates})
    originals = {f"E{index}": hit for index, hit in enumerate(hits, start=1)}
    ordered = [originals[row["id"]] for row in ranking]
    return ordered[:top_k], _usage_tokens(chat_fn)


def make_deepseek_rerank_fn(
    *,
    model: str | None = None,
    env: dict[str, str] | None = None,
) -> RerankFn:
    """Build a DeepSeek-backed rerank function; configuration is read lazily."""
    from dotenv import dotenv_values
    from openai import OpenAI

    values = dict(env) if env is not None else {
        key: str(value)
        for key, value in dotenv_values(
            PROJECT_ROOT / ".env", encoding="utf-8-sig"
        ).items()
        if value is not None
    }
    api_key = (values.get("DEEPSEEK_API_KEY") or "").strip()
    base_url = (values.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").strip()
    resolved_model = (model or values.get("DEEPSEEK_MODEL") or "").strip()
    if not api_key or not resolved_model:
        raise ValueError("DEEPSEEK_API_KEY and DEEPSEEK_MODEL are required")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120, max_retries=0)

    def chat(messages: list[dict]) -> str:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=4096,
            extra_body={"thinking": {"type": "disabled"}},
        )
        chat.last_usage = (  # type: ignore[attr-defined]
            response.usage.model_dump() if response.usage else None
        )
        if not response.choices:
            raise ValueError("rerank model returned no choices")
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("rerank model returned empty content")
        return content

    chat.last_usage = None  # type: ignore[attr-defined]

    def rerank(question: str, hits: list[Hit], top_k: int) -> tuple[list[Hit], int]:
        return rerank_hits(question, hits, chat, top_k=top_k)

    return rerank
