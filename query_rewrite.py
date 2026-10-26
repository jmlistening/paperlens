"""Page-blind Chinese-to-English query rewriting for the product QA path.

The rewrite call sees only the user's question. It never sees retrieved
passages, so it cannot copy gold evidence terms from the corpus. Results are
content-addressed and cached under ``data/query_cache``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "data" / "query_cache"
PROMPT = """你是检索查询翻译器。只输出 JSON 对象：{"query_en": "..."}。
把用户给出的中文问题翻译成简洁的英文检索查询。
只使用问题中出现的概念；不要添加外部术语、数字或答案。
不要回答问题，不要输出解释。"""

_CHINESE = re.compile(r"[㐀-鿿]")


@dataclass(frozen=True)
class RewriteResult:
    original: str
    retrieval_query: str
    rewritten: bool
    cached: bool = False
    error: str | None = None
    prompt_sha256: str | None = None


def contains_chinese(text: str) -> bool:
    return bool(_CHINESE.search(text or ""))


def _prompt_sha256(messages: list[dict]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_key(question: str, model: str, base_url: str) -> str:
    payload = json.dumps(
        {
            "question": question,
            "model": model,
            "base_url": base_url,
            "prompt": PROMPT,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_cache(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = data.get("query_en")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _write_cache(path: Path, value: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(
            json.dumps(
                {
                    "query_en": value,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temp, path)
    except OSError:
        pass


def rewrite_query(
    question: str,
    *,
    chat_fn: Callable[[list[dict]], str] | None = None,
    model: str | None = None,
    env: Mapping[str, str] | None = None,
    cache_dir: Path = DEFAULT_CACHE,
) -> RewriteResult:
    original = (question or "").strip()
    if not original or not contains_chinese(original):
        return RewriteResult(original, original, False)

    if env is not None:
        values = dict(env)
    else:
        try:
            from dotenv import dotenv_values

            values = {
                key: str(value)
                for key, value in dotenv_values(
                    ROOT / ".env", encoding="utf-8-sig"
                ).items()
                if value is not None
            }
        except Exception:
            values = dict(os.environ)
    if str(values.get("PAPERLENS_QUERY_REWRITE", "1")).strip() == "0":
        return RewriteResult(original, original, False)

    resolved_model = str(model or values.get("DEEPSEEK_MODEL", "")).strip()
    base_url = str(values.get("DEEPSEEK_BASE_URL", "")).strip()
    api_key = str(values.get("DEEPSEEK_API_KEY", "")).strip()
    if not all((resolved_model, base_url, api_key)):
        return RewriteResult(
            original, original, False, error="missing DeepSeek config"
        )

    cache_path = Path(cache_dir) / f"{_cache_key(original, resolved_model, base_url)}.json"
    cached = _read_cache(cache_path)
    if cached:
        return RewriteResult(
            original, cached, True, cached=True, prompt_sha256=None
        )

    messages = [
        {"role": "system", "content": PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {"question_zh": original}, ensure_ascii=False
            ),
        },
    ]
    prompt_sha = _prompt_sha256(messages)

    try:
        if chat_fn is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key, base_url=base_url, timeout=60, max_retries=0
            )
            response = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=512,
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = response.choices[0].message.content if response.choices else ""
            raw = content or ""
        else:
            raw = chat_fn(messages)
        payload = json.loads(raw)
        value = payload.get("query_en") if isinstance(payload, dict) else None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("empty query_en")
        value = " ".join(value.split())
        if len(value) > 500:
            raise ValueError("query_en too long")
    except Exception as exc:  # noqa: BLE001 - product must fall back safely
        return RewriteResult(
            original,
            original,
            False,
            error=type(exc).__name__,
            prompt_sha256=prompt_sha,
        )

    _write_cache(cache_path, value)
    return RewriteResult(
        original,
        value,
        True,
        prompt_sha256=prompt_sha,
    )
