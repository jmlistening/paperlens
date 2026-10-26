"""Generate arXiv search groups from the user's plain-language topics.

The model call only translates and expands the user's own topics; it must never
introduce unrelated research directions. Generated groups are validated against
the existing discovery profile schema before they are returned, so the caller
can save topics and groups atomically (or keep the old profile on failure).

No provider configuration is read and no network call is made at import time.
Tests inject a fake ``chat_fn``; production callers use
``make_deepseek_chat_fn``.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path

from fetch_papers import check_terms, parse_profile

ROOT = Path(__file__).resolve().parent

MAX_GROUPS = 3
MAX_ANY_TERMS = 5
MAX_REQUIRED_TERMS = 5
MAX_NAME_LENGTH = 60
GROUP_FIELDS = frozenset({"name", "any_terms", "require_any_terms"})

PROMPT_TEMPLATE = """You turn a user's own research topics into arXiv search queries.
Translate and expand only the user's topics into concise English arXiv search terms.
Do not add unrelated research directions, papers, authors, or assumptions.
Use only concepts that the user's topics state or clearly imply.

Return one JSON object with a "groups" list and no other text:
{{"groups": [{{"name": "...", "any_terms": ["..."], "require_any_terms": ["..."]}}]}}

Rules:
- Return between 1 and {max_groups} groups.
- "name" is a short, non-empty label for the topic.
- "any_terms" holds 1 to 5 unique English terms; every match must include one.
- "require_any_terms" holds 0 to 5 unique English terms; it may be empty.
- Terms are plain keyword phrases (letters, digits, spaces, hyphens), never query expressions.
- Do not invent terms outside the user's topics and do not add other fields.
"""


class TopicQueryError(ValueError):
    """A user-facing topic/query generation error.

    Messages are fixed, safe strings: provider exception text, response
    bodies, credentials, and configuration values are never included.
    """


def build_prompt(max_groups: int = MAX_GROUPS) -> str:
    """Return the system prompt for the requested group limit."""
    return PROMPT_TEMPLATE.format(max_groups=max_groups)


def _checked_max_groups(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAX_GROUPS
    ):
        raise TopicQueryError(
            f"max_groups must be an integer between 1 and {MAX_GROUPS}"
        )
    return value


def _checked_name(value: object, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TopicQueryError(f"group {index}: name must be a non-empty string")
    name = value.strip()
    if len(name) > MAX_NAME_LENGTH:
        raise TopicQueryError(
            f"group {index}: name must be at most {MAX_NAME_LENGTH} characters"
        )
    return name


def _checked_terms(
    value: object,
    field: str,
    index: int,
    *,
    required: bool,
    maximum: int,
) -> list[str]:
    label = f"group {index}: {field}"
    if not isinstance(value, list):
        raise TopicQueryError(f"{label} must be a list of terms")
    # Models occasionally pad terms; normalize whitespace before the schema
    # checks (which reject leading/trailing spaces in the raw form).
    normalized = [
        " ".join(item.split()) if isinstance(item, str) else item
        for item in value
    ]
    try:
        # Reuse the profile schema's term validator so generated groups are
        # guaranteed to load through profile_config.save_active_profile.
        checked = check_terms(normalized, required)
    except (TypeError, ValueError):
        raise TopicQueryError(
            f"{label} must contain plain English arXiv search terms"
        ) from None
    if len(checked) > maximum:
        raise TopicQueryError(f"{label} must contain at most {maximum} unique terms")
    return checked


def _validate_groups(raw_groups: object, max_groups: int) -> list[dict]:
    if not isinstance(raw_groups, list) or not raw_groups:
        raise TopicQueryError("the query generator returned no groups")
    if len(raw_groups) > max_groups:
        raise TopicQueryError(
            f"the query generator returned more than {max_groups} groups"
        )

    groups: list[dict] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_groups, start=1):
        if not isinstance(raw, dict):
            raise TopicQueryError(f"group {index} must be an object")
        unknown = sorted(str(key) for key in raw if key not in GROUP_FIELDS)
        if unknown:
            raise TopicQueryError(
                f"group {index} contains unknown fields: {', '.join(unknown)}"
            )
        name = _checked_name(raw.get("name"), index)
        if name in names:
            raise TopicQueryError(f"group {index}: name must be unique")
        names.add(name)
        groups.append(
            {
                "name": name,
                "any_terms": _checked_terms(
                    raw.get("any_terms"),
                    "any_terms",
                    index,
                    required=True,
                    maximum=MAX_ANY_TERMS,
                ),
                "require_any_terms": _checked_terms(
                    raw.get("require_any_terms", []),
                    "require_any_terms",
                    index,
                    required=False,
                    maximum=MAX_REQUIRED_TERMS,
                ),
            }
        )

    # Final compatibility guard: the same validator used before saving the
    # active profile must accept these groups unchanged.
    try:
        parsed = parse_profile(
            {
                "search": {
                    "lookback_days": 30,
                    "max_candidates": 12,
                    "per_query_limit": 12,
                    "groups": groups,
                }
            }
        )
    except (TypeError, ValueError):
        raise TopicQueryError(
            "generated query groups do not match the profile schema"
        ) from None
    return parsed["groups"]


def _decode_response(raw: object) -> list:
    if not isinstance(raw, str):
        raise TopicQueryError("the query generator returned an invalid response")
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        raise TopicQueryError("the query generator returned invalid JSON") from None
    if not isinstance(payload, dict):
        raise TopicQueryError("the query generator returned an invalid JSON object")
    unknown = sorted(str(key) for key in payload if key != "groups")
    if unknown:
        raise TopicQueryError(
            "the query generator returned unknown top-level fields: "
            + ", ".join(unknown)
        )
    if "groups" not in payload:
        raise TopicQueryError("the query generator response is missing groups")
    return payload["groups"]


def generate_query_groups(
    topics_text: str,
    chat_fn: Callable[[list[dict]], str],
    *,
    max_groups: int = MAX_GROUPS,
) -> list[dict]:
    """Turn plain-language topics into 1—``max_groups`` validated query groups.

    ``chat_fn`` receives OpenAI-style ``messages`` and must return a JSON
    object with a ``groups`` list. Any malformed output, schema violation, or
    provider failure raises :class:`TopicQueryError` with a sanitized message;
    nothing is saved here, so callers can keep the previous profile intact.
    """
    limit = _checked_max_groups(max_groups)
    if not isinstance(topics_text, str) or not topics_text.strip():
        raise TopicQueryError("topics must not be empty")
    if not callable(chat_fn):
        raise TopicQueryError("chat_fn must be callable")

    messages = [
        {"role": "system", "content": build_prompt(limit)},
        {
            "role": "user",
            "content": json.dumps(
                {"topics": topics_text.strip()}, ensure_ascii=False
            ),
        },
    ]
    try:
        raw = chat_fn(messages)
    except TopicQueryError:
        raise
    except Exception:  # noqa: BLE001 - never leak provider details
        raise TopicQueryError("query generation failed") from None

    return _validate_groups(_decode_response(raw), limit)


def _env_values(env: Mapping[str, str] | None) -> dict[str, str]:
    if env is not None:
        return {str(key): str(value) for key, value in env.items()}
    values: dict[str, str] = {}
    try:
        from dotenv import dotenv_values

        values.update(
            {
                key: str(value)
                for key, value in dotenv_values(
                    ROOT / ".env", encoding="utf-8-sig"
                ).items()
                if value is not None
            }
        )
    except Exception:  # noqa: BLE001 - dotenv is optional configuration
        pass
    # An explicit process environment wins over the dotenv file.
    values.update({key: str(value) for key, value in os.environ.items()})
    return values


def make_deepseek_chat_fn(
    *, env: Mapping[str, str] | None = None
) -> Callable[[list[dict]], str]:
    """Return a production ``chat_fn`` backed by the DeepSeek API.

    Signature: ``make_deepseek_chat_fn(*, env: Mapping[str, str] | None = None)
    -> Callable[[list[dict]], str]``.

    Configuration is resolved lazily on the first invocation: ``env`` (when
    supplied) is used as-is; otherwise ``DEEPSEEK_API_KEY``,
    ``DEEPSEEK_BASE_URL``, and ``DEEPSEEK_MODEL`` come from ``.env`` overlaid
    by the process environment. The ``openai`` client is imported only when a
    call is made, so importing this module and building the function never
    touch the network. Callers must not invoke it in offline tests; pass a
    fake ``chat_fn`` to :func:`generate_query_groups` instead.
    """

    def chat(messages: list[dict]) -> str:
        values = _env_values(env)
        api_key = (values.get("DEEPSEEK_API_KEY") or "").strip()
        base_url = (values.get("DEEPSEEK_BASE_URL") or "").strip()
        model = (values.get("DEEPSEEK_MODEL") or "").strip()
        if not all((api_key, base_url, model)):
            raise TopicQueryError("DeepSeek configuration is missing")

        from openai import OpenAI

        with OpenAI(
            api_key=api_key, base_url=base_url, timeout=60, max_retries=0
        ) as client:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=1024,
                extra_body={"thinking": {"type": "disabled"}},
            )
        content = response.choices[0].message.content if response.choices else ""
        if not content or not content.strip():
            raise TopicQueryError("the query generator returned empty content")
        return content

    return chat
