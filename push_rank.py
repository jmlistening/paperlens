"""Ranking, cold start, and digest selection for the optional arXiv push.

The LLM prompt and validation live in :mod:`rank_papers`; this module only
assembles a batch, applies the optional favorite-paper similarity, and selects
a diverse digest. E5 embeddings are opt-in through
``model_config['embedding_model_path']``; otherwise a deterministic lexical
similarity is used, so tests and offline previews never load a model.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import warnings
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

RANK_CACHE_VERSION = "push_rank_cache_v1"
DEFAULT_FAVORITE_WEIGHT = 0.0
DEFAULT_MIN_SCORE = 60
DEFAULT_MAX_PER_CATEGORY = 2
DEFAULT_MAX_PER_DIGEST = 3
FALLBACK_PROMPT = "push_rank_fallback_prompt_v1"

_VERSION_RE = re.compile(r"v\d+$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SECRET_KEY_RE = re.compile(r"(key|secret|password|token|auth)", re.IGNORECASE)

RankFn = Callable[[list[dict], dict, dict], Any]
SimilarityFn = Callable[[str, str], float]


def _paper_key(arxiv_id: Any) -> str:
    value = "" if arxiv_id is None else str(arxiv_id).strip()
    return _VERSION_RE.sub("", value)


def _config_value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def cold_start(feedback: dict | None) -> bool:
    """True when there is no favorite evidence to personalise ranking."""
    if not isinstance(feedback, Mapping):
        return True
    return len(feedback.get("favorites") or []) == 0


def paper_text(paper: Mapping[str, Any]) -> str:
    return " ".join(
        str(paper.get(name) or "") for name in ("title", "abstract")
    ).strip()


def lexical_similarity(left: str, right: str) -> float:
    """Deterministic token cosine, used when E5 is not available."""
    left_tokens = Counter(_TOKEN_RE.findall(str(left).lower()))
    right_tokens = Counter(_TOKEN_RE.findall(str(right).lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    dot = sum(count * right_tokens.get(token, 0) for token, count in left_tokens.items())
    left_norm = math.sqrt(sum(count * count for count in left_tokens.values()))
    right_norm = math.sqrt(sum(count * count for count in right_tokens.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    # Round so identical token bags compare exactly (1.0, not 0.999999...).
    return round(dot / (left_norm * right_norm), 12)


def _resolve_similarity_fn(model_config: Mapping[str, Any]) -> SimilarityFn | None:
    """Load E5 only when a local path is configured and exists."""
    path = model_config.get("embedding_model_path") if isinstance(model_config, Mapping) else None
    if not path:
        return None
    if not Path(path).is_dir():
        warnings.warn(
            "embedding model path is unavailable; using deterministic lexical "
            "similarity",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(path), local_files_only=True)
    except Exception as exc:  # missing package, bad model, incompatible cache
        warnings.warn(
            "E5 is unavailable; using deterministic lexical similarity "
            f"({type(exc).__name__})",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    def similarity(left: str, right: str) -> float:
        import numpy as np

        vectors = model.encode([left, right], normalize_embeddings=True)
        return float(np.dot(vectors[0], vectors[1]))

    return similarity


def favorite_similarity(
    paper: Mapping[str, Any],
    favorites: list[dict],
    *,
    similarity_fn: SimilarityFn | None = None,
    model_config: Mapping[str, Any] | None = None,
) -> float:
    """Highest similarity to a favorite paper, excluding the paper itself."""
    if not favorites:
        return 0.0
    fn = similarity_fn
    if fn is None:
        fn = _resolve_similarity_fn(model_config or {}) or lexical_similarity

    target_key = _paper_key(paper.get("arxiv_id"))
    target_text = paper_text(paper)
    best = 0.0
    for favorite in favorites:
        if not isinstance(favorite, Mapping):
            continue
        if _paper_key(favorite.get("arxiv_id")) == target_key:
            continue
        value = float(fn(target_text, paper_text(favorite)))
        if not math.isfinite(value):
            value = 0.0
        best = max(best, min(1.0, max(0.0, value)))
    return best


def _favorite_weight(profile: Mapping[str, Any], model_config: Mapping[str, Any]) -> float:
    for source in (profile, model_config):
        if isinstance(source, Mapping) and "favorite_weight" in source:
            try:
                value = float(source["favorite_weight"])
            except (TypeError, ValueError) as exc:
                raise ValueError("favorite_weight must be a number") from exc
            if not math.isfinite(value) or value < 0:
                raise ValueError("favorite_weight must be a finite non-negative number")
            return value
    return DEFAULT_FAVORITE_WEIGHT


def _default_prompt_text() -> str:
    try:
        from rank_papers import SYSTEM_PROMPT
        from reading_feedback import FEEDBACK_PROMPT

        return SYSTEM_PROMPT + FEEDBACK_PROMPT
    except Exception:
        # The hash must stay stable even if optional imports are unavailable.
        return FALLBACK_PROMPT


def _default_prompt_hash() -> str:
    return hashlib.sha256(_default_prompt_text().encode("utf-8")).hexdigest()


def _cacheable_model_config(model_config: Mapping[str, Any]) -> dict:
    """Drop credential-like keys so rotating a key never invalidates a cache."""
    if not isinstance(model_config, Mapping):
        return {"value": str(model_config)}
    return {
        str(key): value
        for key, value in model_config.items()
        if not _SECRET_KEY_RE.search(str(key))
    }


def _cacheable_feedback(feedback: Mapping[str, Any]) -> dict:
    favorites = feedback.get("favorites") or []
    canonical_favorites = [
        {
            "arxiv_id": item.get("arxiv_id"),
            "title": item.get("title"),
            "abstract": item.get("abstract"),
        }
        for item in favorites
        if isinstance(item, Mapping)
    ]
    ignored = sorted({_paper_key(key) for key in (feedback.get("ignored_keys") or [])})
    return {"favorites": canonical_favorites, "ignored_keys": ignored}


def rank_cache_key(
    papers: list[dict],
    profile: Mapping[str, Any],
    model_config: Mapping[str, Any],
    *,
    prompt_hash_value: str | None = None,
    feedback: Mapping[str, Any] | None = None,
) -> str:
    """SHA-256 over papers, profile, model, prompt, and feedback evidence."""
    payload = {
        "version": RANK_CACHE_VERSION,
        "prompt_sha256": prompt_hash_value or _default_prompt_hash(),
        "profile": profile,
        "model": _cacheable_model_config(model_config),
        "feedback": _cacheable_feedback(feedback or {}),
        "papers": [
            {
                "arxiv_id": paper.get("arxiv_id"),
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
            }
            for paper in papers
        ],
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _read_cache(path: Path) -> list[dict] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, list):
        return None
    return data


def _write_cache(path: Path, ranked: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.stem + "_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(ranked, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _assessments(raw: Any, papers: list[dict]) -> dict[str, dict]:
    if isinstance(raw, Mapping):
        rows = raw.get("evaluations")
        if rows is None and "score" in raw:
            rows = [raw]
    elif isinstance(raw, list):
        rows = raw
    else:
        rows = None
    if not isinstance(rows, list):
        raise ValueError("rank_fn must return a list or an object with evaluations")

    expected = {paper["arxiv_id"] for paper in papers}
    checked: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each evaluation must be an object")
        arxiv_id = row.get("arxiv_id")
        if not isinstance(arxiv_id, str) or arxiv_id not in expected:
            raise ValueError("rank_fn returned an unknown or missing arxiv_id")
        if arxiv_id in checked:
            raise ValueError("rank_fn returned a duplicate arxiv_id")
        score = row.get("score")
        if type(score) is not int or not 0 <= score <= 100:
            raise ValueError("rank score must be an integer between 0 and 100")
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("rank result must include a reason")
        limitation = row.get("limitation", "")
        if not isinstance(limitation, str):
            raise ValueError("rank limitation must be a string")
        refs = row.get("feedback_refs", [])
        if not isinstance(refs, list) or any(not isinstance(x, str) for x in refs):
            raise ValueError("feedback_refs must be a list of strings")
        checked[arxiv_id] = {
            "score": int(score),
            "reason": reason.strip(),
            "limitation": limitation.strip(),
            "feedback_refs": list(dict.fromkeys(refs)),
        }

    missing = expected - set(checked)
    if missing:
        raise ValueError(f"rank_fn omitted {len(missing)} paper(s)")
    return checked


def _default_rank_fn(
    papers: list[dict],
    profile: dict,
    feedback: dict,
    model_config: Mapping[str, Any],
) -> dict:
    """Real DeepSeek call; never exercised by the offline tests."""
    from openai import OpenAI

    from rank_papers import build_rank_messages, validate_rank_response

    api_key = str(model_config.get("api_key") or "").strip()
    base_url = str(model_config.get("base_url") or "").strip()
    model = str(model_config.get("model") or "").strip()
    if not all((api_key, base_url, model)):
        raise ValueError("model_config requires api_key, base_url, and model")

    with OpenAI(
        api_key=api_key, base_url=base_url, timeout=120, max_retries=0
    ) as client:
        response = client.chat.completions.create(
            model=model,
            messages=build_rank_messages(profile, feedback, papers),
            response_format={"type": "json_object"},
            max_tokens=4096,
            extra_body={"thinking": {"type": "disabled"}},
        )

    if not response.choices:
        raise ValueError("ranking model returned no choices")
    choice = response.choices[0]
    if choice.finish_reason != "stop":
        raise ValueError(f"ranking model did not finish cleanly: {choice.finish_reason}")
    content = choice.message.content
    if not content or not content.strip():
        raise ValueError("ranking model returned empty content")
    raw = json.loads(content)
    return {"evaluations": validate_rank_response(raw, papers, feedback)}


def rank_batch(
    papers: list[dict],
    profile: dict,
    feedback: dict,
    model_config: Mapping[str, Any],
    cache_dir: Path | str,
    rank_fn: RankFn | None = None,
    *,
    similarity_fn: SimilarityFn | None = None,
) -> list[dict]:
    """Score every non-ignored paper and return the ranked batch.

    Each item carries ``llm_score``, ``favorite_similarity``, ``final_score``,
    ``reason``, and ``feedback_refs``. ``min_score`` is intentionally *not*
    applied here: see :func:`select_digest`, which compares it with
    ``llm_score`` only. Results are cached by papers + profile + model +
    prompt hash + feedback.
    """
    profile = profile or {}
    feedback = feedback or {}
    model_config = model_config or {}
    ignored = {_paper_key(key) for key in (feedback.get("ignored_keys") or [])}
    candidates = [
        paper
        for paper in papers
        if _paper_key(paper.get("arxiv_id")) not in ignored
    ]
    if not candidates:
        return []

    cache_path = Path(cache_dir) / (
        rank_cache_key(
            candidates, profile, model_config, feedback=feedback
        )
        + ".json"
    )
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached

    ranker = rank_fn
    if ranker is None:
        ranker = lambda batch, prof, fb: _default_rank_fn(  # noqa: E731
            batch, prof, fb, model_config
        )
    rows = _assessments(ranker(candidates, profile, feedback), candidates)

    favorites = feedback.get("favorites") or []
    weight = _favorite_weight(profile, model_config)
    is_cold = cold_start(feedback)
    sim_fn = similarity_fn
    if not is_cold and sim_fn is None and weight > 0:
        # Loading E5 is pointless when favorites cannot change the score.
        sim_fn = _resolve_similarity_fn(model_config) or lexical_similarity

    ranked: list[dict] = []
    for paper in candidates:
        arxiv_id = paper["arxiv_id"]
        row = rows[arxiv_id]
        llm_score = row["score"]
        if is_cold or sim_fn is None:
            similarity = 0.0
        else:
            similarity = favorite_similarity(
                paper, favorites, similarity_fn=sim_fn
            )
        if weight == 0:
            final_score = float(llm_score)
        else:
            final_score = min(100.0, max(0.0, llm_score + weight * 100.0 * similarity))
        ranked.append(
            {
                **paper,
                "llm_score": llm_score,
                "favorite_similarity": similarity,
                "final_score": final_score,
                "reason": row["reason"],
                "limitation": row["limitation"],
                "feedback_refs": row["feedback_refs"],
                "assessment": {
                    "score": llm_score,
                    "reason": row["reason"],
                    "limitation": row["limitation"],
                    "feedback_refs": row["feedback_refs"],
                },
            }
        )

    ranked.sort(
        key=lambda item: (
            -float(item["final_score"]),
            -float(item["llm_score"]),
            _paper_key(item["arxiv_id"]),
            str(item["arxiv_id"]),
        )
    )
    _write_cache(cache_path, ranked)
    return ranked


def select_digest(
    ranked: list[dict], config: Any, ignored_keys: set[str]
) -> list[dict]:
    """Filter and order ranked papers for one digest.

    ``min_score`` is compared with ``llm_score`` only. Ignored families are
    hard-excluded, and a per-primary-category cap keeps one topic from filling
    the digest.
    """
    ignored = {_paper_key(key) for key in (ignored_keys or set())}
    min_score = int(_config_value(config, "min_score", DEFAULT_MIN_SCORE))
    limit = int(_config_value(config, "max_per_digest", DEFAULT_MAX_PER_DIGEST))
    category_cap = int(
        _config_value(config, "max_per_primary_category", DEFAULT_MAX_PER_CATEGORY)
    )
    if limit < 0 or category_cap < 0:
        raise ValueError("digest limits must be non-negative")

    eligible = [
        item
        for item in ranked
        if _paper_key(item.get("arxiv_id")) not in ignored
        and int(item.get("llm_score", 0)) >= min_score
    ]
    eligible.sort(
        key=lambda item: (
            -float(item.get("final_score", item.get("llm_score", 0))),
            -float(item.get("llm_score", 0)),
            _paper_key(item.get("arxiv_id")),
            str(item.get("arxiv_id", "")),
        )
    )

    selected: list[dict] = []
    per_category: Counter[str] = Counter()
    for item in eligible:
        if len(selected) >= limit:
            break
        categories = item.get("categories") or []
        primary = ""
        if categories:
            primary = str(categories[0])
        elif item.get("primary_category"):
            primary = str(item["primary_category"])
        if primary and per_category[primary] >= category_cap:
            continue
        selected.append(item)
        if primary:
            per_category[primary] += 1
    return selected
