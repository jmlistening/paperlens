"""Read/write and validate the user's personalized discovery profile.

This is the source of the push candidate pool: the user's query groups and
arXiv categories decide which papers can enter ranking. The LLM only orders
the candidates; it never adds topics.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from fetch_papers import parse_profile
from push_fetch import build_queries

ROOT = Path(__file__).resolve().parent
# The personal profile lives under the gitignored data directory; the legacy
# root file is kept only as a local fallback, and the public example is read-only.
ACTIVE_PROFILE_REL = Path("data/push/profile.json")
LEGACY_PROFILE = ROOT / "interests.json"
EXAMPLE_PROFILE = ROOT / "interests.example.json"
DEFAULT_PROFILE = LEGACY_PROFILE
CATEGORY_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z][A-Za-z0-9-]*)?$"
)


class ProfileError(ValueError):
    """A user-facing profile validation error."""


def _string_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ProfileError(f"{name} must be a list of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ProfileError(f"{name} entries must be non-empty strings")
        result.append(item.strip())
    return result


def validate_categories(categories: Iterable[str] | None) -> list[str]:
    checked: list[str] = []
    for value in categories or ():
        if not isinstance(value, str) or not value.strip():
            raise ProfileError("search categories must be non-empty strings")
        category = value.strip()
        if not CATEGORY_RE.fullmatch(category):
            raise ProfileError(f"invalid arXiv category: {value!r}")
        if category not in checked:
            checked.append(category)
    return checked


def validate_profile(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, dict):
        raise ProfileError("profile must be a JSON object")
    try:
        parse_profile(profile)
    except (TypeError, ValueError) as exc:
        raise ProfileError(str(exc)) from exc

    goal = profile.get("research_goal", "")
    if not isinstance(goal, str):
        raise ProfileError("research_goal must be a string")
    topics = _string_list(profile.get("topics"), "topics")
    preferred = _string_list(
        profile.get("preferred_methods"), "preferred_methods"
    )
    less = _string_list(profile.get("less_interested"), "less_interested")
    if not topics:
        raise ProfileError("topics must contain at least one item")
    for name, value, low, high in (
        ("top_k", profile.get("top_k", 3), 1, 10),
        ("min_score", profile.get("min_score", 60), 0, 100),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ProfileError(f"{name} must be an integer")
        if not low <= value <= high:
            raise ProfileError(f"{name} must be between {low} and {high}")

    return {
        **profile,
        "research_goal": goal.strip(),
        "topics": topics,
        "preferred_methods": preferred,
        "less_interested": less,
    }


def load_profile(path: Path = DEFAULT_PROFILE) -> dict[str, Any]:
    path = Path(path)
    try:
        profile = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as exc:
        raise ProfileError(f"cannot read profile: {exc}") from exc
    return validate_profile(profile)


def save_profile(
    profile: dict[str, Any], path: Path = DEFAULT_PROFILE
) -> dict[str, Any]:
    validated = validate_profile(profile)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(validated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return validated


def profile_path(root: Path = ROOT) -> Path:
    """Return the profile to read: active, then legacy, then the example.

    All paths are resolved relative to ``root`` so callers can point the
    resolver at an isolated test or deployment directory. Saving goes through
    ``save_active_profile`` and never writes the legacy or example files.
    """
    root = Path(root)
    active = root / ACTIVE_PROFILE_REL
    if active.is_file():
        return active
    legacy = root / LEGACY_PROFILE.name
    if legacy.is_file():
        return legacy
    return root / EXAMPLE_PROFILE.name


def load_active_profile(root: Path = ROOT) -> dict[str, Any]:
    """Load and validate the profile selected by ``profile_path``."""
    return load_profile(profile_path(root))


def save_active_profile(
    profile: dict[str, Any], root: Path = ROOT
) -> dict[str, Any]:
    """Validate and atomically write the private active profile."""
    return save_profile(profile, Path(root) / ACTIVE_PROFILE_REL)


def profile_sha256(profile: dict[str, Any]) -> str:
    """Return the canonical JSON hash used for preview freshness checks."""
    canonical = json.dumps(
        validate_profile(profile), sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def profile_queries(
    profile: dict[str, Any], categories: Iterable[str] | None = None
) -> list[dict[str, str]]:
    """Return the group queries that define the candidate pool (no network)."""
    validated = validate_profile(profile)
    checked = validate_categories(categories)
    return build_queries(validated, checked)


def split_lines(value: str) -> list[str]:
    return [line.strip() for line in (value or "").splitlines() if line.strip()]


def split_terms(value: str) -> list[str]:
    terms: list[str] = []
    for chunk in (value or "").replace("\n", ",").split(","):
        term = chunk.strip()
        if term and term not in terms:
            terms.append(term)
    return terms
