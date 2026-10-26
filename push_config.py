"""Push configuration and environment validation for PaperLens (Task P1).

The JSON file is the single source of truth for push behaviour and never holds
credentials. SMTP credentials are read only from a mapping supplied by the
caller (normally ``os.environ``); validation never echoes their values and
never writes to stdout/stderr.

This module performs no network, SMTP, or paid-API calls and never reads a
dotenv file or the interests profile.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("push_config.json")

CUSTOM_PROVIDER = "custom"
SUPPORTED_FREQUENCIES = frozenset({"daily"})
SUPPORTED_SECURITY = frozenset({"ssl", "starttls"})

_TIME_LOCAL_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
_SECRET_KEY_PATTERN = re.compile(
    r"(password|passwd|secret|token|api[_-]?key|auth(?:orization)?[_-]?code)",
    re.IGNORECASE,
)
_MAIL_TO_SEPARATOR = re.compile(r"[,;]")


class PushConfigError(ValueError):
    """Raised when push configuration is malformed or contains bad keys."""


@dataclass(frozen=True)
class SmtpPreset:
    """Transport preset for a supported mail provider."""

    host: str
    port: int
    security: str


SMTP_PRESETS: dict[str, SmtpPreset] = {
    "163": SmtpPreset(host="smtp.163.com", port=465, security="ssl"),
    "qq": SmtpPreset(host="smtp.qq.com", port=465, security="ssl"),
}


@dataclass(frozen=True)
class PushConfig:
    """Push pipeline settings (spec section 7.1)."""

    enabled: bool = False
    paused: bool = False
    quiet_mode: bool = False
    frequency: str = "daily"
    time_local: str = "08:30"
    provider: str = "163"
    attach_pdfs: bool = False
    window_hours: int = 72
    overlap_hours: int = 24
    catchup_days: int = 7
    fetch_pool_limit: int = 40
    rank_max_candidates: int = 20
    max_per_digest: int = 3
    max_per_primary_category: int = 2
    min_new_papers: int = 1
    auto_pause_after_failures: int = 3
    search_categories: tuple[str, ...] | None = None


_CONFIG_FIELD_NAMES = frozenset(field.name for field in fields(PushConfig))


def _secret_paths(data: Any, prefix: str = "") -> list[str]:
    """Return dotted key paths whose names look like secrets (values ignored)."""
    paths: list[str] = []
    if isinstance(data, Mapping):
        for key, value in data.items():
            name = str(key)
            path = f"{prefix}{name}"
            if _SECRET_KEY_PATTERN.search(name):
                paths.append(path)
                continue
            paths.extend(_secret_paths(value, f"{path}."))
    elif isinstance(data, (list, tuple)):
        for index, value in enumerate(data):
            paths.extend(_secret_paths(value, f"{prefix}[{index}]."))
    return paths


def _bool_field(data: Mapping[str, Any], name: str, default: bool) -> bool:
    if name not in data:
        return default
    value = data[name]
    if not isinstance(value, bool):
        raise PushConfigError(f"push config field '{name}' must be a boolean")
    return value


def _str_field(data: Mapping[str, Any], name: str, default: str) -> str:
    if name not in data:
        return default
    value = data[name]
    if not isinstance(value, str) or not value.strip():
        raise PushConfigError(f"push config field '{name}' must be a non-empty string")
    return value.strip()


def _int_field(data: Mapping[str, Any], name: str, default: int) -> int:
    if name not in data:
        return default
    value = data[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise PushConfigError(f"push config field '{name}' must be an integer")
    return value


def _categories_field(
    data: Mapping[str, Any], name: str, default: tuple[str, ...] | None
) -> tuple[str, ...] | None:
    if name not in data or data[name] is None:
        return default
    value = data[name]
    if not isinstance(value, (list, tuple)):
        raise PushConfigError(
            f"push config field '{name}' must be a list of non-empty strings"
        )
    categories: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise PushConfigError(
                f"push config field '{name}' must be a list of non-empty strings"
            )
        categories.append(item.strip())
    return tuple(categories) if categories else None


def parse_push_config(data: Mapping[str, Any]) -> PushConfig:
    """Build a ``PushConfig`` from an already-decoded JSON object."""
    if not isinstance(data, Mapping):
        raise PushConfigError("push config must be a JSON object")

    secret_paths = _secret_paths(data)
    if secret_paths:
        raise PushConfigError(
            "push config must not contain secret fields "
            f"({', '.join(sorted(secret_paths))}); "
            "keep SMTP credentials in the process environment"
        )

    unknown = sorted(str(key) for key in data if key not in _CONFIG_FIELD_NAMES)
    if unknown:
        raise PushConfigError(f"unknown push config keys: {', '.join(unknown)}")

    defaults = PushConfig()
    config = PushConfig(
        enabled=_bool_field(data, "enabled", defaults.enabled),
        paused=_bool_field(data, "paused", defaults.paused),
        quiet_mode=_bool_field(data, "quiet_mode", defaults.quiet_mode),
        frequency=_str_field(data, "frequency", defaults.frequency),
        time_local=_str_field(data, "time_local", defaults.time_local),
        provider=_str_field(data, "provider", defaults.provider),
        attach_pdfs=_bool_field(data, "attach_pdfs", defaults.attach_pdfs),
        window_hours=_int_field(data, "window_hours", defaults.window_hours),
        overlap_hours=_int_field(data, "overlap_hours", defaults.overlap_hours),
        catchup_days=_int_field(data, "catchup_days", defaults.catchup_days),
        fetch_pool_limit=_int_field(
            data, "fetch_pool_limit", defaults.fetch_pool_limit
        ),
        rank_max_candidates=_int_field(
            data, "rank_max_candidates", defaults.rank_max_candidates
        ),
        max_per_digest=_int_field(data, "max_per_digest", defaults.max_per_digest),
        max_per_primary_category=_int_field(
            data, "max_per_primary_category", defaults.max_per_primary_category
        ),
        min_new_papers=_int_field(data, "min_new_papers", defaults.min_new_papers),
        auto_pause_after_failures=_int_field(
            data, "auto_pause_after_failures", defaults.auto_pause_after_failures
        ),
        search_categories=_categories_field(
            data, "search_categories", defaults.search_categories
        ),
    )

    if config.provider not in SMTP_PRESETS and config.provider != CUSTOM_PROVIDER:
        raise PushConfigError(
            f"unknown push provider '{config.provider}'; expected one of: "
            f"{', '.join(sorted(SMTP_PRESETS))}, {CUSTOM_PROVIDER}"
        )
    return config


def load_push_config(path: Path = DEFAULT_CONFIG_PATH) -> PushConfig:
    """Read a push config JSON file. Never writes to the file or mutates data."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    raw = config_path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PushConfigError(
            f"push config is not valid JSON: {config_path.name}"
        ) from exc
    return parse_push_config(data)


def _env_value(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "")
    return value.strip() if isinstance(value, str) else ""


def _int_problem(
    name: str, value: Any, minimum: int, maximum: int | None = None
) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return f"{name} must be an integer"
    if maximum is None:
        if value < minimum:
            return f"{name} must be >= {minimum}"
        return None
    if not minimum <= value <= maximum:
        return f"{name} must be between {minimum} and {maximum}"
    return None


def validate_push_config(
    config: PushConfig, env: Mapping[str, str], require_smtp: bool = True
) -> list[str]:
    """Return human-readable configuration problems.

    Messages name missing variables or bad fields only: environment values,
    credentials, and recipient addresses are never included.

    ``require_smtp=False`` validates only the config file's structure. Previews
    use it: they render a forecast without an SMTP connection, so missing
    credentials or a missing ``MAIL_TO`` must not block them. Scheduled and
    send runs keep the default strict validation.
    """
    if not isinstance(config, PushConfig):
        raise TypeError("config must be a PushConfig")
    environment: Mapping[str, str] = env if env is not None else {}

    problems: list[str] = []

    if not isinstance(config.provider, str) or (
        config.provider not in SMTP_PRESETS
        and config.provider != CUSTOM_PROVIDER
    ):
        problems.append(
            f"provider '{config.provider}' is not supported; "
            f"expected one of: {', '.join(sorted(SMTP_PRESETS))}, {CUSTOM_PROVIDER}"
        )
    if not isinstance(config.frequency, str) or (
        config.frequency not in SUPPORTED_FREQUENCIES
    ):
        problems.append(
            f"frequency must be one of: {', '.join(sorted(SUPPORTED_FREQUENCIES))}"
        )
    if not isinstance(config.time_local, str) or not _TIME_LOCAL_PATTERN.match(
        config.time_local
    ):
        problems.append("time_local must be HH:MM in 24-hour local time")

    for name, value, minimum, maximum in (
        ("window_hours", config.window_hours, 1, None),
        ("overlap_hours", config.overlap_hours, 0, None),
        ("catchup_days", config.catchup_days, 1, 365),
        ("fetch_pool_limit", config.fetch_pool_limit, 1, None),
        ("rank_max_candidates", config.rank_max_candidates, 1, None),
        ("max_per_digest", config.max_per_digest, 1, 10),
        ("max_per_primary_category", config.max_per_primary_category, 1, 10),
        ("min_new_papers", config.min_new_papers, 0, None),
        (
            "auto_pause_after_failures",
            config.auto_pause_after_failures,
            1,
            None,
        ),
    ):
        problem = _int_problem(name, value, minimum, maximum)
        if problem:
            problems.append(problem)

    if (
        isinstance(config.rank_max_candidates, int)
        and not isinstance(config.rank_max_candidates, bool)
        and isinstance(config.fetch_pool_limit, int)
        and not isinstance(config.fetch_pool_limit, bool)
        and config.rank_max_candidates > config.fetch_pool_limit
    ):
        problems.append("rank_max_candidates must not exceed fetch_pool_limit")

    if config.search_categories is not None:
        categories = config.search_categories
        if not isinstance(categories, (list, tuple)):
            problems.append("search_categories must be a list of non-empty strings")
        elif any(
            not isinstance(item, str) or not item.strip() for item in categories
        ):
            problems.append("search_categories must be a list of non-empty strings")

    if not require_smtp:
        return problems

    if not config.enabled:
        return problems

    for name in ("SMTP_USER", "SMTP_PASSWORD", "MAIL_TO"):
        if not _env_value(environment, name):
            problems.append(f"{name} is required when push is enabled")

    smtp_user = _env_value(environment, "SMTP_USER")
    if smtp_user and "@" not in smtp_user:
        problems.append("SMTP_USER must be a full email address")

    mail_to = _env_value(environment, "MAIL_TO")
    if mail_to:
        recipients = [
            token.strip()
            for token in _MAIL_TO_SEPARATOR.split(mail_to)
            if token.strip()
        ]
        if not recipients or any("@" not in token for token in recipients):
            problems.append("MAIL_TO must contain valid email addresses")

    if config.provider == CUSTOM_PROVIDER:
        for name in ("SMTP_HOST", "SMTP_PORT", "SMTP_SECURITY"):
            if not _env_value(environment, name):
                problems.append(f"{name} is required when provider is 'custom'")

        port_text = _env_value(environment, "SMTP_PORT")
        if port_text:
            try:
                port = int(port_text)
            except ValueError:
                problems.append("SMTP_PORT must be an integer")
            else:
                if not 1 <= port <= 65535:
                    problems.append("SMTP_PORT must be between 1 and 65535")

        security = _env_value(environment, "SMTP_SECURITY").lower()
        if security and security not in SUPPORTED_SECURITY:
            problems.append(
                "SMTP_SECURITY must be one of: "
                f"{', '.join(sorted(SUPPORTED_SECURITY))}"
            )

    return problems
