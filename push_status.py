"""Read-only, masked status view for the optional PaperLens push.

This module is a thin wrapper over :func:`push_ctl.status_json` and
:func:`push_ctl.history`. It exists so the Streamlit page and CLI users have a
single view that is guaranteed to be free of credentials and full recipient
addresses.

Nothing here performs network, SMTP, or model calls.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from dotenv import dotenv_values

import push_ctl
from mail_transport import mask_email
from push_config import DEFAULT_CONFIG_PATH, load_push_config

ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY_LIMIT = 10

# Statuses that mean "the pipeline produced something a user asked for".
_SUCCESS_STATUSES = frozenset({"preview_ready", "quiet_preview", "sent", "already_sent"})
STALE_AFTER = timedelta(hours=48)

_SECRET_KEY_PATTERN = re.compile(
    r"(password|passwd|secret|api[_-]?key|authorization[_-]?code|auth[_-]?code)",
    re.IGNORECASE,
)
_EMAIL_PATTERN = re.compile(r"[^@\s,;]+@[^@\s,;]+\.[^@\s,;]+")


def load_env(root: Path = ROOT) -> dict[str, str]:
    """Read ``<root>/.env`` without printing or logging any value.

    Only keys and values are returned to the caller; validation and masking
    happen later. A missing file yields an empty mapping.
    """
    values = dotenv_values(Path(root) / ".env", encoding="utf-8-sig")
    return {key: str(value) for key, value in values.items() if value is not None}


def _secret_values(env: Mapping[str, str] | None) -> list[str]:
    """Values that must never appear in a status payload."""
    if not env:
        return []
    secrets: list[str] = []
    for name, value in env.items():
        if not isinstance(value, str) or not value:
            continue
        if _SECRET_KEY_PATTERN.search(str(name)):
            secrets.append(value)
    # Full recipient addresses are masked, not merely removed.
    mail_to = env.get("MAIL_TO")
    if isinstance(mail_to, str):
        for token in re.split(r"[,;]", mail_to):
            token = token.strip()
            if token:
                secrets.append(token)
    return sorted(set(secrets), key=len, reverse=True)


def _redact(value: Any, secrets: list[str]) -> Any:
    """Recursively drop secret-named keys and scrub known secret values."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if _SECRET_KEY_PATTERN.search(str(key)):
                continue
            result[str(key)] = _redact(item, secrets)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, str):
        text = value
        for secret in secrets:
            if not secret:
                continue
            if "@" in secret:
                text = text.replace(secret, mask_email(secret))
            else:
                text = text.replace(secret, "[redacted]")
        # Mask any address that slipped through from another source.
        return _EMAIL_PATTERN.sub(lambda match: mask_email(match.group(0)), text)
    return value


def _next_expected(config: Any, now: datetime | None) -> str | None:
    """Next local occurrence of ``time_local`` while enabled and not paused."""
    if not getattr(config, "enabled", False) or getattr(config, "paused", False):
        return None
    base = now if now is not None else datetime.now()
    if base.tzinfo is not None:
        base = base.astimezone()
    try:
        hour, minute = (int(part) for part in str(config.time_local).split(":"))
        candidate = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
    except (TypeError, ValueError):
        return None
    if candidate <= base:
        candidate += timedelta(days=1)
    return candidate.isoformat()


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed


def _last_success(runs: list[dict[str, Any]]) -> str | None:
    for row in runs:
        if row.get("status") in _SUCCESS_STATUSES:
            return row.get("finished_at") or row.get("started_at")
    return None


def _is_stale(last_success: str | None, now: datetime | None) -> bool:
    timestamp = _parse_timestamp(last_success)
    if timestamp is None:
        return False
    current = now if now is not None else datetime.now()
    if current.tzinfo is None:
        current = current.astimezone()
    else:
        current = current.astimezone()
    return current - timestamp > STALE_AFTER


def _tokens(runs: list[dict[str, Any]]) -> int:
    total = 0
    for row in runs:
        value = row.get("tokens", 0)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            total += value
    return total


def status(
    root: Path = ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return one masked status object suitable for the CLI or Streamlit.

    The returned mapping always contains ``enabled``, ``paused``, ``quiet``,
    ``provider``, ``mail_to`` (masked), ``recipient_verified``,
    ``recipient_source``, ``first_send_approved``, ``auto_paused``, ``counts``,
    and ``runs``. The recipient comes from :mod:`ui_state` and the raw
    address never leaves this function.
    """
    root = Path(root)
    config_path = Path(config_path)
    environment = dict(env) if env is not None else load_env(root)
    config = load_push_config(config_path)

    raw = push_ctl.status_json(root, config_path, environment)
    runs = _redact(push_ctl.history(root, limit=DEFAULT_HISTORY_LIMIT), _secret_values(environment))
    counts = dict(raw.get("counts") or {})
    last_success = _last_success(runs)
    mail_to = raw.get("mail_to")

    result: dict[str, Any] = {
        "enabled": bool(raw.get("enabled")),
        "paused": bool(raw.get("paused")),
        "quiet": bool(raw.get("quiet_mode")),
        "quiet_mode": bool(raw.get("quiet_mode")),
        "provider": raw.get("provider"),
        "mail_to": mask_email(mail_to) if isinstance(mail_to, str) and mail_to else None,
        "recipient_verified": bool(raw.get("recipient_verified")),
        "recipient_source": raw.get("recipient_source"),
        "first_send_approved": bool(raw.get("first_send_approved")),
        "auto_paused": bool(raw.get("auto_paused")),
        "counts": counts,
        "runs": dict(raw.get("runs") or {}),
        "config_problems": list(raw.get("config_problems") or []),
        "queue": int(counts.get("queued", 0)),
        "last_success": last_success,
        "next_expected": _next_expected(config, now),
        "stale": _is_stale(last_success, now),
        "tokens": _tokens(runs),
        "recent_runs": runs,
    }
    return _redact(result, _secret_values(environment))


def recent_runs(
    root: Path = ROOT,
    limit: int = DEFAULT_HISTORY_LIMIT,
    env: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return recent ``run.json`` records with secrets and addresses scrubbed."""
    environment = dict(env) if env is not None else load_env(root)
    rows = push_ctl.history(Path(root), limit=int(limit))
    return _redact(rows, _secret_values(environment))


def status_json(
    config_path: Path = DEFAULT_CONFIG_PATH,
    root: Path = ROOT,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper matching ``push_ctl.status_json``'s name."""
    return status(root=root, config_path=config_path, env=env)


def history(
    root: Path = ROOT,
    limit: int = DEFAULT_HISTORY_LIMIT,
    env: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Compatibility wrapper matching ``push_ctl.history``'s name."""
    return recent_runs(root=root, limit=limit, env=env)


__all__ = [
    "DEFAULT_HISTORY_LIMIT",
    "history",
    "load_env",
    "recent_runs",
    "status",
    "status_json",
]
