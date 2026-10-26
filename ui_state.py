"""Private state for the simple push UI: recipient, verification, preview.

The raw recipient lives only in ``data/push/ui_state.json`` (gitignored). It is
never written to ``push_config.json``, ``.env``, logs, status payloads, or
preview references; previews identify the recipient by :func:`recipient_hash`.

This module has no Streamlit, network, SMTP, or model dependency so it can be
tested offline. Writes are atomic (temporary file plus ``os.replace``).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Mapping

STATE_REL = Path("data/push/ui_state.json")

# Deliberately simple ``user@host.tld`` validation: one local part, a dotted
# hostname, and a letters-only top-level domain of at least two characters.
EMAIL_RE = re.compile(
    r"^[A-Za-z0-9._%+\-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?)*"
    r"\.[A-Za-z]{2,}$"
)

# Preview references must use ``recipient_hash``. These keys would carry the
# raw address, so reject them instead of persisting an email in private state.
_PREVIEW_EMAIL_KEYS = frozenset(
    {"email", "recipient", "mail_to", "recipient_email", "to_address"}
)

_DEFAULTS = {
    "recipient": "",
    "verified_at": None,
    "preview": None,
    "last_manual_send": None,
    "test_email_sent_at": None,
    "test_email_recipient_hash": None,
}


class UiStateError(ValueError):
    """A safe-to-display state error that never contains the raw address."""


def recipient_hash(email: str) -> str:
    """Return the stable, non-reversible recipient reference used by previews."""
    if not isinstance(email, str):
        raise UiStateError("recipient must be text")
    return hashlib.sha256(email.strip().casefold().encode("utf-8")).hexdigest()


def _state_path(root: Path) -> Path:
    return Path(root) / STATE_REL


def _clean_email(email: str) -> str:
    if not isinstance(email, str):
        raise UiStateError("invalid email address")
    cleaned = email.strip()
    if not EMAIL_RE.fullmatch(cleaned):
        # Never echo the input: it may be a near-miss of a private address.
        raise UiStateError("invalid email address")
    return cleaned


def _checked_preview(preview: Mapping) -> dict:
    if not isinstance(preview, Mapping):
        raise UiStateError("preview must be an object")
    data = dict(preview)
    for key in data:
        if str(key).strip().casefold() in _PREVIEW_EMAIL_KEYS:
            raise UiStateError(
                "preview must reference recipient_hash, not an email"
            )
    return data


def _checked_state(state: Mapping) -> dict:
    if not isinstance(state, Mapping):
        raise UiStateError("ui state must be an object")
    data = {**_DEFAULTS, **dict(state)}

    recipient = data.get("recipient")
    if recipient is None:
        recipient = ""
    if not isinstance(recipient, str):
        raise UiStateError("invalid recipient")
    recipient = recipient.strip()
    if recipient and not EMAIL_RE.fullmatch(recipient):
        raise UiStateError("invalid recipient")
    data["recipient"] = recipient

    verified_at = data.get("verified_at")
    if verified_at is not None and not isinstance(verified_at, str):
        raise UiStateError("invalid verification timestamp")

    preview = data.get("preview")
    if preview is not None:
        data["preview"] = _checked_preview(preview)

    last_manual_send = data.get("last_manual_send")
    if last_manual_send is not None and not isinstance(
        last_manual_send, Mapping
    ):
        raise UiStateError("invalid manual send record")

    for key in ("test_email_sent_at", "test_email_recipient_hash"):
        value = data.get(key)
        if value is not None and not isinstance(value, str):
            raise UiStateError("invalid test email receipt")

    return data


def load_state(root: Path) -> dict:
    """Load private UI state; a missing file yields the defaults."""
    path = _state_path(root)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return dict(_DEFAULTS)
    except (OSError, UnicodeError):
        raise UiStateError("cannot read ui state") from None
    try:
        raw = json.loads(text)
    except ValueError:
        # A JSON error message can quote the document (and the address).
        raise UiStateError("cannot read ui state") from None
    return _checked_state(raw)


def save_state(root: Path, state: Mapping) -> dict:
    """Validate and atomically persist private UI state."""
    data = _checked_state(state)
    try:
        payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    except (TypeError, ValueError):
        raise UiStateError("cannot save ui state") from None

    path = _state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise UiStateError("cannot save ui state") from None
    return data


def resolve_recipient(root: Path, env: Mapping[str, str]) -> dict:
    """Resolve the active recipient and its verification state.

    An existing state file always wins, even when it is unreadable: a broken
    UI state fails closed and never silently falls back to ``.env``. Only a
    missing state file uses legacy ``MAIL_TO`` (verified for compatibility).
    """
    path = _state_path(root)
    if path.exists():
        try:
            state = load_state(root)
        except UiStateError:
            return {
                "recipient": "",
                "verified": False,
                "source": "ui_state",
                "state_exists": True,
            }
        recipient = state.get("recipient") or ""
        verified = bool(recipient and state.get("verified_at"))
        return {
            "recipient": recipient,
            "verified": verified,
            "source": "ui_state",
            "state_exists": True,
        }

    values = env or {}
    recipient = str(values.get("MAIL_TO") or "").strip()
    return {
        "recipient": recipient,
        "verified": bool(recipient),
        "source": "env",
        "state_exists": False,
    }


def mail_env(root: Path, env: Mapping[str, str]) -> dict:
    """Return a copy of ``env`` whose ``MAIL_TO`` is the current recipient.

    Verification is intentionally not required: this is also used to send the
    verification email to a newly saved, still-unverified address.
    """
    resolved = resolve_recipient(root, env or {})
    recipient = resolved["recipient"]
    if not recipient:
        raise UiStateError("no recipient configured")
    merged = dict(env or {})
    merged["MAIL_TO"] = recipient
    return merged


def set_recipient(root: Path, email: str) -> dict:
    """Save the current recipient, invalidating verification when it changes.

    Changing the address (including changing back to an older one) clears both
    ``verified_at`` and the preview reference; re-saving the same normalized
    address preserves them.
    """
    cleaned = _clean_email(email)
    state = load_state(root)
    current = state.get("recipient") or ""
    if current and recipient_hash(current) == recipient_hash(cleaned):
        return state

    state["recipient"] = cleaned
    state["verified_at"] = None
    state["preview"] = None
    # A test email proves delivery to the old address only.
    state["test_email_sent_at"] = None
    state["test_email_recipient_hash"] = None
    return save_state(root, state)


def mark_verified(root: Path, email: str, verified_at: str) -> dict:
    """Record a successful verification for the current recipient only."""
    cleaned = _clean_email(email)
    if not isinstance(verified_at, str) or not verified_at.strip():
        raise UiStateError("invalid verification timestamp")

    state = load_state(root)
    current = state.get("recipient") or ""
    if current and recipient_hash(current) != recipient_hash(cleaned):
        # Never echo the address: it is private state.
        raise UiStateError("recipient has changed; verify the current recipient")

    state["recipient"] = current or cleaned
    state["verified_at"] = verified_at.strip()
    return save_state(root, state)


def record_test_email(root: Path, email: str, sent_at: str) -> dict:
    """Remember that a test email was accepted for the current recipient.

    This is deliberately separate from ``verified_at``: the SMTP server
    accepting the test is not the user confirming receipt. The receipt only
    proves which address a test was sent to, so a recipient change clears it.
    """
    cleaned = _clean_email(email)
    if not isinstance(sent_at, str) or not sent_at.strip():
        raise UiStateError("invalid test email timestamp")

    state = load_state(root)
    current = state.get("recipient") or ""
    if current and recipient_hash(current) != recipient_hash(cleaned):
        raise UiStateError("recipient has changed; test the current recipient")

    state["recipient"] = current or cleaned
    state["test_email_sent_at"] = sent_at.strip()
    state["test_email_recipient_hash"] = recipient_hash(cleaned)
    return save_state(root, state)


def clear_test_email(root: Path) -> dict:
    """Drop any recorded test-email receipt (used after a failed send)."""
    if not _state_path(root).exists():
        return dict(_DEFAULTS)
    state = load_state(root)
    state["test_email_sent_at"] = None
    state["test_email_recipient_hash"] = None
    return save_state(root, state)


def test_email_confirmed(root: Path, email: str) -> bool:
    """Whether a test email was accepted for exactly this recipient."""
    if not _state_path(root).exists():
        return False
    try:
        state = load_state(root)
        cleaned = _clean_email(email)
    except UiStateError:
        return False
    sent_at = state.get("test_email_sent_at")
    hashed = state.get("test_email_recipient_hash")
    return bool(
        isinstance(sent_at, str)
        and sent_at
        and hashed == recipient_hash(cleaned)
    )


def set_preview(root: Path, preview: Mapping) -> dict:
    """Store a preview reference (``recipient_hash`` only, never an email)."""
    checked = _checked_preview(preview)
    state = load_state(root)
    state["preview"] = checked
    return save_state(root, state)


def get_preview(root: Path) -> dict | None:
    """Return the stored preview reference, or ``None`` when unusable."""
    if not _state_path(root).exists():
        return None
    try:
        state = load_state(root)
    except UiStateError:
        return None
    preview = state.get("preview")
    return dict(preview) if isinstance(preview, Mapping) else None


def record_manual_send(root: Path, run_id: str, at: str, count: int) -> dict:
    """Remember the last successful manual send for the simple page."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise UiStateError("invalid run id")
    if not isinstance(at, str) or not at.strip():
        raise UiStateError("invalid timestamp")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise UiStateError("invalid paper count")

    state = load_state(root)
    state["last_manual_send"] = {
        "run_id": run_id.strip(),
        "at": at.strip(),
        "count": count,
    }
    return save_state(root, state)
