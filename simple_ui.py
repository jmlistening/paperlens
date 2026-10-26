"""Core logic for the simple daily-subscription page.

This module has no Streamlit dependency so every behaviour can be tested
offline. It never logs or returns the raw recipient: the address lives only in
``data/push/ui_state.json`` and is addressed by :func:`ui_state.recipient_hash`
in preview references. No network, SMTP, or model call happens at import time;
the page injects a ``chat_fn`` for topic-to-query generation.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import push_ctl
import push_status
import ui_state
from mail_transport import mask_email
from profile_config import (
    ProfileError,
    load_active_profile,
    save_active_profile,
    split_lines,
)
from push_config import PushConfigError, load_push_config, parse_push_config
from push_run import PushDeps, RunResult, run_push
from push_state import PushState
from topic_queries import generate_query_groups

_TIME_LOCAL_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

TASK_TIME_ERROR = "计划任务时间未更新"
TASK_INSTALL_ERROR = "计划任务安装失败，未开启自动推送"
TEST_EMAIL_REQUIRED = "请先发送测试邮件并确认收到"
_PREVIEW_STATUSES = frozenset({"preview_ready", "quiet_preview"})


class SimpleUiError(ValueError):
    """A safe, user-facing error for the simple page."""


def _as_mapping(env: Mapping[str, str] | None) -> dict[str, str]:
    return {str(key): str(value) for key, value in (env or {}).items()}


def _checked_count(count: Any) -> int:
    if isinstance(count, bool) or not isinstance(count, int):
        raise SimpleUiError("每次推送篇数需要是 1—10 的整数")
    if not 1 <= count <= 10:
        raise SimpleUiError("每次推送篇数需要在 1—10 之间")
    return count


def _checked_time(time_local: Any) -> str:
    text = str(time_local or "").strip()
    if not _TIME_LOCAL_RE.fullmatch(text):
        raise SimpleUiError("推送时间格式应为 HH:MM")
    return text


def _update_config(config_path: Path, *, count: int, time_local: str) -> None:
    """Write only the simple page's two config fields, atomically."""
    path = Path(config_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise SimpleUiError("无法读取推送配置") from None
    if not isinstance(raw, dict):
        raise SimpleUiError("推送配置格式不正确")
    raw["max_per_digest"] = count
    raw["time_local"] = time_local
    try:
        parse_push_config(raw)
    except PushConfigError as exc:
        raise SimpleUiError(f"推送配置未保存：{exc}") from None
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise SimpleUiError("无法保存推送配置") from None


def save_settings(
    root: Path,
    config_path: Path,
    *,
    topics_text: str,
    email: str,
    count: int,
    time_local: str,
    chat_fn: Callable[[list[dict]], str] | None,
) -> dict[str, Any]:
    """Save topics/recipient/count/time with a topic transaction.

    Topic generation runs *before* any write: when it fails, neither the new
    topic nor its query groups are saved (and the page must not preview). Email,
    count, and time are independent of that transaction. Changing the saved
    recipient while auto-push is on first turns auto-push off and clears
    verification. Changing the time while auto-push stays on re-registers the
    scheduled task; a failure is reported without pretending it took effect.
    """
    root = Path(root)
    config_path = Path(config_path)
    count = _checked_count(count)
    time_local = _checked_time(time_local)
    text = str(topics_text or "").strip()
    topic_lines = split_lines(text)
    if not text or not topic_lines:
        raise SimpleUiError("请填写想关注的主题")

    try:
        profile = load_active_profile(root)
    except FileNotFoundError:
        raise SimpleUiError("找不到兴趣配置") from None
    except ProfileError as exc:
        raise SimpleUiError(f"兴趣配置无效：{exc}") from None

    try:
        config = load_push_config(config_path)
    except (OSError, PushConfigError) as exc:
        raise SimpleUiError(f"无法读取推送配置：{exc}") from None

    # The saved profile may predate ``research_goal``; its newline-joined
    # topics are the equivalent text, so editing count/time must not look like
    # a topic change (which would trigger a paid generation call).
    existing_goal = str(profile.get("research_goal") or "").strip()
    existing_topics_text = "\n".join(profile.get("topics") or [])
    topics_changed = text != existing_goal and text != existing_topics_text

    try:
        state = ui_state.load_state(root)
    except ui_state.UiStateError as exc:
        raise SimpleUiError(str(exc)) from None
    saved_recipient = state.get("recipient") or ""
    new_recipient = str(email or "").strip() or saved_recipient
    if not new_recipient:
        raise SimpleUiError("请填写要推送的邮箱")
    if not ui_state.EMAIL_RE.fullmatch(new_recipient):
        raise SimpleUiError("请输入有效的邮箱地址")
    recipient_changed = (
        ui_state.recipient_hash(new_recipient)
        != ui_state.recipient_hash(saved_recipient)
    )

    groups: list[dict] | None = None
    updated_profile = dict(profile)
    if topics_changed:
        # TopicQueryError propagates unchanged: its messages are sanitized.
        groups = generate_query_groups(text, chat_fn)
        search = dict(profile.get("search") or {})
        search["groups"] = groups
        updated_profile.update(
            research_goal=text,
            topics=topic_lines,
            search=search,
        )
    updated_profile["top_k"] = count

    auto_disabled = False
    if config.enabled and recipient_changed:
        try:
            push_ctl.set_auto_push(False, root, config_path, env={})
        except push_ctl.PushCtlError as exc:
            raise SimpleUiError(f"未能关闭自动推送：{exc}") from None
        auto_disabled = True

    try:
        save_active_profile(updated_profile, root)
    except ProfileError as exc:
        raise SimpleUiError(f"兴趣配置未保存：{exc}") from None

    try:
        ui_state.set_recipient(root, new_recipient)
    except ui_state.UiStateError as exc:
        raise SimpleUiError(f"邮箱未保存：{exc}") from None

    _update_config(config_path, count=count, time_local=time_local)

    task_updated = False
    task_error: str | None = None
    time_changed = config.time_local != time_local
    if time_changed and config.enabled and not auto_disabled:
        try:
            outcome = push_ctl.register_task(root, time_local)
        except push_ctl.PushCtlError:
            task_error = TASK_TIME_ERROR
        else:
            if isinstance(outcome, Mapping) and outcome.get("installed"):
                task_updated = True
            else:
                task_error = TASK_TIME_ERROR

    final_config = load_push_config(config_path)
    return {
        "saved": True,
        "topics_changed": bool(topics_changed),
        "groups": groups,
        "recipient_changed": bool(recipient_changed),
        "auto_disabled": bool(auto_disabled),
        "task_updated": bool(task_updated),
        "task_error": task_error,
        "count": count,
        "time_local": time_local,
        "enabled": bool(final_config.enabled),
    }


def _read_digest(root: Path, run_id: str | None) -> dict[str, Any] | None:
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        return None
    path = Path(root) / "data" / "push" / "runs" / run_id / "digest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _record_preview_reference(
    root: Path,
    result: RunResult,
    *,
    env: Mapping[str, str] | None,
    now: datetime,
) -> dict[str, Any] | None:
    if result.status not in _PREVIEW_STATUSES or not result.run_id:
        return None
    payload = _read_digest(root, result.run_id)
    if payload is None:
        return None
    created_at = payload.get("created_at")
    profile_hash = payload.get("profile_sha256")
    papers = payload.get("papers")
    if (
        not isinstance(created_at, str)
        or not isinstance(profile_hash, str)
        or not isinstance(papers, list)
    ):
        return None
    resolved = ui_state.resolve_recipient(root, _as_mapping(env))
    recipient = resolved["recipient"]
    if not recipient:
        # A preview may run before an email is saved; without a recipient the
        # reference cannot bind freshness, so no reference is written.
        return None
    reference = {
        "run_id": result.run_id,
        "created_at": created_at,
        "profile_sha256": profile_hash,
        "count": int(result.selected_count),
        "recipient_hash": ui_state.recipient_hash(recipient),
        "local_date": now.astimezone().date().isoformat(),
    }
    try:
        ui_state.set_preview(root, reference)
    except ui_state.UiStateError:
        # The preview itself succeeded; without a writable reference a later
        # manual send simply regenerates instead of reusing this forecast.
        return None
    return reference


def preview(
    root: Path,
    config_path: Path,
    env: Mapping[str, str] | None,
    *,
    deps: PushDeps | None = None,
    now: datetime | None = None,
) -> RunResult:
    """Generate a forecast preview and record its freshness reference.

    The preview is the forecast of the next manual send, so it searches the
    same recent ``catchup_days`` window and re-opens non-delivered papers that
    an earlier topic only evaluated.
    """
    root = Path(root)
    config_path = Path(config_path)
    current = now or datetime.now(timezone.utc)
    deps = deps or PushDeps(root=root, env=_as_mapping(env))
    config = load_push_config(config_path)
    result = run_push(
        "preview",
        config_path=config_path,
        now=current,
        deps=deps,
        force=True,
        lookback_days=int(config.catchup_days),
    )
    _record_preview_reference(root, result, env=env, now=current)
    return result


def preview_cards(root: Path, run_id: str | None) -> list[dict[str, str]]:
    """Return only the card fields the simple page may show."""
    payload = _read_digest(Path(root), run_id)
    if payload is None:
        return []
    cards: list[dict[str, str]] = []
    for item in payload.get("papers") or []:
        if not isinstance(item, Mapping):
            continue
        cards.append(
            {
                "title": str(item.get("title") or ""),
                "title_zh": str(item.get("title_zh") or ""),
                "reason": str(item.get("reason") or ""),
                "url": str(item.get("url") or ""),
            }
        )
    return cards


def _first_send_approved(root: Path) -> bool:
    path = Path(root) / "data" / "push" / "frontier.sqlite3"
    if not path.is_file():
        return False
    state = PushState(path)
    state.init_schema()
    return bool(state.get_setting("first_send_approved_at"))


def manual_send(
    root: Path,
    config_path: Path,
    env: Mapping[str, str] | None,
    *,
    deps: PushDeps | None = None,
    now: datetime | None = None,
    confirm: bool = False,
) -> RunResult:
    """Send one digest explicitly; never changes the auto-push switch."""
    if not confirm:
        raise SimpleUiError("请先勾选确认：这会真实发送邮件")
    root = Path(root)
    config_path = Path(config_path)
    current = now or datetime.now(timezone.utc)
    resolved = ui_state.resolve_recipient(root, _as_mapping(env))
    if not resolved["recipient"]:
        raise SimpleUiError("请先保存收件邮箱")
    if not resolved["verified"]:
        raise SimpleUiError("请先发送测试邮件并确认收到")
    if not _first_send_approved(root):
        raise SimpleUiError("请先完成首次发送批准")

    reference = ui_state.get_preview(root)
    preview_run_id = (
        reference.get("run_id") if isinstance(reference, Mapping) else None
    )
    deps = deps or PushDeps(root=root, env=_as_mapping(env))
    config = load_push_config(config_path)
    result = run_push(
        "send",
        config_path=config_path,
        now=current,
        deps=deps,
        manual=True,
        preview_run_id=preview_run_id,
        lookback_days=int(config.catchup_days),
    )
    if result.sent and result.run_id:
        try:
            ui_state.record_manual_send(
                root,
                result.run_id,
                current.isoformat(),
                int(result.selected_count),
            )
        except ui_state.UiStateError:
            # The mail was already accepted; never report the send as failed
            # only because the convenience record could not be written.
            pass
    return result


def _ensure_ui_recipient(
    root: Path, env: Mapping[str, str] | None
) -> str:
    """Return the current recipient and seed private state when only env has it.

    Legacy installs resolve a verified ``MAIL_TO`` from the environment
    without a private state file. The first verification action copies that
    address into ``ui_state`` so the subsequent ``mark_verified`` and preview
    references bind to exactly the address the user tested. This writes state
    only when an action is taken, never while merely rendering the page.
    """
    root = Path(root)
    values = _as_mapping(env)
    resolved = ui_state.resolve_recipient(root, values)
    recipient = resolved["recipient"]
    if not recipient:
        raise SimpleUiError("请先保存要推送的邮箱")
    if not resolved["state_exists"]:
        ui_state.set_recipient(root, recipient)
    return recipient


def test_email(
    root: Path, config_path: Path, env: Mapping[str, str] | None
) -> dict[str, Any]:
    """Send the SMTP test mail to the saved recipient; do not verify it.

    A successful send records a receipt bound to the recipient hash. A failed
    send clears any older receipt, because that older proof must not unlock the
    first-enable confirmation after a visible failure.
    """
    root = Path(root)
    recipient = _ensure_ui_recipient(root, env)
    merged = ui_state.mail_env(root, _as_mapping(env))
    try:
        outcome = push_ctl.test_email(
            Path(config_path), merged, send=True, confirm=True
        )
    except Exception:
        try:
            ui_state.clear_test_email(root)
        except ui_state.UiStateError:
            pass
        raise
    try:
        ui_state.record_test_email(
            root,
            recipient,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    except ui_state.UiStateError:
        raise SimpleUiError(
            "测试邮件已发送，但发送记录未保存，请重试"
        ) from None
    return outcome


def _disable_quietly(
    root: Path, config_path: Path, env: Mapping[str, str] | None
) -> None:
    try:
        push_ctl.set_auto_push(
            False, Path(root), Path(config_path), _as_mapping(env)
        )
    except Exception:  # noqa: BLE001 - failure path already returns an error
        pass


def _verify_and_approve(
    root: Path,
    env: Mapping[str, str] | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify the current recipient and approve the first send.

    Shared by the verify-only action and the first-enable flow. An unverified
    recipient needs a recorded successful test email for that exact address;
    an already-verified recipient does not need a new receipt. A legacy
    ``.env`` recipient (no private state yet) keeps its historical "already
    verified" compatibility. This step never installs the scheduled task,
    never calls ``set_auto_push``, and never writes ``enabled``/``paused``.
    """
    root = Path(root)
    values = _as_mapping(env)
    try:
        resolved = ui_state.resolve_recipient(root, values)
    except ui_state.UiStateError as exc:  # pragma: no cover - defensive
        return {"verified": False, "error": str(exc)}
    recipient = resolved["recipient"]
    if not recipient:
        return {"verified": False, "error": "请先保存要推送的邮箱"}
    if not resolved["state_exists"]:
        try:
            ui_state.set_recipient(root, recipient)
        except ui_state.UiStateError as exc:
            return {"verified": False, "error": str(exc)}
    if not resolved["verified"] and not ui_state.test_email_confirmed(
        root, recipient
    ):
        return {"verified": False, "error": TEST_EMAIL_REQUIRED}

    current = now or datetime.now(timezone.utc)
    try:
        ui_state.mark_verified(
            root, recipient, current.isoformat(timespec="seconds")
        )
    except ui_state.UiStateError as exc:
        return {"verified": False, "error": str(exc)}

    try:
        push_ctl.approve_first_send(root, confirm=True)
    except push_ctl.PushCtlError as exc:
        return {"verified": False, "error": str(exc)}

    return {
        "verified": True,
        "approved": True,
        "recipient": mask_email(recipient),
    }


def confirm_test_and_verify(
    root: Path,
    config_path: Path,
    env: Mapping[str, str] | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify the recipient and approve the first send, without auto-push.

    This is the simple page's "verify only" action: it makes manual sending
    available while leaving the automatic switch off. ``config_path`` is
    accepted for symmetry with the other confirmation helpers but is never
    read or written here.
    """
    del config_path  # intentionally unused: no config or task changes
    return _verify_and_approve(root, env, now=now)


def confirm_test_and_enable(
    root: Path,
    config_path: Path,
    env: Mapping[str, str] | None,
    *,
    register_task_fn: Callable[[Path, str], Mapping[str, Any]] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Complete the first-enable flow; any failure leaves auto-push off.

    Verification and first-send approval are shared with
    :func:`confirm_test_and_verify`; this function then installs the scheduled
    task and turns automatic push on. For an unverified recipient, enabling
    requires a recorded successful test email for that exact address.
    """
    root = Path(root)
    config_path = Path(config_path)
    register = register_task_fn or push_ctl.register_task
    verification = _verify_and_approve(root, env, now=now)
    if not verification.get("verified"):
        return {
            "enabled": False,
            "error": verification.get("error") or "未能验证邮箱",
        }

    config = load_push_config(config_path)
    try:
        outcome = register(root, config.time_local)
    except push_ctl.PushCtlError:
        _disable_quietly(root, config_path, env)
        return {"enabled": False, "task_error": TASK_INSTALL_ERROR}
    if not isinstance(outcome, Mapping) or not outcome.get("installed"):
        _disable_quietly(root, config_path, env)
        return {"enabled": False, "task_error": TASK_INSTALL_ERROR}

    try:
        result = push_ctl.set_auto_push(
            True, root, config_path, _as_mapping(env)
        )
    except push_ctl.PushCtlError as exc:
        _disable_quietly(root, config_path, env)
        return {
            "enabled": False,
            "error": str(exc),
            "task_installed": True,
        }
    return {**result, "enabled": True, "task_installed": True}


def enable(
    root: Path,
    config_path: Path,
    env: Mapping[str, str] | None,
    *,
    register_task_fn: Callable[[Path, str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """One-click enable for a verified, first-send-approved recipient.

    Enabling is never a path around the first-test/approval gate: a recipient
    that has not completed it is sent to the test flow instead.
    """
    root = Path(root)
    config_path = Path(config_path)
    resolved = ui_state.resolve_recipient(root, _as_mapping(env))
    if not resolved["recipient"] or not resolved["verified"]:
        return {"enabled": False, "error": "请先验证收件邮箱"}
    if not _first_send_approved(root):
        return {
            "enabled": False,
            "error": "请先发送测试邮件并确认收到，完成首次发送批准",
        }
    register = register_task_fn or push_ctl.register_task
    config = load_push_config(config_path)
    try:
        outcome = register(root, config.time_local)
    except push_ctl.PushCtlError:
        return {"enabled": False, "task_error": TASK_INSTALL_ERROR}
    if not isinstance(outcome, Mapping) or not outcome.get("installed"):
        return {"enabled": False, "task_error": TASK_INSTALL_ERROR}
    try:
        result = push_ctl.set_auto_push(
            True, root, config_path, _as_mapping(env)
        )
    except push_ctl.PushCtlError as exc:
        return {"enabled": False, "error": str(exc)}
    return {**result, "enabled": True, "task_installed": True}


def disable(
    root: Path, config_path: Path, env: Mapping[str, str] | None
) -> dict[str, Any]:
    """Switch automatic push off without clearing safety flags."""
    return push_ctl.set_auto_push(
        False, Path(root), Path(config_path), _as_mapping(env)
    )


def status(
    root: Path,
    config_path: Path,
    env: Mapping[str, str] | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return the masked, simple-page status object."""
    root = Path(root)
    config_path = Path(config_path)
    report = push_status.status(root, config_path, env, now=now)
    try:
        config = load_push_config(config_path)
    except (OSError, PushConfigError):
        config = None
    try:
        profile = load_active_profile(root)
    except (OSError, ProfileError):
        profile = {}
    try:
        state = ui_state.load_state(root)
    except ui_state.UiStateError:
        state = {}
    resolved = ui_state.resolve_recipient(root, _as_mapping(env))
    topics_text = str(profile.get("research_goal") or "").strip()
    if not topics_text:
        topics_text = "\n".join(profile.get("topics") or [])
    return {
        "enabled": bool(report.get("enabled")),
        "paused": bool(report.get("paused")),
        "auto_paused": bool(report.get("auto_paused")),
        "quiet": bool(report.get("quiet_mode")),
        "mail_to": report.get("mail_to"),
        "recipient_saved": bool(resolved["recipient"]),
        "recipient_stored": bool(state.get("recipient")),
        "recipient_verified": bool(report.get("recipient_verified")),
        "recipient_source": report.get("recipient_source"),
        "test_email_confirmed": bool(
            resolved["recipient"]
            and ui_state.test_email_confirmed(root, resolved["recipient"])
        ),
        "first_send_approved": bool(report.get("first_send_approved")),
        "next_expected": report.get("next_expected"),
        "last_scheduled_run": next(
            (
                dict(item)
                for item in (report.get("recent_runs") or [])
                if item.get("mode") == "scheduled"
            ),
            None,
        ),
        "last_manual_send": state.get("last_manual_send"),
        "count": config.max_per_digest if config is not None else 3,
        "time_local": config.time_local if config is not None else "08:30",
        "topics_text": topics_text,
        "config_problems": list(report.get("config_problems") or []),
    }


def email_pending(
    root: Path,
    email_field: str | None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Whether the input holds a different, not-yet-saved address.

    The comparison uses the recipient the push would actually use: the private
    UI state when it exists, otherwise the legacy ``MAIL_TO`` from ``env``. A
    non-empty field with no resolved recipient is still unsaved, so it is
    pending (which disables auto/manual actions) rather than silently ignored.
    """
    field = str(email_field or "").strip()
    if not field:
        return False
    try:
        resolved = ui_state.resolve_recipient(Path(root), _as_mapping(env))
    except ui_state.UiStateError:
        return True
    saved = resolved["recipient"] or ""
    if not saved:
        return True
    return ui_state.recipient_hash(field) != ui_state.recipient_hash(saved)


def action_state(
    root: Path,
    config_path: Path,
    env: Mapping[str, str] | None,
    *,
    email_field: str | None = "",
    report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return which simple-page actions are currently allowed."""
    report = report if report is not None else status(root, config_path, env)
    pending = email_pending(root, email_field, env)
    verified = bool(report.get("recipient_verified"))
    approved = bool(report.get("first_send_approved"))
    recipient_ready = bool(report.get("recipient_saved")) and not pending
    return {
        "email_pending": pending,
        "verified": verified,
        "first_send_approved": approved,
        "recipient_saved": bool(report.get("recipient_saved")),
        "can_preview": True,
        "can_verify": recipient_ready,
        "can_manual": bool(verified and approved and not pending),
        "can_enable": bool(recipient_ready and verified and approved),
        "enabled": bool(report.get("enabled")),
    }


__all__ = [
    "SimpleUiError",
    "action_state",
    "confirm_test_and_enable",
    "confirm_test_and_verify",
    "disable",
    "email_pending",
    "enable",
    "manual_send",
    "preview",
    "preview_cards",
    "save_settings",
    "status",
    "test_email",
]
