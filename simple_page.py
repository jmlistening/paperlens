"""Default Streamlit page: a simple daily arXiv digest subscription.

The page only renders local configuration and private state. arXiv, model, and
SMTP calls happen exclusively inside the two explicit actions ("preview" and
"send"); rendering never fetches, ranks, or mails anything. Advanced research
and evaluation pages are reached through the light "高级 / 研究模式" entry.

Page configuration and navigation are owned by the application entry; this
module never configures the page itself.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, time
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import push_ctl  # noqa: E402
import push_status  # noqa: E402
import simple_ui  # noqa: E402
import ui_state  # noqa: E402
import ui_theme  # noqa: E402
from topic_queries import TopicQueryError, make_deepseek_chat_fn  # noqa: E402

DEFAULT_TIME = time(8, 30)
# Session key for the upstream values each keyed widget was built from. Streamlit
# ignores ``value=`` once a key exists, so an external config/profile edit is
# only reflected after explicitly syncing the widget's session state.
_SOURCES_KEY = "simple_settings_sources"


def _root() -> Path:
    override = os.environ.get("PAPERLENS_PUSH_ROOT")
    return Path(override).resolve() if override else PROJECT_ROOT


def _config_path(root: Path) -> Path:
    override = os.environ.get("PAPERLENS_PUSH_CONFIG")
    return Path(override).resolve() if override else root / "push_config.json"


def _parse_time(value: str) -> time:
    try:
        hour, minute = (int(part) for part in str(value).split(":"))
    except (TypeError, ValueError):
        return DEFAULT_TIME
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return DEFAULT_TIME
    return time(hour, minute)


def _settings_sources(
    root: Path, env: Mapping[str, str], report: Mapping[str, Any]
) -> dict[str, Any]:
    """The upstream values the keyed widgets mirror.

    A keyed widget keeps its session value across reruns, so only a change to
    its own source may overwrite it. In particular, changing the recipient or
    the auto-switch must never discard unsaved topic text.
    """
    recipient = ui_state.resolve_recipient(root, env).get("recipient") or ""
    return {
        "enabled": bool(report.get("enabled")),
        "count": int(report.get("count") or 3),
        "time_local": str(report.get("time_local") or "08:30"),
        "topics": str(report.get("topics_text") or ""),
        "recipient": ui_state.recipient_hash(recipient) if recipient else "",
    }


def _reconcile_widgets(sources: Mapping[str, Any]) -> None:
    """Mirror external settings into keyed widgets before they are created.

    Rendering only reconciles; it never writes settings. A stale session value
    is therefore harmless, whereas the old render-time auto-save treated it as
    a user action and overwrote an external edit.
    """
    previous = st.session_state.get(_SOURCES_KEY)
    fields = (
        ("enabled", "simple_auto"),
        ("count", "simple_count"),
        ("time_local", "simple_time"),
        ("topics", "simple_topics"),
    )
    if not isinstance(previous, Mapping):
        for source, key in fields:
            st.session_state[key] = _widget_value(source, sources)
    else:
        for source, key in fields:
            if previous.get(source) != sources.get(source):
                st.session_state[key] = _widget_value(source, sources)
    st.session_state[_SOURCES_KEY] = dict(sources)


def _widget_value(source: str, sources: Mapping[str, Any]) -> Any:
    if source == "count":
        return int(sources[source])
    if source == "time_local":
        return _parse_time(str(sources[source]))
    return sources[source]


def _coerce_time_text(value: Any) -> str:
    if isinstance(value, (time, datetime)):
        return value.strftime("%H:%M")
    return str(value or "")


def _set_count(value: int) -> None:
    """Quick-pick only changes the pending value; 确认设置 persists it."""
    st.session_state["simple_count"] = int(value)


def _flash(kind: str, message: str) -> None:
    st.session_state["simple_flash"] = (kind, message)


def _show_flash() -> None:
    item = st.session_state.pop("simple_flash", None)
    if isinstance(item, tuple) and len(item) == 2:
        kind, message = item
        getattr(st, kind, st.info)(message)


def _scheduled_status_label(run: Mapping[str, Any]) -> str:
    """Plain-language result of the last scheduled attempt."""
    status = str(run.get("status") or "")
    counts = run.get("counts") or {}
    if status == "sent":
        return f"已发送 {int(counts.get('sent') or 0)} 篇"
    if status == "already_sent":
        return "这批已发送过，未重复发送"
    if status == "no_digest":
        return "没有合格论文，未发送"
    if status in {"preview_ready", "quiet_preview"}:
        return "仅预览，未发送"
    if status == "failed":
        return "失败，未发送"
    if status == "recipient_unverified":
        return "邮箱未验证，未发送"
    if status in {"rate_limited", "cooldown"}:
        return "抓取受限，未发送"
    if status == "config_error":
        return "配置不完整，未发送"
    if status == "send_not_approved":
        return "未批准首次发送，未发送"
    if status == "auto_paused":
        return "自动暂停中，未发送"
    return status or "未知状态"


def _next_label(report: Mapping[str, Any], can_manual: bool) -> str:
    if report.get("auto_paused"):
        return "发送失败已自动暂停，请先去高级 / 研究模式检查。"
    if report.get("enabled") and not report.get("first_send_approved"):
        return (
            "自动推送已开启，但尚未完成首次发送批准；"
            "请先完成测试邮件确认。"
        )
    if not report.get("enabled"):
        if can_manual:
            return "自动推送已关闭；手动推送仍然可用。"
        return "自动推送已关闭。"
    if report.get("paused"):
        return "自动推送已暂停。"
    if report.get("quiet"):
        return "静默模式只生成预览，不会自动发送。"
    value = report.get("next_expected")
    if not value:
        return "下次发送时间未确定。"
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return f"下次发送：{value}"
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    local = parsed.astimezone()
    now = datetime.now().astimezone()
    if local.date() == now.date():
        day = "今天"
    elif (local.date() - now.date()).days == 1:
        day = "明天"
    else:
        day = local.strftime("%m-%d")
    label = f"下次发送：{day} {local.strftime('%H:%M')}"
    configured = str(report.get("time_local") or "")
    try:
        hour, minute = (int(part) for part in configured.split(":"))
    except (TypeError, ValueError):
        return label
    if local.date() > now.date() and (hour, minute) <= (
        now.hour,
        now.minute,
    ):
        label += "（今天的这个时间已过，改为明天）"
    return label


def _preview_view(root: Path, result: Any) -> dict[str, Any]:
    return {
        "status": str(result.status),
        "count": int(result.selected_count),
        "run_id": result.run_id,
        "cards": simple_ui.preview_cards(root, result.run_id),
    }


def _render_preview(view: Mapping[str, Any] | None) -> None:
    if not view:
        return
    status = str(view.get("status") or "")
    if status in {"preview_ready", "quiet_preview"}:
        cards = list(view.get("cards") or [])
        if not cards:
            st.info("这次预览没有生成论文卡片。")
            return
        st.subheader(f"预览：这次会推 {len(cards)} 篇")
        for card in cards:
            with st.container(border=True):
                title = card.get("title") or "（无英文标题）"
                st.markdown(f"#### {title}")
                title_zh = card.get("title_zh") or ""
                if title_zh:
                    st.markdown(f"**{title_zh}**")
                reason = card.get("reason") or ""
                if reason:
                    st.caption(f"推荐理由：{reason}")
                url = card.get("url") or ""
                if url:
                    st.markdown(f"[在 arXiv 阅读原文]({url})")
        return
    if status == "no_digest":
        st.info("这次没有找到可以推送的新论文。可以调整主题，或稍后再试。")
        return
    if status == "config_error":
        st.warning(
            "推送配置还不完整：请先在高级 / 研究模式里补充邮箱服务配置。"
        )
        return
    if status in {"busy", "rate_limited", "cooldown"}:
        st.warning("预览暂时不可用，请稍后重试。")
        return
    st.warning(f"预览未完成：{status or '未知状态'}")


def _manual_message(result: Any) -> tuple[str, str]:
    status = str(result.status)
    if status == "sent":
        return (
            "success",
            f"今天已手动推送 {int(result.selected_count)} 篇。自动推送开关未改变。",
        )
    if status == "already_sent":
        return ("info", "这批论文之前已经推送过，没有重复发送。")
    if status == "recipient_unverified":
        return ("error", "当前收件邮箱还没有验证，请先发送测试邮件并确认收到。")
    if status == "send_not_approved":
        return ("error", "还没有批准首次发送，请先完成邮箱验证。")
    if status == "auto_paused":
        return ("error", "推送已被自动暂停，请先去高级 / 研究模式检查最近失败。")
    if status == "config_error":
        return ("error", "推送配置不完整，暂时无法真实发送。")
    if status == "no_digest":
        return ("info", "这次没有找到合适的论文，没有发送。")
    if status == "uncertain":
        return ("warning", "发送结果不确定，请检查邮箱；系统不会自动重试。")
    return ("warning", f"手动推送未完成：{status or '未知状态'}")


def _auto_changed() -> None:
    """Handle the one-click auto-push toggle before the page re-renders."""
    root = _root()
    config_path = _config_path(root)
    env = push_status.load_env(root)
    desired = bool(st.session_state.get("simple_auto"))
    try:
        report = simple_ui.status(root, config_path, env)
        if desired == bool(report.get("enabled")):
            return
        if desired:
            pending = simple_ui.email_pending(
                root, st.session_state.get("simple_email", ""), env
            )
            if pending or not report.get("recipient_verified"):
                st.session_state["simple_auto"] = bool(report.get("enabled"))
                _flash("info", "请先保存并验证邮箱，再开启自动推送。")
                return
            if not report.get("first_send_approved"):
                st.session_state["simple_auto"] = bool(report.get("enabled"))
                _flash(
                    "info",
                    "请先发送测试邮件并确认收到，完成首次发送批准后再开启。",
                )
                return
            result = simple_ui.enable(
                root,
                config_path,
                env,
                register_task_fn=push_ctl.register_task,
            )
            if result.get("enabled"):
                _flash("success", "已开启每天自动推送。")
            else:
                st.session_state["simple_auto"] = bool(report.get("enabled"))
                detail = (
                    result.get("error")
                    or result.get("task_error")
                    or "未知原因"
                )
                _flash("error", f"未能开启自动推送：{detail}")
        else:
            try:
                simple_ui.disable(root, config_path, env)
            except Exception as exc:  # noqa: BLE001 - input is local config
                st.session_state["simple_auto"] = bool(report.get("enabled"))
                _flash("error", f"未能关闭自动推送：{type(exc).__name__}")
            else:
                _flash("success", "已关闭自动推送。")
    except Exception as exc:  # noqa: BLE001 - never leak raw configuration
        st.session_state["simple_auto"] = not desired
        _flash("error", f"未能更新自动推送：{type(exc).__name__}")


def _save_settings_for_action(
    root: Path,
    config_path: Path,
    env: Mapping[str, str],
    *,
    topics_text: str,
    email_field: str,
    count: int,
    time_local: str,
) -> dict[str, Any]:
    return simple_ui.save_settings(
        root,
        config_path,
        topics_text=topics_text,
        email=email_field,
        count=int(count),
        time_local=time_local,
        chat_fn=make_deepseek_chat_fn(env=env),
    )


def main() -> None:
    ui_theme.apply_theme()
    st.title("PaperLens")
    st.caption(
        "每天挑几篇你可能关心的 arXiv 论文。"
        "预览会联网并调用模型，但永远不会发邮件。"
    )

    root = _root()
    config_path = _config_path(root)
    env = push_status.load_env(root)
    try:
        report = simple_ui.status(root, config_path, env)
    except Exception as exc:  # noqa: BLE001 - show a short, safe reason
        st.error(f"无法读取推送设置：{type(exc).__name__}")
        st.stop()
        return

    _show_flash()
    for problem in report.get("config_problems") or []:
        st.warning(problem)

    # Keyed widgets ignore ``value=`` after their first render. Sync them from
    # the files before they are created so an external edit (or an advanced-page
    # edit) is shown as-is and is never written back from stale session state.
    _reconcile_widgets(_settings_sources(root, env, report))

    # Settings stay visible and editable. A change stays pending until the
    # page-bottom 确认设置 button writes it; preview/manual only use saved
    # values, so an unsaved topic or address can never be sent by accident.
    email_value = str(st.session_state.get("simple_email", "") or "")
    pending = simple_ui.email_pending(root, email_value, env)

    actions = simple_ui.action_state(
        root,
        config_path,
        env,
        email_field=email_value,
        report=report,
    )

    topics_text = st.text_area(
        "我想关注什么",
        value=str(report.get("topics_text") or ""),
        key="simple_topics",
        height=120,
        help="用中文或英文写一段话，或每行一个主题。",
    )
    email_field = st.text_input(
        "推送到哪个邮箱",
        value="",
        key="simple_email",
        placeholder=(
            f"当前：{report['mail_to']}"
            if report.get("mail_to")
            else "you@example.com"
        ),
        help="修改邮箱会要求重新验证；已保存的收件人见下方摘要。",
    )
    if report.get("mail_to"):
        verified_label = (
            "已验证" if report.get("recipient_verified") else "尚未验证"
        )
        st.caption(f"当前收件人：{report['mail_to']}（{verified_label}）")
    else:
        st.caption("当前收件人：未保存")
    if pending:
        st.warning(
            "请先保存收件邮箱：保存后才能持久化篇数和时间，"
            "并开启自动推送或手动发送。"
        )

    count = st.slider(
        "每次推送几篇",
        min_value=1,
        max_value=10,
        value=int(report["count"]),
        key="simple_count",
    )
    quick_columns = st.columns([1, 1, 1, 1, 6])
    for column, value in zip(quick_columns, (1, 3, 5, 10)):
        with column:
            st.button(
                str(value),
                key=f"simple_count_{value}",
                on_click=_set_count,
                args=(value,),
            )
    st.caption("1 / 3 / 5 / 10 篇快捷选择；改完在页面最下方点确认设置。")

    topics_pending = topics_text.strip() != str(
        report.get("topics_text") or ""
    ).strip()
    count_pending = int(count) != int(report.get("count") or 3)
    raw_time = st.session_state.get("simple_time")
    time_text = (
        _coerce_time_text(raw_time)
        if isinstance(raw_time, (time, datetime))
        else str(report.get("time_local") or "08:30")
    )
    time_pending = time_text != str(report.get("time_local") or "08:30")
    settings_pending = bool(
        pending or topics_pending or count_pending or time_pending
    )
    if settings_pending:
        st.warning(
            "设置已修改，请到页面最下方点「确认设置」保存后再预览或推送。"
        )
    saved_verified = (
        "已验证" if report.get("recipient_verified") else "尚未验证"
    )
    st.info(
        f"当前已保存：{int(report.get('count') or 3)} 篇/次 · "
        f"每天 {report.get('time_local') or '08:30'} · "
        f"收件人 {report.get('mail_to') or '未保存'}"
        + (f"（{saved_verified}）" if report.get("mail_to") else "")
    )

    preview_column, manual_column = st.columns(2)
    with preview_column:
        preview_clicked = st.button(
            "先预览这周会推什么",
            key="simple_preview",
            type="primary",
            disabled=settings_pending,
        )
    confirm_send = st.checkbox(
        "我确认：点击后会真实发送邮件到当前收件人",
        value=False,
        key="simple_confirm_send",
        disabled=settings_pending,
    )
    with manual_column:
        manual_clicked = st.button(
            "现在推送今天的",
            key="simple_manual",
            disabled=not (actions["can_manual"] and confirm_send)
            or settings_pending,
        )
    if not actions["can_manual"]:
        if not actions["recipient_saved"]:
            manual_blocked = "请先保存收件邮箱"
        elif actions["email_pending"]:
            manual_blocked = "有待保存的邮箱"
        elif not actions["verified"]:
            manual_blocked = "未验证邮箱"
        else:
            manual_blocked = "未完成首次发送批准"
        st.caption(f"手动推送暂不可用：{manual_blocked}。")

    if preview_clicked:
        try:
            with st.spinner("正在抓取、排序并生成预览，不会发送邮件……"):
                result = simple_ui.preview(root, config_path, env)
            view = _preview_view(root, result)
        except Exception as exc:  # noqa: BLE001 - type name only, no secrets
            st.error(f"预览未完成：{type(exc).__name__}")
        else:
            st.session_state["simple_preview_view"] = view

    if manual_clicked:
        try:
            result = simple_ui.manual_send(
                root,
                config_path,
                env,
                confirm=confirm_send,
            )
        except simple_ui.SimpleUiError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - type name only, no secrets
            st.error(f"手动推送未完成：{type(exc).__name__}")
        else:
            kind, message = _manual_message(result)
            getattr(st, kind)(message)

    _render_preview(st.session_state.get("simple_preview_view"))

    st.divider()
    needed_approval = not report.get("first_send_approved")
    needed_verification = not report.get("recipient_verified")
    if report.get("recipient_saved") and (
        needed_approval or needed_verification
    ):
        test_confirmed = bool(report.get("test_email_confirmed"))
        if needed_verification:
            st.subheader("首次开启：先验证邮箱")
            st.caption(
                "第一次开启自动推送前，需要发送一封测试邮件并确认收到；"
                "测试邮件不会包含推荐内容。"
            )
        elif test_confirmed:
            st.subheader("首次开启：先批准发送")
            st.caption(
                "邮箱已验证；真实发送前还需要你批准首次发送。"
                "批准不会开启自动推送。"
            )
        else:
            # Legacy .env recipients count as verified but have no receipt;
            # keep the test flow available so they can finish the first
            # enable instead of seeing an approval-only screen with no way
            # to send a test.
            st.subheader("首次开启：先批准发送")
            st.caption(
                "当前邮箱沿用旧配置，还没有测试发送记录；"
                "开启自动推送前请先发送一封测试邮件并确认收到。"
                "也可以只完成首次发送批准（不开启自动推送）。"
            )
        # The test-email controls are for an unverified address, or for a
        # legacy verified address with no receipt. A UI-verified recipient
        # with a receipt only needs approval, so the controls stay hidden.
        if needed_verification or not test_confirmed:
            # Receipt confirmation is only usable after a successful test
            # send. A failed send clears the receipt, so a visible failure
            # cannot leave the confirmation path open.
            if st.button(
                "发送测试邮件",
                key="simple_test_email",
                disabled=not actions["can_verify"],
            ):
                try:
                    simple_ui.test_email(root, config_path, env)
                except Exception as exc:  # noqa: BLE001 - no credentials in output
                    test_confirmed = False
                    st.error(f"测试邮件发送失败：{type(exc).__name__}")
                else:
                    test_confirmed = True
                    st.success("测试邮件已发送。请确认收到后勾选下面的选项。")
            if not test_confirmed:
                st.caption("发送测试邮件成功后，才能勾选确认。")
            received = st.checkbox(
                "我收到了测试邮件",
                value=False,
                key="simple_received",
                disabled=not test_confirmed,
            )
            if st.button(
                "我收到了，开启推送",
                key="simple_confirm_enable",
                disabled=not (
                    received and actions["can_verify"] and test_confirmed
                ),
            ):
                try:
                    result = simple_ui.confirm_test_and_enable(
                        root,
                        config_path,
                        env,
                        register_task_fn=push_ctl.register_task,
                    )
                except Exception as exc:  # noqa: BLE001 - no credentials in output
                    st.error(f"未能开启自动推送：{type(exc).__name__}")
                else:
                    if result.get("enabled"):
                        report["enabled"] = True
                        report["recipient_verified"] = True
                        report["first_send_approved"] = True
                        st.session_state["simple_auto"] = True
                        st.success("邮箱已验证，已开启每天自动推送。")
                    else:
                        detail = (
                            result.get("error")
                            or result.get("task_error")
                            or "未知原因"
                        )
                        st.error(f"未能开启自动推送：{detail}")

        # Verify-only path: it makes manual sending available and never turns
        # auto-push on or installs the daily task. The unverified case reuses
        # the receipt gate above; an already-verified recipient only needs the
        # saved recipient (approval is the remaining gate).
        if needed_verification:
            verify_label = "仅验证邮箱（不开启自动推送）"
            verify_disabled = not (
                received and actions["can_verify"] and test_confirmed
            )
        else:
            verify_label = "完成首次发送批准（不开启自动推送）"
            verify_disabled = not actions["can_verify"]
        if st.button(
            verify_label,
            key="simple_verify_only",
            disabled=verify_disabled,
        ):
            try:
                outcome = simple_ui.confirm_test_and_verify(
                    root, config_path, env
                )
            except Exception as exc:  # noqa: BLE001 - no credentials in output
                st.error(f"邮箱未验证：{type(exc).__name__}")
            else:
                if outcome.get("verified"):
                    report["recipient_verified"] = True
                    report["first_send_approved"] = True
                    _flash(
                        "success",
                        "邮箱已验证，手动推送可用；自动推送开关未改变。",
                    )
                    st.rerun()
                else:
                    detail = outcome.get("error") or "未知原因"
                    st.error(f"未能验证邮箱：{detail}")

    auto_column, time_column = st.columns(2)
    with auto_column:
        st.toggle(
            "每天自动推送",
            value=bool(report.get("enabled")),
            key="simple_auto",
            # An already-enabled switch must always be turnable off, even if
            # the first-send approval is missing; the gate applies to enabling.
            # Enabling waits for pending settings so it cannot install a task
            # for an unconfirmed time.
            disabled=(
                not bool(report.get("enabled"))
                and (settings_pending or not actions["can_enable"])
            ),
            on_change=_auto_changed,
        )
    with time_column:
        time_value = st.time_input(
            "每天几点",
            value=_parse_time(str(report.get("time_local") or "08:30")),
            key="simple_time",
        )
    st.caption(
        "改时间并确认后会按新时间重新注册；同一天改时间可再测一次，"
        "已发送的论文不会重复。"
    )
    st.caption(_next_label(report, bool(actions["can_manual"])))
    last_send = report.get("last_manual_send")
    if isinstance(last_send, Mapping):
        at = str(last_send.get("at") or "")
        try:
            parsed = datetime.fromisoformat(at)
        except (TypeError, ValueError):
            parsed = None
        when = (
            parsed.astimezone().strftime("%m-%d %H:%M")
            if parsed is not None
            else at
        )
        st.caption(
            f"上次手动推送：{when} · {int(last_send.get('count') or 0)} 篇"
        )
    last_scheduled = report.get("last_scheduled_run")
    if isinstance(last_scheduled, Mapping):
        at = str(
            last_scheduled.get("finished_at")
            or last_scheduled.get("started_at")
            or ""
        )
        try:
            parsed = datetime.fromisoformat(at)
        except (TypeError, ValueError):
            parsed = None
        when = (
            parsed.astimezone().strftime("%m-%d %H:%M")
            if parsed is not None
            else at
        )
        st.caption(
            f"上次定时尝试：{when} · "
            f"{_scheduled_status_label(last_scheduled)}"
        )

    st.divider()
    # The one explicit save control sits at the bottom-right of the page.
    summary_column, confirm_column = st.columns([4, 2])
    with summary_column:
        st.caption(
            f"当前已保存：{int(report.get('count') or 3)} 篇/次 · "
            f"每天 {report.get('time_local') or '08:30'} · "
            f"收件人 {report.get('mail_to') or '未保存'}"
        )
    with confirm_column:
        confirm_settings = st.button(
            "确认设置",
            key="simple_confirm_settings",
            type="primary",
        )

    if confirm_settings:
        try:
            saved = _save_settings_for_action(
                root,
                config_path,
                env,
                topics_text=topics_text,
                email_field=email_field,
                count=int(count),
                time_local=_coerce_time_text(time_value),
            )
        except TopicQueryError:
            st.error(
                "主题未保存，检索词生成失败。请重试，或到高级 / 研究模式手动编辑。"
            )
        except simple_ui.SimpleUiError as exc:
            st.error(f"设置未保存：{exc}")
        except Exception as exc:  # noqa: BLE001 - show only a type name
            st.error(f"设置未保存：{type(exc).__name__}")
        else:
            if saved.get("auto_disabled"):
                _flash(
                    "warning", "邮箱已改，自动推送已关闭，验证后重新开启。"
                )
            if saved.get("task_error"):
                _flash(
                    "warning",
                    "计划任务时间未更新，请稍后在高级 / 研究模式重试。",
                )
            _flash("success", "设置已保存。")
            st.rerun()

    advanced_home = PROJECT_ROOT / "advanced" / "home.py"
    if advanced_home.is_file():
        st.page_link("advanced/home.py", label="高级 / 研究模式")
    else:
        st.caption("高级 / 研究模式")


if __name__ == "__main__":
    main()
