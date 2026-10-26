"""PaperLens 推送设置页（可选的前沿 arXiv 推送）。

页面只读取配置和状态，按钮被点击时才执行命令。导入页面本身不会发起
网络、模型或 SMTP 调用；预览按钮永远不会发送邮件。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import push_status  # noqa: E402
import ui_theme  # noqa: E402
from push_config import load_push_config, parse_push_config  # noqa: E402
from push_ctl import (  # noqa: E402
    PushCtlError,
    approve_first_send,
    resume,
    set_auto_push,
    set_paused,
    test_email,
)
from push_run import PushDeps, run_push  # noqa: E402


def _project_root() -> Path:
    override = os.environ.get("PAPERLENS_PUSH_ROOT")
    return Path(override).resolve() if override else PROJECT_ROOT


def _config_path(root: Path) -> Path:
    override = os.environ.get("PAPERLENS_PUSH_CONFIG")
    return Path(override).resolve() if override else root / "push_config.json"


def _env(root: Path) -> dict[str, str]:
    return push_status.load_env(root)


def _write_config(config_path: Path, patch: dict) -> None:
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    raw.update(patch)
    parse_push_config(raw)  # reject unknown keys / bad values before writing
    Path(config_path).write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# The keyed toggle keeps its session value across reruns; remember which
# config value it mirrors so an external edit can be synced before rendering.
_ENABLED_SOURCE_KEY = "push_enabled_source"
_FLASH_KEY = "push_settings_flash"


def _flash(kind: str, message: str) -> None:
    st.session_state[_FLASH_KEY] = (kind, message)


def _show_flash() -> None:
    item = st.session_state.pop(_FLASH_KEY, None)
    if isinstance(item, tuple) and len(item) == 2:
        kind, message = item
        getattr(st, kind, st.info)(message)


def _reconcile_enabled(report: dict) -> None:
    current = bool(report["enabled"])
    if st.session_state.get(_ENABLED_SOURCE_KEY) != current:
        st.session_state["push_enabled_toggle"] = current
    st.session_state[_ENABLED_SOURCE_KEY] = current


def _toggle_changed() -> None:
    """Apply a real user click; rendering never writes the switch."""
    root = _project_root()
    config_path = _config_path(root)
    environment = _env(root)
    desired = bool(st.session_state.get("push_enabled_toggle"))
    try:
        current = bool(load_push_config(config_path).enabled)
    except Exception:  # noqa: BLE001 - a broken config must not flip anything
        current = not desired
    if desired == current:
        return
    try:
        set_auto_push(desired, root, config_path, environment)
    except PushCtlError as exc:
        st.session_state["push_enabled_toggle"] = current
        _flash("error", f"未能修改总开关：{exc}")
    except Exception as exc:  # noqa: BLE001 - show only a safe type name
        st.session_state["push_enabled_toggle"] = current
        _flash("error", f"未能修改总开关：{type(exc).__name__}")
    else:
        _flash("success", "总开关已更新。")


def _provider_label(provider: str) -> str:
    return "QQ" if provider == "qq" else "163"


def _show_preview(run_dir: str | None) -> None:
    if not run_dir:
        return
    html_path = Path(run_dir) / "digest_preview.html"
    if not html_path.is_file():
        return
    with st.expander("预览内容", expanded=True):
        components.html(
            html_path.read_text(encoding="utf-8"), height=720, scrolling=True
        )


def main() -> None:
    ui_theme.apply_theme()
    st.title("推送设置")
    st.caption(
        "可选的前沿 arXiv 推送，默认关闭。此页不显示授权码；预览永远不会发送邮件。"
    )

    root = _project_root()
    config_path = _config_path(root)
    environment = _env(root)

    try:
        config = load_push_config(config_path)
        report = push_status.status(root, config_path, environment)
    except Exception as exc:  # missing or malformed config is shown, not swallowed
        st.error(f"无法读取推送配置：{type(exc).__name__}: {exc}")
        st.stop()
        return

    _show_flash()
    for problem in report.get("config_problems") or []:
        st.warning(problem)
    if report["auto_paused"]:
        st.error("连续失败已自动暂停。请先排查最近运行，再点击“恢复推送”。")

    metrics = st.columns(4)
    metrics[0].metric("总开关", "已开启" if report["enabled"] else "已关闭")
    metrics[1].metric(
        "运行状态", "已暂停" if report["paused"] else ("运行中" if report["enabled"] else "未启用")
    )
    metrics[2].metric("服务商", _provider_label(report["provider"]))
    metrics[3].metric("收件人", report["mail_to"] or "未配置")

    st.caption(
        f"最近成功：{report['last_success'] or '无'} · "
        f"预计下次：{report['next_expected'] or '未启用'} · "
        f"队列：{report['queue']} 篇"
    )

    st.subheader("总开关")
    _reconcile_enabled(report)
    st.toggle(
        "启用每日推送",
        value=report["enabled"],
        key="push_enabled_toggle",
        on_change=_toggle_changed,
    )

    st.subheader("暂停与恢复")
    pause_col, resume_col = st.columns(2)
    if pause_col.button(
        "暂停推送", key="push_pause_button", disabled=report["paused"]
    ):
        set_paused(True, config_path)
        st.success("已暂停；定时任务会直接跳过。")
        st.rerun()
    if resume_col.button(
        "恢复推送",
        key="push_resume_button",
        disabled=not (report["paused"] or report["auto_paused"]),
    ):
        resume(root, config_path)
        st.success("已恢复，并清除了自动暂停标记。")
        st.rerun()

    st.subheader("邮件服务商")
    provider = st.selectbox(
        "服务商",
        ["163", "QQ"],
        index=1 if config.provider == "qq" else 0,
        key="push_provider_select",
    )
    selected_provider = "qq" if provider == "QQ" else "163"
    if st.button("保存服务商", key="push_provider_save"):
        _write_config(config_path, {"provider": selected_provider})
        st.success(f"服务商已保存为 {provider}。授权码仍只从 .env 读取。")
        st.rerun()

    st.subheader("预览与测试")
    preview_col, login_col = st.columns(2)
    if preview_col.button("立即预览（不发信）", key="push_preview_button"):
        with st.spinner("正在抓取、排序并生成预览，请稍候……"):
            result = run_push(
                "preview",
                config_path=config_path,
                deps=PushDeps(root=root, env=environment),
                # A forecast is available while automatic push is off/paused;
                # ``force`` is preview-only and never writes enabled/paused.
                force=True,
            )
        if result.status in {"preview_ready", "quiet_preview"}:
            st.success(
                f"预览已生成：候选 {result.candidate_count} 篇，"
                f"入选 {result.selected_count} 篇。没有发送邮件。"
            )
            _show_preview(result.run_dir)
        elif result.status in {"disabled", "paused", "config_error", "busy"}:
            detail = "; ".join(result.errors)
            st.warning(f"未生成预览：{result.status} {detail}".strip())
        else:
            st.error(f"预览未完成：{result.status} {'; '.join(result.errors)}".strip())

    if login_col.button("测试邮箱登录（不发信）", key="push_test_login_button"):
        try:
            outcome = test_email(config_path, environment)
        except Exception as exc:
            st.error(f"SMTP 登录失败：{type(exc).__name__}: {exc}")
        else:
            st.success(f"SMTP 登录成功：{outcome['host']}:{outcome['port']}（未发送邮件）")

    st.subheader("首次发送批准")
    st.caption(
        "真实发送必须人工批准；计划任务不能绕过。请先用“测试邮箱登录”和一次测试邮件确认收件正常。"
    )
    acknowledged = st.checkbox(
        "我已确认测试邮件能正常收到，同意进行首次真实发送",
        value=False,
        key="push_approve_ack",
    )
    if st.button(
        "批准首次发送",
        key="push_approve_button",
        disabled=not acknowledged,
    ):
        try:
            outcome = approve_first_send(root, confirm=True)
        except PushCtlError as exc:
            st.error(f"批准失败：{exc}")
        else:
            st.success(f"已批准首次发送：{outcome['first_send_approved_at']}")

    st.subheader("最近运行")
    runs = push_status.recent_runs(root, limit=10, env=environment)
    if not runs:
        st.info("还没有运行记录。先生成一次预览。")
    else:
        rows = []
        for row in runs:
            counts = row.get("counts") or {}
            rows.append(
                {
                    "run_id": row.get("run_id"),
                    "模式": row.get("mode"),
                    "状态": row.get("status"),
                    "开始": row.get("started_at"),
                    "结束": row.get("finished_at"),
                    "候选": counts.get("candidates", 0),
                    "入选": counts.get("selected", 0),
                    "已发送": counts.get("sent", 0),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        "安全说明：授权码只保存在 .env；状态与日志会掩码邮箱。"
        "SMTP accepted 只代表服务器接收，不代表已进入收件箱。"
    )


main()
