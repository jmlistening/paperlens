"""Advanced Streamlit page: sent push history (read-only).

The page lists only real sends and never displays recipient addresses, SMTP
settings, tokens, evidence, or evaluation metrics. New runs show structured
Chinese title/summary/reason cards from ``digest.json``; old runs fall back to
the email HTML that was actually sent. Rendering performs no network, model, or
mail call.

Navigation and the single page configuration are owned by the advanced entry
(Task 9); this module only renders its own content.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ui_theme  # noqa: E402
from push_history import load_sent_history  # noqa: E402

_TITLE_ZH_LINE_RE = re.compile(
    r"^[ \t]*中文标题\s*[：:].*(?:\r?\n)?", re.MULTILINE
)
_TITLE_ZH_VALUE_RE = re.compile(r"^[ \t]*中文标题\s*[：:]\s*(.+?)\s*$", re.MULTILINE)
_ARXIV_PREFIXES = (
    "https://arxiv.org/",
    "http://arxiv.org/",
    "https://www.arxiv.org/",
    "http://www.arxiv.org/",
)


def _root() -> Path:
    override = os.environ.get("PAPERLENS_PUSH_ROOT")
    return Path(override).resolve() if override else PROJECT_ROOT


def _summary_without_title(summary: str) -> str:
    """Drop the leading ``中文标题：...`` line so the title is not duplicated.

    The stored digest data is untouched; this only shapes presentation.
    """
    text = (summary or "").replace("\r\n", "\n").replace("\r", "\n")
    return _TITLE_ZH_LINE_RE.sub("", text, count=1).strip()


def _title_from_summary(summary: str) -> str:
    match = _TITLE_ZH_VALUE_RE.search(summary or "")
    return match.group(1).strip() if match else ""


def _display_title_zh(paper: Mapping[str, Any]) -> str:
    title = str(paper.get("title_zh") or "").strip()
    return title or _title_from_summary(str(paper.get("summary") or ""))


def _local_time(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return "时间未知"
    try:
        parsed = datetime.fromisoformat(value.strip())
    except (TypeError, ValueError):
        return value
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M")


def _paper_url(paper: Mapping[str, Any]) -> str:
    url = str(paper.get("url") or "").strip()
    if url.startswith(_ARXIV_PREFIXES):
        return url
    arxiv_id = str(paper.get("arxiv_id") or "").strip()
    return f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""


def _read_email_html(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return Path(value).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _render_paper(paper: Mapping[str, Any]) -> None:
    title = str(paper.get("title") or "").strip() or "（无英文标题）"
    title_zh = _display_title_zh(paper)
    summary = _summary_without_title(str(paper.get("summary") or ""))
    reason = str(paper.get("reason") or "").strip()
    with st.container(border=True):
        st.markdown(f"#### {title}")
        if title_zh:
            st.markdown(f"**{title_zh}**")
        if reason:
            st.markdown(f"**推荐理由：** {reason}")
        if summary:
            st.markdown(summary)
        else:
            st.caption("缺中文摘要")
        url = _paper_url(paper)
        if url:
            st.markdown(f"[在 arXiv 阅读原文]({url})")


def _render_email(item: Mapping[str, Any]) -> None:
    html_text = _read_email_html(item.get("digest_html"))
    if html_text is None:
        st.caption("找不到当时的邮件版 HTML。")
        return
    components.html(html_text, height=720, scrolling=True)


def _render_run(item: Mapping[str, Any]) -> None:
    when = _local_time(str(item.get("sent_at") or ""))
    badge = "手动发送" if item.get("manual") else "自动发送"
    count = item.get("paper_count", 0)
    label = f"{when} · {badge} · {count} 篇"
    with st.expander(label, expanded=False):
        subject = str(item.get("subject") or "").strip()
        if subject:
            st.caption(f"主题：{subject}")
        if item.get("legacy"):
            st.caption("旧运行：没有结构化 digest.json，下面展示当时的邮件版。")
            with st.expander("看邮件版", expanded=True):
                _render_email(item)
            return
        papers = item.get("papers") or []
        if not papers:
            st.caption("这次运行没有记录论文内容。")
        for paper in papers:
            if isinstance(paper, Mapping):
                _render_paper(paper)
        with st.expander("看邮件版", expanded=False):
            _render_email(item)


def main() -> None:
    ui_theme.apply_theme()
    st.title("推送历史")
    st.caption("只读视图：只列出真实发送过的 digest，不会重发。")

    items = load_sent_history(_root())
    if not items:
        st.info("还没有发送记录。预览不会出现在这里，只有真实发送成功后才会记录。")
        return
    for item in items:
        _render_run(item)


if __name__ == "__main__":
    main()
