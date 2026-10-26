"""Advanced home: the single doorway to every research page.

No research content renders here and no network, model, or mail call happens.
The simple daily-push page remains the default destination; this page only
groups links to the hidden advanced pages. Page configuration and sidebar
navigation are owned by the application entry.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ui_theme  # noqa: E402


def main() -> None:
    ui_theme.apply_theme()
    st.title("高级 / 研究模式")
    st.caption(
        "完整的论文流水线都在这里：评测、证据问答、推送配置、精确兴趣组、"
        "论文库、论文发现和发送历史。日常订阅仍然在「每日推送」页完成。"
    )

    st.page_link("simple_page.py", label="返回每日推送", icon="📮")

    st.divider()
    st.subheader("推送与配置")
    st.page_link("advanced/push_settings.py", label="推送设置")
    st.page_link("advanced/interests.py", label="精确兴趣组")
    st.page_link("advanced/history.py", label="推送历史")

    st.subheader("研究与评估")
    st.page_link("advanced/overview.py", label="总览与评测")
    st.page_link("advanced/evidence_qa.py", label="证据问答")

    st.subheader("语料与发现")
    st.page_link("advanced/library.py", label="论文库")
    st.page_link("advanced/discovery.py", label="论文发现")

    st.caption(
        "本页不会联网、不会调用模型、不会发邮件；每个动作都由目标页面上的"
        "按钮显式触发。"
    )


main()
