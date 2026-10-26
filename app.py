"""PaperLens application entry.

The simple daily-push page is the default destination. Research and evaluation
pages live behind one advanced home page and stay hidden from the sidebar so the
default experience remains a plain subscription form. This entry owns the only
page configuration; every page applies the shared theme itself.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="PaperLens",
    page_icon="📚",
    layout="wide",
)

st.navigation(
    [
        st.Page("simple_page.py", title="每日推送", default=True),
        st.Page("advanced/home.py", title="高级 / 研究模式"),
        st.Page("advanced/overview.py", title="总览与评测", visibility="hidden"),
        st.Page("advanced/evidence_qa.py", title="证据问答", visibility="hidden"),
        st.Page("advanced/push_settings.py", title="推送设置", visibility="hidden"),
        st.Page("advanced/interests.py", title="精确兴趣组", visibility="hidden"),
        st.Page("advanced/library.py", title="论文库", visibility="hidden"),
        st.Page("advanced/discovery.py", title="论文发现", visibility="hidden"),
        st.Page("advanced/history.py", title="推送历史", visibility="hidden"),
    ],
    position="sidebar",
).run()
