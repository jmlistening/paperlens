"""Shared visual system for PaperLens Streamlit pages.

The design is a research-instrument / printed-journal hybrid: warm paper,
ink text, one deep-teal signal color, square rules, serif display type, and
quiet controls. It deliberately avoids the generic rounded-card dashboard.
"""
from __future__ import annotations

import streamlit as st

CSS = """
<style>
:root {
  --pl-paper: #F6F3EC;
  --pl-panel: #FFFDF8;
  --pl-ink: #1C2430;
  --pl-muted: #5B6675;
  --pl-line: #D9D3C6;
  --pl-soft-line: #E8E3D8;
  --pl-accent: #0F766E;
  --pl-warn: #B45309;
}
.stApp { background: var(--pl-paper); color: var(--pl-ink); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }
footer { visibility: hidden; }
[data-testid="stSidebar"] {
  background: #EFEBE1;
  border-right: 1px solid var(--pl-line);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { font-family: inherit; }
.block-container { max-width: 1180px; padding-top: 2.0rem; padding-bottom: 3rem; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  font-family: "Noto Serif SC", "Songti SC", "STSong", "SimSun", Georgia, serif;
  color: #172033;
  letter-spacing: -0.015em;
}
.stApp h1 { font-size: clamp(2.2rem, 4vw, 3.1rem); line-height: 1.15; }
.stApp p, .stApp li, .stApp label, .stApp .stMarkdown {
  line-height: 1.7;
}
.stButton > button, .stDownloadButton > button {
  border-radius: 0;
  border: 1px solid var(--pl-ink);
  background: transparent;
  color: var(--pl-ink);
  font-weight: 600;
  padding: 0.45rem 0.9rem;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: var(--pl-ink);
  color: var(--pl-paper);
  border-color: var(--pl-ink);
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
  outline: 2px solid var(--pl-accent);
  outline-offset: 2px;
}
[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 0 !important;
  border-color: var(--pl-line) !important;
  background: var(--pl-panel);
}
.stTextInput input, .stNumberInput input, .stTextArea textarea,
.stSelectbox div[data-baseweb="select"], .stMultiSelect div[data-baseweb="select"] {
  border-radius: 0 !important;
}
[data-baseweb="tab-list"] {
  gap: 1.3rem;
  border-bottom: 1px solid var(--pl-line);
}
[data-baseweb="tab"] {
  border-radius: 0 !important;
  padding-left: 0;
  padding-right: 0;
}
[data-baseweb="tab-highlight"] { background: var(--pl-accent); }
[data-testid="stMetricValue"] {
  font-family: Georgia, "Noto Serif SC", serif;
  font-variant-numeric: tabular-nums;
}
[data-testid="stAlert"] { border-radius: 0; }
[data-testid="stExpander"] details { border-radius: 0; }
[data-testid="stForm"] { border-radius: 0; }
</style>
"""


def apply_theme() -> None:
    """Inject the shared CSS. Call once after st.set_page_config."""
    st.markdown(CSS, unsafe_allow_html=True)
