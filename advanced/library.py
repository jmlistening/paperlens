"""PaperLens local paper library status page."""
import json
import os
import subprocess
import sys
from pathlib import Path


import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import ui_theme  # noqa: E402
from library_status import scan_library  # noqa: E402

ui_theme.apply_theme()
st.title("本地论文库")
st.caption("查看入库状态，或将全部本地原文接入问答。使用本地嵌入模型，不调用大模型 API 或 arXiv。")

if st.button("累计入库全部本地原文", type="primary"):
    st.session_state.pop("library_status_report", None)
    output = st.empty()
    lines = []
    try:
        with st.spinner("正在解析、清洗并构建索引，请保持本页打开……"):
            process = subprocess.Popen(
                [sys.executable, "-u", str(ROOT / "ingest_library.py")],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            for line in process.stdout:
                lines.append(line.rstrip())
                output.code("\n".join(lines[-18:]), language=None)
            if process.wait() != 0:
                raise RuntimeError("入库未完成，请检查上方日志；不要将旧索引当成本次成功。")
            st.session_state["library_status_report"] = scan_library(ROOT)
        st.success("累计入库检查完成，下面是当前实际状态。")
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")

if st.button("检查 / 刷新论文库状态"):
    st.session_state.pop("library_status_report", None)
    try:
        with st.spinner("正在检查本地文件与索引……"):
            st.session_state["library_status_report"] = scan_library(ROOT)
    except Exception as exc:
        st.error(f"检查未完成：{type(exc).__name__}: {exc}")

report = st.session_state.get("library_status_report")
if report is None:
    st.info("点击上方按钮，查看已下载论文中哪些已经接入问答。")
    st.stop()

st.caption(f"检查时间（UTC）：{report['checked_at']}。文件变化后请重新检查。")
left, middle, right = st.columns(3)
left.metric("本地原文 PDF", report["pdf_count"])
middle.metric("已入有效索引", report["ready_count"])
right.metric("本地待处理", sum(p["pdf_exists"] and not p["ready"] for p in report["papers"]))
for warning in report["warnings"]:
    st.warning(warning)
st.info("“已入有效索引”表示该论文通过本地文件与索引一致性检查，不代表模型必然能回答所有问题。")

choice = st.selectbox("状态筛选", ["全部", "已入有效索引", "需要处理"])
rows = report["papers"]
if choice == "已入有效索引":
    rows = [p for p in rows if p["ready"]]
elif choice == "需要处理":
    rows = [p for p in rows if not p["ready"]]
for paper in rows:
    with st.expander(f"{paper['arxiv_id']} · {paper['status']} · {paper['title']}"):
        st.write({"原始片段": paper["raw_chunks"], "清洗片段": paper["clean_chunks"],
                  "有效索引中的向量": paper["vectors"]})
        st.write(paper["next_action"])
        if paper["ready"]:
            st.caption("回到证据问答页，在论文筛选框中选择上面的完整编号。")
        st.code(paper["arxiv_id"], language=None)
if not rows:
    st.write("当前筛选下没有论文。")
if report["unrecognized_pdf_files"]:
    with st.expander("未按原文编号识别的 PDF"):
        st.write(report["unrecognized_pdf_files"])
        st.caption("翻译副本或其他文件名暂不自动认作原文，避免重复入库。")
st.download_button("下载本次状态报告", json.dumps(report, ensure_ascii=False, indent=2),
                   file_name="paperlens_library_status.json", mime="application/json")
