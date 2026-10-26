import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ui_theme  # noqa: E402

RAG = ROOT / "data" / "rag"


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def history_items():
    """读取最近 50 次已保存的回答。"""
    records = []

    paths = sorted(
        (RAG / "answers").glob("*/answer_result.json"),
        reverse=True,
    )

    for path in paths:
        try:
            record = read_json(path)
            when = datetime.fromisoformat(
                record["created_at"]
            ).astimezone()

            label = (
                f"{when:%m-%d %H:%M:%S}"
                f" · {record['query'][:45]}"
            )
            records.append((str(path), label))

        except (OSError, ValueError, KeyError, TypeError):
            continue

        if len(records) == 50:
            break

    return records


def run_question(query, paper, mode="fixed"):
    """调用现有统一入口，并展示运行过程。"""
    command = [
        sys.executable,
        "-u",
        str(ROOT / ("run_agent.py" if mode == "agent" else "run_qa.py")),
        query,
    ]

    if paper:
        command += ["--paper", paper]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    logs = []
    run_dir = None

    status = st.status(
        "正在查找论文证据……",
        expanded=False,
    )

    with status:
        log_area = st.empty()

        with subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        ) as process:

            for line in process.stdout:
                logs.append(line.rstrip())
                if line.startswith("阶段："):
                    stage = line.split("：", 1)[1].strip()
                    status.update(label=stage)
                    log_area.write(stage)

                if line.startswith("本次完整记录："):
                    run_dir = Path(
                        line.split("：", 1)[1].strip()
                    ).resolve()

                if line.startswith(
                    "正在运行：rerank_evidence.py"
                ):
                    status.update(
                        label="正在筛选更有帮助的证据……"
                    )

                elif line.startswith(
                    "正在运行：answer_question.py"
                ):
                    status.update(
                        label="正在生成带引用的回答……"
                    )

            code = process.wait()

    st.session_state["last_log"] = "\n".join(logs)

    if code != 0:
        status.update(
            label="本次问答未完成，请查看运行详情",
            state="error",
            expanded=True,
        )
        return None

    allowed = (RAG / ("agent_runs" if mode == "agent" else "qa_runs")).resolve()

    if (
        run_dir is None
        or not run_dir.is_relative_to(allowed)
    ):
        status.update(
            label="未找到本次运行记录",
            state="error",
        )
        return None

    report = read_json(run_dir / "run.json")

    if report.get("query") != query or (report.get("paper_filter") or "") != (paper or ""):
        raise ValueError("运行记录与本次问题不一致。")

    if report.get("status") == "insufficient_evidence":
        status.update(
            label="本次没有找到可用证据",
            state="complete",
            expanded=False,
        )
        st.warning("请调整问题或补充相关论文后再试。")
        return None

    answer_path = run_dir / "answer_result.json"

    if (
        report.get("status") != "completed"
        or not answer_path.exists()
    ):
        status.update(
            label="本次尚未生成完整答案",
            state="error",
        )
        return None

    status.update(
        label="回答已生成，引用可展开核对",
        state="complete",
        expanded=False,
    )
    return str(answer_path)


def show_answer(path):
    """展示中文回答、原文引用和证据附件。"""
    record = read_json(path)
    answer = record["answer"]
    evidence = record["evidence"]

    st.subheader(record["query"])

    names = {
        "answered": "证据可回答",
        "partial": "只能部分回答",
        "insufficient": "证据不足",
    }

    status_name = names.get(
        answer["status"], answer["status"]
    )
    st.caption(
        f"模型自评：{status_name} · {record['model']}"
    )

    agent = record.get("agent")
    if agent:
        stats = st.columns(3)
        stats[0].metric("检索轮数", agent["retrieval_rounds"])
        stats[1].metric("模型调用", agent["model_calls"])
        usage = agent.get("usage", {})
        stats[2].metric("已记录 Token", usage.get("known_total_tokens", 0))
        if not usage.get("complete", False):
            st.caption("部分请求未返回用量，上面的 Token 不是完整总量。")
        st.caption("Agent 模式 · 行动与停止原因可在执行记录中查看。")
    else:
        st.caption("固定 RAG 模式 · 单次检索与重排后回答。")

    tab_answer, tab_evidence, tab_trace = st.tabs([
        "回答与引用", "检索证据", "执行记录",
    ])

    with tab_answer:
        quality = record.get("answer_quality")
        if quality:
            if quality.get("needs_review"):
                st.warning("本次回答仍有待核对项，请查看下方“回答检查”。")
            if quality.get("quote_extensions"):
                st.caption(f"已从同一证据连续补入 {len(quality['quote_extensions'])} 处公式；原始摘录可在回答检查中查看。")
        for number, point in enumerate(
            answer["points"], 1
        ):
            st.write(f"{number}. {point['text']}")

            with st.expander(
                f"查看第 {number} 点的原文依据"
            ):
                for citation in point["citations"]:
                    source = evidence[citation["id"]]

                    st.caption(
                        f"{citation['id']} · "
                        f"{source['title']} · "
                        f"PDF 第 {source['page_number']} 页"
                    )
                    st.text(citation["quote"])

        if answer["limitations"]:
            st.markdown("**证据局限**")

            for item in answer["limitations"]:
                st.write(f"• {item}")

        st.caption(
            "引用原句存在，不代表中文解释一定正确；"
            "可展开依据自行核对。"
        )
        for change in record.get("schema_normalizations", []):
            st.info(change["message"])
        if quality:
            with st.expander("回答检查 · 引用边界与覆盖自查", expanded=quality.get("needs_review", False)):
                st.caption("覆盖内容由模型自查；程序只检查格式、回答序号及状态一致性，不能证明完整性或正确性。句界检查也可能漏报或误报。")
                for warning in quality.get("warnings", []):
                    st.warning(warning["message"])
                from answer_quality import coverage_display_rows
                rows = coverage_display_rows(answer.get("coverage"), len(answer["points"]))
                if rows:
                    st.caption("以下逐项展示模型原始自查；一项格式有误不会隐藏其他项，也不会自动改写其状态。")
                    st.dataframe(rows, hide_index=True)
                    for row in rows:
                        if row["核对说明"]:
                            st.warning(f"自查第 {row['自查项']} 项（{row['问题要点']}）：{row['核对说明']}")
                    # Derived view is exported separately; historical answer/quality
                    # fields and on-disk records remain unchanged.
                    record["coverage_display_review"] = {"version": "coverage_display_v1",
                                                          "source": "answer.coverage", "rows": rows,
                                                          "semantic_verification": "not_performed"}
                elif quality.get("coverage_validation") == "not_applicable_no_evidence":
                    st.write("未取得可用证据，本次没有生成模型回答。")
                for item in quality.get("quote_extensions", []):
                    st.write(f"第 {item['point']} 点 · {item['evidence_id']} · 已连续补入相邻公式")
                    st.caption("模型原始摘录")
                    st.text(item["original_quote"])
                    st.caption("展示的完整摘录（只允许空白规范化）")
                    st.text(item["expanded_quote"])

    with tab_evidence:
        downloaded = set()

        for eid, source in evidence.items():
            label = (
                f"{eid} · PDF 第 {source['page_number']} 页"
                f" · {source['title']}"
            )

            with st.expander(label):
                st.text(source["text"])

                url = source.get("source_url", "")
                if url.startswith(
                    "https://arxiv.org/abs/"
                ):
                    st.link_button(
                        "查看 arXiv 论文页面", url
                    )

                pdf = (
                    ROOT / source["pdf_path"]
                ).resolve()
                pdf_folder = (
                    ROOT / "data" / "pdfs"
                ).resolve()

                if (
                    pdf.is_relative_to(pdf_folder)
                    and pdf.is_file()
                    and str(pdf) not in downloaded
                ):
                    st.download_button(
                        "下载原文 PDF",
                        pdf.read_bytes(),
                        file_name=pdf.name,
                        mime="application/pdf",
                        key=f"pdf_{eid}",
                    )
                    downloaded.add(str(pdf))

    with tab_trace:
        if not agent:
            st.info("这是固定 RAG 的回答，没有 Agent 决策记录。")
        else:
            stops = {
                "sufficient": "模型判断证据已覆盖问题",
                "model_stopped_with_gaps": "保留证据缺口，结束检索",
                "round_budget": "达到检索轮数上限",
                "call_budget": "达到模型调用预算，预留最终回答",
                "repeated_query": "拦截重复查询",
                "no_new_evidence": "补检索没有带来新证据",
                "empty_retrieval": "没有检索到可用证据",
            }
            st.info("停止原因：" + stops.get(agent["stop_reason"], agent["stop_reason"]))
            for event in record.get("agent_trace", []):
                if event["kind"] == "retrieval":
                    tool = "语义检索" if event["tool"] == "search_dense" else "关键词检索"
                    with st.expander(f"第 {event['round']} 轮 · {tool} · 新增 {event['new_count']} 条证据"):
                        st.write("检索词：", event["query"])
                        st.write("返回候选：", event["candidate_count"])
                        st.dataframe(event["sources"], hide_index=True)
                elif event["kind"] == "decision":
                    with st.expander(f"第 {event['round']} 轮 · 证据评估与行动"):
                        st.write("覆盖自评：", event["assessment"])
                        st.write("行动依据：", event["reason"])
                        st.write("下一步：", event["action"])
                        st.write("选择证据：", event["selected_ids"])
                        if event["missing"]:
                            st.write("尚缺内容：", event["missing"])
            st.caption("覆盖判断由模型给出；执行记录不等于答案正确性的证明。")

    st.download_button(
        "导出完整回答与执行记录（JSON）",
        json.dumps(record, ensure_ascii=False, indent=2),
        file_name="paperlens_answer_record.json", mime="application/json",
    )
    preview = Path(path).with_name("answer_preview.txt")

    if preview.exists():
        st.download_button(
            "导出本次回答",
            preview.read_bytes(),
            file_name="paperlens_answer.txt",
            mime="text/plain; charset=utf-8",
        )


def main():
    ui_theme.apply_theme()

    st.markdown("""<style>
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stText"] {font-family: inherit; line-height: 1.7;}
    [data-testid="stForm"] {border-radius: 0; padding: 1.3rem;
        border: 1px solid #D9D3C6; background: #FFFDF8;}
    [data-testid="stMetric"] {padding: .8rem 1rem; border-radius: 0;
        background: #FFFDF8; border: 1px solid #D9D3C6;}
    </style>""", unsafe_allow_html=True)
    st.title("PaperLens")
    st.caption(
        "用论文原文回答问题，并保留可核对的出处。"
    )

    records = history_items()

    with st.sidebar:
        st.header("论文问答")
        st.caption("当前为本机运行的个人版。")
        st.subheader("历史记录")

        if records:
            labels = dict(records)

            choice = st.selectbox(
                "最近 50 次回答",
                list(labels),
                format_func=labels.get,
            )

            if st.button("打开所选回答"):
                st.session_state["active_answer"] = choice

        else:
            st.info(
                "完成第一次问答后，记录会显示在这里。"
            )

        st.caption(
            "打开历史记录、查看引用和下载文件"
            "不会调用模型。"
        )

    papers = {"": "全部已入库论文"}
    corpus_path = RAG / "corpus_clean.json"

    if corpus_path.exists():
        for chunk in read_json(corpus_path)["chunks"]:
            papers[chunk["arxiv_id"]] = chunk["title"]

    ready = (
        (ROOT / "run_qa.py").exists()
        and (RAG / "vector_store.npz").exists()
    )

    if not ready:
        st.warning(
            "请先完成论文入库，并准备好 run_qa.py。"
        )

    left, right = st.columns([2, 1])
    left.caption(f"已入库 {len(papers) - 1} 篇论文 · 回答依据来自本地原文")
    right.caption("发现 → 入库 → 证据问答")
    mode_label = st.radio(
        "问答模式", ["Agent · 证据补检索", "固定 RAG · 对照模式"], horizontal=True,
        help="Agent 会评估证据并决定是否继续检索；固定模式保留原有处理流程。",
    )
    mode = "agent" if mode_label.startswith("Agent") else "fixed"
    if mode == "agent" and not (ROOT / "run_agent.py").is_file():
        ready = False
        st.warning("请先完整复制 Agent 更新包。")
    st.caption("Agent：最多 3 轮本地检索、4 次模型调用；证据不足时保留缺口。"
               if mode == "agent" else "固定 RAG：通常调用 2 次模型，分别用于重排与回答。")

    with st.form("question_form"):
        query = st.text_area(
            "想了解什么？",
            placeholder=(
                "例如：CARA 如何评估检索证据的因果相关性？"
            ),
            max_chars=2000,
        )

        paper = st.selectbox(
            "检索范围",
            list(papers),
            format_func=papers.get,
        )

        st.caption("点击提交才会调用模型，按实际 API 用量计费。查看历史和导出记录不调用模型。")

        submitted = st.form_submit_button(
            "查找证据并回答",
            type="primary",
            disabled=not ready,
        )

    if submitted:
        if not query.strip():
            st.warning("请先输入问题。")

        else:
            st.session_state.pop("active_answer", None)
            st.session_state.pop("last_log", None)

            try:
                result = run_question(
                    query.strip(), paper, mode
                )

                if result:
                    st.session_state[
                        "active_answer"
                    ] = result

            except Exception as error:
                st.error(f"问答未完成：{error}")

    if st.session_state.get("last_log"):
        with st.expander("运行日志（排查问题时查看）", expanded=False):
            st.code(st.session_state["last_log"], language="text")

    active = st.session_state.get("active_answer")

    if active:
        try:
            show_answer(active)

        except (
            OSError, ValueError, KeyError, TypeError
        ) as error:
            st.error(f"无法展示这条记录：{error}")

    else:
        st.info(
            "输入问题开始检索，或从左侧打开已有回答。"
        )


if __name__ == "__main__":
    main()
