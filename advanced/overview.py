"""PaperLens overview: system state, evaluation snapshot, and push status."""
from __future__ import annotations

import html
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import push_status  # noqa: E402
import ui_theme  # noqa: E402
from push_config import load_push_config  # noqa: E402

TASK_NAME = "PaperLens Frontier Push"
EVAL_DIR = PROJECT_ROOT / "eval" / "benchmark_v2" / "reports" / "benchmark_v2_120_retrieval"
ANSWER_EVAL_DIR = (
    PROJECT_ROOT / "eval" / "benchmark_v2" / "reports" / "answer_eval_v2_120"
)
REFUSAL_DIR = PROJECT_ROOT / "eval" / "benchmark_v2" / "reports" / "answer_eval_refusal"
SYSTEM_LABELS = {
    "bm25_top6": "BM25",
    "dense_top6": "Dense · English query",
    "dense_zh_top6": "Dense · Chinese query",
    "hybrid_rrf_top6": "RRF hybrid",
    "hybrid_rrf_rerank_top6": "RRF + DeepSeek rerank",
}


def _root() -> Path:
    override = os.environ.get("PAPERLENS_PUSH_ROOT")
    return Path(override).resolve() if override else PROJECT_ROOT


def _config_path(root: Path) -> Path:
    override = os.environ.get("PAPERLENS_PUSH_CONFIG")
    return Path(override).resolve() if override else root / "push_config.json"


def _query_task() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["schtasks", "/Query", "/TN", TASK_NAME, "/FO", "LIST"],
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, type(exc).__name__
    output = (result.stdout or result.stderr or "").strip()
    return result.returncode == 0, output[:800]


def _run_ps(script: Path, root: Path) -> tuple[int, str]:
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "RemoteSigned",
        "-File",
        str(script),
        "-ProjectDir",
        str(root),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, type(exc).__name__
    output = (result.stdout or result.stderr or "").strip()
    return result.returncode, output[:1200]


def _load_optional_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_refusal() -> dict | None:
    path = REFUSAL_DIR / "refusal_summary.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_answer_eval() -> dict | None:
    path = ANSWER_EVAL_DIR / "aggregate.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_eval_systems() -> list[dict]:
    systems: list[dict] = []
    for name in ("results.json", "rerank_results.json"):
        path = EVAL_DIR / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for row in payload.get("systems") or []:
            if isinstance(row, dict) and row.get("status") == "completed":
                systems.append(row)
    return systems


def _recent_delivered(root: Path, limit: int = 3) -> list[dict]:
    path = root / "data" / "push" / "frontier.sqlite3"
    if not path.is_file():
        return []
    try:
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT title, arxiv_id, delivered_at FROM papers "
                "WHERE status='delivered' ORDER BY delivered_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return [dict(row) for row in rows]


def _metric(row: dict, key: str) -> str:
    value = (row.get("metrics") or {}).get(key)
    if not isinstance(value, dict):
        return "—"
    mean = value.get("mean")
    low = value.get("ci_low")
    high = value.get("ci_high")
    if not all(isinstance(item, (int, float)) for item in (mean, low, high)):
        return "—"
    return f"{mean:.3f} <span class=\"pl-ci\">[{low:.3f}, {high:.3f}]</span>"


def _stat_html(number: str, label: str) -> str:
    return (
        '<div class="pl-stat">'
        f'<div class="pl-n">{html.escape(number)}</div>'
        f'<div class="pl-l">{html.escape(label)}</div>'
        "</div>"
    )


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --pl-paper: #F6F3EC;
          --pl-ink: #1C2430;
          --pl-muted: #5B6675;
          --pl-line: #D9D3C6;
          --pl-accent: #0F766E;
          --pl-warn: #B45309;
        }
        .stApp { background: var(--pl-paper); color: var(--pl-ink); }
        [data-testid="stSidebar"] { background: #EFEBE1; }
        .block-container { max-width: 1180px; padding-top: 2.0rem; padding-bottom: 3rem; }
        .pl-hero { border-top: 3px solid var(--pl-accent); border-bottom: 1px solid var(--pl-line); padding-top: 1.1rem; padding-bottom: 1.5rem; margin-bottom: 0.6rem; }
        .pl-kicker { color: var(--pl-accent); font: 600 0.82rem/1.2 "Segoe UI", sans-serif; }
        .pl-hero h1 {
          font-family: "Noto Serif SC", "Songti SC", "SimSun", Georgia, serif;
          font-weight: 600; font-size: clamp(2.5rem, 5vw, 4.4rem);
          line-height: 1.08; letter-spacing: -0.025em; margin: 0.55rem 0 0.8rem;
        }
        .pl-lede { max-width: 46rem; color: var(--pl-muted); font-size: 1.03rem; line-height: 1.75; margin: 0; }
        .pl-strip { display: flex; flex-wrap: wrap; border-bottom: 1px solid var(--pl-line); margin: 0.8rem 0 1.8rem; }
        .pl-stat { flex: 1; min-width: 160px; padding: 0.9rem 1.1rem 1rem; border-right: 1px solid var(--pl-line); }
        .pl-stat:first-child { padding-left: 0; }
        .pl-stat:last-child { border-right: 0; }
        .pl-n { font-family: Georgia, "Noto Serif SC", serif; font-size: 2rem; line-height: 1; }
        .pl-l { color: var(--pl-muted); font-size: 0.78rem; margin-top: 0.4rem; }
        .pl-panel-head h3 { font-family: Georgia, "Noto Serif SC", serif; font-size: 1.45rem; margin: 0 0 0.35rem; }
        .pl-panel-head p { color: var(--pl-muted); line-height: 1.65; margin: 0 0 0.8rem; }
        .pl-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-left: 0.45rem; vertical-align: middle; }
        .pl-dot.on { background: var(--pl-accent); }
        .pl-dot.off { background: #B9B2A5; }
        .pl-dot.warn { background: var(--pl-warn); }
        .pl-recent { margin: 0.35rem 0 0; padding: 0; list-style: none; }
        .pl-recent li { border-top: 1px solid var(--pl-soft-line); padding: 0.7rem 0; line-height: 1.55; }
        .pl-recent li:first-child { border-top: 0; }
        .pl-recent a { color: var(--pl-ink); text-decoration: underline; text-decoration-color: var(--pl-line); text-underline-offset: 3px; }
        .pl-recent .pl-meta { color: var(--pl-muted); font-size: 0.78rem; }
        .pl-table tbody tr:hover { background: #F1ECE1; }
        .pl-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
        .pl-table th { text-align: left; color: var(--pl-muted); font-weight: 600; border-bottom: 1px solid var(--pl-line); padding: 0.6rem 0.45rem; }
        .pl-table td { border-bottom: 1px solid #E8E3D8; padding: 0.6rem 0.45rem; vertical-align: top; }
        .pl-table td.num { font-family: Consolas, "SFMono-Regular", monospace; white-space: nowrap; }
        .pl-ci { color: var(--pl-muted); font-size: 0.78rem; }
        .pl-note { border-left: 3px solid var(--pl-accent); padding: 0.15rem 0 0.15rem 0.8rem; color: var(--pl-muted); font-size: 0.86rem; line-height: 1.65; }
        .pl-warn { border-left: 3px solid var(--pl-warn); padding: 0.15rem 0 0.15rem 0.8rem; color: var(--pl-warn); font-size: 0.86rem; line-height: 1.65; }
        .pl-section { margin-top: 1.9rem; }
        .stButton > button { border-radius: 0; border: 1px solid var(--pl-ink); background: transparent; color: var(--pl-ink); }
        .stButton > button:hover { background: var(--pl-ink); color: var(--pl-paper); border-color: var(--pl-ink); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    ui_theme.apply_theme()
    _inject_css()

    root = _root()
    config_path = _config_path(root)
    environment = push_status.load_env(root)

    st.markdown(
        """
        <div class="pl-hero">
          <div class="pl-kicker">PaperLens</div>
          <h1>你的论文雷达，<br>由你校准。</h1>
          <p class="pl-lede">从你定义的 arXiv 候选池，到可追溯的证据问答，再到每日 163 推送。
          一条能审计、能评测、也能停下来的研究流水线。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        config = load_push_config(config_path)
        report = push_status.status(root, config_path, environment)
    except Exception as exc:
        st.error(f"无法读取推送状态：{type(exc).__name__}: {exc}")
        st.stop()
        return

    systems = _load_eval_systems()
    rerank = next(
        (row for row in systems if row.get("system") == "hybrid_rrf_rerank_top6"),
        None,
    )
    rerank_ndcg = "—"
    if rerank:
        value = (rerank.get("metrics") or {}).get("ndcg@10") or {}
        if isinstance(value.get("mean"), (int, float)):
            rerank_ndcg = f"{value['mean']:.3f}"

    st.markdown(
        '<div class="pl-strip">'
        + _stat_html("48 / 2,125", "indexed / prepared papers")
        + _stat_html("60", "held-out questions")
        + _stat_html(rerank_ndcg, "rerank nDCG@10 / n=60")
        + _stat_html("769", "offline tests")
        + "</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        with st.container(border=True):
            dot = (
                "warn"
                if report["auto_paused"]
                or (report["enabled"] and not report["first_send_approved"])
                else "on" if report["enabled"] else "off"
            )
            st.markdown(
                '<div class="pl-panel-head">'
                f'<h3>自动推送 <span class="pl-dot {dot}"></span></h3>'
                "<p>开关由「每日推送」页统一管理，这里只展示状态。"
                "要每天自动运行，还需要安装 Windows 计划任务；"
                "之前仍可用手动预览和发送。</p></div>",
                unsafe_allow_html=True,
            )
            if st.button("去「每日推送」页开关推送", key="dash_open_simple"):
                st.switch_page("simple_page.py")

            if report["auto_paused"]:
                st.markdown(
                    '<div class="pl-warn">上一次投递结果不确定，系统已自动暂停；'
                    "请到「推送设置」排查后再恢复。</div>",
                    unsafe_allow_html=True,
                )
            if report["enabled"] and not report["first_send_approved"]:
                st.markdown(
                    '<div class="pl-warn">自动推送已开启，但首次发送尚未批准；'
                    "计划任务会拒绝发信。</div>",
                    unsafe_allow_html=True,
                )
            st.caption(
                f"服务商 {report['provider']} ｜ 收件人 {report['mail_to'] or '未配置'} "
                f"｜ 上次成功 {report['last_success'] or '无'}"
            )

    with right:
        installed, task_detail = _query_task()
        with st.container(border=True):
            st.markdown(
                '<div class="pl-panel-head">'
                f'<h3>计划任务 <span class="pl-dot {"on" if installed else "off"}"></span></h3>'
                f"<p>任务 <strong>{TASK_NAME}</strong>："
                f"{'已安装' if installed else '未安装'}。注册脚本使用当前 Windows "
                "用户、交互式登录，不保存密码。</p></div>",
                unsafe_allow_html=True,
            )
            if task_detail and not installed:
                st.caption(f"查询结果：{task_detail[:240]}")

            confirm = st.checkbox(
                "我确认：开启后会在每天配置时间自动发送邮件",
                key="dash_schedule_confirm",
            )
            install_col, remove_col = st.columns(2)
            if install_col.button(
                "安装每日计划任务",
                key="dash_install_task",
                disabled=not confirm,
            ):
                code, output = _run_ps(
                    root / "scripts" / "register_push_task.ps1", root
                )
                if code == 0:
                    st.success("计划任务已安装。")
                    st.rerun()
                else:
                    st.error(f"安装失败：{output or code}")
                    st.code(
                        "powershell -NoProfile -ExecutionPolicy RemoteSigned "
                        f"-File scripts/register_push_task.ps1 -ProjectDir {root}"
                    )
            if remove_col.button(
                "移除计划任务",
                key="dash_remove_task",
                disabled=not installed,
            ):
                code, output = _run_ps(
                    root / "scripts" / "unregister_push_task.ps1", root
                )
                if code == 0:
                    st.success("计划任务已移除。")
                    st.rerun()
                else:
                    st.error(f"移除失败：{output or code}")

            if installed:
                st.markdown(
                    '<div class="pl-note">计划任务已安装。它只在自动推送开启、'
                    "且非暂停状态时发信；关机/未登录时会错过 on-time，"
                    "由 Task Scheduler 的补跑策略处理。</div>",
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="pl-section"></div>', unsafe_allow_html=True)
    st.markdown("### 60 题检索评测")
    st.caption(
        "Benchmark v2.120 ｜ 40 evaluated + 80 distractor papers ｜ model-reviewed ｜ "
        "n=60 ｜ 83 gold parent chunks ｜ 描述性配对 bootstrap，无预注册/多重比较校正。"
    )
    if not systems:
        st.info("还没有 v2 评测结果文件。")
    else:
        rows = []
        for row in systems:
            name = SYSTEM_LABELS.get(row.get("system"), str(row.get("system")))
            rows.append(
                "<tr>"
                f"<td>{html.escape(name)}</td>"
                f'<td class="num">{_metric(row, "recall@6")}</td>'
                f'<td class="num">{_metric(row, "ndcg@10")}</td>'
                f'<td class="num">{_metric(row, "mrr@10")}</td>'
                f'<td class="num">{int(row.get("n_failed") or 0)}</td>'
                "</tr>"
            )
        st.markdown(
            '<div style="overflow-x:auto;"><table class="pl-table"><thead><tr>'
            "<th>系统</th><th>Recall@6</th><th>nDCG@10</th><th>MRR@10</th><th>失败</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="pl-note">BM25 仍是最强的单路 baseline；plain RRF 没有超过 BM25；'
            'DeepSeek rerank 在 120 篇 distractor 语料上把 nDCG@10 从 0.660 提到 0.830、'
            'MRR@10 从 0.645 提到 0.838，但 Recall 提升的 CI 跨 0。'
            '这些是固定 benchmark 上的描述性结果，不是预注册的普遍显著提升。</p>',
            unsafe_allow_html=True,
        )

    answer_eval = _load_answer_eval()
    if answer_eval:
        st.markdown('<div class="pl-section"></div>', unsafe_allow_html=True)
        st.markdown("### 答案级 citation 评测")
        st.caption(
            "复用 120 篇 rerank 证据，n=60。只测 schema、逐字引文、"
            "cited gold parent 和 gold-point citation；不是 answer accuracy / "
            "faithfulness，没有 judge 或人工复核。"
        )
        answer_labels = {
            "schema_success": "Schema 通过",
            "quote_integrity": "逐字引文通过",
            "cited_gold_parent_recall": "Cited gold parent",
            "gold_point_citation_recall": "Gold-point citation",
        }
        answer_rows = [
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f'<td class="num">{_metric(answer_eval, key)}</td>'
            "</tr>"
            for key, label in answer_labels.items()
        ]
        precision = _load_optional_json(
            ANSWER_EVAL_DIR / "citation_precision.json"
        )
        if precision:
            for key, label in (
                ("cited_parent_precision", "Cited gold parent precision"),
                (
                    "citation_occurrence_gold_quote_precision",
                    "Citation occurrence gold-quote precision",
                ),
            ):
                value = precision.get(key)
                if isinstance(value, list) and len(value) == 2:
                    answer_rows.append(
                        "<tr>"
                        f"<td>{html.escape(label)}</td>"
                        f'<td class="num">{float(value[0]):.3f} '
                        f'<span class="pl-ci">(n={int(value[1])})</span></td>'
                        "</tr>"
                    )
        st.markdown(
            '<div style="overflow-x:auto;"><table class="pl-table"><thead><tr>'
            "<th>指标</th><th>Mean [95% CI]</th>"
            "</tr></thead><tbody>"
            + "".join(answer_rows)
            + "</tbody></table></div>",
            unsafe_allow_html=True,
        )
        counts = answer_eval.get("status_counts") or {}
        st.caption(
            "状态："
            + " / ".join(
                f"{name} {int(counts.get(name, 0))}"
                for name in ("answered", "partial", "insufficient", "invalid")
            )
            + f"；失败 {int(answer_eval.get('n_failed', 0))}/"
            f"{int(answer_eval.get('n', 0))} 按 0 计。"
        )
        refusal = _load_refusal()
        if refusal:
            st.markdown(
                '<p class="pl-note">Out-of-scope refusal 检查：'
                f"{int(refusal.get('strict_refusal_insufficient', 0))}/"
                f"{int(refusal.get('n', 0))} 返回 insufficient；"
                "这是模板化外部信息题，不是完整拒答基准。</p>",
                unsafe_allow_html=True,
            )

    recent = _recent_delivered(root)
    if recent:
        st.markdown('<div class="pl-section"></div>', unsafe_allow_html=True)
        st.markdown("### 最近推送")
        items = []
        for row in recent:
            title = html.escape(row.get("title") or "Untitled")
            arxiv_id = html.escape(row.get("arxiv_id") or "")
            when = html.escape(str(row.get("delivered_at") or ""))
            when = when.replace("T", " ")[:16]
            items.append(
                f'<li><a href="https://arxiv.org/abs/{arxiv_id}">{title}</a>'
                f'<div class="pl-meta">{arxiv_id} / {when} UTC</div></li>'
            )
        st.markdown(
            '<ul class="pl-recent">' + "".join(items) + "</ul>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="pl-section"></div>', unsafe_allow_html=True)
    st.markdown("### 现在能做什么")
    cols = st.columns(3)
    if cols[0].button("打开证据问答", key="dash_open_qa"):
        st.switch_page("advanced/evidence_qa.py")
    if cols[1].button("推送设置与预览", key="dash_open_push"):
        st.switch_page("advanced/push_settings.py")
    if cols[2].button("编辑我的前沿定义", key="dash_open_profile"):
        st.switch_page("advanced/interests.py")

    st.markdown(
        '<p class="pl-small">已知限制：评测只有检索层，没有 answer accuracy / '
        'faithfulness / citation / refusal；benchmark 是 model-reviewed，不是人工核验；'
        '60 题来自同一批 40 篇论文；rerank 有 1/60 失败按 0 计；尚未接 CI/GitHub 发布。</p>',
        unsafe_allow_html=True,
    )


main()
