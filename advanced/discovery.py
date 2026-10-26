import hashlib
import json
import re
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st


# 当前文件在 advanced 文件夹中，向上一层就是项目目录。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import ui_theme  # noqa: E402

DB = ROOT / "data" / "reading.sqlite3"

STATES = {
    "unread": "未读",
    "read": "已读",
    "ignored": "已忽略",
}


# ---------- 第一部分：数据库 ----------

@contextmanager
def database():
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.row_factory = sqlite3.Row

    try:
        with conn:
            # 保存论文信息，以及当前的阅读状态。
            conn.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    paper_key TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    favorite INTEGER NOT NULL DEFAULT 0,
                    reading_state TEXT NOT NULL DEFAULT 'unread',
                    note TEXT NOT NULL DEFAULT '',
                    feedback_at TEXT NOT NULL DEFAULT ''
                )
            """)

            # 记录已经导入的推荐文件，避免每次刷新重复导入。
            conn.execute("""
                CREATE TABLE IF NOT EXISTS imports (
                    source_hash TEXT PRIMARY KEY
                )
            """)

            # 保存用户反馈事件，方便以后分析推荐效果。
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY,
                    paper_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    feedback_json TEXT NOT NULL
                )
            """)

            yield conn
    finally:
        conn.close()


# ---------- 第二部分：导入论文、保存反馈 ----------

def import_recommendations(path):
    if not path.exists():
        return

    raw = path.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw.decode("utf-8-sig"))

    papers = data.get("ranked_papers")
    if not isinstance(papers, list):
        raise ValueError("推荐文件缺少 ranked_papers。")

    selected = set(data.get("selected_ids", []))
    rows = []
    seen = set()

    for paper in papers:
        pid = paper["arxiv_id"]

        if not isinstance(pid, str) or not re.fullmatch(
            r"(?:\d{4}\.\d{4,5}|[A-Za-z.-]+/\d{7})(?:v\d+)?",
            pid,
        ):
            raise ValueError("推荐文件中的 arXiv 编号无效。")

        # 去掉版本号，让同一篇论文的新版本保留原来的阅读记录。
        key = re.sub(r"v\d+$", "", pid)

        if key in seen:
            raise ValueError("同一轮推荐包含重复论文。")
        seen.add(key)

        score = paper["assessment"]["score"]
        if type(score) is not int or not 0 <= score <= 100:
            raise ValueError("兴趣匹配分格式无效。")

        for field in ("title", "abstract"):
            if not isinstance(paper.get(field), str):
                raise ValueError(f"论文缺少 {field}。")

        snapshot = {
            "paper": paper,
            "selected": pid in selected,
            "recommended_at": data.get("created_at", "未知"),
        }

        rows.append((
            key,
            json.dumps(snapshot, ensure_ascii=False),
        ))

    with database() as conn:
        already_imported = conn.execute(
            "SELECT 1 FROM imports WHERE source_hash=?",
            (source_hash,),
        ).fetchone()

        if already_imported:
            return

        for key, payload in rows:
            # 更新推荐信息，保留收藏、阅读状态和笔记。
            conn.execute("""
                INSERT INTO papers(paper_key, data_json)
                VALUES (?, ?)
                ON CONFLICT(paper_key)
                DO UPDATE SET data_json=excluded.data_json
            """, (key, payload))

        conn.execute(
            "INSERT INTO imports(source_hash) VALUES (?)",
            (source_hash,),
        )


def load_papers():
    with database() as conn:
        rows = [
            dict(row)
            for row in conn.execute("SELECT * FROM papers")
        ]

    for row in rows:
        row["snapshot"] = json.loads(row.pop("data_json"))

    return sorted(
        rows,
        key=lambda row: (
            -row["snapshot"]["paper"]["assessment"]["score"],
            row["paper_key"],
        ),
    )


def save_feedback(key, favorite, state, note):
    if state not in STATES or len(note) > 1000:
        raise ValueError("阅读状态或笔记格式无效。")

    # 忽略和收藏互斥：忽略时取消收藏。
    favorite = bool(favorite) and state != "ignored"
    now = datetime.now(timezone.utc).isoformat()

    feedback = {
        "favorite": favorite,
        "reading_state": state,
        "note": note.strip(),
    }

    with database() as conn:
        result = conn.execute("""
            UPDATE papers
            SET favorite=?, reading_state=?, note=?, feedback_at=?
            WHERE paper_key=?
        """, (
            int(favorite),
            state,
            note.strip(),
            now,
            key,
        ))

        if result.rowcount != 1:
            raise ValueError("找不到这篇论文。")

        conn.execute("""
            INSERT INTO feedback_events(
                paper_key, created_at, feedback_json
            )
            VALUES (?, ?, ?)
        """, (
            key,
            now,
            json.dumps(feedback, ensure_ascii=False),
        ))


# ---------- 第三部分：网页 ----------

def main():
    ui_theme.apply_theme()

    st.title("论文发现与阅读管理")
    st.caption("查看已生成的推荐，积累自己的阅读记录。")

    try:
        import_recommendations(
            ROOT / "data" / "recommendations.json"
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        sqlite3.Error,
    ) as error:
        st.warning(
            f"本次推荐导入失败，仍可查看已保存记录：{error}"
        )

    rows = load_papers()

    if not rows:
        st.info(
            "暂时没有论文记录，请先生成 "
            "data/recommendations.json。"
        )
        return

    columns = st.columns(4)
    columns[0].metric("已记录论文", len(rows))
    columns[1].metric(
        "收藏",
        sum(row["favorite"] for row in rows),
    )
    columns[2].metric(
        "已读",
        sum(row["reading_state"] == "read" for row in rows),
    )
    columns[3].metric(
        "已忽略",
        sum(row["reading_state"] == "ignored" for row in rows),
    )

    if message := st.session_state.pop("reading_notice", None):
        st.success(message)

    st.info(
        "兴趣匹配分来自上次推荐。修改收藏或忽略状态后，"
        "需要重新运行推荐程序，才会生成新的推荐结果。"
    )

    view = st.selectbox(
        "查看范围",
        ["全部（不含忽略）", "收藏", "未读", "已读", "已忽略"],
    )

    query = st.text_input(
        "搜索标题或摘要"
    ).strip().casefold()

    filtered = []

    for row in rows:
        state = row["reading_state"]

        if view == "全部（不含忽略）" and state == "ignored":
            continue

        if view == "收藏" and not row["favorite"]:
            continue

        target = {
            "未读": "unread",
            "已读": "read",
            "已忽略": "ignored",
        }.get(view)

        if target and state != target:
            continue

        paper = row["snapshot"]["paper"]
        searchable = (
            paper["title"] + " " + paper["abstract"]
        ).casefold()

        if query and query not in searchable:
            continue

        filtered.append(row)

    if not filtered:
        st.info("当前筛选条件下没有论文。")
        return

    page = st.number_input(
        "页码（每页 10 篇）",
        min_value=1,
        max_value=max(1, (len(filtered) + 9) // 10),
        value=1,
    )

    for row in filtered[(page - 1) * 10:page * 10]:
        snapshot = row["snapshot"]
        paper = snapshot["paper"]
        assessment = paper["assessment"]

        with st.container(border=True):
            st.subheader(paper["title"])

            status_text = (
                f"{paper['arxiv_id']} · "
                f"{STATES[row['reading_state']]}"
            )
            if row["favorite"]:
                status_text += " · 已收藏"
            st.caption(status_text)

            st.write(
                f"记录中的兴趣匹配分：{assessment['score']}/100"
                "（不是论文质量分）"
            )
            st.write(
                "推荐理由：" + assessment.get("reason", "未提供")
            )
            st.write(
                "仍需确认：" + assessment.get("limitation", "未提供")
            )

            selection_text = (
                "该轮入选" if snapshot["selected"] else "该轮候选"
            )
            st.caption(
                f"推荐记录时间：{snapshot['recommended_at']} · "
                f"{selection_text}"
            )

            with st.expander("查看摘要"):
                st.write(paper["abstract"])

            st.link_button(
                "打开 arXiv 论文",
                "https://arxiv.org/abs/" + paper["arxiv_id"],
            )

            form_key = (
                f"feedback_{row['paper_key']}_{row['feedback_at']}"
            )

            with st.form(form_key):
                favorite = st.checkbox(
                    "收藏这篇论文",
                    value=bool(row["favorite"]),
                )

                state = st.selectbox(
                    "阅读状态",
                    list(STATES),
                    index=list(STATES).index(row["reading_state"]),
                    format_func=STATES.get,
                )

                note = st.text_area(
                    "阅读笔记或忽略原因（可选）",
                    value=row["note"],
                    max_chars=1000,
                )

                st.caption(
                    "忽略后会取消收藏，并隐藏在默认列表中；"
                    "可在“已忽略”中恢复。"
                )

                submitted = st.form_submit_button("保存阅读记录")

            if submitted:
                try:
                    save_feedback(
                        row["paper_key"], favorite, state, note
                    )
                except (ValueError, sqlite3.Error) as error:
                    st.error(f"保存失败：{error}")
                else:
                    st.session_state["reading_notice"] = (
                        "阅读记录已保存。"
                    )
                    st.rerun()


if __name__ == "__main__":
    main()