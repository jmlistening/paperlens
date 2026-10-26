"""Streamlit page: personalize the frontier definition used by the arXiv push.

The candidate pool is built from exactly these query groups and arXiv
categories. The LLM only ranks/explains candidates inside that pool; it never
adds a research direction on its own.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import profile_config  # noqa: E402
import ui_theme  # noqa: E402
from push_config import load_push_config, parse_push_config  # noqa: E402

DEFAULT_PUSH_CONFIG = PROJECT_ROOT / "push_config.json"


def _root() -> Path:
    override = os.environ.get("PAPERLENS_PUSH_ROOT")
    return Path(override).resolve() if override else PROJECT_ROOT


def _profile_path(root: Path) -> Path:
    override = os.environ.get("PAPERLENS_PROFILE_PATH")
    if override:
        return Path(override).resolve()
    return profile_config.profile_path(root)


def _save_profile(profile: dict, root: Path, path: Path) -> dict:
    """Write to the private active path unless an explicit override is set.

    The public ``interests.example.json`` is read-only, so a page opened
    without any local profile migrates its first save to ``data/push/profile.json``.
    """
    if os.environ.get("PAPERLENS_PROFILE_PATH"):
        return profile_config.save_profile(profile, path)
    return profile_config.save_active_profile(profile, root)


def _config_path(root: Path) -> Path:
    override = os.environ.get("PAPERLENS_PUSH_CONFIG")
    return Path(override).resolve() if override else root / "push_config.json"


def _save_push_links(
    path: Path,
    categories: list[str],
    max_per_digest: int,
    rank_max_candidates: int,
) -> None:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw["search_categories"] = categories
    raw["max_per_digest"] = int(max_per_digest)
    raw["rank_max_candidates"] = int(rank_max_candidates)
    parse_push_config(raw)  # reject bad keys/types before writing
    Path(path).write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _group_rows(profile: dict) -> list[dict]:
    rows = []
    for group in profile.get("search", {}).get("groups", []):
        rows.append(
            {
                "name": group.get("name", ""),
                "any_terms": ", ".join(group.get("any_terms") or []),
                "require_any_terms": ", ".join(
                    group.get("require_any_terms") or []
                ),
            }
        )
    return rows or [{"name": "", "any_terms": "", "require_any_terms": ""}]


def _parse_group_rows(rows: pd.DataFrame) -> list[dict]:
    groups = []
    for row in rows.to_dict("records"):
        name = str(row.get("name") or "").strip()
        terms = profile_config.split_terms(str(row.get("any_terms") or ""))
        required = profile_config.split_terms(
            str(row.get("require_any_terms") or "")
        )
        if not name and not terms and not required:
            continue
        groups.append(
            {
                "name": name,
                "any_terms": terms,
                "require_any_terms": required,
            }
        )
    return groups


def _show_queries(profile: dict, categories: list[str]) -> None:
    try:
        queries = profile_config.profile_queries(profile, categories)
    except Exception as exc:
        st.warning(f"查询预览失败：{exc}")
        return
    st.caption("下面就是本轮候选池的全部检索条件；不会有隐藏的“前沿”主题。")
    for query in queries:
        st.code(f"{query['name']}\n{query['query']}", language="text")


def main() -> None:
    ui_theme.apply_theme()
    st.title("兴趣配置")
    st.caption(
        "推送的候选池只来自你写的查询组和 arXiv 分类。"
        "模型只在候选池里排序和解释，不会自行添加研究方向。"
    )

    root = _root()
    profile_path = _profile_path(root)
    config_path = _config_path(root)

    try:
        profile = profile_config.load_profile(profile_path)
        config = load_push_config(config_path)
    except Exception as exc:
        st.error(f"无法读取兴趣/推送配置：{type(exc).__name__}: {exc}")
        st.stop()
        return

    categories = list(config.search_categories or [])
    st.info(
        "想改变“前沿”的定义，就在这里改。收藏/已读/忽略会作为排序反馈，"
        "但不会把未命中查询组的论文加进来。"
    )

    with st.form("interest_profile_form"):
        st.subheader("研究方向")
        research_goal = st.text_area(
            "研究目标", value=profile.get("research_goal", ""), height=80
        )
        topics = st.text_area(
            "关注主题（一行一个）",
            value="\n".join(profile.get("topics") or []),
            height=120,
        )
        preferred = st.text_area(
            "偏好方法（一行一个）",
            value="\n".join(profile.get("preferred_methods") or []),
            height=100,
        )
        less = st.text_area(
            "不太感兴趣（一行一个）",
            value="\n".join(profile.get("less_interested") or []),
            height=100,
        )

        st.subheader("检索范围")
        col1, col2, col3, col4, col5 = st.columns(5)
        top_k = col1.number_input(
            "每封入选篇数", min_value=1, max_value=10,
            value=int(profile.get("top_k", 3)),
            help="同时写入推送的 max_per_digest，实际取两者较小值。",
        )
        min_score = col2.number_input(
            "最低兴趣分", min_value=0, max_value=100,
            value=int(profile.get("min_score", 60)),
        )
        lookback = col3.number_input(
            "手动发现回看天数", min_value=1, max_value=365,
            value=int(profile["search"].get("lookback_days", 30)),
            help="只影响「论文发现」页的手动抓取；每日推送窗口在推送设置里。",
        )
        max_candidates = col4.number_input(
            "每轮候选上限", min_value=1, max_value=20,
            value=int(profile["search"].get("max_candidates", 12)),
            help="同时写入推送的 rank_max_candidates，实际取两者较小值。",
        )
        per_query = col5.number_input(
            "每组抓取上限", min_value=1, max_value=30,
            value=int(profile["search"].get("per_query_limit", 12)),
        )

        categories_text = st.text_input(
            "arXiv 分类（逗号分隔，可留空）",
            value=", ".join(categories),
            help="例如 cs.CL, cs.IR, cs.LG",
        )

        st.subheader("查询组（决定候选池）")
        st.caption(
            "只有命中这些查询组的论文才会进入候选池；topics/preferred_methods "
            "只参与排序，不会扩大或缩小候选池。每组用英文检索词，多个词用逗号分隔，最多 5 组。"
        )
        edited = st.data_editor(
            pd.DataFrame(_group_rows(profile)),
            num_rows="dynamic",
            width="stretch",
            column_config={
                "name": "组名",
                "any_terms": "任意匹配词（逗号分隔）",
                "require_any_terms": "必须同时匹配的词（可选）",
            },
            key="interest_groups_editor",
        )

        submitted = st.form_submit_button("保存兴趣配置")
        if submitted:
            try:
                new_profile = dict(profile)
                new_profile.update(
                    {
                        "research_goal": research_goal,
                        "topics": profile_config.split_lines(topics),
                        "preferred_methods": profile_config.split_lines(preferred),
                        "less_interested": profile_config.split_lines(less),
                        "top_k": int(top_k),
                        "min_score": int(min_score),
                    }
                )
                new_profile["search"] = {
                    **profile.get("search", {}),
                    "lookback_days": int(lookback),
                    "max_candidates": int(max_candidates),
                    "per_query_limit": int(per_query),
                    "groups": _parse_group_rows(edited),
                }
                saved = _save_profile(new_profile, root, profile_path)
                new_categories = profile_config.validate_categories(
                    profile_config.split_terms(categories_text)
                )
                _save_push_links(
                    config_path,
                    new_categories,
                    int(top_k),
                    int(max_candidates),
                )
            except Exception as exc:
                st.error(f"保存失败：{type(exc).__name__}: {exc}")
            else:
                st.success("兴趣配置已保存。候选池会从下一轮抓取开始变化。")
                profile = saved
                categories = new_categories

    st.subheader("当前查询预览")
    _show_queries(profile, categories)

    st.divider()
    st.caption(
        "配置文件：" + str(profile_path) + "；分类保存在 push_config.json。"
        "本页不会联网、不会调用模型、不会发邮件。"
    )


main()
