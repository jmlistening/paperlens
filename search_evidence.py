import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import query_rewrite
from prepare_corpus import atomic_write
from search_vector import load_store

ROOT = Path(__file__).resolve().parent

# 识别独立的参考文献标题，以及附录标题。
HEADING = re.compile(
    r"^[ \t]*(?:(?:\d+(?:\.\d+)*\.?)[ \t]+)?"
    r"(?P<ref>references|bibliography)[ \t]*$"
    r"|^[ \t]*(?:[A-Z][ \t\n]+)?appendix(?:[ \t]+[^\n]*)?[ \t]*$",
    re.I | re.M,
)


def section_events(chunks):
    """记录每篇论文中参考文献、附录的起始位置。"""
    events = {}

    for chunk in chunks:
        paper = events.setdefault(chunk["arxiv_id"], {})

        for match in HEADING.finditer(chunk["text"]):
            position = (
                chunk["page_number"],
                chunk["group_number"],
                chunk["char_start"] + match.start(),
            )
            paper[position] = (
                "references" if match.group("ref") else "appendix"
            )

    return {
        paper: sorted(items.items())
        for paper, items in events.items()
    }


def exclusion_reason(unit, events):
    """返回过滤原因；返回 None 表示保留。"""
    start = (
        unit["page_number"],
        unit["group_number"],
        unit["char_start"],
    )
    end = (
        unit["page_number"],
        unit["group_number"],
        unit["char_end"],
    )

    state = "body"

    for position, section in events.get(unit["arxiv_id"], []):
        if position <= start:
            state = section
        elif position < end and section == "references":
            # 正文与参考文献混在同一片段，暂时整段跳过。
            return "contains_reference_heading"

    if state == "references":
        return "reference_section"

    if len(unit["text"].strip()) < 80:
        return "short_fragment"

    return None


def select_hits(index, units, vector, events, top_k, paper=None):
    """先对完整索引排序，再过滤、去重并取前 K 条。"""
    scores, positions = index.search(vector, index.ntotal)
    hits, excluded, seen = [], [], set()

    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue

        unit = units[int(position)]

        if paper and unit["arxiv_id"] != paper:
            continue

        reason = exclusion_reason(unit, events)
        if reason:
            excluded.append({
                "chunk_id": unit["chunk_id"],
                "reason": reason,
            })
            continue

        parent = unit["parent_chunk_id"]
        if parent in seen:
            continue

        seen.add(parent)
        hits.append({
            **unit,
            "rank": len(hits) + 1,
            "score": float(score),
        })

    return hits[:top_k], excluded


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    import torch
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser(
        description="PaperLens 过滤噪声后的证据检索"
    )
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--paper")
    args = parser.parse_args()

    query = args.query.strip()
    if not query or len(query) > 2000 or not 1 <= args.top_k <= 20:
        parser.error("问题须为 1～2000 字符，top-k 须为 1～20。")

    rewrite = query_rewrite.rewrite_query(query)
    retrieval_query = rewrite.retrieval_query
    if rewrite.rewritten:
        print(f"查询改写：{retrieval_query}", flush=True)

    # 复用已有的索引读取和语料一致性检查。
    index, metadata = load_store(ROOT)
    units = metadata["units"]

    if args.paper and args.paper not in {
        unit["arxiv_id"] for unit in units
    }:
        parser.error("索引中没有这个完整论文编号。")

    folder = ROOT / "data" / "rag"
    corpus = json.loads(
        (folder / "corpus_clean.json").read_text(encoding="utf-8")
    )
    events = section_events(corpus["chunks"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"索引：{index.ntotal} 条；运行设备：{device}",
        flush=True,
    )

    # 继续使用已经下载的模型，只编码当前问题。
    model = SentenceTransformer(
        metadata["model_name"],
        device=device,
        revision=metadata.get("model_revision"),
        cache_folder=str(ROOT / "data" / "models"),
        local_files_only=True,
    )

    text = metadata["query_prefix"] + retrieval_query
    token_count = len(
        model.tokenizer(text, truncation=False)["input_ids"]
    )
    max_tokens = min(
        metadata["max_tokens"],
        int(model.max_seq_length),
    )
    if token_count > max_tokens and retrieval_query != query:
        retrieval_query = query
        text = metadata["query_prefix"] + query
        token_count = len(
            model.tokenizer(text, truncation=False)["input_ids"]
        )
        rewrite = query_rewrite.RewriteResult(
            query, query, False, error="rewritten query too long"
        )
    if token_count > max_tokens:
        raise ValueError("问题过长，请缩短后再检索。")

    vector = model.encode(
        [text],
        prompt="",
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    vector = np.ascontiguousarray(vector, dtype="float32")

    if (
        vector.shape != (1, index.d)
        or not np.isfinite(vector).all()
        or not np.allclose(np.linalg.norm(vector), 1.0, atol=1e-4)
    ):
        raise ValueError("查询向量异常。")

    hits, excluded = select_hits(
        index, units, vector, events, args.top_k, args.paper
    )

    counts = Counter(item["reason"] for item in excluded)
    print("过滤统计（当前检索范围的全部向量）：", dict(counts))

    for hit in hits:
        print(f"\n[{hit['rank']}] {hit['title']}")
        print(
            f"PDF 第 {hit['page_number']} 页"
            f" | 相似度 {hit['score']:.4f}"
        )
        print(hit["text"])

    if len(hits) < args.top_k:
        print(
            f"可用证据不足 {args.top_k} 条，"
            f"实际返回 {len(hits)} 条。"
        )

    output = folder / "evidence_results.json"
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "e5_faiss_section_filter_v1",
        "model_name": metadata["model_name"],
        "model_revision": metadata.get("model_revision"),
        "corpus_sha256": metadata["corpus_sha256"],
        "query": query,
        "retrieval_query": retrieval_query,
        "query_rewrite": {
            "applied": rewrite.rewritten,
            "cached": rewrite.cached,
            "error": rewrite.error,
            "prompt_sha256": rewrite.prompt_sha256,
        },
        "top_k": args.top_k,
        "paper_filter": args.paper,
        "filter": {
            "min_chars": 80,
            "reference_sections": True,
        },
        "hits": hits,
        "excluded": excluded,
    }

    atomic_write(
        output,
        json.dumps(record, ensure_ascii=False, indent=2),
    )
    print(f"\n已保存：{output}")


if __name__ == "__main__":
    main()