import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from prepare_corpus import atomic_write, sha256
from library_transaction import check_available

ROOT = Path(__file__).resolve().parent


def load_store(root=ROOT):
    check_available(root)
    folder = Path(root) / "data" / "rag"

    with np.load(
        folder / "vector_store.npz",
        allow_pickle=False,
    ) as saved:
        metadata = json.loads(saved["metadata"].item())
        index = faiss.deserialize_index(saved["index"])

    raw = (folder / "corpus_clean.json").read_bytes()
    original = (folder / "corpus.json").read_bytes()

    if metadata["source_corpus_sha256"] != sha256(original):
        raise ValueError("原始语料已更新，请重新清洗并建库。")

    if metadata["corpus_sha256"] != sha256(raw):
        raise ValueError(
            "清洗语料已更新，请重新运行 build_vector_index.py"
        )

    units = metadata["units"]

    if (
        index.ntotal != len(units)
        or index.ntotal == 0
        or index.d != metadata["dimension"]
        or len(units) != metadata["vector_count"]
    ):
        raise ValueError(
            "索引数量、维度与出处不一致，请重新建库。"
        )

    if (
        index.metric_type != faiss.METRIC_INNER_PRODUCT
        or not metadata["normalized"]
    ):
        raise ValueError("当前脚本要求归一化的内积索引。")

    if len({u["chunk_id"] for u in units}) != len(units):
        raise ValueError("发现重复向量编号。")

    # 检查索引中的文本、论文编号和页码是否对应原片段。
    parents = {
        c["chunk_id"]: c
        for c in json.loads(raw)["chunks"]
    }

    for unit in units:
        parent = parents[unit["parent_chunk_id"]]
        a = unit["char_start"] - parent["char_start"]
        b = unit["char_end"] - parent["char_start"]

        if (
            not 0 <= a < b <= len(parent["text"])
            or parent["text"][a:b] != unit["text"]
            or unit["arxiv_id"] != parent["arxiv_id"]
            or unit["page_number"] != parent["page_number"]
        ):
            raise ValueError(
                "向量对应的文本或页码与语料不一致。"
            )

    check_available(root)
    return index, metadata


def rank_hits(index, units, query_vector, top_k, paper=None):
    # 当前索引很小，先取全部排序，再过滤和去重。
    scores, positions = index.search(
        query_vector, index.ntotal
    )
    hits, seen = [], set()

    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue

        unit = units[int(position)]

        if paper and unit["arxiv_id"] != paper:
            continue

        # 同一个原片段拆出来的子片段，只保留分数最高的一条。
        parent_id = unit["parent_chunk_id"]
        if parent_id in seen:
            continue

        seen.add(parent_id)
        hits.append({
            **unit,
            "rank": len(hits) + 1,
            "score": float(score),
        })

        if len(hits) >= top_k:
            break

    return hits


def main():
    import torch
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser(
        description="PaperLens 中英文向量检索"
    )
    parser.add_argument(
        "query", help="用英文双引号包住问题"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--paper",
        help="可选：限定完整 arXiv 编号，例如 2610.22017v1",
    )
    args = parser.parse_args()

    query = args.query.strip()

    if not query or len(query) > 2000:
        parser.error("问题不能为空，且不能超过 2000 字符。")

    if not 1 <= args.top_k <= 20:
        parser.error("--top-k 必须在 1 到 20 之间。")

    index, metadata = load_store()
    units = metadata["units"]

    if (
        args.paper
        and args.paper not in {u["arxiv_id"] for u in units}
    ):
        parser.error(
            "索引中没有这个论文编号，请检查编号及版本后缀。"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"已读取 {index.ntotal} 条向量；"
        f"模型运行设备：{device}",
        flush=True,
    )

    model = SentenceTransformer(
        metadata["model_name"],
        device=device,
        revision=metadata.get("model_revision"),
        cache_folder=str(ROOT / "data" / "models"),
        local_files_only=True,
    )

    # E5 查询必须使用 query: 前缀。
    text = metadata["query_prefix"] + query
    length = len(
        model.tokenizer(
            text, truncation=False
        )["input_ids"]
    )

    if length > min(
        metadata["max_tokens"],
        int(model.max_seq_length),
    ):
        raise ValueError(
            "问题超出模型输入长度，请缩短问题后再检索。"
        )

    vector = model.encode(
        [text],
        prompt="",
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    vector = np.ascontiguousarray(
        vector, dtype="float32"
    )

    if (
        vector.shape != (1, index.d)
        or not np.isfinite(vector).all()
        or not np.allclose(
            np.linalg.norm(vector), 1.0, atol=1e-4
        )
    ):
        raise ValueError(
            "查询向量异常，或与索引维度不一致。"
        )

    hits = rank_hits(
        index, units, vector, args.top_k, args.paper
    )

    print(f"\n问题：{query}")
    print("以下是按相似度排序的候选证据，不是最终答案。")

    for hit in hits:
        print(f"\n[{hit['rank']}] {hit['title']}")
        print(
            f"arXiv：{hit['arxiv_id']} | "
            f"PDF 第 {hit['page_number']} 页"
        )
        print(
            f"余弦相似度：{hit['score']:.4f}"
            "（不是正确率）"
        )
        print(hit["text"])

    output = ROOT / "data" / "rag" / "vector_results.json"
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "e5_faiss_cosine",
        "model_name": metadata["model_name"],
        "model_revision": metadata.get("model_revision"),
        "corpus_sha256": metadata["corpus_sha256"],
        "query": query,
        "top_k": args.top_k,
        "paper_filter": args.paper,
        "hits": hits,
    }

    atomic_write(
        output,
        json.dumps(record, ensure_ascii=False, indent=2),
    )
    print(f"\n完整结果已保存：{output}")


if __name__ == "__main__":
    main()
