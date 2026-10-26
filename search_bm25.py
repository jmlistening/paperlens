import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from rank_bm25 import BM25Okapi
from prepare_corpus import atomic_write, sha256

ROOT = Path(__file__).resolve().parent

STOP_WORDS = set(
    "a an the is are was were be been of to in on for "
    "and or with by as at from what how does do".split()
)


def tokenize(text):
    # 正文和查询使用相同的分词规则。
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [word for word in words if word not in STOP_WORDS]


def load_corpus(root):
    folder = root / "data" / "rag"
    raw = (folder / "corpus_clean.json").read_bytes()
    corpus = json.loads(raw)

    # 避免原始语料更新后，误用旧的清洗结果。
    original = (folder / "corpus.json").read_bytes()
    if corpus.get("source_corpus_sha256") != sha256(original):
        raise ValueError(
            "原始语料已变化，请先重新运行 python clean_corpus.py"
        )

    chunks = corpus["chunks"]
    if not chunks:
        raise ValueError("清洗语料没有文本片段。")

    ids = [chunk["chunk_id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("发现重复片段编号，请检查语料。")

    # 只排除没有英文或数字词的片段，短片段暂时保留。
    pairs = [
        (chunk, tokenize(chunk["text"]))
        for chunk in chunks
    ]
    pairs = [(chunk, words) for chunk, words in pairs if words]

    if not pairs:
        raise ValueError("没有可建立索引的英文或数字内容。")

    return raw, pairs


def search(pairs, query, top_k):
    if re.search(r"[\u3400-\u9fff]", query):
        raise ValueError(
            "这一版先用英文关键词检索；中文语义检索在下一阶段加入。"
        )

    words = list(dict.fromkeys(tokenize(query)))
    if not words:
        raise ValueError("请输入有效的英文关键词。")

    chunks = [pair[0] for pair in pairs]
    documents = [pair[1] for pair in pairs]

    # 每次运行在内存中建立 BM25 索引，目前没有建立向量库。
    index = BM25Okapi(documents, k1=1.5, b=0.75)
    scores = index.get_scores(words)

    # 不为了凑够 Top-K 而加入完全没有关键词命中的片段。
    matched = [
        i for i, doc in enumerate(documents)
        if set(words).intersection(doc)
    ]
    ranked = sorted(
        matched,
        key=lambda i: (-float(scores[i]), chunks[i]["chunk_id"]),
    )

    return [
        {
            "rank": rank,
            "score": float(scores[i]),
            **chunks[i],
        }
        for rank, i in enumerate(ranked[:top_k], 1)
    ]


def main(root=ROOT):
    parser = argparse.ArgumentParser(
        description="PaperLens 英文 BM25 证据检索"
    )
    parser.add_argument("query", help="用英文双引号包住查询内容")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not 1 <= args.top_k <= 20:
        parser.error("--top-k 必须在 1 到 20 之间")

    root = Path(root)
    raw, pairs = load_corpus(root)
    hits = search(pairs, args.query, args.top_k)

    print(f"已索引 {len(pairs)} 个片段；检索词：{args.query}")

    for hit in hits:
        print(f"\n[{hit['rank']}] {hit['title']}")
        print(
            f"arXiv：{hit['arxiv_id']} | "
            f"PDF 第 {hit['page_number']} 页"
        )
        print(
            f"BM25 分数：{hit['score']:.4f}"
            "（不是置信度或准确率）"
        )
        print(hit["text"])

    if not hits:
        print(
            "没有关键词匹配结果；"
            "这不等于论文中一定没有相关答案。"
        )

    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "BM25Okapi",
        "package_version": "rank_bm25==0.2.2",
        "k1": 1.5,
        "b": 0.75,
        "tokenizer": "ascii_lower_stopwords_v1",
        "corpus_sha256": sha256(raw),
        "indexed_count": len(pairs),
        "query": args.query,
        "top_k": args.top_k,
        "hits": hits,
    }

    output = root / "data" / "rag" / "bm25_results.json"
    atomic_write(
        output,
        json.dumps(record, ensure_ascii=False, indent=2),
    )
    print(f"\n本次结果已保存：{output}")


if __name__ == "__main__":
    main()