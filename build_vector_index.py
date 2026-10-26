import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from prepare_corpus import sha256

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "intfloat/multilingual-e5-small"


def make_segments(chunk, tokenizer, limit):
    # 长片段递归拆分，并保留部分重叠。
    # 避免超出模型长度后，尾部文字被静默截断。
    text = chunk["text"]
    pending, ranges = [(0, len(text))], []

    while pending:
        start, end = pending.pop()
        part = text[start:end]
        size = len(
            tokenizer(
                "passage: " + part,
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]
        )

        if size <= limit:
            ranges.append((start, end))
            continue

        if end - start < 8:
            raise ValueError(
                "极短文本仍超出 token 限制，请检查分词器。"
            )

        middle = (start + end) // 2
        space = text.rfind(
            " ", start + (end - start) // 3, middle
        )
        if space > start:
            middle = space

        overlap = min(
            80,
            (middle - start) // 4,
            (end - middle) // 4,
        )
        pending.extend([
            (start, middle + overlap),
            (middle - overlap, end),
        ])

    units = []

    for number, (start, end) in enumerate(sorted(ranges), 1):
        absolute_start = chunk["char_start"] + start
        absolute_end = chunk["char_start"] + end

        units.append({
            **chunk,
            "parent_chunk_id": chunk["chunk_id"],
            "chunk_id": f"{chunk['chunk_id']}:v{number}",
            "text": text[start:end],
            "char_start": absolute_start,
            "char_end": absolute_end,
            "sources": [
                s for s in chunk["sources"]
                if (
                    s["start"] < absolute_end
                    and s["end"] > absolute_start
                )
            ],
        })

    return units


def save_store(path, index, metadata):
    # 索引和出处一起保存，成功写完后再替换旧文件。
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp"
    )

    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream,
                index=faiss.serialize_index(index),
                metadata=np.array(
                    json.dumps(metadata, ensure_ascii=False)
                ),
            )
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(name, path)

    finally:
        Path(name).unlink(missing_ok=True)


def main(root=ROOT, model_cache=None, device="auto"):
    import torch
    from sentence_transformers import SentenceTransformer

    root = Path(root).resolve()
    folder = root / "data" / "rag"
    raw = (folder / "corpus_clean.json").read_bytes()
    corpus = json.loads(raw)
    original = (folder / "corpus.json").read_bytes()

    if corpus["source_corpus_sha256"] != sha256(original):
        raise ValueError(
            "清洗语料已过期，请先运行 python clean_corpus.py"
        )

    chunks = corpus["chunks"]
    if (
        not chunks
        or len({c["chunk_id"] for c in chunks}) != len(chunks)
    ):
        raise ValueError("语料为空或存在重复片段编号。")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("device 必须为 auto、cpu 或 cuda。")

    print("正在从本地缓存加载模型……", flush=True)

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        cache_folder=str(model_cache or root / "data" / "models"),
        local_files_only=True,
    )

    print("模型运行设备：", model.device)

    limit = min(512, int(model.max_seq_length))
    units = []

    for chunk in chunks:
        units.extend(
            make_segments(chunk, model.tokenizer, limit)
        )

    print(
        f"原片段 {len(chunks)} 个；"
        f"实际编码片段 {len(units)} 个。"
    )

    vectors = model.encode(
        ["passage: " + unit["text"] for unit in units],
        prompt="",
        batch_size=8,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    vectors = np.ascontiguousarray(
        vectors, dtype="float32"
    )

    if (
        vectors.shape != (len(units), 384)
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("向量形状或数值异常。")

    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("向量没有正确归一化。")

    index = faiss.IndexFlatIP(384)
    index.add(vectors)

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "model_revision": getattr(
            model[0].auto_model.config,
            "_commit_hash",
            None,
        ),
        "corpus_sha256": sha256(raw),
        "source_corpus_sha256": sha256(original),
        "max_tokens": limit,
        "normalized": True,
        "dimension": 384,
        "index_type": "IndexFlatIP",
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
        "parent_chunk_count": len(chunks),
        "vector_count": len(units),
        "units": units,
    }

    if (folder / "corpus_clean.json").read_bytes() != raw:
        raise ValueError(
            "建库过程中清洗语料发生变化，请重试。"
        )

    if (folder / "corpus.json").read_bytes() != original:
        raise ValueError(
            "建库过程中原始语料发生变化，"
            "请重新清洗后重试。"
        )

    output = folder / "vector_store.npz"
    save_store(output, index, metadata)

    print(
        f"\n建库成功：{index.ntotal} 条向量，"
        f"每条 {index.d} 维。"
    )
    print(f"保存位置：{output}")
    print(
        "已保存向量索引与出处。"
        "语料变化后再重新运行本脚本。"
    )


if __name__ == "__main__":
    main()
