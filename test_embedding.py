from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "intfloat/multilingual-e5-small"


def main():
    print("正在加载多语言嵌入模型……", flush=True)
    print("首次运行需要下载模型，后续会复用本地缓存。", flush=True)

    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda",
        cache_folder=str(ROOT / "data" / "models"),
    )

    # 以下是测试句子，不是从论文中提取的证据。
    query = "智能体如何记住随时间变化的信息？"
    passages = [
        "An agent stores changing facts as versioned memory records.",
        "This recipe explains how to bake a chocolate cake.",
        "The weather forecast predicts rain tomorrow.",
    ]

    texts = ["query: " + query]
    texts += ["passage: " + text for text in passages]

    vectors = model.encode(
        texts,
        batch_size=4,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    if vectors.shape != (4, 384):
        raise ValueError(f"向量形状不符合预期：{vectors.shape}")

    if not np.isfinite(vectors).all():
        raise ValueError("向量中出现了无效数值。")

    # 向量已经归一化，点积等于余弦相似度。
    scores = vectors[1:] @ vectors[0]

    print("\n模型加载与编码成功！")
    print("向量形状：", vectors.shape)
    print("中文问题：", query)

    for index in np.argsort(-scores):
        print(f"\n相似度：{scores[index]:.4f}")
        print(passages[index])

    print("\n相似度不是正确率；这只是环境与跨语言编码测试。")


if __name__ == "__main__":
    main()