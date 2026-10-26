import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from prepare_corpus import atomic_write, sha256, split_text

ROOT = Path(__file__).resolve().parent


def clean_text(raw, vocabulary):
    # 仅在论文中出现过完整拼写时，尝试修复行末断词。
    def repair(match):
        left, right = match.group(1), match.group(2)
        joined = left + right
        return joined if joined.lower() in vocabulary else left + "-" + right

    raw = raw.replace("\u00ad", "")
    raw = re.sub(r"([A-Za-z]+)-[ \t]*\r?\n[ \t]*([a-z]+)", repair, raw)
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def column_cut(blocks):
    # 用长段落的位置判断是否有两个正文栏，避免把单栏强行拆开。
    long_blocks = [b for b in blocks if len(b["text"]) >= 180]
    starts = sorted(set(b["bbox"][0] for b in long_blocks))
    gaps = [(b - a, (a + b) / 2) for a, b in zip(starts, starts[1:])]
    if not gaps:
        return None

    gap, boundary = max(gaps)
    left = [b for b in long_blocks if b["bbox"][0] < boundary]
    right = [b for b in long_blocks if b["bbox"][0] >= boundary]
    if gap < 90 or len(left) < 2 or len(right) < 2:
        return None

    left_end = median(b["bbox"][2] for b in left)
    right_start = median(b["bbox"][0] for b in right)
    if right_start - left_end < 6:
        return None

    return (left_end + right_start) / 2


def reading_groups(blocks):
    cut = column_cut(blocks)
    top_down = lambda b: (b["bbox"][1], b["bbox"][0])
    if cut is None:
        return "single_or_uncertain", [sorted(blocks, key=top_down)]

    # 跨栏标题/图注作为分区边界；每个分区先左栏、再右栏。
    wide = sorted(
        [
            b for b in blocks
            if b["bbox"][0] < cut - 5 and b["bbox"][2] > cut + 5
        ],
        key=top_down,
    )
    remaining = [b for b in blocks if b not in wide]
    groups = []

    for separator in wide + [None]:
        stop = separator["bbox"][1] if separator else float("inf")
        band = [b for b in remaining if b["bbox"][1] < stop]
        remaining = [b for b in remaining if b["bbox"][1] >= stop]

        for is_left in (True, False):
            column = [
                b for b in band
                if (
                    (b["bbox"][0] + b["bbox"][2]) / 2 < cut
                ) == is_left
            ]
            if column:
                groups.append(sorted(column, key=top_down))

        if separator:
            groups.append([separator])

    return "two_columns_heuristic", groups


def main(root=ROOT):
    folder = Path(root) / "data" / "rag"
    raw = (folder / "corpus.json").read_bytes()
    source = json.loads(raw)

    result = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_corpus_sha256": sha256(raw),
        "recommendations_sha256": source["recommendations_sha256"],
        "selected_ids": source["selected_ids"],
        "method": "coordinate heuristic; same-page same-region chunks; no OCR",
        "papers": [],
        "chunks": [],
    }
    preview = [
        "PaperLens 清洗核对稿：页码为 PDF 物理页码。",
        "分栏为启发式判断；图表、公式、跨页连接仍需核对。",
    ]

    for paper in source["papers"]:
        vocabulary = set(
            re.findall(
                r"(?<![A-Za-z-])[A-Za-z]{3,}(?![A-Za-z-])",
                "\n".join(
                    b["raw_text"]
                    for p in paper["pages"]
                    for b in p["blocks"]
                ).lower(),
            )
        )

        cleaned = {
            k: paper[k]
            for k in ("arxiv_id", "title", "pdf_path", "pdf_sha256")
        }
        cleaned["pages"] = []
        start_count = len(result["chunks"])
        preview.extend(["\n" + "=" * 60, paper["title"]])

        for page in paper["pages"]:
            number = page["page_number"]
            blocks, excluded = [], []
            bottom = max(
                (b["bbox"][3] for b in page["blocks"]),
                default=0,
            )

            for block in page["blocks"]:
                item = dict(block)
                item["text"] = clean_text(block["raw_text"], vocabulary)
                x0, y0, x1, y1 = item["bbox"]
                reason = None

                if not item["text"]:
                    reason = "empty_text"
                elif item["text"] == str(number) and y0 >= bottom - 25:
                    reason = "probable_page_number"
                elif (
                    item["text"].lower().startswith("arxiv:")
                    and y1 - y0 > 3 * (x1 - x0)
                ):
                    reason = "vertical_arxiv_label"

                if reason:
                    excluded.append({**item, "reason": reason})
                else:
                    blocks.append(item)

            layout, groups = reading_groups(blocks)
            cleaned["pages"].append({
                "page_number": number,
                "layout": layout,
                "blocks": [b for g in groups for b in g],
                "excluded_blocks": excluded,
            })
            preview.append(
                f"\n--- PDF 第 {number} 页 | {layout} ---"
            )

            for group_number, group in enumerate(groups, 1):
                text, spans = "", []

                for block in group:
                    text += "\n\n" if text else ""
                    begin = len(text)
                    text += block["text"]
                    spans.append({
                        "block_number": block["block_number"],
                        "bbox": block["bbox"],
                        "start": begin,
                        "end": len(text),
                    })

                preview.append(text)

                for piece, (a, b, content) in enumerate(
                    split_text(text), 1
                ):
                    sources = [
                        s for s in spans
                        if s["start"] < b and s["end"] > a
                    ]
                    result["chunks"].append({
                        "chunk_id": (
                            f"{paper['arxiv_id']}:{sha256(raw)[:12]}"
                            f":clean2:p{number}:g{group_number}:c{piece}"
                        ),
                        "arxiv_id": paper["arxiv_id"],
                        "title": paper["title"],
                        "pdf_path": paper["pdf_path"],
                        "pdf_sha256": paper["pdf_sha256"],
                        "source_url": (
                            f"https://arxiv.org/abs/{paper['arxiv_id']}"
                        ),
                        "page_number": number,
                        "group_number": group_number,
                        "char_start": a,
                        "char_end": b,
                        "text": content,
                        "sources": sources,
                        "short_fragment": len(content.strip()) < 80,
                    })

        cleaned["chunk_count"] = len(result["chunks"]) - start_count
        result["papers"].append(cleaned)
        print(
            f"{paper['arxiv_id']}："
            f"{paper['chunk_count']} → {cleaned['chunk_count']} 个片段"
        )

    result["paper_count"] = len(result["papers"])
    result["chunk_count"] = len(result["chunks"])

    if (folder / "corpus.json").read_bytes() != raw:
        raise ValueError(
            "原始语料在运行期间发生变化，请停止其他任务后重试。"
        )

    atomic_write(
        folder / "corpus_clean.json",
        json.dumps(result, ensure_ascii=False, indent=2),
    )
    atomic_write(
        folder / "cleaning_preview.txt",
        "\n\n".join(preview),
    )

    short = sum(c["short_fragment"] for c in result["chunks"])
    print(
        f"完成：{result['chunk_count']} 个片段，"
        f"其中 {short} 个短片段已标记，未直接删除。"
    )
    print("原始 corpus.json 保留。请检查 data/rag/cleaning_preview.txt。")


if __name__ == "__main__":
    main()