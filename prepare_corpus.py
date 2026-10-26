"""PaperLens: turn selected original PDFs into page-addressable RAG evidence.

Run: python prepare_corpus.py
Dependency: python -m pip install PyMuPDF
No LLM, network, OCR or email calls. Input PDFs are read-only.
This is a text extraction baseline, not layout/table/formula understanding.
"""

import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHUNK_CHARS = 1400
OVERLAP_CHARS = 180
SCHEMA_VERSION = 1


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def split_text(text, limit=CHUNK_CHARS, overlap=OVERLAP_CHARS):
    """Yield exact [start,end) slices of normalized block text, not tokens."""
    if not 0 <= overlap < limit:
        raise ValueError("Require 0 <= overlap < limit")
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            # Prefer a word boundary in the latter half, without losing text.
            boundary = text.rfind(" ", start + limit // 2, end)
            if boundary > start + overlap:
                end = boundary
        yield start, end, text[start:end]
        if end == len(text):
            break
        start = max(start + 1, end - overlap)


def load_selection(root):
    path = root / "data" / "recommendations.json"
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    selected = data["selected_ids"]
    ranked = data["ranked_papers"]
    if not isinstance(selected, list) or not all(isinstance(x, str) for x in selected):
        raise ValueError("selected_ids 必须是论文编号列表。")
    if len(selected) != len(set(selected)):
        raise ValueError("推荐清单中存在重复编号。")
    if not selected:
        raise ValueError("本轮没有入选论文；未更新正文语料。")
    paper_map = {}
    for paper in ranked:
        paper_id = paper["arxiv_id"]
        if paper_id in paper_map:
            raise ValueError("ranked_papers 中存在重复编号。")
        paper_map[paper_id] = paper
    papers = []
    for paper_id in selected:
        if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[A-Za-z][A-Za-z0-9.-]*/\d{7})(?:v\d+)?", paper_id):
            raise ValueError(f"无效的 arXiv 编号：{paper_id}")
        paper = paper_map[paper_id]
        if not isinstance(paper.get("title"), str) or not paper["title"].strip():
            raise ValueError(f"论文标题缺失：{paper_id}")
        papers.append(paper)
    return path, raw, papers


def extract_paper(root, paper, pymupdf, pdf_path=None):
    paper_id = paper["arxiv_id"]
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", paper_id) + ".pdf"
    if pdf_path is None:
        pdf_path = root / "data" / "pdfs" / filename
        recorded_path = pdf_path.relative_to(root).as_posix()
    else:
        # Backward-compatible extension for alternate corpora (e.g. the
        # held-out benchmark): parse this exact file, but keep the recorded
        # path relative to ``root`` whenever it lives underneath it.
        pdf_path = Path(pdf_path)
        try:
            recorded_path = pdf_path.resolve().relative_to(
                Path(root).resolve()
            ).as_posix()
        except ValueError:
            recorded_path = pdf_path.resolve().as_posix()
    pdf_bytes = pdf_path.read_bytes()
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError(f"文件不是 PDF：{filename}")
    digest = sha256(pdf_bytes)
    pages, chunks, warnings = [], [], []
    # Parse the same bytes whose hash is recorded, even if another process replaces the file.
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.needs_pass:
            raise ValueError(f"PDF 需要密码：{filename}")
        for page_index, page in enumerate(doc):
            page_number = page_index + 1  # physical PDF page, not printed page label
            blocks = []
            for block in page.get_text("blocks", sort=True):
                if block[6] != 0:  # exclude image metadata
                    continue
                raw_text = block[4]
                text = re.sub(r"\s+", " ", raw_text.replace("\x00", " ")).strip()
                if not text:
                    continue
                block_number = len(blocks) + 1
                bbox = [round(float(value), 2) for value in block[:4]]
                blocks.append({"block_number": block_number, "bbox": bbox,
                               "raw_text": raw_text, "text": text})
                for piece, (start, end, content) in enumerate(split_text(text), 1):
                    chunks.append({
                        "chunk_id": f"{paper_id}:{digest[:16]}:p{page_number}:b{block_number}:c{piece}",
                        "arxiv_id": paper_id, "title": paper["title"],
                        "page_number": page_number, "block_number": block_number,
                        "bbox": bbox, "char_start": start, "char_end": end,
                        "text": content, "pdf_path": recorded_path,
                        "source_url": f"https://arxiv.org/abs/{paper_id}",
                        "pdf_sha256": digest,
                    })
            char_count = sum(len(block["text"]) for block in blocks)
            page_warnings = []
            if char_count < 80:
                page_warnings.append("可提取文字少于 80 字符：请检查是否为图片页、扫描页或空白页。")
            text = "\n\n".join(block["text"] for block in blocks)
            if text and text.count("\ufffd") / len(text) > 0.01:
                page_warnings.append("替代字符较多，文字编码可能异常，请对照原 PDF。")
            for warning in page_warnings:
                warnings.append(f"PDF 第 {page_number} 页：{warning}")
            pages.append({"page_number": page_number, "char_count": char_count,
                          "blocks": blocks, "warnings": page_warnings})
    if not chunks:
        raise ValueError(f"未提取到文字：{filename}。当前版本不做 OCR。")
    return {
        "arxiv_id": paper_id, "title": paper["title"],
        "pdf_path": recorded_path, "pdf_sha256": digest,
        "page_count": len(pages), "chunk_count": len(chunks),
        "pages": pages, "warnings": warnings,
    }, chunks


def preview_text(corpus):
    lines = ["PaperLens 正文提取核对稿", f"生成时间：{corpus['created_at']}",
             "页码 = PDF 文件从 1 开始的物理页码；不一定等于论文印刷页码。",
             "请对照原 PDF 检查分栏、公式、表格；当前不识别图片文字。", ""]
    for paper in corpus["papers"]:
        lines += ["=" * 60, paper["title"], paper["arxiv_id"], *paper["warnings"]]
        for page in paper["pages"]:
            lines.append(f"\n--- PDF 第 {page['page_number']} 页 ---")
            for block in page["blocks"]:
                lines += [f"[文本块 {block['block_number']}]", block["text"], ""]
    return "\n".join(lines)


def main(root=ROOT):
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("缺少依赖，请运行：python -m pip install PyMuPDF") from exc
    root = Path(root).resolve()
    selection_path, raw_selection, papers = load_selection(root)
    documents, chunks = [], []
    for index, paper in enumerate(papers, 1):
        print(f"\n[{index}/{len(papers)}] {paper['title']}", flush=True)
        document, paper_chunks = extract_paper(root, paper, pymupdf)
        documents.append(document)
        chunks.extend(paper_chunks)
        print(f"页数：{document['page_count']}；文本片段：{len(paper_chunks)}")
        for warning in document["warnings"]:
            print(f"  注意：{warning}")
    if selection_path.read_bytes() != raw_selection:
        raise ValueError("处理期间推荐清单发生变化，请停止其他任务后重试。")
    corpus = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recommendations_sha256": sha256(raw_selection),
        "selected_ids": [paper["arxiv_id"] for paper in papers],
        "extractor": {"name": "PyMuPDF", "version": pymupdf.VersionBind,
                      "method": "page-local text blocks; sort=True; no OCR",
                      "max_chunk_chars": CHUNK_CHARS, "overlap_chars": OVERLAP_CHARS},
        "scope": "Current selected PDFs only. Extracted source text, not verified answers.",
        "paper_count": len(documents), "chunk_count": len(chunks),
        "papers": documents, "chunks": chunks,
    }
    output = root / "data" / "rag"
    # JSON is the authoritative snapshot; each file is replaced atomically.
    atomic_write(output / "corpus.json", json.dumps(corpus, ensure_ascii=False, indent=2))
    atomic_write(output / "extraction_preview.txt", preview_text(corpus))
    print(f"\n正文语料已生成：{len(documents)} 篇，{len(chunks)} 个文本片段。")
    print(f"语料文件：{output / 'corpus.json'}")
    print(f"核对文本：{output / 'extraction_preview.txt'}")
    print("下一步：核对提取内容，再构建检索索引。当前尚未建立向量库。")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n正文准备未完成：{type(exc).__name__}: {exc}", file=sys.stderr)
        print("请解决错误后重试；不要把此前留下的语料当成本轮成功结果。", file=sys.stderr)
        sys.exit(1)
