"""Read-only local PDF/corpus/index inventory; no API calls or model loading."""
import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from library_transaction import check_available

ROOT = Path(__file__).resolve().parent
ID_PATTERN = r"(?:\d{4}\.\d{4,5}|[A-Za-z][A-Za-z0-9.-]*/\d{7})(?:v\d+)?"


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pdf_filename(paper_id):
    return re.sub(r"[^A-Za-z0-9._-]", "_", paper_id) + ".pdf"


def read_corpus(path, warnings):
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8-sig"))
        papers, chunks = result["papers"], result["chunks"]
        if not isinstance(papers, list) or not isinstance(chunks, list):
            raise ValueError("papers/chunks 必须为列表")
        ids = [p["arxiv_id"] for p in papers]
        if len(ids) != len(set(ids)):
            raise ValueError("存在重复论文编号")
        for paper in papers:
            if not re.fullmatch(ID_PATTERN, paper["arxiv_id"]):
                raise ValueError("论文编号无效")
            if not re.fullmatch(r"[0-9a-f]{64}", paper["pdf_sha256"]):
                raise ValueError("PDF 校验值缺失或无效")
        for chunk in chunks:
            if chunk["arxiv_id"] not in ids:
                raise ValueError("片段所属论文不存在")
        return result
    except Exception as exc:
        warnings.append(f"{path.name} 无法读取：{type(exc).__name__}: {exc}")
        return None


def scan_library(root=ROOT):
    root = Path(root).resolve()
    check_available(root)
    folder = root / "data" / "rag"
    pdf_folder = root / "data" / "pdfs"
    warnings = []
    # Record source bytes before and after checking to detect a concurrent rebuild.
    tracked = [folder / name for name in (
        "corpus.json", "corpus_clean.json", "vector_store.npz")]
    before = {p: file_hash(p) if p.is_file() else None for p in tracked}
    raw = read_corpus(folder / "corpus.json", warnings)
    clean = read_corpus(folder / "corpus_clean.json", warnings)
    raw_papers = {p["arxiv_id"]: p for p in (raw or {}).get("papers", [])}
    clean_papers = {p["arxiv_id"]: p for p in (clean or {}).get("papers", [])}
    raw_counts = Counter(c["arxiv_id"] for c in (raw or {}).get("chunks", []))
    clean_counts = Counter(c["arxiv_id"] for c in (clean or {}).get("chunks", []))
    clean_current = bool(raw and clean and clean.get("source_corpus_sha256") ==
                         before[folder / "corpus.json"])
    if clean and not clean_current:
        warnings.append("清洗语料与当前原始语料不匹配，需要重新清洗。")
    index_valid, metadata = False, {}
    index_path = folder / "vector_store.npz"
    if index_path.exists():
        try:
            # Reuse the exact integrity checks already used by question answering.
            # This does not construct SentenceTransformer or call an API.
            from search_vector import load_store
            _, metadata = load_store(root)
            if not raw or not clean or not clean_current:
                raise ValueError("语料结构或清洗来源校验失败")
            index_valid = True
        except Exception as exc:
            warnings.append(f"向量索引校验失败：{type(exc).__name__}: {exc}")
    else:
        warnings.append("尚无 vector_store.npz 向量索引。")
    units = metadata.get("units", []) if index_valid else []
    vector_counts = Counter(u["arxiv_id"] for u in units)
    unit_hashes = {}
    for unit in units:
        unit_hashes.setdefault(unit["arxiv_id"], set()).add(unit.get("pdf_sha256"))

    # Titles for downloaded but not yet extracted papers are optional metadata.
    titles = {}
    rec_path = root / "data" / "recommendations.json"
    if rec_path.is_file():
        try:
            for paper in json.loads(rec_path.read_text(encoding="utf-8-sig"))["ranked_papers"]:
                titles[paper["arxiv_id"]] = paper["title"]
        except Exception:
            warnings.append("推荐元数据不可用；不影响本地 PDF/索引状态检查。")
    expected = {pdf_filename(pid): pid for pid in raw_papers.keys() | clean_papers.keys()}
    files, unrecognized = {}, []
    if pdf_folder.is_dir():
        for path in sorted(pdf_folder.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".pdf":
                continue
            pid = expected.get(path.name)
            if not pid:
                candidate = path.stem
                if re.fullmatch(ID_PATTERN, candidate):
                    pid = candidate
                elif "_" in candidate and re.fullmatch(ID_PATTERN, candidate.replace("_", "/", 1)):
                    pid = candidate.replace("_", "/", 1)
            if not pid:
                unrecognized.append(path.name)
                continue
            if pid in files:
                raise ValueError(f"同一编号对应多个 PDF，请先检查文件名：{pid}")
            files[pid] = path
    rows = []
    for pid in sorted(files.keys() | raw_papers.keys() | clean_papers.keys()):
        path, original, cleaned = files.get(pid), raw_papers.get(pid), clean_papers.get(pid)
        digest, pdf_error = None, None
        if path:
            try:
                if not path.resolve().is_relative_to(pdf_folder.resolve()):
                    raise ValueError("PDF 路径指向 data/pdfs 之外")
                with path.open("rb") as stream:
                    if stream.read(5) != b"%PDF-":
                        raise ValueError("文件头不是 PDF")
                digest = file_hash(path)
            except Exception as exc:
                pdf_error = str(exc)
        raw_current = bool(digest and original and original["pdf_sha256"] == digest
                           and raw_counts[pid] > 0)
        cleaned_current = bool(raw_current and clean_current and cleaned and
                               cleaned["pdf_sha256"] == digest and clean_counts[pid] > 0)
        ready = bool(cleaned_current and index_valid and vector_counts[pid] > 0
                     and unit_hashes.get(pid) == {digest})
        if not path:
            status, action = "原文缺失", "恢复对应版本的原文 PDF，再检查状态"
        elif pdf_error:
            status, action = "PDF 异常", pdf_error
        elif original and original["pdf_sha256"] != digest:
            status, action = "PDF 已变化", "重新提取、清洗并建索引"
        elif not raw_current:
            status, action = "待提取正文", "尚未纳入正文语料；等待累计入库步骤"
        elif not cleaned_current:
            status, action = "待清洗", "清洗当前正文语料，再建索引"
        elif not ready:
            status, action = "待建索引或修复", "查看上方校验提示，再构建索引"
        else:
            status, action = "已入有效索引", "可在问答首页选择此论文；不代表所有问题都有答案"
        title = (original or cleaned or {}).get("title") or titles.get(pid) or "标题暂缺"
        rows.append({"arxiv_id": pid, "title": title, "pdf_exists": bool(path),
                     "pdf_sha256": digest, "raw_chunks": raw_counts[pid],
                     "clean_chunks": clean_counts[pid], "vectors": vector_counts[pid],
                     "ready": ready, "status": status, "next_action": action})
    after = {p: file_hash(p) if p.is_file() else None for p in tracked}
    if before != after:
        raise RuntimeError("检查期间语料或索引发生变化，请等其他任务完成后重新检查。")
    check_available(root)
    return {"schema_version": 1, "checked_at": datetime.now(timezone.utc).isoformat(),
            "pdf_count": len(files), "ready_count": sum(r["ready"] for r in rows),
            "index_valid": index_valid, "warnings": warnings, "papers": rows,
            "unrecognized_pdf_files": unrecognized,
            "scope": "Local inventory and consistency checks only; no answer-quality evaluation."}


def main():
    parser = argparse.ArgumentParser(description="PaperLens 本地论文库状态检查")
    parser.add_argument("--json", action="store_true", help="在终端输出完整 JSON")
    args = parser.parse_args()
    report = scan_library()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print(f"本地原文 PDF：{report['pdf_count']} 篇")
    print(f"已入有效索引：{report['ready_count']} 篇")
    print("此次只检查状态，没有抓取、翻译、调用大模型或改动现有语料。")
    for warning in report["warnings"]:
        print(f"提示：{warning}")
    for row in report["papers"]:
        print(f"\n{row['arxiv_id']} | {row['status']} | "
              f"原始片段 {row['raw_chunks']} / 清洗片段 {row['clean_chunks']} / 向量 {row['vectors']}")
        print(f"  {row['next_action']}")
    if report["unrecognized_pdf_files"]:
        print("\n未按原文编号识别的 PDF（例如翻译副本）：")
        print("\n".join(report["unrecognized_pdf_files"]))


if __name__ == "__main__":
    main()
