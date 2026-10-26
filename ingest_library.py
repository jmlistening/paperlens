"""Cumulative local-PDF ingestion; full snapshot rebuild, no API or email calls."""
import argparse
from importlib.metadata import version, PackageNotFoundError
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from library_status import ID_PATTERN, file_hash, pdf_filename, read_corpus, scan_library
from library_transaction import check_available, publish, recover, state
from prepare_corpus import (CHUNK_CHARS, OVERLAP_CHARS, atomic_write,
                            extract_paper, preview_text, sha256)
from run_pipeline import pipeline_lock

ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2))


def inventory(root):
    """Read originals independently of index availability; never drop indexed PDFs silently."""
    folder = root / "data" / "pdfs"
    papers, skipped = {}, []
    for path in sorted(folder.glob("*")):
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue
        pid = path.stem
        if not re.fullmatch(ID_PATTERN, pid):
            pid = pid.replace("_", "/", 1)
        if not re.fullmatch(ID_PATTERN, pid):
            skipped.append(path.name)
            continue
        if path.name != pdf_filename(pid):
            raise ValueError(f"原文文件名应为 {pdf_filename(pid)}：{path.name}")
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError(f"PDF 路径指向原文目录之外：{path.name}")
        with path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise ValueError(f"文件头不是 PDF：{path.name}")
        if pid in papers:
            raise ValueError(f"重复论文编号：{pid}")
        papers[pid] = file_hash(path)
    if not papers:
        raise ValueError("data/pdfs 中没有可识别的原文 PDF。")
    for name in ("corpus.json", "corpus_clean.json"):
        path = root / "data" / "rag" / name
        if path.exists():
            warnings = []
            corpus = read_corpus(path, warnings)
            if corpus is None:
                raise ValueError("；".join(warnings))
            missing = {p["arxiv_id"] for p in corpus["papers"]} - papers.keys()
            if missing:
                raise ValueError("已有语料的原文缺失，停止以防缩小论文库：" + ", ".join(sorted(missing)))
    return papers, skipped


def titles(root):
    # Read only paper metadata files, not credentials, personal notes or answer logs.
    data, result = root / "data", {}
    paths = sorted((data / "runs").glob("*/previous/papers.json"))
    paths += sorted((data / "runs").glob("*/previous/recommendations.json"))
    paths += [data / "papers.json", data / "recommendations.json",
              data / "rag" / "corpus.json", data / "library_titles.json"]
    for path in paths:
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            rows = value if isinstance(value, list) else value.get("ranked_papers", value.get("papers", []))
            for row in rows:
                if isinstance(row, dict) and isinstance(row.get("title"), str) and row["title"].strip():
                    result[row["arxiv_id"]] = row["title"].strip()
        except (ValueError, KeyError, TypeError):
            print(f"提示：跳过无法识别的标题元数据 {path.relative_to(root)}")
    return result


def fingerprint():
    names = ("prepare_corpus.py", "clean_corpus.py", "build_vector_index.py", "ingest_library.py")
    return {name: file_hash(ROOT / name) for name in names}


def runtime_versions():
    result = {}
    for name in ("PyMuPDF", "sentence-transformers", "transformers", "torch", "faiss-cpu", "numpy"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "unknown"
    return result


def run(root=ROOT, dry_run=False, force=False, recover_only=False, device="auto"):
    root = Path(root).resolve()
    if dry_run:
        check_available(root)
        sources, skipped = inventory(root)
        report = scan_library(root)
        print(f"预览：本地原文 {len(sources)} 篇，已入有效索引 {report['ready_count']} 篇。")
        for row in report["papers"]:
            print(f"  {row['arxiv_id']} | {row['status']}")
        print(f"未识别文件：{skipped or '无'}")
        print("正式运行将累计重建全部本地原文的索引；此预览未写文件、未加载模型。")
        return report
    with pipeline_lock(root):
        if recover(root):
            print("已恢复上次中断前的论文库。")
        if recover_only:
            print("恢复检查完成；未重建索引。")
            return {"status": "recovery_checked"}
        sources, skipped = inventory(root)
        code = fingerprint()
        versions = runtime_versions()
        baseline = state(root)
        manifest_path = root / "data" / "rag" / "library_manifest.json"
        if not force and manifest_path.is_file():
            try:
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            except ValueError:
                previous = {}
            if (previous.get("source_hashes") == sources and previous.get("code_hashes") == code
                    and previous.get("runtime_versions") == versions):
                report = scan_library(root)
                outputs_match = all(baseline.get(k) == v for k, v in previous.get("output_hashes", {}).items())
                if outputs_match and previous.get("output_hashes") and report["ready_count"] == len(sources):
                    print(f"无需重建：{len(sources)} 篇原文及处理代码未变化，索引校验通过。")
                    return {"status": "unchanged", "paper_count": len(sources)}
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = root / "data" / "library_runs" / run_id
        work = run_dir / "work"
        (work / "data" / "pdfs").mkdir(parents=True)
        record = {"schema_version": 1, "started_at": now(), "status": "running",
                  "source_hashes": sources, "code_hashes": code, "runtime_versions": versions, "skipped": skipped,
                  "method": "cumulative full rebuild", "api_calls": 0}
        write_json(run_dir / "run.json", record)
        print(f"运行记录：{run_dir}", flush=True)
        try:
            import pymupdf
            import clean_corpus
            import build_vector_index
            known_titles = titles(root)
            documents, chunks = [], []
            for number, (pid, expected) in enumerate(sources.items(), 1):
                target = work / "data" / "pdfs" / pdf_filename(pid)
                shutil.copy2(root / "data" / "pdfs" / target.name, target)
                if file_hash(target) != expected:
                    raise ValueError(f"复制时原文发生变化：{pid}")
                paper = {"arxiv_id": pid, "title": known_titles.get(pid, f"{pid}（标题待补充）")}
                print(f"[{number}/{len(sources)}] 提取 {paper['title']}", flush=True)
                document, pieces = extract_paper(work, paper, pymupdf)
                documents.append(document)
                chunks.extend(pieces)
            # Compatibility field refers to the staged selection manifest, never the current recommendation file.
            selection = {"selected_ids": list(sources), "source_hashes": sources}
            selection_bytes = json.dumps(selection, sort_keys=True).encode("utf-8")
            atomic_write(work / "data" / "library_selection.json", selection_bytes.decode("utf-8"))
            corpus = {"schema_version": 1, "created_at": now(),
                      "recommendations_sha256": sha256(selection_bytes),
                      "selection_source": "data/library_selection.json in this run's work directory",
                      "selected_ids": list(sources),
                      "extractor": {"name": "PyMuPDF", "version": pymupdf.VersionBind,
                                    "method": "page-local text blocks; sort=True; no OCR",
                                    "max_chunk_chars": CHUNK_CHARS, "overlap_chars": OVERLAP_CHARS},
                      "scope": "All recognized local original PDFs; cumulative full rebuild, no OCR.",
                      "paper_count": len(documents), "chunk_count": len(chunks),
                      "papers": documents, "chunks": chunks}
            staged = work / "data" / "rag"
            write_json(staged / "corpus.json", corpus)
            atomic_write(staged / "extraction_preview.txt", preview_text(corpus))
            print("正在清洗全部原文……", flush=True)
            clean_corpus.main(work)
            build_vector_index.main(work, model_cache=root / "data" / "models", device=device)
            report = scan_library(work)
            if not report["index_valid"] or report["ready_count"] != len(sources):
                raise ValueError("候选论文库校验未全部通过；正式文件保持原样。")
            if inventory(root)[0] != sources:
                raise ValueError("构建期间原文集合发生变化，停止发布。")
            snapshot = {"schema_version": 1, "created_at": now(), "source_hashes": sources,
                        "code_hashes": code, "runtime_versions": versions, "method": "cumulative full rebuild",
                        "output_hashes": {k: v for k, v in state(work).items() if k != "library_manifest.json"},
                        "run": run_dir.relative_to(root).as_posix()}
            write_json(staged / "library_manifest.json", snapshot)
            publish(root, work, run_dir, baseline)
            record.update(status="completed", completed_at=now(), paper_count=len(sources),
                          page_count=sum(d["page_count"] for d in documents),
                          raw_chunks=len(chunks), clean_chunks=sum(p["clean_chunks"] for p in report["papers"]),
                          vectors=sum(p["vectors"] for p in report["papers"]), library=report)
            print(f"累计入库成功：{len(sources)} 篇，{record['vectors']} 条向量。", flush=True)
        except BaseException as exc:
            record.update(status="failed", failed_at=now(), error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            write_json(run_dir / "run.json", record)
        return record


def main():
    parser = argparse.ArgumentParser(description="PaperLens 累计入库：使用本地 PDF 与本地嵌入模型")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只显示现状，不写入数据或加载模型")
    mode.add_argument("--recover", action="store_true", help="仅恢复上次中断的发布")
    mode.add_argument("--force", action="store_true", help="原文未变化时也重新构建")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        run(dry_run=args.dry_run, force=args.force, recover_only=args.recover, device=args.device)
    except Exception as exc:
        print(f"累计入库未完成：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
