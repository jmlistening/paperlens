"""Prepare PaperLens PDFs locally; retain page evidence. No network or LLM calls."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
import uuid
import zipfile

ROOT = Path(__file__).resolve().parent
VERSION = "dataset_prepare_v1"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def json_bytes(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")


def normalized(text):
    # Preserve page boundaries, line breaks, hyphens, formulas and case.
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def base_record(row):
    return {k: row.get(k) for k in ("family_id", "arxiv_id", "title", "abstract",
            "authors", "published", "updated", "doi", "categories", "query_matches",
            "url", "pdf_url", "license_status", "exclude_from_test")}


def extract(row, base, out):
    import fitz
    pid = row["arxiv_id"]
    if not re.fullmatch(r"\d{4}\.\d{4,5}v[1-9]\d*", pid):
        raise ValueError("Invalid versioned paper ID")
    raw = (base / "pdfs" / (pid + ".pdf")).read_bytes()
    pdf_hash = digest(raw)
    if pdf_hash != row.get("pdf", {}).get("sha256"):
        raise ValueError("PDF differs from collected SHA256; no replacement performed")
    fingerprint = {"version": VERSION, "pymupdf": fitz.VersionBind, "pdf_sha256": pdf_hash,
                   "metadata_sha256": digest(json_bytes(base_record(row)))}
    pages_path = out / "pages" / (pid + ".jsonl")
    meta_path = out / "records" / (pid + ".json")
    if pages_path.exists() and meta_path.exists():
        old = load(meta_path)
        if old.get("fingerprint") == fingerprint and old.get("pages_sha256") == digest(pages_path.read_bytes()):
            return old, True
    lines, lengths, previews, page_texts = [], [], {}, []
    with fitz.open(stream=raw, filetype="pdf") as doc:
        if doc.needs_pass or not doc.page_count:
            raise ValueError("Encrypted or empty PDF")
        sample_pages = {0, doc.page_count // 2, doc.page_count - 1}
        for i, page in enumerate(doc):
            original = page.get_text("text", sort=False)
            text = normalized(original)
            page_texts.append(text)
            lengths.append(len(text.strip()))
            line = {"paper_id": pid, "pdf_sha256": pdf_hash, "pdf_page": i + 1,
                    "printed_page_label": page.get_label(), "extracted_text": original,
                    "normalized_text": text, "text_sha256": digest(text.encode("utf-8"))}
            lines.append(json.dumps(line, ensure_ascii=False) + "\n")
            if i in sample_pages:
                previews[str(i + 1)] = text[:600]
    full = "\n".join(page_texts)
    total = sum(lengths)
    low = [i + 1 for i, n in enumerate(lengths) if n < 100]
    flags = []
    if total < 2000 or len(low) / len(lengths) > .4:
        flags.append("sparse_text_requires_review")
    if low:
        flags.append("some_low_text_pages")
    if len(lengths) > 100:
        flags.append("long_document_over_100_pages")
    if full.count("\ufffd") / max(1, len(full)) > .001:
        flags.append("replacement_characters")
    title_terms = words(row["title"])
    title_overlap = len(title_terms & words("\n".join(page_texts[:2]))) / max(1, len(title_terms))
    if len(title_terms) >= 5 and title_overlap < .6:
        flags.append("title_not_well_matched_in_first_two_pages")
    record = base_record(row)
    content = "".join(lines).encode("utf-8")
    record.update({"fingerprint": fingerprint, "status": "prepared", "split": "unassigned",
                   "evidence_verified": False, "page_count": len(lengths), "text_chars": total,
                   "low_text_pages": low, "flags": flags, "title_token_overlap": round(title_overlap, 4),
                   "normalized_full_text_sha256": digest(re.sub(r"\s+", " ", full).strip().encode("utf-8")),
                   "pages_file": "pages/" + pages_path.name, "pages_sha256": digest(content),
                   "sample_page_excerpts": previews, "prepared_at": stamp()})
    atomic(pages_path, content)
    atomic(meta_path, json_bytes(record))
    return record, False


def duplicate_candidates(rows):
    groups, exact = {}, []
    fields = {
        "same_doi": lambda r: re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", (r.get("doi") or "").strip().casefold()),
        "same_normalized_title": lambda r: " ".join(re.findall(r"\w+", r["title"].casefold())),
        "same_pdf_bytes": lambda r: r.get("fingerprint", {}).get("pdf_sha256"),
        "same_extracted_text": lambda r: r.get("normalized_full_text_sha256") if r.get("text_chars", 0) >= 2000 else None,
    }
    for reason, key_fn in fields.items():
        groups = defaultdict(list)
        for row in rows:
            key = key_fn(row)
            if key:
                groups[key].append(row["arxiv_id"])
        exact.extend({"reason": reason, "paper_ids": ids, "decision": "pending_review"}
                     for ids in groups.values() if len(ids) > 1)
    titles = [words(r["title"]) for r in rows]
    abstracts = [words(r.get("abstract") or "") for r in rows]
    for i in range(len(rows)):
        for j in range(i):
            a, b = titles[i], titles[j]
            title_j = len(a & b) / max(1, len(a | b))
            a, b = abstracts[i], abstracts[j]
            abstract_j = len(a & b) / max(1, len(a | b))
            if (min(len(titles[i]), len(titles[j])) >= 6 and title_j >= .85) or (min(len(a), len(b)) >= 30 and abstract_j >= .8):
                exact.append({"reason": "similar_title_or_abstract", "paper_ids": [rows[j]["arxiv_id"], rows[i]["arxiv_id"]],
                              "title_jaccard": round(title_j, 4), "abstract_jaccard": round(abstract_j, 4),
                              "decision": "pending_review"})
    return exact


def prepare(base, out):
    state_path = base / "state.json"
    raw = state_path.read_bytes()
    state = json.loads(raw.decode("utf-8-sig"))
    rows = list(state["papers"].values())
    out.mkdir(parents=True, exist_ok=True)
    records, interrupted = [], False
    for i, row in enumerate(rows, 1):
        record = base_record(row)
        record.update({"status": "source_unavailable", "split": "unassigned", "evidence_verified": False,
                       "error": row.get("error") or row.get("status"), "flags": []})
        reused = False
        if row["status"] == "downloaded":
            try:
                record, reused = extract(row, base, out)
            except KeyboardInterrupt:
                interrupted = True
                break
            except Exception as exc:
                record.update(status="preparation_failed", error=type(exc).__name__ + ": " + str(exc)[:400])
        records.append(record)
        print(f'[{i}/{len(rows)}] {row["arxiv_id"]}: {record["status"]}' + (' (reused)' if reused else ''), flush=True)
    print("Checking duplicate candidates...", flush=True)
    candidates = duplicate_candidates(records)
    involved = {p for c in candidates for p in c["paper_ids"]}
    for r in records:
        r["duplicate_review_pending"] = r["arxiv_id"] in involved
        # Do not freeze any test split until duplicate and relevance reviews finish.
    prepared = [r for r in records if r["status"] == "prepared"]
    summary = {"version": VERSION, "created_at": stamp(), "source_state_sha256": digest(raw),
               "code_sha256": digest(Path(__file__).read_bytes()),
               "run_status": "interrupted" if interrupted else "completed_with_issues" if any(r['status'] != 'prepared' for r in records) else "completed",
               "source_metadata_count": len(rows), "processed_records": len(records),
               "unprocessed_records": len(rows) - len(records),
               "status_counts": dict(Counter(r["status"] for r in records)),
               "prepared_pages": sum(r["page_count"] for r in prepared),
               "prepared_text_chars": sum(r["text_chars"] for r in prepared),
               "flagged_prepared_papers": sum(bool(r["flags"]) for r in prepared),
               "flag_counts": dict(Counter(f for r in prepared for f in r["flags"])),
               "duplicate_candidate_groups_or_pairs": len(candidates),
               "development_excluded_papers": sum(bool(r["exclude_from_test"]) for r in records),
               "published_months": dict(sorted(Counter((r.get("published") or "unknown")[:7] for r in records).items())),
               "verified_papers": 0, "candidate_qa": 0, "verified_qa": 0, "frozen_test_qa": 0,
               "limitations": ["Flags and lexical duplicate candidates require review; no semantic deduplication claimed.",
                               "PDF extraction keeps physical page numbers; no OCR/table/formula reconstruction.",
                               "Prepared papers are not automatically relevant or evidence-verified.",
                               "No source files removed; no retrieval index changed; no split frozen."]}
    atomic(out / "manifest.json", json_bytes({"summary": summary, "papers": records}))
    atomic(out / "duplicate_candidates.json", json_bytes(candidates))
    atomic(out / "source_state_snapshot.json", raw)
    report_dir = out / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + ".zip")
    with zipfile.ZipFile(report_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("summary.json", json_bytes(summary))
        z.writestr("papers_review.json", json_bytes(records))
        z.writestr("duplicate_candidates.json", json_bytes(candidates))
        z.writestr("source_config.json", json_bytes(state["config"]))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("REPORT ZIP: " + str(report_path.resolve()), flush=True)
    return summary, report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    base = ROOT / "data" / "dataset_v1"
    if not args.run:
        state = load(base / "state.json")
        print("Offline preview: " + str(dict(Counter(r["status"] for r in state["papers"].values()))))
        print("Run: python prepare_dataset.py --run")
        return
    import fitz
    from run_pipeline import pipeline_lock
    with pipeline_lock(ROOT):
        prepare(base, base / "prepared_v1")


if __name__ == "__main__":
    main()
