"""PaperLens corpus collector v1. No LLM calls. Default: offline preview."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parent
VERSION = "corpus_collector_v1"
IMPLEMENTATION_VERSION = "corpus_collector_v1_1_configurable_pdf_limit"
NS = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
HOSTS = {"export.arxiv.org", "arxiv.org", "www.arxiv.org", "browse.arxiv.org"}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identity(value):
    match = re.fullmatch(r"(\d{4}\.\d{4,5})(v[1-9]\d*)", value)
    if not match:
        raise ValueError("Expected a versioned modern arXiv ID: " + value)
    return match.group(1)


def parse_feed(raw):
    root = ET.fromstring(raw)
    if root.tag != "{" + NS["a"] + "}feed":
        raise ValueError("Response is not an Atom feed")
    total = root.findtext("o:totalResults", namespaces=NS)
    rows = []
    for entry in root.findall("a:entry", NS):
        url = entry.findtext("a:id", "", NS)
        pid = url.rsplit("/", 1)[-1]
        family = identity(pid)  # Also rejects HTTP-200 API error entries.
        title = " ".join(entry.findtext("a:title", "", NS).split())
        abstract = " ".join(entry.findtext("a:summary", "", NS).split())
        if not title or not abstract:
            raise ValueError("Missing title/abstract for " + pid)
        rows.append({"family_id": family, "arxiv_id": pid, "title": title,
                     "abstract": abstract, "authors": [a.findtext("a:name", "", NS)
                     for a in entry.findall("a:author", NS)],
                     "published": entry.findtext("a:published", "", NS),
                     "updated": entry.findtext("a:updated", "", NS),
                     "categories": [a.get("term") for a in entry.findall("a:category", NS)],
                     "doi": entry.findtext("{http://arxiv.org/schemas/atom}doi"),
                     "url": "https://arxiv.org/abs/" + pid,
                     "pdf_url": "https://export.arxiv.org/pdf/" + pid,
                     "license_status": "not_verified", "fetched_at": now()})
    if total is None or not total.isdigit():
        raise ValueError("Missing valid totalResults")
    return rows, int(total)


def checked_url(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in HOSTS or parsed.port not in (None, 443):
        raise ValueError("Non-official or non-HTTPS redirect rejected")
    return url


class OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, code, msg, headers, checked_url(newurl))


class Paused(RuntimeError):
    pass


class HTTP:
    def __init__(self, data):
        self.rate = data / "dataset_arxiv_rate.json"
        self.cooldown = data / "arxiv_cooldown.json"
        self.opener = urllib.request.build_opener(OfficialRedirect())

    def get(self, url, max_bytes):
        current = time.time()
        if self.cooldown.exists():
            until = datetime.fromisoformat(read(self.cooldown)["retry_not_before_utc"]).timestamp()
            if until > current:
                raise Paused("arXiv cooldown until " + datetime.fromtimestamp(until, timezone.utc).isoformat())
        last = read(self.rate).get("last_request_epoch", 0) if self.rate.exists() else 0
        delay = max(0, 6 - (current - last))
        if delay > 60:
            raise Paused("System clock changed; check clock before resuming")
        if delay:
            time.sleep(delay)
        atomic(self.rate, {"last_request_epoch": time.time()})
        req = urllib.request.Request(checked_url(url), headers={"User-Agent": "PaperLensResearchCorpus/1.0", "Accept-Encoding": "identity"})
        try:
            with self.opener.open(req, timeout=90) as response:
                checked_url(response.geturl())
                chunks, size = [], 0
                while True:
                    block = response.read(min(65536, max_bytes + 1 - size))
                    if not block:
                        break
                    chunks.append(block)
                    size += len(block)
                    if size > max_bytes:
                        raise ValueError("Response exceeds configured size limit")
                expected = response.headers.get("Content-Length")
                if expected and int(expected) != size:
                    raise OSError("Incomplete response; safe to resume")
                return b"".join(chunks)
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429, 503):
                wait = 900
                retry = exc.headers.get("Retry-After", "")
                try:
                    wait = max(wait, int(retry))
                except ValueError:
                    try:
                        wait = max(wait, parsedate_to_datetime(retry).timestamp() - time.time())
                    except (ValueError, TypeError, OverflowError):
                        pass
                until = datetime.fromtimestamp(time.time() + wait, timezone.utc).isoformat()
                atomic(self.cooldown, {"retry_not_before_utc": until, "http_status": exc.code})
                raise Paused(f"HTTP {exc.code}; stopped, retry no earlier than {until}") from None
            raise


def initial_state(config):
    return {"version": VERSION, "created_at": now(), "config": config,
            "cursor": 0, "papers": {}, "duplicate_hits": 0,
            "queries": [{"name": q["name"], "query": q["query"], "start": 0,
                         "done": False, "buffer": [], "page_hashes": []} for q in config["queries"]]}


def gather(state, limit, http, base, save):
    queries = state["queries"]
    while len(state["papers"]) < limit:
        if all(q["done"] and not q["buffer"] for q in queries):
            return
        q = queries[state["cursor"] % len(queries)]
        state["cursor"] += 1
        if not q["buffer"] and not q["done"]:
            query = q["query"] + " AND submittedDate:[" + state["config"]["date_start"] + " TO " + state["config"]["date_end"] + "]"
            url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
                "search_query": query, "start": q["start"], "max_results": 25,
                "sortBy": "submittedDate", "sortOrder": "descending"})
            raw = http.get(url, 5 * 1024 * 1024)
            rows, total = parse_feed(raw)
            digest = sha(json.dumps([r["arxiv_id"] for r in rows]).encode())
            if rows and digest in q["page_hashes"]:
                raise ValueError("Repeated API page; stopped to avoid endless pagination")
            name = f'{q["name"]}_{q["start"]:06d}.xml'
            (base / "feeds").mkdir(exist_ok=True)
            (base / "feeds" / name).write_bytes(raw)
            q["page_hashes"].append(digest)
            q["total_reported"] = total
            q["start"] += len(rows)
            q["done"] = not rows or q["start"] >= total
            q["buffer"] = rows
            save()
            print(f'Metadata: {q["name"]}, fetched {q["start"]}/{total}; unique {len(state["papers"])}', flush=True)
        if q["buffer"]:
            row = q["buffer"].pop(0)
            key = row["family_id"]
            if key in state["papers"]:
                state["duplicate_hits"] += 1
                old = state["papers"][key]
                if q["name"] not in old["query_matches"]:
                    old["query_matches"].append(q["name"])
            else:
                row.update({"query_matches": [q["name"]], "status": "pending",
                            "split": "unassigned", "exclude_from_test": key in state["config"]["development_families"]})
                state["papers"][key] = row
            save()


def inspect_pdf(raw):
    import fitz
    if not raw.startswith(b"%PDF-"):
        raise ValueError("Response is not a PDF")
    with fitz.open(stream=raw, filetype="pdf") as doc:
        if doc.needs_pass or doc.page_count < 1:
            raise ValueError("Encrypted or empty PDF")
        lengths, preview = [], ""
        for page in doc:
            txt = page.get_text("text")
            lengths.append(len(txt.strip()))
            if len(preview) < 1200:
                preview += txt[:1200 - len(preview)]
        chars = sum(lengths)
        low = sum(n < 100 for n in lengths)
        return {"sha256": sha(raw), "bytes": len(raw), "pages": len(lengths),
                "text_chars": chars, "low_text_pages": low,
                "parse_review_required": chars < 2000 or low / len(lengths) > 0.4,
                "text_preview": preview, "quality_status": "heuristic_only_not_verified"}


def download(state, limit, http, base, save, retry_failed=False, max_pdf_mb=40):
    folder = base / "pdfs"
    folder.mkdir(exist_ok=True)
    for index, row in enumerate(list(state["papers"].values())[:limit], 1):
        path = folder / (row["arxiv_id"] + ".pdf")
        if row["status"] == "failed" and not retry_failed:
            continue
        try:
            if path.exists():
                raw = path.read_bytes()
                if row.get("pdf", {}).get("sha256") not in (None, sha(raw)):
                    raise ValueError("Local PDF hash changed; retained for inspection")
            else:
                raw = http.get(row["pdf_url"], max_pdf_mb * 1024 * 1024)
            info = inspect_pdf(raw)
            if not path.exists():
                tmp = path.with_suffix(".pdf.part")
                tmp.write_bytes(raw)
                os.replace(tmp, path)
            row.update({"status": "downloaded", "pdf": info, "checked_at": now()})
            row.pop("error", None)
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            row.update({"status": "failed", "error": "PDF HTTP 404"})
        except (ValueError, RuntimeError) as exc:
            if isinstance(exc, Paused):
                raise
            row.update({"status": "failed", "error": str(exc)[:400]})
        save()
        print(f'[{index}/{min(limit, len(state["papers"]))}] {row["arxiv_id"]}: {row["status"]}', flush=True)


def report(state, base, status, error, limit, max_pdf_mb=40):
    papers = list(state["papers"].values())
    summary = {"version": VERSION, "implementation_version": IMPLEMENTATION_VERSION,
               "max_pdf_mib": max_pdf_mb, "created_at": now(), "run_status": status,
               "error": error, "checkpoint_metadata_target": limit,
               "eventual_paper_target": state["config"]["target_unique_papers"],
               "unique_metadata": len(papers), "status_counts": dict(Counter(p["status"] for p in papers)),
               "parse_review_required": sum(p.get("pdf", {}).get("parse_review_required", False) for p in papers),
               "downloaded_parse_screen_pass": sum(p["status"] == "downloaded" and not p["pdf"]["parse_review_required"] for p in papers),
               "verified_papers": 0, "candidate_qa": 0, "verified_qa": 0, "frozen_test_qa": 0,
               "duplicate_hits": state["duplicate_hits"],
               "topic_first_match_counts": dict(Counter(p["query_matches"][0] for p in papers)),
               "collector_sha256": sha(Path(__file__).read_bytes()),
               "notes": ["Keyword-selected, recent-first sample; not representative.",
                         "Topic matches are retrieval provenance, not human labels.",
                         "No QA generated; no train/dev/test split frozen; parse screen is heuristic."]}
    out = base / "reports" / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + ".zip")
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, value in [("summary.json", summary), ("state.json", state), ("papers.json", papers)]:
            z.writestr(name, json.dumps(value, ensure_ascii=False, indent=2))
        for feed in sorted((base / "feeds").glob("*.xml")):
            z.write(feed, "feeds/" + feed.name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("REPORT ZIP: " + str(out.resolve()), flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--limit", type=int, default=100, help="Cumulative metadata checkpoint, not additional count")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-pdf-mb", type=int, default=40,
                        help="Per-PDF size cap in MiB (1..512); default 40")
    args = parser.parse_args()
    if not 1 <= args.max_pdf_mb <= 512:
        parser.error("max-pdf-mb must be 1..512 MiB")
    config = read(ROOT / "dataset_config" / "corpus_v1.json")
    if not 1 <= args.limit <= config["target_unique_papers"]:
        parser.error("limit must be 1..1000")
    if not args.run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        print("Offline preview only. Start: python collect_dataset.py --run --limit 100")
        return 0
    import fitz  # Fail before network if PDF parser is not installed.
    from run_pipeline import pipeline_lock  # Import only; does not invoke pipeline/mail.
    with pipeline_lock(ROOT):
        base = ROOT / "data" / "dataset_v1"
        base.mkdir(parents=True, exist_ok=True)
        path = base / "state.json"
        state = read(path) if path.exists() else initial_state(config)
        if state["config"] != config or state["version"] != VERSION:
            raise ValueError("Frozen configuration/version differs; restore original config before resume")
        save = lambda: atomic(path, state)
        save()
        status, error = "completed_checkpoint", None
        try:
            http = HTTP(ROOT / "data")
            gather(state, args.limit, http, base, save)
            download(state, args.limit, http, base, save, args.retry_failed, args.max_pdf_mb)
            selected = list(state["papers"].values())[:args.limit]
            if len(selected) < args.limit or any(p["status"] != "downloaded" for p in selected):
                status = "checkpoint_incomplete"
        except KeyboardInterrupt:
            status, error = "interrupted", "User interruption; rerun same command to resume"
        except Exception as exc:
            status, error = "paused", type(exc).__name__ + ": " + str(exc)[:600]
        finally:
            save()
            report(state, base, status, error, args.limit, args.max_pdf_mb)
        return 0 if status == "completed_checkpoint" else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        sys.exit(2)
