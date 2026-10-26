"""Task E2: build the held-out benchmark test index from re-parsed PDFs.

The dev index in ``data/rag`` and the held-out test index must be directly
comparable. This module therefore runs every held-out PDF through the *same*
pipeline as the dev index:

1. ``prepare_corpus.extract_paper``: PyMuPDF page/block extraction from the
   original PDF bytes (never the pre-extracted ``prepared_v1/pages``).
2. ``clean_corpus.clean_text`` + ``clean_corpus.reading_groups`` (which uses
   ``clean_corpus.column_cut``): hyphen repair and the two-column heuristic.
3. ``prepare_corpus.split_text``: parent chunks with the requested chunker
   parameters.
4. ``build_vector_index.make_segments``: child units that preserve absolute
   parent offsets.
5. E5-small (or an injected embedder) + a normalized inner-product FAISS
   index, saved with ``build_vector_index.save_store``.

Outputs (inside ``out_dir``): ``corpus.json``, ``corpus_clean.json``,
``vector_store.npz``, ``manifest.json``.

``data/rag`` is read-only here and writing into it is refused. Tests inject a
fake embedder; the default E5 model is only loaded when ``embedder is None``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import clean_corpus  # noqa: E402
import prepare_corpus  # noqa: E402
from build_vector_index import MODEL_NAME, make_segments, save_store  # noqa: E402

DEFAULT_MODEL_CACHE = PROJECT_ROOT / "data" / "models"
DEFAULT_SPLITS = PROJECT_ROOT / "eval" / "benchmark_v2" / "splits.draft.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "eval" / "benchmark_v2" / "index"
DEFAULT_CHUNKER_CONFIG = {
    "chunk_chars": prepare_corpus.CHUNK_CHARS,
    "overlap_chars": prepare_corpus.OVERLAP_CHARS,
    "max_tokens": 512,
}
SCHEMA_VERSION = 1
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
INDEX_TYPE = "IndexFlatIP"

_ARXIV_ID = re.compile(
    r"(?:\d{4}\.\d{4,5}|[A-Za-z][A-Za-z0-9.-]*/\d{7})(?:v\d+)?"
)
_VERSION_SUFFIX = re.compile(r"v\d+$")
_VOCABULARY = re.compile(r"(?<![A-Za-z-])[A-Za-z]{3,}(?![A-Za-z-])")


class TestIndexError(RuntimeError):
    """Raised when a held-out PDF cannot be turned into index evidence."""


# ---------------------------------------------------------------------------
# Configuration and identity helpers
# ---------------------------------------------------------------------------


def family_of(paper_id: str) -> str:
    """Return the version-stripped arXiv family (base ID)."""
    return _VERSION_SUFFIX.sub("", paper_id)


def _normalize_chunker_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("chunker_config must be a dict")
    unknown = set(config) - set(DEFAULT_CHUNKER_CONFIG)
    if unknown:
        raise ValueError(
            "unknown chunker_config keys: " + ", ".join(sorted(unknown))
        )
    merged = {**DEFAULT_CHUNKER_CONFIG, **config}
    try:
        chunk_chars = int(merged["chunk_chars"])
        overlap_chars = int(merged["overlap_chars"])
        max_tokens = int(merged["max_tokens"])
    except (TypeError, ValueError) as exc:
        raise ValueError("chunker_config values must be integers") from exc

    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if not 0 <= overlap_chars < chunk_chars:
        raise ValueError("Require 0 <= overlap_chars < chunk_chars")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    return {
        "chunk_chars": chunk_chars,
        "overlap_chars": overlap_chars,
        "max_tokens": max_tokens,
    }


def _checked_out_dir(out_dir: Any) -> Path:
    if out_dir is None:
        raise ValueError("out_dir is required")
    out = Path(out_dir)
    protected = (PROJECT_ROOT / "data" / "rag").resolve()
    resolved = out.resolve()
    if resolved == protected or protected in resolved.parents:
        raise ValueError(
            f"Refusing to write the held-out test index into data/rag: {out}"
        )
    return out


def _paper_id_of(path: Path) -> str:
    paper_id = path.stem
    if not _ARXIV_ID.fullmatch(paper_id):
        raise TestIndexError(
            f"Cannot derive an arXiv ID from PDF {path.name}: "
            f"expected the versioned ID as the filename stem, got {paper_id!r}"
        )
    return paper_id


def _validate_identifiers(paths: Sequence[Path]) -> list[str]:
    seen_ids: dict[str, Path] = {}
    seen_families: dict[str, Path] = {}
    paper_ids: list[str] = []
    for path in paths:
        paper_id = _paper_id_of(path)
        if paper_id in seen_ids:
            raise TestIndexError(
                f"Duplicate paper id {paper_id}: {seen_ids[paper_id]} and {path}"
            )
        family = family_of(paper_id)
        if family in seen_families:
            raise TestIndexError(
                f"Duplicate paper family {family}: "
                f"{seen_families[family]} and {path}"
            )
        seen_ids[paper_id] = path
        seen_families[family] = path
        paper_ids.append(paper_id)
    return paper_ids


def _chunker_sha256(config: dict, max_tokens: int) -> str:
    payload = {
        "chunk_chars": config["chunk_chars"],
        "overlap_chars": config["overlap_chars"],
        "max_tokens": max_tokens,
        "segmenter": "build_vector_index.make_segments",
        "splitter": "prepare_corpus.split_text",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _selection_sha256(documents: Sequence[dict]) -> str:
    payload = [
        {"arxiv_id": document["arxiv_id"], "pdf_sha256": document["pdf_sha256"]}
        for document in documents
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _index_sha256(units: Sequence[dict], vectors: np.ndarray) -> str:
    """Deterministic content hash of child-unit order plus vector values."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            [unit["chunk_id"] for unit in units],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(np.ascontiguousarray(vectors.astype("<f4", copy=False)).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Step 1: PDF -> raw corpus
# ---------------------------------------------------------------------------


def _pdf_title(pdf_bytes: bytes, pymupdf) -> str:
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            title = ((document.metadata or {}).get("title") or "").strip()
    except Exception:
        return ""
    return title


def _parse_pdf(path: Path, pymupdf) -> tuple[dict, list[dict]]:
    if not path.is_file():
        raise TestIndexError(f"Cannot parse PDF {path.name}: file not found ({path})")

    try:
        pdf_bytes = path.read_bytes()
    except OSError as exc:
        raise TestIndexError(
            f"Cannot parse PDF {path.name}: unreadable ({path}): {exc}"
        ) from exc

    if not pdf_bytes.startswith(b"%PDF-"):
        raise TestIndexError(f"Cannot parse PDF {path.name}: not a PDF file ({path})")

    paper = {
        "arxiv_id": _paper_id_of(path),
        "title": _pdf_title(pdf_bytes, pymupdf) or path.stem,
    }
    try:
        document, chunks = prepare_corpus.extract_paper(
            PROJECT_ROOT, paper, pymupdf, pdf_path=path
        )
    except Exception as exc:
        raise TestIndexError(
            f"Cannot parse PDF {path.name} ({path}): "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return document, chunks


def _build_corpus(
    documents: Sequence[dict],
    chunks: Sequence[dict],
    chunker_config: dict,
    pymupdf,
    created_at: str,
) -> dict:
    return {
        "schema_version": prepare_corpus.SCHEMA_VERSION,
        "created_at": created_at,
        "builder": "eval/benchmark_v2/build_test_index.py",
        "source": "re-parsed original PDFs; prepared_v1 pages are not used",
        "scope": "Held-out benchmark test corpus; extracted source text, not verified answers.",
        "recommendations_sha256": _selection_sha256(documents),
        "selected_ids": [document["arxiv_id"] for document in documents],
        "extractor": {
            "name": "PyMuPDF",
            "version": pymupdf.VersionBind,
            "method": "page-local text blocks; sort=True; no OCR",
            "max_chunk_chars": chunker_config["chunk_chars"],
            "overlap_chars": chunker_config["overlap_chars"],
        },
        "paper_count": len(documents),
        "chunk_count": len(chunks),
        "papers": list(documents),
        "chunks": list(chunks),
    }


# ---------------------------------------------------------------------------
# Step 2: raw corpus -> cleaned corpus (same logic as clean_corpus.main)
# ---------------------------------------------------------------------------


def _clean_corpus(corpus: dict, corpus_raw: bytes, chunker_config: dict) -> dict:
    chunk_chars = chunker_config["chunk_chars"]
    overlap_chars = chunker_config["overlap_chars"]
    corpus_sha = prepare_corpus.sha256(corpus_raw)
    # Chunk IDs are salted with the deterministic PDF-selection digest, not
    # with the corpus.json file hash: the file carries ``created_at`` and a
    # wall-clock-dependent salt would silently invalidate frozen qrels on a
    # rebuild. ``source_corpus_sha256`` still records the exact file hash.
    id_salt = corpus["recommendations_sha256"][:12]

    result = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_corpus_sha256": corpus_sha,
        "recommendations_sha256": corpus["recommendations_sha256"],
        "selected_ids": corpus["selected_ids"],
        "method": "coordinate heuristic; same-page same-region chunks; no OCR",
        "chunker": {
            "chunk_chars": chunk_chars,
            "overlap_chars": overlap_chars,
        },
        "papers": [],
        "chunks": [],
    }

    for paper in corpus["papers"]:
        vocabulary = set(
            _VOCABULARY.findall(
                "\n".join(
                    block["raw_text"]
                    for page in paper["pages"]
                    for block in page["blocks"]
                ).lower()
            )
        )

        cleaned = {
            key: paper[key]
            for key in ("arxiv_id", "title", "pdf_path", "pdf_sha256")
        }
        cleaned["pages"] = []
        start_count = len(result["chunks"])

        for page in paper["pages"]:
            number = page["page_number"]
            blocks: list[dict] = []
            excluded: list[dict] = []
            bottom = max(
                (block["bbox"][3] for block in page["blocks"]),
                default=0,
            )

            for block in page["blocks"]:
                item = dict(block)
                item["text"] = clean_corpus.clean_text(
                    block["raw_text"], vocabulary
                )
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

            layout, groups = clean_corpus.reading_groups(blocks)
            cleaned["pages"].append(
                {
                    "page_number": number,
                    "layout": layout,
                    "blocks": [block for group in groups for block in group],
                    "excluded_blocks": excluded,
                }
            )

            for group_number, group in enumerate(groups, 1):
                text = ""
                spans: list[dict] = []

                for block in group:
                    text += "\n\n" if text else ""
                    begin = len(text)
                    text += block["text"]
                    spans.append(
                        {
                            "block_number": block["block_number"],
                            "bbox": block["bbox"],
                            "start": begin,
                            "end": len(text),
                        }
                    )

                for piece, (start, end, content) in enumerate(
                    prepare_corpus.split_text(
                        text, chunk_chars, overlap_chars
                    ),
                    1,
                ):
                    sources = [
                        span
                        for span in spans
                        if span["start"] < end and span["end"] > start
                    ]
                    result["chunks"].append(
                        {
                            "chunk_id": (
                                f"{paper['arxiv_id']}:{id_salt}"
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
                            "char_start": start,
                            "char_end": end,
                            "text": content,
                            "sources": sources,
                            "short_fragment": len(content.strip()) < 80,
                        }
                    )

        cleaned["chunk_count"] = len(result["chunks"]) - start_count
        result["papers"].append(cleaned)

    result["paper_count"] = len(result["papers"])
    result["chunk_count"] = len(result["chunks"])
    return result


# ---------------------------------------------------------------------------
# Step 3: cleaned corpus -> child units -> vectors -> FAISS
# ---------------------------------------------------------------------------


def _model_identity(embedder: object) -> tuple[str, str | None]:
    name = (
        getattr(embedder, "model_name", None)
        or getattr(embedder, "model_name_or_path", None)
        or MODEL_NAME
    )
    revision = getattr(embedder, "model_revision", None)
    if revision is None:
        revision = getattr(embedder, "revision", None)
    return str(name), revision


def _build_vectors(
    clean: dict, embedder: object, chunker_config: dict
) -> tuple[list[dict], np.ndarray, int]:
    tokenizer = getattr(embedder, "tokenizer", None)
    if tokenizer is None or not callable(tokenizer):
        raise ValueError("embedder must expose a callable tokenizer")

    model_limit = getattr(embedder, "max_seq_length", None)
    limit = int(chunker_config["max_tokens"])
    if model_limit is not None:
        limit = min(limit, int(model_limit))
    if limit <= 0:
        raise ValueError("embedder max_seq_length must be positive")

    chunks = clean["chunks"]
    if not chunks:
        raise ValueError("cleaned corpus is empty; refusing to build an index")

    units: list[dict] = []
    for chunk in chunks:
        units.extend(make_segments(chunk, tokenizer, limit))
    if not units:
        raise ValueError("no child units were produced from the cleaned corpus")
    if len({unit["chunk_id"] for unit in units}) != len(units):
        raise ValueError("duplicate child unit chunk_id detected")

    texts = [PASSAGE_PREFIX + unit["text"] for unit in units]
    vectors = embedder.encode(
        texts,
        prompt="",
        batch_size=8,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    vectors = np.ascontiguousarray(vectors, dtype="float32")

    if vectors.ndim != 2 or vectors.shape[0] != len(units):
        raise ValueError(
            "embedder returned an unexpected vector shape: "
            f"{vectors.shape} for {len(units)} units"
        )
    dimension = int(vectors.shape[1])
    if dimension <= 0:
        raise ValueError("embedder returned zero-dimensional vectors")
    if not np.isfinite(vectors).all():
        raise ValueError("embedder returned NaN or infinite vectors")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(
            "embedder must return L2-normalized vectors "
            "(normalize_embeddings=True)"
        )

    get_dimension = getattr(embedder, "get_sentence_embedding_dimension", None)
    if callable(get_dimension):
        declared = get_dimension()
        if declared is not None and int(declared) != dimension:
            raise ValueError(
                f"embedder dimension mismatch: declared {declared}, "
                f"encoded {dimension}"
            )

    return units, vectors, limit


def _load_default_embedder(model_cache: Path | None = None) -> object:
    """Load E5-small strictly from the local cache; never downloads."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        cache_folder=str(model_cache or DEFAULT_MODEL_CACHE),
        local_files_only=True,
    )
    model.model_name = MODEL_NAME
    model.model_revision = getattr(model[0].auto_model.config, "_commit_hash", None)
    return model


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_test_index(
    pdf_paths: list[Path],
    out_dir: Path,
    chunker_config: dict,
    embedder: object | None = None,
    *,
    model_cache: Path | None = None,
) -> dict:
    """Build a held-out test index from ``pdf_paths`` into ``out_dir``.

    ``embedder`` must mirror the SentenceTransformer surface (``tokenizer``,
    ``max_seq_length``, ``encode``, optionally ``get_sentence_embedding_dimension``)
    and return L2-normalized float vectors. When it is ``None``, E5-small is
    loaded from ``data/models`` with ``local_files_only=True``.

    Returns the manifest that was written. Raises ``TestIndexError`` naming
    the offending file when a PDF cannot be parsed; papers are never dropped.
    """
    paths = [Path(path) for path in pdf_paths]
    if not paths:
        raise ValueError("pdf_paths must contain at least one PDF")

    config = _normalize_chunker_config(chunker_config)
    out = _checked_out_dir(out_dir)
    _validate_identifiers(paths)

    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "缺少依赖，请运行：python -m pip install PyMuPDF"
        ) from exc

    created_at = datetime.now(timezone.utc).isoformat()
    documents: list[dict] = []
    chunks: list[dict] = []
    for path in paths:
        document, paper_chunks = _parse_pdf(path, pymupdf)
        documents.append(document)
        chunks.extend(paper_chunks)

    corpus = _build_corpus(documents, chunks, config, pymupdf, created_at)
    corpus_text = json.dumps(corpus, ensure_ascii=False, indent=2)
    corpus_raw = corpus_text.encode("utf-8")
    prepare_corpus.atomic_write(out / "corpus.json", corpus_text)

    clean = _clean_corpus(corpus, corpus_raw, config)
    clean_text = json.dumps(clean, ensure_ascii=False, indent=2)
    clean_raw = clean_text.encode("utf-8")
    prepare_corpus.atomic_write(out / "corpus_clean.json", clean_text)

    if embedder is None:
        embedder = _load_default_embedder(model_cache)

    units, vectors, max_tokens = _build_vectors(clean, embedder, config)
    dimension = int(vectors.shape[1])

    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    if index.ntotal != len(units):
        raise RuntimeError("FAISS index count does not match child units")

    model_name, model_revision = _model_identity(embedder)
    metadata = {
        "schema_version": 1,
        "created_at": created_at,
        "model_name": model_name,
        "model_revision": model_revision,
        "corpus_sha256": prepare_corpus.sha256(clean_raw),
        "source_corpus_sha256": prepare_corpus.sha256(corpus_raw),
        "max_tokens": max_tokens,
        "normalized": True,
        "dimension": dimension,
        "index_type": INDEX_TYPE,
        "passage_prefix": PASSAGE_PREFIX,
        "query_prefix": QUERY_PREFIX,
        "parent_chunk_count": len(clean["chunks"]),
        "vector_count": len(units),
        "units": units,
    }
    vector_store_path = out / "vector_store.npz"
    save_store(vector_store_path, index, metadata)

    chunks_per_paper: dict[str, int] = {}
    for chunk in clean["chunks"]:
        chunks_per_paper[chunk["arxiv_id"]] = (
            chunks_per_paper.get(chunk["arxiv_id"], 0) + 1
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "purpose": (
            "Held-out benchmark test index built from re-parsed PDFs with the "
            "same prepare/clean pipeline as the dev index"
        ),
        "builder": "eval/benchmark_v2/build_test_index.py",
        "paper_count": len(documents),
        "paper_ids": [document["arxiv_id"] for document in documents],
        "families": [family_of(document["arxiv_id"]) for document in documents],
        "papers": [
            {
                "paper_id": document["arxiv_id"],
                "family": family_of(document["arxiv_id"]),
                "title": document["title"],
                "pdf_file": Path(document["pdf_path"]).name,
                "pdf_path": document["pdf_path"],
                "pdf_sha256": document["pdf_sha256"],
                "page_count": document["page_count"],
                "parent_chunk_count": chunks_per_paper.get(
                    document["arxiv_id"], 0
                ),
            }
            for document in documents
        ],
        "chunker": {
            "chunk_chars": config["chunk_chars"],
            "overlap_chars": config["overlap_chars"],
            "max_tokens": max_tokens,
        },
        "chunker_sha256": _chunker_sha256(config, max_tokens),
        "model": {
            "name": model_name,
            "revision": model_revision,
            "dimension": dimension,
            "max_tokens": max_tokens,
            "normalized": True,
            "passage_prefix": PASSAGE_PREFIX,
            "query_prefix": QUERY_PREFIX,
        },
        "corpus": {
            "corpus_sha256": prepare_corpus.sha256(corpus_raw),
            "corpus_clean_sha256": prepare_corpus.sha256(clean_raw),
            "parent_chunk_count": len(clean["chunks"]),
            "vector_count": len(units),
        },
        "vectors": {
            "count": len(units),
            "dimension": dimension,
            "index_type": INDEX_TYPE,
            "normalized": True,
        },
        "index_sha256": _index_sha256(units, vectors),
        "vector_store_sha256": prepare_corpus.sha256(
            vector_store_path.read_bytes()
        ),
    }
    prepare_corpus.atomic_write(
        out / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2),
    )
    return manifest


# ---------------------------------------------------------------------------
# CLI (Task E7 uses this; not exercised by offline tests)
# ---------------------------------------------------------------------------


def _splits_pdf_paths(splits_path: Path) -> list[Path]:
    if not splits_path.is_file():
        raise FileNotFoundError(f"split file not found: {splits_path}")
    with splits_path.open("r", encoding="utf-8") as handle:
        splits = json.load(handle)
    rows = splits.get("test") or []
    if not rows:
        raise ValueError(f"no test rows in {splits_path}")

    paths: list[Path] = []
    missing: list[str] = []
    for row in rows:
        relative = row.get("pdf_path")
        if not relative:
            raise ValueError(
                f"test row {row.get('arxiv_id')!r} has no pdf_path"
            )
        path = Path(relative)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            missing.append(f"{row.get('arxiv_id')}: {path}")
        paths.append(path)
    if missing:
        raise TestIndexError(
            "PDFs missing for the test split; replace from reserve instead of "
            "shrinking the set:\n  " + "\n  ".join(missing)
        )
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the held-out benchmark test index from the split's PDFs "
            "(Task E2). Never writes to data/rag."
        )
    )
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--model-cache", type=Path, default=DEFAULT_MODEL_CACHE
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_CHUNKER_CONFIG["max_tokens"],
    )
    args = parser.parse_args(argv)

    try:
        pdf_paths = _splits_pdf_paths(args.splits.resolve())
        manifest = build_test_index(
            pdf_paths,
            args.out_dir.resolve(),
            {
                "chunk_chars": DEFAULT_CHUNKER_CONFIG["chunk_chars"],
                "overlap_chars": DEFAULT_CHUNKER_CONFIG["overlap_chars"],
                "max_tokens": args.max_tokens,
            },
            embedder=None,
            model_cache=args.model_cache.resolve(),
        )
    except (TestIndexError, FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    summary = {
        "papers": manifest["paper_count"],
        "vectors": manifest["vectors"]["count"],
        "dimension": manifest["vectors"]["dimension"],
        "index_sha256": manifest["index_sha256"],
        "out_dir": str(args.out_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("data/rag was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
