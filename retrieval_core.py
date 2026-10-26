"""Dense, BM25, and reciprocal-rank-fusion retrieval for Eval-lite."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from prepare_corpus import sha256
from search_bm25 import tokenize

CHINESE_RE = re.compile(r"[㐀-鿿]")


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    parent_chunk_id: str
    paper_id: str
    score: float
    source: str
    text: str = ""
    page_number: int | None = None


def _strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", str(arxiv_id))


def load_index(index_dir: Path) -> tuple[faiss.Index, dict, dict[str, dict]]:
    """Load an arbitrary prepared index directory, with the same checks as load_store."""
    index_dir = Path(index_dir)
    with np.load(index_dir / "vector_store.npz", allow_pickle=False) as saved:
        metadata = json.loads(saved["metadata"].item())
        index = faiss.deserialize_index(saved["index"])

    corpus_raw = (index_dir / "corpus.json").read_bytes()
    clean_raw = (index_dir / "corpus_clean.json").read_bytes()
    if metadata.get("source_corpus_sha256") != sha256(corpus_raw):
        raise ValueError("original corpus hash does not match the index")
    if metadata.get("corpus_sha256") != sha256(clean_raw):
        raise ValueError("clean corpus hash does not match the index")

    units = metadata.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("index has no units")
    if (
        index.ntotal != len(units)
        or index.ntotal == 0
        or index.d != metadata.get("dimension")
        or len(units) != metadata.get("vector_count")
    ):
        raise ValueError("index count/dimension does not match metadata")
    if index.metric_type != faiss.METRIC_INNER_PRODUCT or not metadata.get("normalized"):
        raise ValueError("index must be normalized inner product")
    if len({unit["chunk_id"] for unit in units}) != len(units):
        raise ValueError("duplicate unit ids")

    clean = json.loads(clean_raw)
    parents = {chunk["chunk_id"]: chunk for chunk in clean["chunks"]}
    for unit in units:
        parent = parents.get(unit["parent_chunk_id"])
        if parent is None:
            raise ValueError("unit references a missing parent chunk")
        start = unit["char_start"] - parent["char_start"]
        end = unit["char_end"] - parent["char_start"]
        if (
            not 0 <= start < end <= len(parent["text"])
            or parent["text"][start:end] != unit["text"]
            or unit["arxiv_id"] != parent["arxiv_id"]
            or unit["page_number"] != parent["page_number"]
        ):
            raise ValueError("unit text/offsets do not match the parent chunk")
    return index, metadata, parents


def _embed(embedder: Any, text: str) -> np.ndarray:
    if hasattr(embedder, "encode"):
        vector = embedder.encode([text], normalize_embeddings=True)
    else:
        vector = embedder(text)
    array = np.asarray(vector, dtype="float32")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] != 1:
        raise ValueError("embedder must return one vector")
    return array


def retrieve_dense(
    index: faiss.Index,
    metadata: Mapping[str, Any],
    query: str,
    k: int,
    *,
    embedder: Any | None = None,
    paper_filter: str | None = None,
) -> list[Hit]:
    if k <= 0:
        return []
    if embedder is None:
        from sentence_transformers import SentenceTransformer

        root = Path(__file__).resolve().parent
        embedder = SentenceTransformer(
            metadata["model_name"],
            device="cpu",
            cache_folder=str(root / "data" / "models"),
            local_files_only=True,
        )
    prefix = metadata.get("query_prefix", "query: ")
    query_vector = _embed(embedder, prefix + query)
    scores, positions = index.search(query_vector, index.ntotal)
    hits: list[Hit] = []
    seen: set[str] = set()
    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue
        unit = metadata["units"][int(position)]
        if paper_filter is not None and (
            unit["arxiv_id"] != paper_filter
            and _strip_version(unit["arxiv_id"]) != _strip_version(paper_filter)
        ):
            continue
        parent_id = unit["parent_chunk_id"]
        if parent_id in seen:
            continue
        seen.add(parent_id)
        hits.append(
            Hit(
                chunk_id=unit["chunk_id"],
                parent_chunk_id=parent_id,
                paper_id=unit["arxiv_id"],
                score=float(score),
                source="dense",
                text=unit["text"],
                page_number=unit.get("page_number"),
            )
        )
        if len(hits) >= k:
            break
    return hits


def retrieve_bm25(
    clean_corpus: Mapping[str, Any],
    query_en: str,
    k: int,
    *,
    paper_filter: str | None = None,
) -> list[Hit]:
    if k <= 0:
        return []
    if CHINESE_RE.search(query_en or ""):
        raise ValueError("BM25 requires an English query")
    query_terms = list(dict.fromkeys(tokenize(query_en)))
    if not query_terms:
        raise ValueError("BM25 query has no English or numeric terms")

    chunks = list(clean_corpus.get("chunks") or [])
    pairs = []
    for chunk in chunks:
        if paper_filter is not None and (
            chunk["arxiv_id"] != paper_filter
            and _strip_version(chunk["arxiv_id"]) != _strip_version(paper_filter)
        ):
            continue
        terms = tokenize(chunk.get("text", ""))
        if terms:
            pairs.append((chunk, terms))
    if not pairs:
        return []
    documents = [terms for _, terms in pairs]
    scores = BM25Okapi(documents, k1=1.5, b=0.75).get_scores(query_terms)
    matched = [
        index
        for index, (_, terms) in enumerate(pairs)
        if set(query_terms).intersection(terms)
    ]
    ranked = sorted(
        matched,
        key=lambda index: (
            -float(scores[index]),
            pairs[index][0]["chunk_id"],
        ),
    )
    hits: list[Hit] = []
    for index in ranked[:k]:
        chunk = pairs[index][0]
        hits.append(
            Hit(
                chunk_id=chunk["chunk_id"],
                parent_chunk_id=chunk["chunk_id"],
                paper_id=chunk["arxiv_id"],
                score=float(scores[index]),
                source="bm25",
                text=chunk.get("text", ""),
                page_number=chunk.get("page_number"),
            )
        )
    return hits


def rrf_fuse(
    dense: Sequence[Hit], bm25: Sequence[Hit], k: int = 60, top_n: int = 6
) -> list[Hit]:
    if k < 0 or top_n < 0:
        raise ValueError("k and top_n must be non-negative")
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for hits in (dense, bm25):
        for rank, hit in enumerate(hits, start=1):
            key = hit.parent_chunk_id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            best.setdefault(key, hit)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return [
        Hit(
            chunk_id=best[key].chunk_id,
            parent_chunk_id=key,
            paper_id=best[key].paper_id,
            score=scores[key],
            source="rrf",
            text=best[key].text,
            page_number=best[key].page_number,
        )
        for key in ordered[:top_n]
    ]


def retrieve(
    mode: str,
    query_zh: str,
    query_en: str,
    index: faiss.Index,
    metadata: Mapping[str, Any],
    clean_corpus: Mapping[str, Any],
    k: int,
    *,
    embedder: Any | None = None,
    paper_filter: str | None = None,
) -> list[Hit]:
    if mode == "dense":
        return retrieve_dense(
            index, metadata, query_zh, k, embedder=embedder,
            paper_filter=paper_filter,
        )
    if mode == "bm25":
        return retrieve_bm25(
            clean_corpus, query_en or query_zh, k, paper_filter=paper_filter
        )
    if mode == "hybrid":
        pool = max(50, k)
        dense = retrieve_dense(
            index, metadata, query_zh, pool, embedder=embedder,
            paper_filter=paper_filter,
        )
        bm25 = retrieve_bm25(
            clean_corpus, query_en or query_zh, pool, paper_filter=paper_filter
        )
        return rrf_fuse(dense, bm25, top_n=k)
    raise ValueError("mode must be dense, bm25, or hybrid")
