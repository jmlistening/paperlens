"""Offline tests for dense/BM25/RRF retrieval (Task E4)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_vector_index import save_store  # noqa: E402
from prepare_corpus import sha256  # noqa: E402
from retrieval_core import (  # noqa: E402
    Hit,
    load_index,
    retrieve_bm25,
    retrieve_dense,
    rrf_fuse,
)

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"


def make_synthetic_index(tmp: Path):
    tmp.mkdir(parents=True, exist_ok=True)
    parent_a = {
        "chunk_id": "p1", "arxiv_id": "2601.1v1", "title": "A",
        "page_number": 1, "char_start": 0, "text": "alpha retrieval augmented generation",
    }
    parent_b = {
        "chunk_id": "p2", "arxiv_id": "2601.2v1", "title": "B",
        "page_number": 2, "char_start": 0, "text": "traffic signal control agent",
    }
    corpus = {"chunks": [parent_a, parent_b]}
    clean = {"chunks": [parent_a, parent_b]}

    def unit(uid, parent, start, end):
        return {
            "chunk_id": uid,
            "parent_chunk_id": parent["chunk_id"],
            "arxiv_id": parent["arxiv_id"],
            "page_number": parent["page_number"],
            "char_start": start,
            "char_end": end,
            "text": parent["text"][start:end],
        }

    units = [
        unit("u1", parent_a, 0, 5),
        unit("u1b", parent_a, 6, 15),
        unit("u2", parent_b, 0, 7),
    ]
    corpus_bytes = json.dumps(corpus).encode("utf-8")
    clean_bytes = json.dumps(clean).encode("utf-8")
    (tmp / "corpus.json").write_bytes(corpus_bytes)
    (tmp / "corpus_clean.json").write_bytes(clean_bytes)

    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype="float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    metadata = {
        "schema_version": 1,
        "model_name": "fake-e5",
        "model_revision": None,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "max_tokens": 512,
        "normalized": True,
        "dimension": 2,
        "index_type": "IndexFlatIP",
        "parent_chunk_count": 2,
        "vector_count": 3,
        "source_corpus_sha256": sha256(corpus_bytes),
        "corpus_sha256": sha256(clean_bytes),
        "units": units,
    }
    save_store(tmp / "vector_store.npz", index, metadata)
    return tmp, clean, units


class RetrievalCoreTests(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.index_dir, self.clean, self.units = make_synthetic_index(
            Path(self.tmp.name) / "index"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_index_verifies_hashes_and_parent_offsets(self):
        index, metadata, parents = load_index(self.index_dir)
        self.assertEqual(index.ntotal, 3)
        self.assertEqual(metadata["dimension"], 2)
        self.assertIn("p1", parents)
        (self.index_dir / "corpus.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_index(self.index_dir)

    def test_dense_dedups_parent_and_honors_paper_filter(self):
        index, metadata, _ = load_index(self.index_dir)
        embedder = lambda text: np.array([[1.0, 0.0]], dtype="float32")  # noqa: E731
        hits = retrieve_dense(index, metadata, "query", 3, embedder=embedder)
        self.assertEqual([hit.parent_chunk_id for hit in hits], ["p1", "p2"])
        filtered = retrieve_dense(
            index, metadata, "query", 3, embedder=embedder,
            paper_filter="2601.2v1",
        )
        self.assertEqual([hit.parent_chunk_id for hit in filtered], ["p2"])

    def test_bm25_finds_english_terms_and_rejects_chinese(self):
        hits = retrieve_bm25(self.clean, "retrieval augmented generation", 3)
        self.assertEqual(hits[0].parent_chunk_id, "p1")
        with self.assertRaises(ValueError):
            retrieve_bm25(self.clean, "检索增强生成", 3)

    def test_rrf_fusion_orders_by_reciprocal_rank(self):
        dense = [
            Hit("d1", "p1", "2601.1v1", 0.9, "dense"),
            Hit("d2", "p2", "2601.2v1", 0.8, "dense"),
        ]
        bm25 = [
            Hit("b2", "p2", "2601.2v1", 5.0, "bm25"),
            Hit("b3", "p3", "2601.3v1", 4.0, "bm25"),
        ]
        fused = rrf_fuse(dense, bm25, k=60, top_n=3)
        self.assertEqual([hit.parent_chunk_id for hit in fused], ["p2", "p1", "p3"])
        self.assertAlmostEqual(fused[0].score, 1 / 62 + 1 / 61)


if __name__ == "__main__":
    unittest.main()
