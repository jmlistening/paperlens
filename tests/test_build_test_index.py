"""Tests for Task E2: build a held-out test index from PDFs.

Offline only. Two tiny PDFs are generated with PyMuPDF under ``tests/.tmp/``
and a fake embedder is injected, so no model is ever downloaded and no network
call is made. ``<repo>/data/rag`` is only read for hashing.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import tempfile
import textwrap
import unittest
import zlib
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import clean_corpus  # noqa: E402
import prepare_corpus  # noqa: E402
from eval.benchmark_v2.build_test_index import (  # noqa: E402
    TestIndexError,
    build_test_index,
)

TMP_BASE = Path(__file__).resolve().parent / ".tmp"
DATA_RAG = ROOT / "data" / "rag"

CHUNKER_CONFIG = {"chunk_chars": 300, "overlap_chars": 40, "max_tokens": 8}
FAKE_DIMENSION = 8

ALPHA_TEXT = "ZEBRA-ALPHA marker. " + " ".join(
    f"Alpha sentence {index} discusses dense retrieval and evidence ranking."
    for index in range(30)
)
BETA_TEXT = "ZEBRA-BETA marker. " + " ".join(
    f"Beta sentence {index} studies hybrid search and citation integrity."
    for index in range(30)
)


class FakeEmbedder:
    """Mirror of the SentenceTransformer surface used by the builder."""

    model_name = "fake/e5-test"
    model_revision = "fake-revision-1"
    max_seq_length = 8

    def __init__(self, dimension: int = FAKE_DIMENSION) -> None:
        self.dimension = dimension
        self.encode_calls = 0
        self.encoded_texts: list[str] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def tokenizer(self, text, truncation=False, add_special_tokens=True):
        tokens = text.split()
        if truncation and len(tokens) > self.max_seq_length:
            tokens = tokens[: self.max_seq_length]
        return {"input_ids": list(range(len(tokens)))}

    def encode(
        self,
        sentences,
        prompt="",
        batch_size=8,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
        **kwargs,
    ):
        self.encode_calls += 1
        self.encoded_texts.extend(sentences)
        rows = []
        for text in sentences:
            seed = zlib.crc32(text.encode("utf-8")) or 1
            rng = random.Random(seed)
            vector = np.array(
                [rng.uniform(-1.0, 1.0) for _ in range(self.dimension)],
                dtype="float32",
            )
            norm = float(np.linalg.norm(vector))
            rows.append(vector / norm if norm else vector)
        if not rows:
            return np.zeros((0, self.dimension), dtype="float32")
        return np.vstack(rows)


def write_pdf(path: Path, title: str, text: str) -> None:
    """Write a small one-page PDF whose text lands in several blocks."""
    import pymupdf

    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    y = 60.0
    for line in textwrap.wrap(text, width=92):
        page.insert_text((50, y), line, fontsize=9)
        y += 12.0
        if y > 800:
            break
    document.set_metadata({"title": title})
    document.save(str(path))
    document.close()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_rag_hashes() -> dict[str, str]:
    names = (
        "corpus.json",
        "corpus_clean.json",
        "vector_store.npz",
        "library_manifest.json",
        "extraction_preview.txt",
        "cleaning_preview.txt",
    )
    return {
        name: sha256_file(DATA_RAG / name)
        for name in names
        if (DATA_RAG / name).is_file()
    }


class BuildTestIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        TMP_BASE.mkdir(parents=True, exist_ok=True)
        self._tmp = tempfile.TemporaryDirectory(dir=TMP_BASE)
        self.tmp = Path(self._tmp.name)
        self.pdf_dir = self.tmp / "pdfs"
        self.pdf_dir.mkdir()
        self.alpha_pdf = self.pdf_dir / "2601.00001v1.pdf"
        self.beta_pdf = self.pdf_dir / "2101.00002v2.pdf"
        write_pdf(self.alpha_pdf, "Alpha Fixture Title", ALPHA_TEXT)
        write_pdf(self.beta_pdf, "Beta Fixture Title", BETA_TEXT)
        self.out_dir = self.tmp / "index"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def build(self, **overrides) -> dict:
        embedder = overrides.pop("embedder", None)
        if embedder is None:
            embedder = FakeEmbedder()
        params = {
            "pdf_paths": [self.alpha_pdf, self.beta_pdf],
            "out_dir": self.out_dir,
            "chunker_config": dict(CHUNKER_CONFIG),
            "embedder": embedder,
        }
        params.update(overrides)
        return build_test_index(**params)

    # -- outputs ---------------------------------------------------------

    def test_writes_required_outputs(self) -> None:
        manifest = self.build()
        for name in (
            "corpus.json",
            "corpus_clean.json",
            "vector_store.npz",
            "manifest.json",
        ):
            self.assertTrue((self.out_dir / name).is_file(), name)
        self.assertEqual(manifest["paper_count"], 2)

    def test_corpus_is_reparsed_from_pdfs(self) -> None:
        with (
            mock.patch.object(
                prepare_corpus, "extract_paper", wraps=prepare_corpus.extract_paper
            ) as extract,
            mock.patch.object(
                clean_corpus, "clean_text", wraps=clean_corpus.clean_text
            ) as clean_text,
            mock.patch.object(
                clean_corpus, "reading_groups", wraps=clean_corpus.reading_groups
            ) as reading_groups,
            mock.patch.object(
                clean_corpus, "column_cut", wraps=clean_corpus.column_cut
            ) as column_cut,
        ):
            self.build()

        self.assertEqual(extract.call_count, 2)
        for call in extract.call_args_list:
            self.assertIn("pdf_path", call.kwargs)
        self.assertGreater(clean_text.call_count, 0)
        self.assertGreater(reading_groups.call_count, 0)
        self.assertGreater(column_cut.call_count, 0)

        corpus = json.loads((self.out_dir / "corpus.json").read_text(encoding="utf-8"))
        clean = json.loads(
            (self.out_dir / "corpus_clean.json").read_text(encoding="utf-8")
        )
        for payload in (corpus, clean):
            text = " ".join(chunk["text"] for chunk in payload["chunks"])
            self.assertIn("ZEBRA-ALPHA", text)
            self.assertIn("ZEBRA-BETA", text)
        self.assertEqual(
            {paper["arxiv_id"] for paper in corpus["papers"]},
            {"2601.00001v1", "2101.00002v2"},
        )
        # The source text must not come from prepared_v1 pages: every chunk
        # carries the PDF hash of its own re-parsed PDF.
        for chunk in clean["chunks"]:
            expected = sha256_file(self.pdf_dir / f"{chunk['arxiv_id']}.pdf")
            self.assertEqual(chunk["pdf_sha256"], expected)

    def test_parent_child_offsets_are_preserved(self) -> None:
        self.build()
        clean = json.loads(
            (self.out_dir / "corpus_clean.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (self.out_dir / "manifest.json").read_text(encoding="utf-8")
        )
        with np.load(self.out_dir / "vector_store.npz", allow_pickle=False) as saved:
            metadata = json.loads(saved["metadata"].item())

        parents = {chunk["chunk_id"]: chunk for chunk in clean["chunks"]}
        units = metadata["units"]
        self.assertEqual(len(units), manifest["vectors"]["count"])
        self.assertGreater(len(units), len(parents))

        children_per_parent: dict[str, int] = {}
        for unit in units:
            parent = parents[unit["parent_chunk_id"]]
            start = unit["char_start"] - parent["char_start"]
            end = unit["char_end"] - parent["char_start"]
            self.assertGreaterEqual(start, 0)
            self.assertLess(start, end)
            self.assertLessEqual(end, len(parent["text"]))
            self.assertEqual(parent["text"][start:end], unit["text"])
            self.assertEqual(unit["arxiv_id"], parent["arxiv_id"])
            self.assertEqual(unit["page_number"], parent["page_number"])
            self.assertTrue(unit["chunk_id"].startswith(parent["chunk_id"] + ":v"))
            children_per_parent[parent["chunk_id"]] = (
                children_per_parent.get(parent["chunk_id"], 0) + 1
            )

        self.assertTrue(
            any(count > 1 for count in children_per_parent.values()),
            "expected at least one parent split into multiple child units",
        )

    def test_manifest_records_hashes_and_counts(self) -> None:
        embedder = FakeEmbedder()
        manifest = self.build(embedder=embedder)

        for key in (
            "schema_version",
            "created_at",
            "paper_count",
            "paper_ids",
            "families",
            "papers",
            "chunker",
            "chunker_sha256",
            "model",
            "corpus",
            "vectors",
            "index_sha256",
        ):
            self.assertIn(key, manifest)

        self.assertEqual(manifest["paper_ids"], ["2601.00001v1", "2101.00002v2"])
        self.assertEqual(manifest["families"], ["2601.00001", "2101.00002"])

        by_id = {paper["paper_id"]: paper for paper in manifest["papers"]}
        self.assertEqual(
            by_id["2601.00001v1"]["pdf_sha256"], sha256_file(self.alpha_pdf)
        )
        self.assertEqual(
            by_id["2101.00002v2"]["pdf_sha256"], sha256_file(self.beta_pdf)
        )
        self.assertEqual(by_id["2601.00001v1"]["family"], "2601.00001")
        self.assertEqual(by_id["2601.00001v1"]["title"], "Alpha Fixture Title")

        self.assertEqual(manifest["chunker"]["chunk_chars"], 300)
        self.assertEqual(manifest["chunker"]["overlap_chars"], 40)
        self.assertEqual(manifest["chunker"]["max_tokens"], 8)
        self.assertEqual(len(manifest["chunker_sha256"]), 64)

        self.assertEqual(manifest["model"]["name"], "fake/e5-test")
        self.assertEqual(manifest["model"]["revision"], "fake-revision-1")
        self.assertEqual(manifest["model"]["dimension"], FAKE_DIMENSION)
        self.assertEqual(manifest["vectors"]["dimension"], FAKE_DIMENSION)
        self.assertEqual(manifest["vectors"]["index_type"], "IndexFlatIP")
        self.assertGreater(manifest["vectors"]["count"], 0)

        for value in (
            manifest["index_sha256"],
            manifest["corpus"]["corpus_sha256"],
            manifest["corpus"]["corpus_clean_sha256"],
        ):
            self.assertRegex(value, r"^[0-9a-f]{64}$")

        corpus_bytes = (self.out_dir / "corpus.json").read_bytes()
        clean_bytes = (self.out_dir / "corpus_clean.json").read_bytes()
        self.assertEqual(
            manifest["corpus"]["corpus_sha256"],
            hashlib.sha256(corpus_bytes).hexdigest(),
        )
        self.assertEqual(
            manifest["corpus"]["corpus_clean_sha256"],
            hashlib.sha256(clean_bytes).hexdigest(),
        )

        self.assertEqual(embedder.encode_calls, 1)
        self.assertTrue(embedder.encoded_texts)
        for text in embedder.encoded_texts:
            self.assertTrue(text.startswith("passage: "))

    def test_output_is_compatible_with_load_store(self) -> None:
        from search_vector import load_store

        fake_root = self.tmp / "fake_project"
        out_dir = fake_root / "data" / "rag"
        manifest = self.build(out_dir=out_dir)

        index, metadata = load_store(fake_root)

        self.assertEqual(index.ntotal, manifest["vectors"]["count"])
        self.assertEqual(index.d, FAKE_DIMENSION)
        self.assertEqual(metadata["model_name"], "fake/e5-test")
        self.assertEqual(metadata["vector_count"], manifest["vectors"]["count"])
        self.assertEqual(len(metadata["units"]), manifest["vectors"]["count"])
        self.assertEqual(metadata["corpus_sha256"], manifest["corpus"]["corpus_clean_sha256"])
        self.assertEqual(
            metadata["source_corpus_sha256"], manifest["corpus"]["corpus_sha256"]
        )

    def test_index_hash_is_deterministic(self) -> None:
        first = self.build(out_dir=self.tmp / "index_a")
        second = self.build(out_dir=self.tmp / "index_b")
        self.assertEqual(first["index_sha256"], second["index_sha256"])
        # Parent chunk ids must be stable across rebuilds, or frozen qrels
        # would silently break when the index is rebuilt later.
        first_clean = json.loads(
            (self.tmp / "index_a" / "corpus_clean.json").read_text(encoding="utf-8")
        )
        second_clean = json.loads(
            (self.tmp / "index_b" / "corpus_clean.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [chunk["chunk_id"] for chunk in first_clean["chunks"]],
            [chunk["chunk_id"] for chunk in second_clean["chunks"]],
        )
        self.assertEqual(
            [paper["pdf_sha256"] for paper in first["papers"]],
            [paper["pdf_sha256"] for paper in second["papers"]],
        )

    # -- safety ----------------------------------------------------------

    def test_data_rag_is_never_modified(self) -> None:
        before = data_rag_hashes()
        if not before:
            self.skipTest("local data/rag fixture is not part of the public repo")
        self.build()
        self.assertEqual(data_rag_hashes(), before)

    def test_refuses_real_data_rag_as_output(self) -> None:
        before = data_rag_hashes()
        with self.assertRaises(ValueError) as ctx:
            self.build(out_dir=ROOT / "data" / "rag")
        self.assertIn("data/rag", str(ctx.exception).lower().replace("\\", "/"))
        self.assertEqual(data_rag_hashes(), before)
        self.assertFalse((DATA_RAG / ".library_transaction.json").exists())

    # -- errors ----------------------------------------------------------

    def test_unparseable_pdf_raises_naming_the_file(self) -> None:
        broken = self.pdf_dir / "broken.pdf"
        broken.write_bytes(b"this is not a pdf at all")
        with self.assertRaises(TestIndexError) as ctx:
            self.build(pdf_paths=[self.alpha_pdf, broken])
        self.assertIn("broken.pdf", str(ctx.exception))
        self.assertFalse((self.out_dir / "manifest.json").exists())

    def test_missing_pdf_raises_naming_the_file(self) -> None:
        missing = self.pdf_dir / "missing.pdf"
        with self.assertRaises(TestIndexError) as ctx:
            self.build(pdf_paths=[missing])
        self.assertIn("missing.pdf", str(ctx.exception))

    def test_pdf_without_text_raises_naming_the_file(self) -> None:
        import pymupdf

        blank = self.pdf_dir / "blank.pdf"
        document = pymupdf.open()
        document.new_page()
        document.save(str(blank))
        document.close()
        with self.assertRaises(TestIndexError) as ctx:
            self.build(pdf_paths=[blank])
        self.assertIn("blank.pdf", str(ctx.exception))

    def test_invalid_chunker_config_is_rejected(self) -> None:
        bad = dict(CHUNKER_CONFIG)
        bad["overlap_chars"] = bad["chunk_chars"]
        with self.assertRaises(ValueError):
            self.build(chunker_config=bad)

    def test_duplicate_paper_family_is_rejected(self) -> None:
        duplicate = self.pdf_dir / "2601.00001v2.pdf"
        write_pdf(duplicate, "Alpha Duplicate", ALPHA_TEXT)
        with self.assertRaises(TestIndexError) as ctx:
            self.build(pdf_paths=[self.alpha_pdf, duplicate])
        self.assertIn("family", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
