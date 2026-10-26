"""Tests for Task E5b: parent-chunk qrels and benchmark freeze scaffolding.

Offline only. All fixtures are written under ``tests/.tmp/``. The qrels builder
reads a fake clean corpus shaped like ``eval/benchmark_v2/index/corpus_clean.json``
and the freeze manifest hashes only the supplied artifacts.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.benchmark_v2.build_qrels import build_qrels  # noqa: E402
from eval.benchmark_v2.freeze import FreezeError, freeze_benchmark  # noqa: E402

TMP_BASE = Path(__file__).resolve().parent / ".tmp"
TMP_BASE.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def make_questions() -> list[dict]:
    return [
        {
            "question_id": "q001",
            "paper_ids": ["2607.00001v1"],
            "question_zh": "哪种检索方法被使用？",
            "query_en": "Which retrieval method is used?",
            "type": "factual",
            "answerability": "answerable",
            "gold_points": [
                {
                    "requirement": "name the retrieval method",
                    "quote": "sparse retrieval",
                    "pdf_page": 2,
                    "section": "Method",
                }
            ],
            "missing_aspects": [],
            "source_title": "Test Paper",
            "generator": {
                "model": "fake-model",
                "prompt_sha256": "a" * 64,
                "created_at": "2026-10-25T00:00:00+00:00",
            },
            "review": {"status": "pending", "reviewer": None, "timestamp": None},
        },
        {
            "question_id": "q002",
            "paper_ids": ["2607.00001v1"],
            "question_zh": "论文是否报告了 ImageNet 准确率？",
            "query_en": "Does the paper report ImageNet accuracy?",
            "type": "unanswerable",
            "answerability": "unanswerable",
            "gold_points": [
                {
                    "requirement": "this quote must not create qrels",
                    "quote": "sparse retrieval",
                    "pdf_page": 2,
                    "section": "Method",
                }
            ],
            "missing_aspects": ["no ImageNet experiment"],
            "source_title": "Test Paper",
            "generator": {
                "model": "fake-model",
                "prompt_sha256": "b" * 64,
                "created_at": "2026-10-25T00:00:00+00:00",
            },
            "review": {"status": "pending", "reviewer": None, "timestamp": None},
        },
    ]


def make_clean_corpus() -> dict:
    return {
        "schema_version": 2,
        "papers": [{"arxiv_id": "2607.00001v1", "title": "Test Paper"}],
        "chunks": [
            {
                "chunk_id": "2607.00001v1:clean:p1:g1:c1",
                "arxiv_id": "2607.00001v1",
                "page_number": 2,
                "text": "The method uses sparse retrieval and 12 tokens.",
            },
            {
                "chunk_id": "2607.00001v1:clean:p2:g2:c2",
                "arxiv_id": "2607.00001v1",
                "page_number": 2,
                "text": "Another parent chunk repeats sparse   retrieval here.",
            },
            {
                "chunk_id": "2607.00001v1:clean:p3:g3:c3",
                "arxiv_id": "2607.00001v1",
                "page_number": 3,
                "text": "Unrelated discussion about evaluation.",
            },
        ],
        "paper_count": 1,
        "chunk_count": 3,
    }


class BuildQrelsTests(unittest.TestCase):
    def test_quote_maps_to_all_parent_chunks_containing_it(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_BASE) as tmp:
            folder = Path(tmp)
            questions_path = folder / "questions.jsonl"
            corpus_path = folder / "corpus_clean.json"
            out_path = folder / "qrels.jsonl"
            write_jsonl(questions_path, make_questions())
            write_json(corpus_path, make_clean_corpus())

            summary = build_qrels(questions_path, corpus_path, out_path)

            rows = read_jsonl(out_path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                rows[0]["gold_parent_ids"],
                [
                    "2607.00001v1:clean:p1:g1:c1",
                    "2607.00001v1:clean:p2:g2:c2",
                ],
            )
            self.assertEqual(rows[0]["answerability"], "answerable")
            self.assertEqual(rows[0]["missing_aspects"], [])
            self.assertEqual(summary["question_count"], 2)
            self.assertEqual(summary["with_gold"], 1)

    def test_unanswerable_question_returns_empty_gold_list(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_BASE) as tmp:
            folder = Path(tmp)
            questions_path = folder / "questions.jsonl"
            corpus_path = folder / "corpus_clean.json"
            out_path = folder / "qrels.jsonl"
            write_jsonl(questions_path, make_questions())
            write_json(corpus_path, make_clean_corpus())

            build_qrels(questions_path, corpus_path, out_path)

            rows = read_jsonl(out_path)
            unanswerable = next(row for row in rows if row["question_id"] == "q002")
            self.assertEqual(unanswerable["gold_parent_ids"], [])
            self.assertEqual(unanswerable["answerability"], "unanswerable")
            self.assertEqual(
                unanswerable["missing_aspects"], ["no ImageNet experiment"]
            )

    def test_qrels_records_have_exact_schema(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_BASE) as tmp:
            folder = Path(tmp)
            questions_path = folder / "questions.jsonl"
            corpus_path = folder / "corpus_clean.json"
            out_path = folder / "qrels.jsonl"
            write_jsonl(questions_path, make_questions())
            write_json(corpus_path, make_clean_corpus())

            build_qrels(questions_path, corpus_path, out_path)

            for row in read_jsonl(out_path):
                self.assertEqual(
                    set(row),
                    {
                        "question_id",
                        "gold_parent_ids",
                        "answerability",
                        "missing_aspects",
                    },
                )

    def test_unmatched_quote_is_reported_without_failing(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_BASE) as tmp:
            folder = Path(tmp)
            questions = make_questions()
            questions[0]["gold_points"][0]["quote"] = "quote absent from corpus"
            questions_path = folder / "questions.jsonl"
            corpus_path = folder / "corpus_clean.json"
            out_path = folder / "qrels.jsonl"
            write_jsonl(questions_path, questions)
            write_json(corpus_path, make_clean_corpus())

            summary = build_qrels(questions_path, corpus_path, out_path)

            rows = read_jsonl(out_path)
            self.assertEqual(rows[0]["gold_parent_ids"], [])
            self.assertEqual(summary["with_gold"], 0)
            self.assertEqual(len(summary["unmatched_quotes"]), 1)
            self.assertEqual(
                summary["unmatched_quotes"][0]["question_id"], "q001"
            )

    def test_duplicate_chunk_ids_are_deduplicated_and_sorted(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_BASE) as tmp:
            folder = Path(tmp)
            questions = make_questions()[:1]
            questions[0]["gold_points"].append(
                {
                    "requirement": "same quote again",
                    "quote": "sparse retrieval",
                    "pdf_page": 2,
                    "section": "Method",
                }
            )
            questions_path = folder / "questions.jsonl"
            corpus_path = folder / "corpus_clean.json"
            out_path = folder / "qrels.jsonl"
            write_jsonl(questions_path, questions)
            write_json(corpus_path, make_clean_corpus())

            build_qrels(questions_path, corpus_path, out_path)

            row = read_jsonl(out_path)[0]
            self.assertEqual(row["gold_parent_ids"], sorted(set(row["gold_parent_ids"])))
            self.assertEqual(len(row["gold_parent_ids"]), 2)


class FreezeTests(unittest.TestCase):
    def _fixture(self, folder: Path) -> tuple[Path, Path, Path, Path]:
        questions = make_questions()
        qrels = [
            {
                "question_id": "q001",
                "gold_parent_ids": ["c1"],
                "answerability": "answerable",
                "missing_aspects": [],
            },
            {
                "question_id": "q002",
                "gold_parent_ids": [],
                "answerability": "unanswerable",
                "missing_aspects": ["no ImageNet experiment"],
            },
        ]
        questions_path = folder / "questions.jsonl"
        qrels_path = folder / "qrels.jsonl"
        splits_path = folder / "splits.json"
        write_jsonl(questions_path, questions)
        write_jsonl(qrels_path, qrels)
        write_json(splits_path, {"test": [{"arxiv_id": "2607.00001v1"}]})
        return questions_path, qrels_path, splits_path, folder / "manifest_v1.json"

    def test_manifest_records_hashes_and_counts(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_BASE) as tmp:
            folder = Path(tmp)
            questions_path, qrels_path, splits_path, out_path = self._fixture(folder)

            manifest = freeze_benchmark(
                questions_path,
                qrels_path,
                out_path,
                extra_artifacts={"splits": splits_path},
            )

            self.assertTrue(out_path.exists())
            on_disk = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "frozen")
            self.assertEqual(manifest["schema_version"], 1)
            self.assertRegex(manifest["freeze_id"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                on_disk["artifacts"]["questions"]["sha256"],
                hashlib.sha256(questions_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                on_disk["artifacts"]["qrels"]["sha256"],
                hashlib.sha256(qrels_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                on_disk["artifacts"]["questions"]["count"], 2
            )
            self.assertEqual(on_disk["artifacts"]["qrels"]["count"], 2)
            self.assertEqual(
                on_disk["artifacts"]["questions"]["answerability"],
                {"answerable": 1, "unanswerable": 1},
            )
            self.assertEqual(
                on_disk["artifacts"]["splits"]["sha256"],
                hashlib.sha256(splits_path.read_bytes()).hexdigest(),
            )

    def test_directory_artifact_is_hashed(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_BASE) as tmp:
            folder = Path(tmp)
            questions_path, qrels_path, _, out_path = self._fixture(folder)
            pages_dir = folder / "pages"
            (pages_dir / "nested").mkdir(parents=True)
            (pages_dir / "a.jsonl").write_text('{"pdf_page": 1}\n', encoding="utf-8")
            (pages_dir / "nested" / "b.jsonl").write_text(
                '{"pdf_page": 2}\n', encoding="utf-8"
            )

            manifest = freeze_benchmark(
                questions_path,
                qrels_path,
                out_path,
                extra_artifacts={"pages_dir": pages_dir},
            )

            artifact = manifest["artifacts"]["pages_dir"]
            self.assertEqual(artifact["kind"], "directory")
            self.assertEqual(artifact["file_count"], 2)
            self.assertRegex(artifact["sha256"], r"^[0-9a-f]{64}$")

    def test_refreezing_unchanged_inputs_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_BASE) as tmp:
            folder = Path(tmp)
            questions_path, qrels_path, _, out_path = self._fixture(folder)

            first = freeze_benchmark(questions_path, qrels_path, out_path)
            second = freeze_benchmark(questions_path, qrels_path, out_path)

            self.assertEqual(first["freeze_id"], second["freeze_id"])
            self.assertTrue(second["already_frozen"])

    def test_changed_inputs_require_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_BASE) as tmp:
            folder = Path(tmp)
            questions_path, qrels_path, _, out_path = self._fixture(folder)
            freeze_benchmark(questions_path, qrels_path, out_path)

            with questions_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "question_id": "q003",
                            "answerability": "answerable",
                            "gold_points": [],
                            "missing_aspects": [],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            with self.assertRaises(FreezeError):
                freeze_benchmark(questions_path, qrels_path, out_path)

            rewritten = freeze_benchmark(
                questions_path, qrels_path, out_path, overwrite=True
            )
            self.assertEqual(rewritten["artifacts"]["questions"]["count"], 3)

    def test_missing_input_raises_file_not_found(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_BASE) as tmp:
            folder = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                freeze_benchmark(
                    folder / "missing.jsonl",
                    folder / "missing_qrels.jsonl",
                    folder / "manifest_v1.json",
                )


if __name__ == "__main__":
    unittest.main()
