"""Tests for Task E5: deterministic Eval-lite metrics.

All expected values below are hand-calculated. No model calls, no network, and
no reads or writes under ``data/``; fixtures live in ``tests/.tmp/`` only.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.metrics_v2.metrics import (  # noqa: E402
    all_gold_at_k,
    mrr_at_k,
    ndcg_at_k,
    quote_integrity,
    recall_at_k,
    schema_success,
)
from eval.metrics_v2.score_run import score_run  # noqa: E402

TMP_BASE = Path(__file__).resolve().parent / ".tmp"


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class RecallTests(unittest.TestCase):
    def test_recall_at_3_is_two_thirds(self) -> None:
        gold = {"a", "b", "c"}
        retrieved = ["x", "a", "b", "y", "c"]
        self.assertAlmostEqual(recall_at_k(gold, retrieved, 3), 2 / 3, places=12)
        self.assertAlmostEqual(recall_at_k(gold, retrieved, 5), 1.0, places=12)

    def test_recall_counts_only_top_k(self) -> None:
        gold = {"a", "b"}
        retrieved = ["x", "y", "a", "b"]
        self.assertEqual(recall_at_k(gold, retrieved, 2), 0.0)
        self.assertAlmostEqual(recall_at_k(gold, retrieved, 3), 0.5, places=12)

    def test_empty_gold_and_retrieved_are_zero(self) -> None:
        self.assertEqual(recall_at_k(set(), ["a"], 3), 0.0)
        self.assertEqual(recall_at_k({"a"}, [], 3), 0.0)
        self.assertEqual(recall_at_k({"a"}, ["a"], 0), 0.0)

    def test_negative_k_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            recall_at_k({"a"}, ["a"], -1)


class AllGoldTests(unittest.TestCase):
    def test_all_gold_requires_every_gold_chunk(self) -> None:
        self.assertEqual(all_gold_at_k({"a", "b"}, ["a", "b", "c"], 2), 1.0)
        self.assertEqual(all_gold_at_k({"a", "b"}, ["a", "c"], 3), 0.0)
        self.assertEqual(all_gold_at_k({"a", "b"}, ["a"], 3), 0.0)

    def test_all_gold_empty_inputs_are_zero(self) -> None:
        self.assertEqual(all_gold_at_k(set(), ["a"], 3), 0.0)
        self.assertEqual(all_gold_at_k({"a"}, [], 3), 0.0)
        self.assertEqual(all_gold_at_k({"a"}, ["a"], 0), 0.0)


class MrrTests(unittest.TestCase):
    def test_mrr_at_10_is_one_half_for_second_position_hit(self) -> None:
        gold = {"p5", "p9"}
        retrieved = ["p1", "p5", "p2", "p9", "p3"]
        self.assertEqual(mrr_at_k(gold, retrieved, 10), 0.5)

    def test_mrr_uses_first_relevant_rank_and_respects_k(self) -> None:
        gold = {"a"}
        self.assertEqual(mrr_at_k(gold, ["a", "b"], 10), 1.0)
        self.assertEqual(mrr_at_k(gold, ["b", "a"], 10), 0.5)
        self.assertEqual(mrr_at_k(gold, ["b", "a"], 1), 0.0)

    def test_mrr_empty_inputs_are_zero(self) -> None:
        self.assertEqual(mrr_at_k(set(), ["a"], 10), 0.0)
        self.assertEqual(mrr_at_k({"a"}, [], 10), 0.0)


class NdcgTests(unittest.TestCase):
    def test_ndcg_known_ranking_at_position_two(self) -> None:
        # Gold a has gain 2**2-1 = 3. It appears at rank 2:
        #   DCG  = 3 / log2(3)
        #   IDCG = 3 / log2(2) = 3
        #   nDCG = 1 / log2(3)
        self.assertAlmostEqual(
            ndcg_at_k({"a"}, ["b", "a"], 2),
            1.0 / math.log2(3),
            places=12,
        )

    def test_ndcg_is_one_for_ideal_ranking(self) -> None:
        self.assertEqual(ndcg_at_k({"a", "b"}, ["a", "b", "c"], 10), 1.0)

    def test_ndcg_graded_relevance_uses_exponential_gain(self) -> None:
        # rel: a=2 (gain 3), b=1 (gain 1). Retrieved b then a.
        dcg = 1 / math.log2(2) + 3 / math.log2(3)
        idcg = 3 / math.log2(2) + 1 / math.log2(3)
        self.assertAlmostEqual(
            ndcg_at_k({"a"}, ["b", "a"], 2, rel={"a": 2, "b": 1}),
            dcg / idcg,
            places=12,
        )

    def test_duplicate_hits_do_not_inflate_ndcg(self) -> None:
        # Only the first occurrence of a hit earns gain; the repeat is zero.
        self.assertEqual(ndcg_at_k({"a"}, ["a", "a"], 2), 1.0)

    def test_ndcg_empty_inputs_are_zero(self) -> None:
        self.assertEqual(ndcg_at_k(set(), ["a"], 10), 0.0)
        self.assertEqual(ndcg_at_k({"a"}, [], 10), 0.0)
        self.assertEqual(ndcg_at_k({"a"}, ["a"], 0), 0.0)

    def test_ndcg_rejects_negative_relevance(self) -> None:
        with self.assertRaises(ValueError):
            ndcg_at_k({"a"}, ["a"], 10, rel={"a": -1})


class QuoteIntegrityTests(unittest.TestCase):
    def test_whitespace_normalized_quotes_are_integral(self) -> None:
        citations = [
            {
                "quote": "use  retrieval augmented\tgeneration",
                "source_text": "We use retrieval\n   augmented generation in this paper.",
            },
            {
                "quote": "in this paper",
                "source_text": "We use retrieval augmented generation in this paper.",
            },
        ]
        self.assertEqual(quote_integrity(citations), 1.0)

    def test_rewritten_quote_is_not_integral(self) -> None:
        citations = [
            {"quote": "We use retrieval augmented generation in the paper.",
             "source_text": "We use retrieval augmented generation in this paper."}
        ]
        self.assertEqual(quote_integrity(citations), 0.0)

    def test_case_changes_and_missing_sources_fail(self) -> None:
        self.assertEqual(
            quote_integrity([{"quote": "we use", "source_text": "We use"}]),
            0.0,
        )
        self.assertEqual(quote_integrity([{"quote": "anything at all"}]), 0.0)
        self.assertEqual(quote_integrity([]), 0.0)

    def test_mixed_citations_score_the_fraction(self) -> None:
        citations = [
            {"quote": "Alpha beta", "source_text": "Alpha beta gamma"},
            {"quote": "Alpha  beta", "source_text": "Alpha beta gamma"},
            {"quote": "Alpha delta", "source_text": "Alpha beta gamma"},
        ]
        self.assertAlmostEqual(quote_integrity(citations), 2 / 3, places=12)


class SchemaSuccessTests(unittest.TestCase):
    def test_all_valid_runs_succeed(self) -> None:
        runs = [
            {"status": "ok", "schema_ok": True},
            {"status": "answered", "schema_ok": True},
        ]
        self.assertEqual(schema_success(runs), 1.0)

    def test_failed_run_counts_as_zero(self) -> None:
        runs = [{"status": "ok", "schema_ok": True}, {"status": "failed"}]
        self.assertEqual(schema_success(runs), 0.5)
        self.assertEqual(schema_success([{"status": "failed"}]), 0.0)
        self.assertEqual(schema_success([]), 0.0)

    def test_schema_invalid_status_overrides_schema_ok(self) -> None:
        self.assertEqual(
            schema_success([{"status": "schema_invalid", "schema_ok": True}]),
            0.0,
        )

    def test_partial_and_insufficient_answers_are_schema_valid(self) -> None:
        runs = [{"status": "partial"}, {"status": "insufficient"}]
        self.assertEqual(schema_success(runs), 1.0)


class ScoreRunTests(unittest.TestCase):
    def setUp(self) -> None:
        TMP_BASE.mkdir(exist_ok=True)

    def test_score_run_reads_jsonl_fixtures_and_counts_failure(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_BASE) as tmp:
            root = Path(tmp)
            gold_path = root / "gold.jsonl"
            write_jsonl(
                gold_path,
                [
                    {"question_id": "q1", "gold_parent_ids": ["p1", "p2", "p3"],
                     "answerability": "answerable"},
                    {"question_id": "q2", "gold_parent_ids": ["p9"],
                     "answerability": "answerable"},
                    {"question_id": "q3", "gold_parent_ids": [],
                     "answerability": "unanswerable"},
                ],
            )
            run_dir = root / "run"
            run_dir.mkdir()
            write_jsonl(
                run_dir / "answers.jsonl",
                [
                    {
                        "question_id": "q1",
                        "status": "ok",
                        "schema_ok": True,
                        "retrieved_parent_ids": ["p1", "p2", "p3"],
                        "citations": [
                            {"quote": "Alpha beta",
                             "source_text": "Alpha beta gamma"},
                            {"quote": "Alpha delta",
                             "source_text": "Alpha beta gamma"},
                        ],
                    },
                    {
                        "question_id": "q2",
                        "status": "failed",
                        "retrieved_parent_ids": [],
                        "citations": [],
                    },
                    {
                        "question_id": "q3",
                        "status": "ok",
                        "schema_ok": True,
                        # Retrieved evidence for an unanswerable question is
                        # not part of any retrieval denominator.
                        "retrieved_parent_ids": ["p2"],
                        "citations": [],
                    },
                ],
            )

            result = score_run(run_dir, gold_path)

            self.assertEqual(result["n"], 3)
            self.assertEqual(result["n_retrieval"], 2)
            self.assertEqual(result["n_unanswerable"], 1)
            self.assertEqual(result["n_failed"], 1)
            self.assertEqual(result["n_schema_ok"], 2)
            self.assertEqual(result["n_citations"], 2)
            self.assertEqual(result["missing_run_ids"], [])
            self.assertEqual(result["extra_run_ids"], [])

            metrics = result["metrics"]
            # q1 scores 1.0 on every retrieval metric; failed q2 scores 0.
            self.assertAlmostEqual(metrics["recall@3"], 0.5, places=12)
            self.assertAlmostEqual(metrics["recall@6"], 0.5, places=12)
            self.assertAlmostEqual(metrics["recall@10"], 0.5, places=12)
            self.assertAlmostEqual(metrics["all_gold@6"], 0.5, places=12)
            self.assertAlmostEqual(metrics["mrr@10"], 0.5, places=12)
            self.assertAlmostEqual(metrics["ndcg@10"], 0.5, places=12)
            self.assertAlmostEqual(metrics["quote_integrity"], 0.5, places=12)
            self.assertAlmostEqual(metrics["schema_success"], 2 / 3, places=12)

            by_id = {row["question_id"]: row for row in result["per_question"]}
            self.assertTrue(by_id["q1"]["retrieval_scored"])
            self.assertEqual(by_id["q1"]["metrics"]["recall@3"], 1.0)
            self.assertEqual(by_id["q2"]["status"], "failed")
            self.assertFalse(by_id["q2"]["schema_ok"])
            self.assertEqual(by_id["q2"]["metrics"]["recall@3"], 0.0)
            self.assertFalse(by_id["q3"]["retrieval_scored"])
            self.assertIsNone(by_id["q3"]["metrics"])
            self.assertIsNone(by_id["q3"]["quote_integrity"])

    def test_score_run_reads_json_fixtures_and_missing_run_is_failure(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_BASE) as tmp:
            root = Path(tmp)
            gold_path = root / "gold.json"
            write_json(
                gold_path,
                {
                    "questions": [
                        {"question_id": "q1", "gold_parent_ids": ["a"]},
                        {"question_id": "q2", "gold_parent_ids": ["b"]},
                    ]
                },
            )
            run_dir = root / "run"
            run_dir.mkdir()
            # Filename supplies the question id when the record omits it.
            write_json(
                run_dir / "q1.json",
                {
                    "status": "ok",
                    "schema_ok": True,
                    "retrieved_parent_ids": ["a", "x"],
                    "citations": [],
                },
            )
            # Aggregate run files are not per-question records.
            write_json(run_dir / "run.json", {"status": "ok", "n": 1})

            result = score_run(run_dir, gold_path)

            self.assertEqual(result["n"], 2)
            self.assertEqual(result["n_retrieval"], 2)
            self.assertEqual(result["n_failed"], 1)
            self.assertEqual(result["missing_run_ids"], ["q2"])
            self.assertEqual(result["extra_run_ids"], [])
            self.assertAlmostEqual(
                result["metrics"]["recall@3"], 0.5, places=12
            )
            self.assertAlmostEqual(
                result["metrics"]["schema_success"], 0.5, places=12
            )
            self.assertEqual(result["metrics"]["quote_integrity"], 0.0)

            by_id = {row["question_id"]: row for row in result["per_question"]}
            self.assertEqual(by_id["q2"]["status"], "missing_run")
            self.assertFalse(by_id["q2"]["schema_ok"])
            self.assertEqual(by_id["q2"]["metrics"]["recall@3"], 0.0)

    def test_score_run_resolves_citations_nested_in_answer_points(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_BASE) as tmp:
            root = Path(tmp)
            gold_path = root / "gold.jsonl"
            write_jsonl(
                gold_path,
                [{"question_id": "q1", "gold_parent_ids": ["a"]}],
            )
            run_dir = root / "run"
            run_dir.mkdir()
            write_jsonl(
                run_dir / "answers.jsonl",
                [
                    {
                        "question_id": "q1",
                        "status": "partial",
                        "retrieved_parent_ids": ["a"],
                        "evidence": {"E1": {"text": "Needle in a haystack."}},
                        "answer": {
                            "status": "partial",
                            "points": [
                                {
                                    "text": "答案要点",
                                    "citations": [
                                        {"id": "E1", "quote": "Needle in a  haystack"},
                                    ],
                                }
                            ],
                        },
                    }
                ],
            )

            result = score_run(run_dir, gold_path)
            self.assertEqual(result["n_citations"], 1)
            self.assertEqual(result["metrics"]["quote_integrity"], 1.0)


if __name__ == "__main__":
    unittest.main()
