"""Offline tests for deterministic answer-level (citation-level) evaluation.

Every test here builds a synthetic benchmark index, a fake retrieval run and a
fake ``answer_fn``. No network, no model, no ``data/`` and no real benchmark
files are touched.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answer_question import PROMPT  # noqa: E402
from eval.answer_eval import (  # noqa: E402
    make_deepseek_answer_fn,
    resolve_retrieval_run,
    run_answer_eval,
)

CHUNKS = [
    {
        "chunk_id": "p1",
        "title": "Paper One",
        "page_number": 3,
        "text": "Alpha evidence sentence. Extra context follows here.",
    },
    {
        "chunk_id": "p2",
        "title": "Paper Two",
        "page_number": 5,
        "text": "Beta evidence sentence. More context follows there.",
    },
]

QUESTIONS = [
    {
        "question_id": "q1",
        "question_zh": "问题一",
        "gold_points": [
            {"requirement": "alpha", "quote": "Alpha evidence sentence."},
            {"requirement": "extra", "quote": "Extra context follows here."},
        ],
        "answerability": "answerable",
    },
    {
        "question_id": "q2",
        "question_zh": "问题二",
        "gold_points": [
            {"requirement": "beta", "quote": "Beta evidence sentence."},
        ],
        "answerability": "answerable",
    },
    {
        "question_id": "q3",
        "question_zh": "问题三",
        "gold_points": [],
        "answerability": "unanswerable",
    },
]

QRELS = [
    {"question_id": "q1", "gold_parent_ids": ["p1"], "answerability": "answerable"},
    {"question_id": "q2", "gold_parent_ids": ["p2"], "answerability": "answerable"},
    {"question_id": "q3", "gold_parent_ids": [], "answerability": "unanswerable"},
]

RETRIEVAL = {
    "q1": {"status": "ok", "retrieved_parent_ids": ["p1", "p2"]},
    "q2": {"status": "ok", "retrieved_parent_ids": ["p2", "p1"]},
    "q3": {"status": "ok", "retrieved_parent_ids": ["p1"]},
}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def valid_answer(
    *,
    quote: str = "Alpha evidence sentence.",
    evidence_id: str = "E1",
    status: str = "answered",
) -> dict:
    if status == "insufficient":
        return {"status": "insufficient", "points": [], "limitations": ["证据不足"]}
    return {
        "status": status,
        "points": [
            {
                "text": "证据支持的要点",
                "citations": [{"id": evidence_id, "quote": quote}],
            }
        ],
        "limitations": [] if status == "answered" else ["证据不完整"],
    }


class FakeAnswer:
    """Route a response by the question in the user message; record usage."""

    def __init__(self, responses: dict[str, object], *, tokens: int = 17) -> None:
        self.responses = responses
        self.tokens = tokens
        self.calls: list[list[dict]] = []
        self.last_usage: dict | None = None
        self.model = "fake-answer-model"

    def __call__(self, messages: list[dict]) -> str | dict:
        payload = json.loads(messages[-1]["content"])
        self.calls.append(messages)
        value = self.responses[payload["question"]]
        if isinstance(value, BaseException):
            raise value
        self.last_usage = {"total_tokens": self.tokens}
        return value


class AnswerEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path(tempfile.mkdtemp(prefix="answer_eval_"))
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        self.benchmark = self.base / "benchmark"
        self.run_dir = self.base / "runs" / "fake_system"
        write_jsonl(self.benchmark / "questions.jsonl", QUESTIONS)
        write_jsonl(self.benchmark / "qrels.jsonl", QRELS)
        write_json(
            self.benchmark / "index" / "corpus_clean.json",
            {"schema_version": 1, "chunks": CHUNKS},
        )
        for question_id, record in RETRIEVAL.items():
            write_json(
                self.run_dir / f"{question_id}.json",
                {"question_id": question_id, **record},
            )

    def run_eval(self, response, *, limit=None, out=None):
        answer = response if isinstance(response, FakeAnswer) else FakeAnswer(response)
        out_dir = out or (self.base / "out")
        aggregate = run_answer_eval(
            self.benchmark,
            self.base / "runs",
            out_dir,
            answer,
            retrieval_system="fake_system",
            limit=limit,
        )
        return aggregate, answer, out_dir

    # -- happy path -----------------------------------------------------
    def test_valid_answer_with_gold_citation(self) -> None:
        aggregate, answer, out_dir = self.run_eval(
            {"问题一": valid_answer()}, limit=1
        )

        record = json.loads((out_dir / "q1.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["answer_status"], "answered")
        self.assertEqual(record["metrics"]["schema_success"], 1.0)
        self.assertEqual(record["metrics"]["quote_integrity"], 1.0)
        self.assertEqual(record["metrics"]["cited_gold_parent_recall"], 1.0)
        self.assertEqual(record["metrics"]["gold_point_citation_recall"], 1.0)
        self.assertEqual(record["cited_parent_ids"], ["p1"])
        self.assertEqual(record["retrieved_parent_ids"], ["p1", "p2"])
        self.assertEqual(record["tokens"], 17)

        metrics = aggregate["metrics"]
        self.assertAlmostEqual(metrics["schema_success"]["mean"], 1.0)
        self.assertEqual(metrics["schema_success"]["n"], 1)
        self.assertEqual(aggregate["status_counts"]["answered"], 1)
        self.assertEqual(aggregate["n"], 1)
        self.assertEqual(aggregate["n_failed"], 0)
        self.assertEqual(aggregate["tokens"]["total"], 17)
        self.assertIn("p50", aggregate["latency_ms"])
        self.assertIn("p95", aggregate["latency_ms"])
        self.assertIn("NOT answer accuracy", aggregate["disclaimer"])
        self.assertIn("NOT answer accuracy", (out_dir / "report.md").read_text(encoding="utf-8"))

        # Same payload as answer_question.main, no gold leakage.
        messages = answer.calls[0]
        self.assertEqual(messages[0], {"role": "system", "content": PROMPT})
        payload = json.loads(messages[1]["content"])
        self.assertEqual(
            payload,
            {
                "question": "问题一",
                "evidence": [
                    {
                        "id": "E1",
                        "title": "Paper One",
                        "text": "Alpha evidence sentence. Extra context follows here.",
                    },
                    {
                        "id": "E2",
                        "title": "Paper Two",
                        "text": "Beta evidence sentence. More context follows there.",
                    },
                ],
            },
        )

    def test_answer_fn_may_return_a_json_string(self) -> None:
        aggregate, _, out_dir = self.run_eval(
            {"问题一": json.dumps(valid_answer())}, limit=1
        )
        record = json.loads((out_dir / "q1.json").read_text(encoding="utf-8"))
        self.assertEqual(record["metrics"]["schema_success"], 1.0)
        self.assertEqual(aggregate["metrics"]["schema_success"]["mean"], 1.0)

    # -- invalid outputs -------------------------------------------------
    def test_invalid_json_is_a_scored_failure(self) -> None:
        aggregate, _, out_dir = self.run_eval({"问题一": "{not json"}, limit=1)
        record = json.loads((out_dir / "q1.json").read_text(encoding="utf-8"))

        self.assertEqual(record["status"], "ok")  # the model call itself happened
        self.assertEqual(record["answer_status"], "invalid")
        self.assertEqual(record["raw_response"], "{not json")
        self.assertEqual(record["metrics"]["schema_success"], 0.0)
        self.assertEqual(record["metrics"]["quote_integrity"], 0.0)
        self.assertEqual(record["metrics"]["cited_gold_parent_recall"], 0.0)
        self.assertEqual(record["metrics"]["gold_point_citation_recall"], 0.0)
        self.assertEqual(aggregate["failures"], ["q1"])
        self.assertEqual(aggregate["n_failed"], 1)
        self.assertEqual(aggregate["status_counts"]["invalid"], 1)
        self.assertEqual(aggregate["metrics"]["schema_success"]["mean"], 0.0)

    def test_fabricated_quote_fails_validation(self) -> None:
        aggregate, _, out_dir = self.run_eval(
            {"问题一": valid_answer(quote="This sentence is not in the evidence.")},
            limit=1,
        )
        record = json.loads((out_dir / "q1.json").read_text(encoding="utf-8"))
        self.assertEqual(record["metrics"]["schema_success"], 0.0)
        self.assertEqual(record["metrics"]["quote_integrity"], 0.0)
        self.assertEqual(record["answer_status"], "invalid")
        self.assertIn("q1", aggregate["failures"])

    def test_answer_fn_exception_is_a_safe_failure(self) -> None:
        aggregate, _, out_dir = self.run_eval(
            {"问题一": RuntimeError("backend exploded")}, limit=1
        )
        record = json.loads((out_dir / "q1.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["answer_status"], "invalid")
        self.assertIsNone(record["raw_response"])
        self.assertIn("backend exploded", record["error"])
        self.assertEqual(record["metrics"]["schema_success"], 0.0)
        self.assertEqual(aggregate["failures"], ["q1"])

    # -- citation-level metrics -----------------------------------------
    def test_non_gold_citation_has_zero_cited_recall(self) -> None:
        _, _, out_dir = self.run_eval(
            {
                "问题一": valid_answer(
                    quote="Beta evidence sentence.", evidence_id="E2"
                )
            },
            limit=1,
        )
        record = json.loads((out_dir / "q1.json").read_text(encoding="utf-8"))
        self.assertEqual(record["metrics"]["schema_success"], 1.0)
        self.assertEqual(record["metrics"]["quote_integrity"], 1.0)
        self.assertEqual(record["metrics"]["cited_gold_parent_recall"], 0.0)
        self.assertEqual(record["metrics"]["gold_point_citation_recall"], 0.0)
        self.assertEqual(record["cited_parent_ids"], ["p2"])

    def test_empty_gold_metrics_are_not_applicable(self) -> None:
        aggregate, _, out_dir = self.run_eval(
            {
                "问题一": valid_answer(),
                "问题二": valid_answer(
                    quote="Beta evidence sentence.", evidence_id="E1"
                ),
                "问题三": valid_answer(status="insufficient"),
            },
            limit=3,
        )
        record = json.loads((out_dir / "q3.json").read_text(encoding="utf-8"))
        self.assertEqual(record["metrics"]["schema_success"], 1.0)
        self.assertIsNone(record["metrics"]["cited_gold_parent_recall"])
        self.assertIsNone(record["metrics"]["gold_point_citation_recall"])
        self.assertEqual(record["answer_status"], "insufficient")
        self.assertEqual(aggregate["status_counts"]["insufficient"], 1)
        self.assertEqual(aggregate["status_counts"]["answered"], 2)
        # Empty-gold questions are excluded from ratio metrics only.
        self.assertEqual(aggregate["metrics"]["cited_gold_parent_recall"]["n"], 2)
        self.assertEqual(aggregate["metrics"]["schema_success"]["n"], 3)

    def test_retrieval_failure_does_not_call_the_model(self) -> None:
        write_json(
            self.run_dir / "q1.json",
            {
                "question_id": "q1",
                "status": "failed",
                "error": "ValueError: rerank failed",
                "retrieved_parent_ids": [],
            },
        )
        aggregate, answer, out_dir = self.run_eval({"问题一": valid_answer()}, limit=1)
        record = json.loads((out_dir / "q1.json").read_text(encoding="utf-8"))
        self.assertEqual(answer.calls, [])
        self.assertEqual(record["status"], "failed")
        self.assertIn("retrieval", record["error"])
        self.assertEqual(record["metrics"]["schema_success"], 0.0)
        self.assertEqual(aggregate["failures"], ["q1"])

    # -- aggregate statistics -------------------------------------------
    def test_aggregate_bootstrap_ci_is_deterministic(self) -> None:
        responses = {"问题一": valid_answer(), "问题二": "{broken"}
        first, _, out1 = self.run_eval(responses, limit=2, out=self.base / "out1")
        second, _, _ = self.run_eval(responses, limit=2, out=self.base / "out2")

        interval = first["metrics"]["schema_success"]
        self.assertEqual(interval["mean"], 0.5)
        self.assertEqual(interval["n"], 2)
        self.assertGreaterEqual(interval["ci_low"], 0.0)
        self.assertLessEqual(interval["ci_high"], 1.0)
        self.assertEqual(interval, second["metrics"]["schema_success"])
        self.assertEqual(first["tokens"]["total"], 34)
        self.assertEqual(first["tokens"]["mean"], 17.0)
        self.assertEqual(first["latency_ms"]["n"], 2)
        self.assertEqual(first["failures"], ["q2"])

    def test_resume_skips_finished_checkpoints(self) -> None:
        responses = {
            "问题一": valid_answer(),
            "问题二": valid_answer(quote="Beta evidence sentence.", evidence_id="E1"),
            "问题三": valid_answer(status="insufficient"),
        }
        first, answer, out_dir = self.run_eval(responses, limit=3)
        self.assertEqual(len(answer.calls), 3)

        second, same_answer, _ = self.run_eval(responses, limit=3, out=out_dir)
        self.assertEqual(len(same_answer.calls), 0)
        self.assertEqual(second["n"], 3)
        self.assertEqual(second["n_failed"], 0)
        self.assertEqual(second["status_counts"], first["status_counts"])

    def test_resolve_retrieval_run_accepts_root_or_system_dir(self) -> None:
        run_dir, root = resolve_retrieval_run(self.base / "runs", "fake_system")
        self.assertEqual(run_dir, self.run_dir)
        self.assertEqual(root, self.base / "runs")

        direct_dir, direct_root = resolve_retrieval_run(self.run_dir, "fake_system")
        self.assertEqual(direct_dir, self.run_dir)
        self.assertEqual(direct_root, self.base / "runs")

    def test_default_answer_fn_requires_configuration(self) -> None:
        with self.assertRaises(ValueError):
            make_deepseek_answer_fn(env={})


if __name__ == "__main__":
    unittest.main()
