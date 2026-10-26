"""Offline tests for the one-shot answer-checkpoint repair pass.

Every test copies synthetic checkpoints and a synthetic benchmark into a
temporary directory and injects a fake ``answer_fn``. No network, no model, no
``data/`` and no real answer-eval checkpoint is touched.
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
from eval.answer_repair import (  # noqa: E402
    make_repair_answer_fn,
    run_repair,
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
    {
        "chunk_id": "p3",
        "title": "Paper Three",
        "page_number": 7,
        "text": "Gamma evidence sentence. Further context follows here.",
    },
]

QUESTIONS = [
    {
        "question_id": "q1",
        "question_zh": "问题一",
        "gold_points": [
            {"requirement": "alpha", "quote": "Alpha evidence sentence."}
        ],
    },
    {
        "question_id": "q2",
        "question_zh": "问题二",
        "gold_points": [
            {"requirement": "beta", "quote": "Beta evidence sentence."}
        ],
    },
    {
        "question_id": "q3",
        "question_zh": "问题三",
        "gold_points": [],
    },
    {
        "question_id": "q4",
        "question_zh": "问题四",
        "gold_points": [
            {"requirement": "gamma", "quote": "Gamma evidence sentence."}
        ],
    },
    {
        "question_id": "q5",
        "question_zh": "问题五",
        "gold_points": [
            {"requirement": "alpha", "quote": "Alpha evidence sentence."}
        ],
    },
]

QRELS = [
    {"question_id": "q1", "gold_parent_ids": ["p1"]},
    {"question_id": "q2", "gold_parent_ids": ["p2"]},
    {"question_id": "q3", "gold_parent_ids": []},
    {"question_id": "q4", "gold_parent_ids": ["p3"]},
    {"question_id": "q5", "gold_parent_ids": ["p1"]},
]

EVIDENCE_P1 = [
    {
        "id": "E1",
        "parent_chunk_id": "p1",
        "title": "Paper One",
        "page_number": 3,
    },
    {
        "id": "E2",
        "parent_chunk_id": "p2",
        "title": "Paper Two",
        "page_number": 5,
    },
]

EVIDENCE_P2 = [
    {
        "id": "E1",
        "parent_chunk_id": "p2",
        "title": "Paper Two",
        "page_number": 5,
    }
]

INVALID_RAW_Q1 = (
    '{"status":"answered","points":[{"text":"x",'
    '"citations":[{"id":"E1","quote":"not in the evidence at all"}]}],'
    '"limitations":[]}'
)
INVALID_RAW_Q2 = (
    '{"status":"answered","points":[{"text":"y",'
    '"citations":[{"id":"E1","quote":"also not in the evidence"}]}],'
    '"limitations":[]}'
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def valid_answer(
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


def checkpoint(
    *,
    question_id: str,
    status: str = "ok",
    answer_status: str = "invalid",
    error: str | None = "E1 的引文不在原文中，回答未通过校验。",
    raw_response: object | None = INVALID_RAW_Q1,
    evidence: list[dict] | None = None,
    retrieved: list[str] | None = None,
    gold: list[str] | None = None,
    tokens: int = 100,
    latency_ms: float = 50.0,
    usage: dict | None = None,
    attempt: int = 1,
    metrics: dict | None = None,
) -> dict:
    evidence = list(evidence or [])
    retrieved = list(retrieved if retrieved is not None else [
        unit["parent_chunk_id"] for unit in evidence
    ])
    gold = list(gold or [])
    zero_metrics = {
        "schema_success": 0.0,
        "quote_integrity": 0.0,
        "cited_gold_parent_recall": 0.0,
        "gold_point_citation_recall": 0.0,
    }
    return {
        "question_id": question_id,
        "status": status,
        "answer_status": answer_status,
        "error": error,
        "model": "old-model",
        "raw_response": raw_response,
        "usage": usage if usage is not None else {"total_tokens": tokens},
        "tokens": tokens,
        "latency_ms": latency_ms,
        "retrieved_parent_ids": retrieved,
        "evidence": evidence,
        "cited_evidence_ids": [],
        "cited_parent_ids": [],
        "gold_parent_ids": gold,
        "gold_point_count": 0,
        "metrics": metrics if metrics is not None else dict(zero_metrics),
        "attempt": attempt,
        "finished_at": "2026-01-01T00:00:00+00:00",
    }


class FakeRepairAnswer:
    """Route by the question in the original payload; record every call."""

    def __init__(
        self,
        responses: dict[str, object],
        *,
        tokens: int = 17,
        model: str = "fake-repair-model",
    ) -> None:
        self.responses = responses
        self.tokens = tokens
        self.calls: list[list[dict]] = []
        self.last_usage: dict | None = None
        self.model = model

    def __call__(self, messages: list[dict]) -> str | dict:
        # messages[1] is always the original payload, even in repair mode.
        payload = json.loads(messages[1]["content"])
        self.calls.append(messages)
        value = self.responses[payload["question"]]
        if isinstance(value, BaseException):
            raise value
        self.last_usage = {"total_tokens": self.tokens}
        return value


class AnswerRepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path(tempfile.mkdtemp(prefix="answer_repair_"))
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        self.benchmark = self.base / "benchmark"
        self.eval_dir = self.base / "answer_eval"
        write_jsonl(self.benchmark / "questions.jsonl", QUESTIONS)
        write_jsonl(self.benchmark / "qrels.jsonl", QRELS)
        write_json(
            self.benchmark / "index" / "corpus_clean.json",
            {"schema_version": 1, "chunks": CHUNKS},
        )

        self.checkpoints = {
            "q1": checkpoint(
                question_id="q1",
                error="E1 的引文不在原文中，回答未通过校验。",
                raw_response=INVALID_RAW_Q1,
                evidence=EVIDENCE_P1,
                gold=["p1"],
                tokens=100,
                latency_ms=50.0,
            ),
            "q2": checkpoint(
                question_id="q2",
                error="引用原句长度无效。",
                raw_response=INVALID_RAW_Q2,
                evidence=EVIDENCE_P2,
                gold=["p2"],
                tokens=200,
                latency_ms=60.0,
            ),
            "q3": checkpoint(
                question_id="q3",
                status="failed",
                error="retrieval run unavailable for q3: 重排序结果数量不完整。",
                raw_response=None,
                evidence=[],
                retrieved=[],
                gold=[],
                tokens=0,
                latency_ms=0.0,
            ),
            "q4": checkpoint(
                question_id="q4",
                status="ok",
                answer_status="answered",
                error=None,
                raw_response=json.dumps(
                    valid_answer("Gamma evidence sentence.", "E1"),
                    ensure_ascii=False,
                ),
                evidence=[
                    {
                        "id": "E1",
                        "parent_chunk_id": "p3",
                        "title": "Paper Three",
                        "page_number": 7,
                    }
                ],
                gold=["p3"],
                tokens=30,
                latency_ms=10.0,
                metrics={
                    "schema_success": 1.0,
                    "quote_integrity": 1.0,
                    "cited_gold_parent_recall": 1.0,
                    "gold_point_citation_recall": 1.0,
                },
            ),
            "q5": checkpoint(
                question_id="q5",
                status="failed",
                error="APIConnectionError: Connection error.",
                raw_response=None,
                evidence=EVIDENCE_P1,
                gold=["p1"],
                tokens=0,
                latency_ms=0.0,
            ),
        }
        for qid, record in self.checkpoints.items():
            write_json(self.eval_dir / f"{qid}.json", record)
        write_json(self.eval_dir / "aggregate.json", {"version": "test"})

    # -- helpers --------------------------------------------------------
    def read_checkpoint(self, qid: str) -> dict:
        return json.loads(
            (self.eval_dir / f"{qid}.json").read_text(encoding="utf-8")
        )

    def run_repair(self, responses=None, *, ids=None, dry_run=False, tokens=17):
        answer = None
        if not dry_run:
            answer = FakeRepairAnswer(responses or {}, tokens=tokens)
        report = run_repair(
            self.eval_dir,
            self.benchmark,
            answer_fn=answer,
            ids=ids,
            dry_run=dry_run,
        )
        return report, answer

    # -- valid repair ---------------------------------------------------
    def test_valid_repair_updates_only_that_checkpoint(self) -> None:
        report, answer = self.run_repair(
            {"问题一": valid_answer()}, ids=["q1"]
        )

        self.assertEqual(report["counts"]["repaired"], 1)
        self.assertEqual(report["counts"]["still_invalid"], 0)
        self.assertEqual(report["repaired"][0]["question_id"], "q1")
        self.assertEqual(len(answer.calls), 1)

        record = self.read_checkpoint("q1")
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["answer_status"], "answered")
        self.assertIsNone(record["error"])
        self.assertEqual(record["metrics"]["schema_success"], 1.0)
        self.assertEqual(record["metrics"]["quote_integrity"], 1.0)
        self.assertEqual(record["metrics"]["cited_gold_parent_recall"], 1.0)
        self.assertEqual(record["metrics"]["gold_point_citation_recall"], 1.0)
        self.assertEqual(record["cited_parent_ids"], ["p1"])
        self.assertEqual(record["attempt"], 2)
        self.assertEqual(
            record["previous_errors"],
            ["E1 的引文不在原文中，回答未通过校验。"],
        )
        self.assertIn("repaired_at", record)
        self.assertEqual(record["repair"]["mode"], "repair")
        self.assertEqual(record["repair"]["previous_tokens"], 100)
        self.assertEqual(record["model"], "fake-repair-model")

        # The original payload is rebuilt exactly, and the repair message
        # carries the previous error and previous raw response.
        messages = answer.calls[0]
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0], {"role": "system", "content": PROMPT})
        self.assertEqual(
            json.loads(messages[1]["content"]),
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
        repair_text = messages[2]["content"]
        self.assertIn("E1 的引文不在原文中", repair_text)
        self.assertIn(INVALID_RAW_Q1, repair_text)
        self.assertIn("10～800", repair_text)

        # Other checkpoints are byte-for-byte untouched.
        untouched = self.read_checkpoint("q2")
        self.assertEqual(untouched, self.checkpoints["q2"])

    def test_network_only_failure_retries_original_messages(self) -> None:
        report, answer = self.run_repair(
            {"问题五": valid_answer()}, ids=["q5"]
        )
        self.assertEqual(report["counts"]["repaired"], 1)
        self.assertEqual(len(answer.calls), 1)
        messages = answer.calls[0]
        self.assertEqual(len(messages), 2)  # original messages only, no repair text
        self.assertEqual(
            json.loads(messages[1]["content"])["question"], "问题五"
        )

        record = self.read_checkpoint("q5")
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["answer_status"], "answered")
        self.assertEqual(record["attempt"], 2)
        self.assertEqual(record["repair"]["mode"], "retry_original")
        self.assertEqual(record["tokens"], 17)

    # -- still invalid --------------------------------------------------
    def test_still_invalid_is_preserved_and_recorded(self) -> None:
        fabricated = valid_answer("This sentence is not in the evidence.")
        report, _ = self.run_repair({"问题二": fabricated}, ids=["q2"])

        self.assertEqual(report["counts"]["repaired"], 0)
        self.assertEqual(report["counts"]["still_invalid"], 1)
        self.assertEqual(report["still_invalid"][0]["question_id"], "q2")

        record = self.read_checkpoint("q2")
        # The original failure is not hidden or overwritten.
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["answer_status"], "invalid")
        self.assertEqual(record["error"], "引用原句长度无效。")
        self.assertEqual(record["raw_response"], INVALID_RAW_Q2)
        self.assertEqual(record["tokens"], 200)
        self.assertEqual(record["metrics"]["schema_success"], 0.0)
        # The failed repair attempt is visible.
        attempt = record["repaired_attempt"]
        self.assertEqual(attempt["attempt"], 2)
        self.assertEqual(attempt["answer_status"], "invalid")
        self.assertTrue(attempt["error"])
        self.assertIn("引文", attempt["error"])
        self.assertEqual(attempt["tokens"], 17)
        self.assertEqual(attempt["mode"], "repair")

    # -- skips ----------------------------------------------------------
    def test_no_evidence_is_skipped_never_repaired(self) -> None:
        report, answer = self.run_repair({"问题三": valid_answer()}, ids=["q3"])

        self.assertEqual(answer.calls, [])
        self.assertEqual(report["counts"]["repaired"], 0)
        self.assertEqual(report["counts"]["skipped"], 1)
        skipped = report["skipped"][0]
        self.assertEqual(skipped["question_id"], "q3")
        self.assertIn("no usable evidence", skipped["reason"])
        self.assertEqual(self.read_checkpoint("q3"), self.checkpoints["q3"])

    def test_requested_missing_checkpoint_is_reported_skipped(self) -> None:
        report, answer = self.run_repair({"问题一": valid_answer()}, ids=["q999"])
        self.assertEqual(answer.calls, [])
        self.assertEqual(report["counts"]["repaired"], 0)
        self.assertEqual(report["skipped"][0]["question_id"], "q999")
        self.assertIn("not found", report["skipped"][0]["reason"])

    # -- already valid --------------------------------------------------
    def test_already_valid_checkpoint_is_a_noop(self) -> None:
        before = self.read_checkpoint("q4")
        report, answer = self.run_repair({"问题四": valid_answer()}, ids=["q4"])

        self.assertEqual(answer.calls, [])
        self.assertEqual(report["counts"]["repaired"], 0)
        self.assertEqual(report["counts"]["already_valid"], 1)
        self.assertEqual(report["already_valid"][0]["question_id"], "q4")
        self.assertEqual(self.read_checkpoint("q4"), before)

    def test_full_scan_skips_valid_and_failed_without_evidence(self) -> None:
        report, answer = self.run_repair(
            {
                "问题一": valid_answer(),
                "问题二": valid_answer("This sentence is not in the evidence."),
                "问题五": valid_answer(),
            }
        )
        self.assertEqual(
            sorted(item["question_id"] for item in report["repaired"]),
            ["q1", "q5"],
        )
        self.assertEqual(
            [item["question_id"] for item in report["still_invalid"]], ["q2"]
        )
        self.assertEqual(
            [item["question_id"] for item in report["skipped"]], ["q3"]
        )
        self.assertEqual(
            [item["question_id"] for item in report["already_valid"]], ["q4"]
        )
        self.assertEqual(len(answer.calls), 3)

    # -- dry run --------------------------------------------------------
    def test_dry_run_calls_nothing_and_writes_nothing(self) -> None:
        before_files = {
            path.name: path.read_bytes() for path in self.eval_dir.glob("*.json")
        }
        report, _ = self.run_repair(None, dry_run=True)

        after_files = {
            path.name: path.read_bytes() for path in self.eval_dir.glob("*.json")
        }
        self.assertEqual(before_files, after_files)
        self.assertFalse((self.eval_dir / "repair_report.json").exists())
        self.assertFalse((self.eval_dir / "repair_report.md").exists())
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["counts"]["repaired"], 0)
        self.assertEqual(
            sorted(item["question_id"] for item in report["would_repair"]),
            ["q1", "q2", "q5"],
        )
        self.assertEqual(
            [item["question_id"] for item in report["skipped"]], ["q3"]
        )

    # -- token / latency accumulation -----------------------------------
    def test_tokens_and_latency_accumulate(self) -> None:
        report, _ = self.run_repair(
            {"问题一": valid_answer()}, ids=["q1"], tokens=23
        )
        record = self.read_checkpoint("q1")
        self.assertEqual(record["tokens"], 100 + 23)
        self.assertEqual(record["usage"], {"total_tokens": 23})
        self.assertGreaterEqual(record["latency_ms"], 50.0)
        self.assertEqual(record["repair"]["previous_latency_ms"], 50.0)
        self.assertEqual(report["repaired"][0]["tokens_before"], 100)
        self.assertEqual(report["repaired"][0]["tokens_after"], 123)

    # -- report ---------------------------------------------------------
    def test_repair_report_written_json_and_markdown(self) -> None:
        self.run_repair(
            {
                "问题一": valid_answer(),
                "问题二": valid_answer("This sentence is not in the evidence."),
                "问题五": valid_answer(),
            }
        )
        json_path = self.eval_dir / "repair_report.json"
        md_path = self.eval_dir / "repair_report.md"
        self.assertTrue(json_path.is_file())
        self.assertTrue(md_path.is_file())

        report = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(report["version"], "answer_repair_v1")
        self.assertEqual(
            [item["question_id"] for item in report["repaired"]], ["q1", "q5"]
        )
        self.assertEqual(
            [item["question_id"] for item in report["still_invalid"]], ["q2"]
        )
        self.assertEqual(
            [item["question_id"] for item in report["skipped"]], ["q3"]
        )
        self.assertTrue(all("reason" in item for item in report["skipped"]))
        self.assertEqual(report["counts"]["selected"], 4)

        markdown = md_path.read_text(encoding="utf-8")
        self.assertIn("# Answer Repair Report", markdown)
        self.assertIn("## Repaired", markdown)
        self.assertIn("## Skipped (not repaired)", markdown)
        self.assertIn("## Still invalid", markdown)
        self.assertIn("q1", markdown)
        self.assertIn("q2", markdown)
        self.assertIn("q3", markdown)

    # -- configuration --------------------------------------------------
    def test_repair_answer_fn_requires_configuration(self) -> None:
        with self.assertRaises(ValueError):
            make_repair_answer_fn(env={})
        with self.assertRaises(ValueError):
            make_repair_answer_fn(max_tokens=0, env={})


if __name__ == "__main__":
    unittest.main()
