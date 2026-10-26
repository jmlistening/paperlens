"""Offline tests for the fixed-RAG vs bounded-agent A/B harness.

Every test uses synthetic questions, a fake corpus, fake retrieval records,
fake tools and fake answer/decision backends. No network, no model download and
no writes under ``data/``.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent_tools  # noqa: E402
from agent_policy import DECISION_PROMPT  # noqa: E402
from agent_tools import LocalTools  # noqa: E402
from answer_question import PROMPT as ANSWER_PROMPT  # noqa: E402
from eval.agent_eval import (  # noqa: E402
    EVIDENCE_BUDGET,
    _validate_agent_index,
    make_deepseek_answer_fn,
    run_ab_eval,
)
from prepare_corpus import sha256  # noqa: E402
from run_agent import AgentRun  # noqa: E402

TEXT1 = (
    "Alpha evidence sentence describes the retrieval pipeline in detail and "
    "stays well above the minimum fragment threshold used by the index."
)
TEXT2 = (
    "Beta evidence sentence describes the second stage of the pipeline and "
    "stays well above the minimum fragment threshold used by the index."
)
GOLD1 = "Alpha evidence sentence describes the retrieval pipeline"
GOLD2 = "Beta evidence sentence describes the second stage"
SECRET = "Secret gamma sentence that the fake tools never retrieve."


def make_unit(number: int, *, text: str | None = None, paper: str = "paper-a") -> dict:
    text = text or (
        f"Evidence passage {number} describes the retrieval pipeline in detail "
        "and stays well above the minimum fragment threshold used by the index."
    )
    return {
        "chunk_id": f"u{number}",
        "parent_chunk_id": f"p{number}",
        "arxiv_id": paper,
        "title": f"Paper {number}",
        "page_number": number,
        "group_number": 1,
        "char_start": 0,
        "char_end": len(text),
        "text": text,
        "pdf_path": f"data/pdfs/{paper}.pdf",
        "pdf_sha256": "a" * 64,
        "source_url": f"https://arxiv.org/abs/{paper}",
        "short_fragment": False,
    }


U1 = make_unit(1, text=TEXT1)
U2 = make_unit(2, text=TEXT2)
UNITS = [U1, U2] + [make_unit(n) for n in range(3, 9)]


def parent_chunk(unit: dict) -> dict:
    return {
        "chunk_id": unit["parent_chunk_id"],
        "arxiv_id": unit["arxiv_id"],
        "title": unit["title"],
        "page_number": unit["page_number"],
        "group_number": unit["group_number"],
        "char_start": unit["char_start"],
        "char_end": unit["char_end"],
        "text": unit["text"],
    }


QUESTIONS = [
    {
        "question_id": "q1",
        "question_zh": "问题一",
        "gold_points": [
            {"requirement": "alpha", "quote": GOLD1},
        ],
        "answerability": "answerable",
    },
    {
        "question_id": "q2",
        "question_zh": "问题二",
        "gold_points": [
            {"requirement": "beta", "quote": GOLD2},
        ],
        "answerability": "answerable",
    },
]

QRELS = [
    {"question_id": "q1", "gold_parent_ids": ["p1"], "answerability": "answerable"},
    {"question_id": "q2", "gold_parent_ids": ["p2"], "answerability": "answerable"},
]


def finish(ids=("C1",), assessment="sufficient", missing=None) -> dict:
    return {
        "assessment": assessment,
        "missing": list(missing or []),
        "selected_ids": list(ids),
        "action": "finish",
        "query": "",
        "reason": "现有证据足够，结束检索",
    }


def search(tool="search_bm25", query="beta pipeline", ids=("C1",)) -> dict:
    return {
        "assessment": "partial",
        "missing": ["缺少第二阶段"],
        "selected_ids": list(ids),
        "action": tool,
        "query": query,
        "reason": "补查第二阶段",
    }


def answer_ok(quote=GOLD1, evidence_id="E1") -> dict:
    return {
        "status": "answered",
        "points": [
            {
                "text": "证据支持的要点",
                "citations": [{"id": evidence_id, "quote": quote}],
            }
        ],
        "limitations": [],
    }


def answer_two() -> dict:
    return {
        "status": "answered",
        "points": [
            {
                "text": "第一阶段",
                "citations": [{"id": "E1", "quote": GOLD1}],
            },
            {
                "text": "第二阶段",
                "citations": [{"id": "E2", "quote": GOLD2}],
            },
        ],
        "limitations": [],
    }


def answer_insufficient() -> dict:
    return {"status": "insufficient", "points": [], "limitations": ["证据不足"]}


ANSWERS = {"问题一": answer_ok(GOLD1), "问题二": answer_ok(GOLD2)}


class FakeTools:
    """LocalTools-shaped fake returning queued hits in order."""

    def __init__(self, results, units=None):
        self.metadata = {"corpus_sha256": "c" * 64}
        self.canonical = {unit["chunk_id"]: dict(unit) for unit in (units or UNITS)}
        self.results = list(results)
        self.calls = []

    def check_current(self):
        return None

    def search(self, tool, query, top_k):
        self.calls.append((tool, query, top_k))
        value = self.results.pop(0)
        if isinstance(value, BaseException):
            raise value
        return [{**dict(hit), "rank": index + 1, "score": 1.0}
                for index, hit in enumerate(value)]


class FakeDecision:
    simulated = True
    model = "fake-decision"

    def __init__(self, values, tokens=5):
        self.values = list(values)
        self.tokens = tokens
        self.payloads = []
        self.last_usage = None

    def __call__(self, prompt, payload, max_tokens):
        assert prompt == DECISION_PROMPT
        self.payloads.append(payload)
        value = self.values.pop(0)
        if isinstance(value, BaseException):
            raise value
        self.last_usage = {"total_tokens": self.tokens}
        content = value if isinstance(value, str) else json.dumps(value)
        return {
            "model": self.model,
            "finish_reason": "stop",
            "content": content,
            "usage": {"total_tokens": self.tokens},
        }


class FakeAnswer:
    model = "fake-answer"

    def __init__(self, responses=None, tokens=17):
        self.responses = dict(responses or ANSWERS)
        self.tokens = tokens
        self.calls = []
        self.last_usage = None

    def __call__(self, messages):
        payload = json.loads(messages[-1]["content"])
        self.calls.append(messages)
        value = self.responses[payload["question"]]
        if isinstance(value, BaseException):
            raise value
        self.last_usage = {"total_tokens": self.tokens}
        return value


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class AgentEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path(tempfile.mkdtemp(prefix="agent_eval_"))
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        self.benchmark = self.base / "benchmark"
        self.run_root = self.base / "runs"
        self.run_dir = self.run_root / "fake_system"
        write_jsonl(self.benchmark / "questions.jsonl", QUESTIONS)
        write_jsonl(self.benchmark / "qrels.jsonl", QRELS)
        write_json(
            self.benchmark / "index" / "corpus_clean.json",
            {"schema_version": 1, "chunks": [parent_chunk(U1), parent_chunk(U2)]},
        )
        write_json(
            self.run_dir / "q1.json",
            {"question_id": "q1", "status": "ok", "retrieved_parent_ids": ["p1"]},
        )
        write_json(
            self.run_dir / "q2.json",
            {"question_id": "q2", "status": "ok", "retrieved_parent_ids": ["p2"]},
        )

    def run_eval(
        self,
        systems,
        *,
        decisions=None,
        answers=None,
        tools_results=None,
        tools=None,
        backend=None,
        limit=None,
        out=None,
        question_ids=None,
    ):
        answer = answers if isinstance(answers, FakeAnswer) else FakeAnswer(answers)
        backend = backend or FakeDecision(decisions or [finish()])
        tools = tools or FakeTools(tools_results or [[U1], [U2]])
        out_dir = out or (self.base / "out")
        summary = run_ab_eval(
            self.benchmark,
            self.run_root,
            out_dir,
            answer_fn=answer,
            systems=systems,
            decision_backend=backend,
            tools=tools,
            retrieval_system="fake_system",
            limit=limit,
            question_ids=question_ids,
        )
        return summary, answer, backend, tools, out_dir

    # -- fixed control ---------------------------------------------------
    def test_fixed_uses_precomputed_retrieval_and_the_shared_answer_prompt(self):
        summary, answer, backend, tools, out_dir = self.run_eval("fixed_rag", limit=2)

        self.assertEqual(tools.calls, [])
        self.assertEqual(backend.payloads, [])
        self.assertEqual(len(answer.calls), 2)

        record = json.loads((out_dir / "fixed_rag" / "q1.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["stop_reason"], "precomputed_retrieval")
        self.assertEqual(record["retrieved_parent_ids"], ["p1"])
        self.assertEqual(record["tool_calls"], 0)
        self.assertEqual(record["retrieval_rounds"], 0)
        self.assertEqual(record["metrics"], {
            "schema_success": 1.0,
            "quote_integrity": 1.0,
            "cited_gold_parent_recall": 1.0,
            "gold_point_citation_recall": 1.0,
        })

        messages = answer.calls[0]
        self.assertEqual(messages[0], {"role": "system", "content": ANSWER_PROMPT})
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["question"], "问题一")
        self.assertEqual([unit["id"] for unit in payload["evidence"]], ["E1"])
        self.assertEqual(payload["evidence"][0]["text"], TEXT1)

        self.assertEqual(summary["systems"]["fixed_rag"]["n"], 2)
        self.assertEqual(summary["systems"]["fixed_rag"]["n_failed"], 0)
        self.assertEqual(
            summary["systems"]["fixed_rag"]["metrics"]["schema_success"]["mean"], 1.0
        )

    def test_fixed_label_uses_actual_single_directory(self):
        summary = run_ab_eval(
            self.benchmark,
            self.run_dir,
            self.base / "label",
            answer_fn=FakeAnswer(),
            systems="fixed_rag",
            retrieval_system="wrong_label",
            limit=1,
        )
        self.assertEqual(summary["retrieval_system"], "fake_system")
        self.assertEqual(
            summary["systems"]["fixed_rag"]["retrieval_system"],
            "fake_system",
        )

    def test_benchmark_fingerprint_is_recorded(self):
        summary, _, _, _, out_dir = self.run_eval("fixed_rag", limit=1)

        fingerprint = summary["benchmark_fingerprint"]
        self.assertRegex(fingerprint, r"^[0-9a-f]{64}$")
        files = summary["benchmark_files"]
        self.assertEqual(
            files["questions_sha256"],
            sha256((self.benchmark / "questions.jsonl").read_bytes()),
        )
        self.assertEqual(
            files["qrels_sha256"],
            sha256((self.benchmark / "qrels.jsonl").read_bytes()),
        )
        self.assertEqual(
            files["corpus_clean_sha256"],
            sha256(
                (self.benchmark / "index" / "corpus_clean.json").read_bytes()
            ),
        )
        self.assertIn("vector_store_sha256", files)

        record = json.loads(
            (out_dir / "fixed_rag" / "q1.json").read_text(encoding="utf-8")
        )
        self.assertRegex(record["run_signature"], r"^[0-9a-f]{64}$")
        aggregate = json.loads(
            (out_dir / "fixed_rag" / "aggregate.json").read_text(encoding="utf-8")
        )
        self.assertEqual(aggregate["benchmark_fingerprint"], fingerprint)

    def test_agent_index_mismatch_is_rejected(self):
        other = self.base / "other_index"
        other.mkdir()
        (other / "corpus_clean.json").write_text(
            json.dumps({"chunks": []}), encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            _validate_agent_index(self.benchmark, other)
        _validate_agent_index(self.benchmark, self.benchmark / "index")

    def test_stale_usage_is_not_charged_to_a_failed_question(self):
        class FlakyAnswer(FakeAnswer):
            def __init__(self):
                super().__init__()
                self.attempts = 0

            def __call__(self, messages):
                self.attempts += 1
                if self.attempts == 2:
                    raise RuntimeError("second answer call fails")
                return super().__call__(messages)

        flaky = FlakyAnswer()
        _, _, _, _, out_dir = self.run_eval("fixed_rag", answers=flaky, limit=2)
        q1 = json.loads(
            (out_dir / "fixed_rag" / "q1.json").read_text(encoding="utf-8")
        )
        q2 = json.loads(
            (out_dir / "fixed_rag" / "q2.json").read_text(encoding="utf-8")
        )
        aggregate = json.loads(
            (out_dir / "fixed_rag" / "aggregate.json").read_text(encoding="utf-8")
        )
        self.assertEqual(q1["tokens"], 17)
        self.assertEqual(q2["status"], "failed")
        self.assertEqual(q2["stop_reason"], "answer_error")
        self.assertEqual(q2["tokens"], 0)
        self.assertEqual(aggregate["tokens"]["total"], 17)

    def test_missing_finish_reason_is_rejected(self):
        class NoFinishDecision(FakeDecision):
            def __call__(self, prompt, payload, max_tokens):
                raw = super().__call__(prompt, payload, max_tokens)
                raw.pop("finish_reason", None)
                return raw

        _, answer, _, _, out_dir = self.run_eval(
            "agent_single",
            backend=NoFinishDecision([finish()]),
            tools_results=[[U1]],
            limit=1,
        )
        record = json.loads(
            (out_dir / "agent_single" / "q1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["stop_reason"], "decision_error")
        self.assertEqual(record["tokens"], 5)
        self.assertEqual(answer.calls, [])

    def test_question_ids_filter_preserves_benchmark_order(self):
        summary, _, _, _, out_dir = self.run_eval(
            "fixed_rag", question_ids="q2", out=self.base / "ids"
        )
        self.assertEqual(summary["systems"]["fixed_rag"]["n"], 1)
        self.assertTrue((out_dir / "fixed_rag" / "q2.json").is_file())
        self.assertFalse((out_dir / "fixed_rag" / "q1.json").exists())

        ordered, _, _, _, _ = self.run_eval(
            "fixed_rag",
            question_ids="q2,q1",
            out=self.base / "ids_order",
        )
        self.assertEqual(
            [
                item["question_id"]
                for item in ordered["systems"]["fixed_rag"]["per_question"]
            ],
            ["q1", "q2"],
        )

        limited, _, _, _, limited_out = self.run_eval(
            "fixed_rag",
            question_ids="q2",
            limit=1,
            out=self.base / "ids_limit",
        )
        self.assertEqual(limited["systems"]["fixed_rag"]["n"], 1)
        self.assertTrue((limited_out / "fixed_rag" / "q2.json").is_file())
        self.assertFalse((limited_out / "fixed_rag" / "q1.json").exists())

        with self.assertRaises(ValueError):
            self.run_eval(
                "fixed_rag", question_ids="q999", out=self.base / "ids_bad"
            )

    def test_missing_qrels_is_rejected(self):
        write_jsonl(self.benchmark / "qrels.jsonl", [QRELS[0]])
        with self.assertRaises(ValueError):
            self.run_eval(
                "fixed_rag", limit=2, out=self.base / "missing_qrels"
            )

    # -- agent single ----------------------------------------------------
    def test_agent_single_one_decision_then_answer(self):
        summary, answer, backend, tools, out_dir = self.run_eval(
            "agent_single",
            decisions=[finish(), finish()],
            tools_results=[[U1], [U2]],
            limit=2,
        )
        record = json.loads((out_dir / "agent_single" / "q1.json").read_text(encoding="utf-8"))

        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["stop_reason"], "sufficient")
        self.assertEqual(record["tool_calls"], 1)
        self.assertEqual(record["retrieval_rounds"], 1)
        self.assertEqual(record["tokens"], 5 + 17)
        self.assertEqual(tools.calls[0], ("search_dense", "问题一", 20))
        self.assertEqual(record["retrieved_parent_ids"], ["p1"])
        self.assertEqual(record["metrics"]["schema_success"], 1.0)
        self.assertEqual(record["metrics"]["cited_gold_parent_recall"], 1.0)

        payload = json.loads(answer.calls[0][-1]["content"])
        self.assertEqual([unit["id"] for unit in payload["evidence"]], ["E1"])
        self.assertEqual(payload["evidence"][0]["text"], TEXT1)
        self.assertEqual(answer.calls[0][0], {"role": "system", "content": ANSWER_PROMPT})
        self.assertEqual(
            summary["systems"]["agent_single"]["tool_calls"]["total"], 2
        )
        self.assertEqual(summary["retrieval_system"], "local_tools")
        self.assertEqual(
            summary["systems"]["agent_single"]["retrieval_system"],
            "local_tools",
        )
        self.assertTrue(
            summary["systems"]["agent_single"]["agent_index"]["injected_tools"]
        )

    def test_harness_abstention_is_marked_and_excluded_from_schema_quote(self):
        _, answer, _, _, out_dir = self.run_eval(
            "agent_single",
            decisions=[
                finish(ids=("C1",)),
                finish(ids=(), assessment="insufficient", missing=["没有证据"]),
            ],
            tools_results=[[U1], [U2]],
            limit=2,
        )
        q1 = json.loads(
            (out_dir / "agent_single" / "q1.json").read_text(encoding="utf-8")
        )
        q2 = json.loads(
            (out_dir / "agent_single" / "q2.json").read_text(encoding="utf-8")
        )
        aggregate = json.loads(
            (out_dir / "agent_single" / "aggregate.json").read_text(encoding="utf-8")
        )
        self.assertEqual(q1["answer_source"], "model")
        self.assertEqual(q2["answer_source"], "harness_abstention")
        self.assertEqual(q2["status"], "ok")
        self.assertEqual(q2["answer_status"], "insufficient")
        self.assertIsNone(q2["metrics"]["schema_success"])
        self.assertIsNone(q2["metrics"]["quote_integrity"])
        self.assertIsNone(q2["answer_latency_ms"])
        self.assertEqual(len(answer.calls), 1)
        self.assertEqual(aggregate["n"], 2)
        self.assertEqual(aggregate["n_failed"], 0)
        self.assertEqual(aggregate["status_counts"]["answered"], 1)
        self.assertEqual(aggregate["status_counts"]["insufficient"], 1)
        self.assertEqual(aggregate["metrics"]["schema_success"]["n"], 1)
        self.assertEqual(aggregate["metrics"]["schema_success"]["mean"], 1.0)
        self.assertEqual(aggregate["answer_latency_ms"]["n"], 1)

    # -- agent multi -----------------------------------------------------
    def test_agent_multi_uses_bm25_and_two_rounds(self):
        summary, answer, backend, tools, out_dir = self.run_eval(
            "agent_multi",
            decisions=[search(), finish(ids=("C1", "C2"))],
            answers={"问题一": answer_two()},
            tools_results=[[U1], [U2]],
            limit=1,
        )
        record = json.loads((out_dir / "agent_multi" / "q1.json").read_text(encoding="utf-8"))

        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["stop_reason"], "sufficient")
        self.assertEqual(record["tool_calls"], 2)
        self.assertEqual(record["retrieval_rounds"], 2)
        self.assertEqual(record["tokens"], 5 + 5 + 17)
        self.assertEqual(tools.calls[0][:2], ("search_dense", "问题一"))
        self.assertEqual(tools.calls[1][:2], ("search_bm25", "beta pipeline"))
        self.assertEqual(record["retrieved_parent_ids"], ["p1", "p2"])
        self.assertEqual(record["cited_parent_ids"], ["p1", "p2"])

        payload = json.loads(answer.calls[0][-1]["content"])
        self.assertEqual([unit["id"] for unit in payload["evidence"]], ["E1", "E2"])
        self.assertEqual(payload["evidence"][1]["text"], TEXT2)
        self.assertEqual(
            summary["systems"]["agent_multi"]["retrieval_rounds"]["max"], 2
        )

    def test_agent_multi_allows_two_supplemental_searches(self):
        _, _, _, tools, out_dir = self.run_eval(
            "agent_multi",
            decisions=[
                search(query="beta pipeline"),
                search(ids=("C1",), query="gamma pipeline"),
                finish(ids=("C1", "C2", "C3")),
            ],
            answers={"问题一": answer_insufficient()},
            tools_results=[[U1], [U2], [UNITS[2]]],
            limit=1,
        )
        record = json.loads(
            (out_dir / "agent_multi" / "q1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["stop_reason"], "sufficient")
        self.assertEqual(record["tool_calls"], 3)
        self.assertEqual(record["retrieval_rounds"], 3)
        self.assertEqual(len(tools.calls), 3)

    def test_agent_multi_third_search_request_is_round_budget(self):
        _, _, _, tools, out_dir = self.run_eval(
            "agent_multi",
            decisions=[
                search(query="beta pipeline"),
                search(ids=("C1",), query="gamma pipeline"),
                search(ids=("C1",), query="delta pipeline"),
            ],
            answers={"问题一": answer_insufficient()},
            tools_results=[[U1], [U2], [UNITS[2]]],
            limit=1,
        )
        record = json.loads(
            (out_dir / "agent_multi" / "q1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["stop_reason"], "round_budget")
        self.assertEqual(record["tool_calls"], 3)
        self.assertEqual(record["retrieval_rounds"], 3)
        self.assertEqual(len(tools.calls), 3)

    # -- evidence budget -------------------------------------------------
    def test_fixed_and_agent_share_the_six_unit_evidence_cap(self):
        units = UNITS[:8]
        write_json(
            self.benchmark / "index" / "corpus_clean.json",
            {"schema_version": 1, "chunks": [parent_chunk(unit) for unit in units]},
        )
        write_json(
            self.run_dir / "q1.json",
            {
                "question_id": "q1",
                "status": "ok",
                "retrieved_parent_ids": [unit["parent_chunk_id"] for unit in units],
            },
        )

        _, fixed_answer, _, _, fixed_out = self.run_eval(
            "fixed_rag", limit=1, out=self.base / "fixed_cap"
        )
        fixed_payload = json.loads(fixed_answer.calls[0][-1]["content"])
        self.assertEqual(len(fixed_payload["evidence"]), EVIDENCE_BUDGET)
        self.assertEqual(
            [unit["id"] for unit in fixed_payload["evidence"]],
            [f"E{i}" for i in range(1, EVIDENCE_BUDGET + 1)],
        )

        six_ids = [f"C{i}" for i in range(1, EVIDENCE_BUDGET + 1)]
        _, agent_answer, _, _, agent_out = self.run_eval(
            "agent_single",
            decisions=[finish(ids=six_ids)],
            tools_results=[units],
            limit=1,
            out=self.base / "agent_cap",
        )
        agent_payload = json.loads(agent_answer.calls[0][-1]["content"])
        self.assertEqual(len(agent_payload["evidence"]), EVIDENCE_BUDGET)

        # A seventh selection is rejected before any answer call.
        _, extra_answer, _, _, extra_out = self.run_eval(
            "agent_single",
            decisions=[finish(ids=[f"C{i}" for i in range(1, 8)])],
            tools_results=[units],
            limit=1,
            out=self.base / "agent_over_cap",
        )
        record = json.loads(
            (extra_out / "agent_single" / "q1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["stop_reason"], "invalid_decision")
        self.assertEqual(record["metrics"]["schema_success"], 0.0)
        self.assertEqual(extra_answer.calls, [])

    # -- tool errors -----------------------------------------------------
    def test_tool_error_is_recorded_as_a_stop_reason(self):
        _, answer, _, tools, out_dir = self.run_eval(
            "agent_multi",
            decisions=[search()],
            answers={"问题一": answer_ok(GOLD1)},
            tools_results=[[U1], RuntimeError("offline tool failure")],
            limit=1,
        )
        record = json.loads((out_dir / "agent_multi" / "q1.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["stop_reason"], "tool_error")
        self.assertEqual(record["tool_calls"], 1)
        self.assertEqual(record["retrieval_rounds"], 1)
        self.assertEqual(record["retrieved_parent_ids"], ["p1"])
        self.assertTrue(any(item["kind"] == "tool_error" for item in record["trace"]))
        self.assertEqual(len(answer.calls), 1)

    def test_initial_tool_error_fails_without_an_answer_call(self):
        _, answer, _, _, out_dir = self.run_eval(
            "agent_single",
            decisions=[finish()],
            tools_results=[RuntimeError("offline initial failure")],
            limit=1,
        )
        record = json.loads((out_dir / "agent_single" / "q1.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["stop_reason"], "tool_error")
        self.assertEqual(record["metrics"]["schema_success"], 0.0)
        self.assertEqual(answer.calls, [])

    def test_index_change_is_recorded_as_a_failure(self):
        class ChangingTools(FakeTools):
            def __init__(self):
                super().__init__([[U1]])
                self.checks = 0

            def check_current(self):
                self.checks += 1
                if self.checks >= 2:
                    raise ValueError("index changed mid-question")

        tools = ChangingTools()
        _, answer, _, _, out_dir = self.run_eval(
            "agent_single",
            decisions=[finish()],
            tools=tools,
            limit=1,
        )
        record = json.loads(
            (out_dir / "agent_single" / "q1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["stop_reason"], "index_changed")
        self.assertEqual(answer.calls, [])

    def test_index_change_during_search_is_index_changed(self):
        class ChangedTools(FakeTools):
            def __init__(self):
                super().__init__([[U1]])
                self.changed = False

            def check_current(self):
                if self.changed:
                    raise ValueError("index changed")

            def search(self, tool, query, top_k):
                self.changed = True
                raise ValueError("corpus/index changed mid-search")

        tools = ChangedTools()
        _, answer, _, _, out_dir = self.run_eval(
            "agent_single",
            decisions=[finish()],
            tools=tools,
            limit=1,
        )
        record = json.loads(
            (out_dir / "agent_single" / "q1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["stop_reason"], "index_changed")
        self.assertEqual(answer.calls, [])

    # -- no gold leakage -------------------------------------------------
    def test_gold_data_never_enters_decision_or_answer_prompts(self):
        question = {
            "question_id": "q3",
            "question_zh": "问题三",
            "gold_points": [{"requirement": "secret", "quote": SECRET}],
            "answerability": "answerable",
        }
        qrels = {
            "question_id": "q3",
            "gold_parent_ids": ["psecret"],
            "answerability": "answerable",
        }
        write_jsonl(self.benchmark / "questions.jsonl", [question])
        write_jsonl(self.benchmark / "qrels.jsonl", [qrels])
        write_json(
            self.run_dir / "q3.json",
            {"question_id": "q3", "status": "ok", "retrieved_parent_ids": ["p1"]},
        )
        responses = {"问题三": answer_insufficient()}

        _, fixed_answer, _, _, _ = self.run_eval(
            "fixed_rag",
            answers=responses,
            limit=1,
            out=self.base / "leak_fixed",
        )
        fixed_text = json.dumps(fixed_answer.calls, ensure_ascii=False)
        self.assertNotIn(SECRET, fixed_text)
        self.assertNotIn("psecret", fixed_text)

        _, answer, backend, _, _ = self.run_eval(
            "agent_single",
            decisions=[finish(ids=("C1",))],
            answers=responses,
            tools_results=[[U1]],
            limit=1,
            out=self.base / "leak_agent",
        )
        decision_payload = backend.payloads[0]
        self.assertNotIn("gold_points", decision_payload)
        self.assertNotIn("gold_parent_ids", decision_payload)
        decision_text = json.dumps(decision_payload, ensure_ascii=False)
        self.assertNotIn(SECRET, decision_text)
        self.assertNotIn("psecret", decision_text)
        self.assertNotIn(SECRET, json.dumps(answer.calls, ensure_ascii=False))
        self.assertNotIn("gold_points", json.dumps(answer.calls, ensure_ascii=False))
        self.assertNotIn("gold_parent_ids", json.dumps(answer.calls, ensure_ascii=False))

    # -- outputs and resume ---------------------------------------------
    def test_checkpoints_aggregate_report_and_resume(self):
        summary, answer, backend, tools, out_dir = self.run_eval(
            "fixed_rag,agent_single",
            decisions=[finish(), finish()],
            tools_results=[[U1], [U2]],
            limit=2,
        )
        for system in ("fixed_rag", "agent_single"):
            for name in ("q1.json", "q2.json", "aggregate.json", "report.md"):
                self.assertTrue((out_dir / system / name).is_file(), f"{system}/{name}")
            aggregate = json.loads(
                (out_dir / system / "aggregate.json").read_text(encoding="utf-8")
            )
            self.assertEqual(aggregate["n"], 2)
            self.assertEqual(aggregate["n_failed"], 0)
            for key in ("schema_success", "quote_integrity",
                        "cited_gold_parent_recall", "gold_point_citation_recall"):
                self.assertEqual(aggregate["metrics"][key]["mean"], 1.0)
            self.assertIn("tool_calls", aggregate)
            self.assertIn("retrieval_rounds", aggregate)
            self.assertIn("tokens", aggregate)
            self.assertIn("latency_ms", aggregate)
            self.assertIn("NOT answer accuracy", (out_dir / system / "report.md").read_text(encoding="utf-8"))

        self.assertTrue((out_dir / "summary.json").is_file())
        self.assertTrue((out_dir / "report.md").is_file())
        self.assertEqual(set(summary["systems"]), {"fixed_rag", "agent_single"})
        self.assertIn("fixed_rag", (out_dir / "report.md").read_text(encoding="utf-8"))
        self.assertFalse((self.base / "data").exists())

        second_answer = FakeAnswer()
        second_backend = FakeDecision([finish(), finish()])
        second_tools = FakeTools([[U1], [U2]])
        second = run_ab_eval(
            self.benchmark,
            self.run_root,
            out_dir,
            answer_fn=second_answer,
            systems="fixed_rag,agent_single",
            decision_backend=second_backend,
            tools=second_tools,
            retrieval_system="fake_system",
            limit=2,
        )
        self.assertEqual(second_answer.calls, [])
        self.assertEqual(second_backend.payloads, [])
        self.assertEqual(second_tools.calls, [])
        self.assertEqual(second["systems"]["agent_single"]["n"], 2)

    def test_out_dir_must_differ_from_retrieval_run(self):
        with self.assertRaises(ValueError):
            self.run_eval("fixed_rag", limit=1, out=self.run_root)
        with self.assertRaises(ValueError):
            self.run_eval("fixed_rag", limit=1, out=self.run_dir)

    def test_resume_is_invalidated_by_a_model_change(self):
        _, first_answer, _, _, out_dir = self.run_eval("fixed_rag", limit=1)
        self.assertEqual(len(first_answer.calls), 1)

        other = FakeAnswer()
        other.model = "other-answer"
        second = run_ab_eval(
            self.benchmark,
            self.run_root,
            out_dir,
            answer_fn=other,
            systems="fixed_rag",
            retrieval_system="fake_system",
            limit=1,
        )
        self.assertEqual(len(other.calls), 1)
        record = json.loads(
            (out_dir / "fixed_rag" / "q1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(record["model"], "other-answer")
        self.assertEqual(
            second["systems"]["fixed_rag"]["answer_model"], "other-answer"
        )

    def test_resume_is_invalidated_by_retrieval_config_change(self):
        _, first_answer, _, _, out_dir = self.run_eval("agent_single", limit=1)
        self.assertEqual(len(first_answer.calls), 1)

        second_answer = FakeAnswer()
        second_backend = FakeDecision([finish()])
        second_tools = FakeTools([[U1]])
        with mock.patch("eval.agent_eval.DEFAULT_TOP_K", 19):
            second = run_ab_eval(
                self.benchmark,
                self.run_root,
                out_dir,
                answer_fn=second_answer,
                systems="agent_single",
                decision_backend=second_backend,
                tools=second_tools,
                retrieval_system="fake_system",
                limit=1,
            )
        self.assertEqual(len(second_answer.calls), 1)
        self.assertEqual(second_tools.calls[0][2], 19)
        self.assertEqual(second["systems"]["agent_single"]["n"], 1)

    def test_resume_is_invalidated_by_scoring_version_change(self):
        _, first_answer, _, _, out_dir = self.run_eval("fixed_rag", limit=1)
        self.assertEqual(len(first_answer.calls), 1)

        other = FakeAnswer()
        with mock.patch("eval.agent_eval.ANSWER_EVAL_VERSION", "other-version"):
            run_ab_eval(
                self.benchmark,
                self.run_root,
                out_dir,
                answer_fn=other,
                systems="fixed_rag",
                retrieval_system="fake_system",
                limit=1,
            )
        self.assertEqual(len(other.calls), 1)

    def test_injected_tools_are_used_instead_of_production_tools(self):
        with mock.patch.object(
            agent_tools, "LocalTools", side_effect=AssertionError("production tools built")
        ):
            summary, _, _, tools, _ = self.run_eval(
                "agent_single",
                decisions=[finish()],
                tools_results=[[U1]],
                limit=1,
            )
        self.assertEqual(summary["systems"]["agent_single"]["n"], 1)
        self.assertEqual(len(tools.calls), 1)

    # -- LocalTools evaluation mode -------------------------------------
    def _write_local_tools_fixture(self, root: Path, rag_dir: Path) -> tuple[dict, dict]:
        text = TEXT1
        unit = {
            "chunk_id": "u1",
            "parent_chunk_id": "p1",
            "arxiv_id": "paper-a",
            "title": "Paper 1",
            "page_number": 1,
            "group_number": 1,
            "char_start": 0,
            "char_end": len(text),
            "text": text,
            "pdf_path": "data/dataset_v1/pdfs/paper-a.pdf",
            "pdf_sha256": "b" * 64,
            "source_url": "https://arxiv.org/abs/paper-a",
            "short_fragment": False,
        }
        chunk = parent_chunk(unit)
        rag_dir.mkdir(parents=True, exist_ok=True)
        (rag_dir / "corpus_clean.json").write_text(
            json.dumps({"chunks": [chunk]}), encoding="utf-8"
        )
        (rag_dir / "corpus.json").write_text("{}", encoding="utf-8")
        (rag_dir / "vector_store.npz").write_bytes(b"")
        return unit, {"units": [unit], "corpus_sha256": "c" * 64}

    def test_local_tools_eval_mode_skips_lock_and_source_pdfs(self):
        root = self.base / "eval_root"
        rag_dir = self.base / "frozen_index"
        unit, metadata = self._write_local_tools_fixture(root, rag_dir)

        with mock.patch.object(
            agent_tools, "_load_eval_store", return_value=(object(), metadata)
        ) as loader, mock.patch.object(
            agent_tools, "check_available", side_effect=AssertionError("journal checked")
        ):
            tools = LocalTools(root, rag_dir=rag_dir, verify_sources=False)
            tools.check_current()

        self.assertEqual(Path(loader.call_args.args[0]).resolve(), rag_dir.resolve())
        self.assertEqual(len(tools.eligible), 1)
        self.assertEqual(tools.canonical["u1"]["parent_chunk_id"], "p1")
        # The unit's pdf_path points outside data/pdfs and does not exist; that
        # proves source checks were skipped.
        self.assertFalse((root / unit["pdf_path"]).exists())

    def test_local_tools_production_defaults_still_use_load_store_and_journal(self):
        prod = self.base / "prod"
        rag_dir = prod / "data" / "rag"
        rag_dir.mkdir(parents=True)
        pdf = prod / "data" / "pdfs" / "paper-a.pdf"
        pdf.parent.mkdir(parents=True)
        pdf.write_bytes(b"pdf-bytes")
        text = TEXT1
        unit = {
            "chunk_id": "u1",
            "parent_chunk_id": "p1",
            "arxiv_id": "paper-a",
            "title": "Paper 1",
            "page_number": 1,
            "group_number": 1,
            "char_start": 0,
            "char_end": len(text),
            "text": text,
            "pdf_path": "data/pdfs/paper-a.pdf",
            "pdf_sha256": sha256(b"pdf-bytes"),
            "source_url": "https://arxiv.org/abs/paper-a",
            "short_fragment": False,
        }
        metadata = {"units": [unit], "corpus_sha256": "c" * 64}
        (rag_dir / "corpus_clean.json").write_text(
            json.dumps({"chunks": [parent_chunk(unit)]}), encoding="utf-8"
        )
        (rag_dir / "corpus.json").write_text("{}", encoding="utf-8")
        (rag_dir / "vector_store.npz").write_bytes(b"")

        with mock.patch.object(
            agent_tools, "load_store", return_value=(object(), metadata)
        ) as load_store, mock.patch.object(agent_tools, "check_available") as check_available:
            tools = LocalTools(prod)

        self.assertEqual(Path(load_store.call_args.args[0]).resolve(), prod.resolve())
        check_available.assert_called()
        self.assertEqual(len(tools.eligible), 1)

    def test_local_tools_rag_dir_pointing_at_live_is_not_eval_mode(self):
        prod = self.base / "prod_live"
        rag_dir = prod / "data" / "rag"
        rag_dir.mkdir(parents=True)
        (rag_dir / "corpus_clean.json").write_text(
            json.dumps({"chunks": [parent_chunk(U1)]}), encoding="utf-8"
        )
        (rag_dir / "corpus.json").write_text("{}", encoding="utf-8")
        (rag_dir / "vector_store.npz").write_bytes(b"")
        metadata = {"units": [dict(U1)], "corpus_sha256": "c" * 64}

        with mock.patch.object(
            agent_tools, "load_store", return_value=(object(), metadata)
        ) as load_store, mock.patch.object(
            agent_tools, "_load_eval_store", side_effect=AssertionError("eval mode")
        ), mock.patch.object(agent_tools, "check_available") as check_available:
            tools = LocalTools(prod, rag_dir=rag_dir, verify_sources=False)
            tools.check_current()

        self.assertEqual(Path(load_store.call_args.args[0]).resolve(), prod.resolve())
        check_available.assert_called()
        self.assertTrue(tools.production_layout)

    def test_local_tools_frozen_index_rejects_verify_sources_true(self):
        root = self.base / "guard_root"
        frozen = self.base / "guard_frozen"
        with self.assertRaisesRegex(ValueError, "verify_sources=False"):
            LocalTools(root, rag_dir=frozen, verify_sources=True)

    def test_deepseek_answer_fn_rejects_non_stop_finish(self):
        from types import SimpleNamespace

        response = SimpleNamespace(
            model="fake-model",
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 3}),
            choices=[
                SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(
                        content=(
                            '{"status":"insufficient","points":[],'
                            '"limitations":["x"]}'
                        )
                    ),
                )
            ],
        )
        with mock.patch("openai.OpenAI") as openai_cls:
            openai_cls.return_value.chat.completions.create.return_value = response
            answer_fn = make_deepseek_answer_fn(
                env={
                    "DEEPSEEK_API_KEY": "test-key",
                    "DEEPSEEK_MODEL": "fake-model",
                }
            )
            with self.assertRaises(ValueError):
                answer_fn([{"role": "user", "content": "x"}])

    # -- AgentRun output redirection ------------------------------------
    def test_agent_run_output_dir_does_not_write_into_data_rag(self):
        class CombinedBackend:
            simulated = False

            def __call__(self, prompt, payload, max_tokens):
                if prompt == DECISION_PROMPT:
                    content = json.dumps(finish())
                else:
                    content = json.dumps(answer_ok(GOLD1))
                return {
                    "model": "fake",
                    "finish_reason": "stop",
                    "content": content,
                    "usage": {"total_tokens": 10},
                }

        root = self.base / "agent_root"
        out_dir = self.base / "agent_output"
        tools = FakeTools([[U1]])
        backend = CombinedBackend()
        agent = AgentRun(
            root, "问题一", None, tools, backend,
            max_rounds=1, max_calls=2, output_dir=out_dir,
        )
        result = agent.run()

        self.assertTrue((agent.folder / "answer_result.json").is_file())
        self.assertTrue((agent.history_folder / "answer_result.json").is_file())
        self.assertTrue(
            Path(result["agent"]["run"]).resolve().is_relative_to(out_dir.resolve())
        )
        self.assertFalse((root / "data").exists())

        # Defaults still point at the live library layout.
        default = AgentRun(root, "问题一", None, FakeTools([[U1]]), CombinedBackend(),
                           max_rounds=1, max_calls=2)
        self.assertEqual(
            default.folder.parent,
            root / "data" / "rag" / "agent_runs",
        )
        self.assertEqual(
            default.history_folder.parent,
            root / "data" / "rag" / "answers",
        )


if __name__ == "__main__":
    unittest.main()
