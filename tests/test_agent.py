"""Offline tests of control flow and failure boundaries; no real model/API calls."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_agent import AgentRun


def unit(number):
    return {"chunk_id": f"chunk{number}", "parent_chunk_id": f"parent{number}",
            "arxiv_id": "2610.19398v1", "title": "Test paper",
            "page_number": number, "text": f"Evidence number {number} describes retrieval and a finite passage budget.",
            "pdf_path": "data/pdfs/2610.19398v1.pdf", "pdf_sha256": "a" * 64,
            "source_url": "https://arxiv.org/abs/2610.19398v1"}


class Tools:
    def __init__(self, results=None):
        self.metadata = {"corpus_sha256": "b" * 64}
        self.canonical = {u["chunk_id"]: u for u in (unit(1), unit(2), unit(3))}
        self.results = results if results is not None else [[unit(1)], [unit(2)]]
        self.calls, self.changed = [], False

    def check_current(self):
        if self.changed:
            raise ValueError("snapshot changed")

    def search(self, tool, query, top_k):
        self.check_current()
        self.calls.append((tool, query))
        return self.results.pop(0)


class Backend:
    simulated = True

    def __init__(self, values):
        self.values, self.payloads = list(values), []

    def __call__(self, prompt, payload, max_tokens):
        self.payloads.append(payload)
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        if callable(value):
            value = value()
        if isinstance(value, tuple):
            return {"model": "offline-test", "content": value[0], "finish_reason": value[1], "usage": None}
        return {"model": "offline-test", "content": json.dumps(value), "finish_reason": "stop",
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}


def decision(action="finish", ids=None, sufficient=True, query=""):
    return {"assessment": "sufficient" if sufficient else "partial", "selected_ids": ids or ["C1"],
            "missing": [] if sufficient else ["缺少停止条件"], "action": action, "query": query,
            "reason": "检查证据覆盖，决定是否补充检索"}


def answer(number=1, bad=False):
    return {"status": "answered", "points": [{"text": "测试答案", "citations": [
        {"id": "E1", "quote": "This quote was never present in the evidence." if bad else unit(number)["text"]}]}],
        "limitations": []}


class AgentTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def create(self, responses, tools=None, **limits):
        self.tools = tools or Tools()
        self.backend = Backend(responses)
        self.agent = AgentRun(self.root, "Original question", "2610.19398v1", self.tools, self.backend, **limits)
        return self.agent

    def test_sufficient_stops_without_supplement_and_simulation_never_enters_history(self):
        result = self.create([decision(), answer()]).run()
        self.assertEqual(result["agent"]["model_calls"], 2)
        self.assertEqual(len(self.tools.calls), 1)
        self.assertFalse((self.root / "data/rag/answers").exists())

    def test_bm25_supplement_changes_selected_evidence_and_keeps_original_question(self):
        result = self.create([decision("search_bm25", sufficient=False, query="passage budget"),
                              decision(ids=["C2"]), answer(2)]).run()
        self.assertEqual(result["evidence"]["E1"]["page_number"], 2)
        self.assertEqual(self.tools.calls[1], ("search_bm25", "passage budget"))
        self.assertTrue(all(p["question"] == "Original question" for p in self.backend.payloads))
        self.assertEqual(result["agent"]["model_calls"], 3)
        self.assertEqual(result["agent"]["usage"]["known_total_tokens"], 60)

    def test_dense_supplement_is_allowed(self):
        result = self.create([decision("search_dense", sufficient=False, query="How does retrieval terminate?"),
                              decision(ids=["C2"]), answer(2)]).run()
        self.assertEqual(result["agent"]["retrieval_rounds"], 2)
        self.assertEqual(self.tools.calls[1][0], "search_dense")

    def test_round_budget_reserves_answer_and_preserves_gap(self):
        result = self.create([decision("search_bm25", sufficient=False, query="budget"), answer()], max_rounds=1).run()
        self.assertEqual(result["agent"]["stop_reason"], "round_budget")
        self.assertEqual(result["answer"]["status"], "partial")
        self.assertIn("缺少停止条件", result["answer"]["limitations"])
        self.assertEqual(len(self.tools.calls), 1)

    def test_call_budget_does_not_consume_reserved_answer_call(self):
        result = self.create([decision("search_bm25", sufficient=False, query="budget"), answer()], max_calls=2).run()
        self.assertEqual(result["agent"]["stop_reason"], "call_budget")
        self.assertEqual(len(self.backend.payloads), 2)

    def test_three_rounds_and_four_calls_are_hard_limits(self):
        tools = Tools([[unit(1)], [unit(2)], [unit(3)]])
        result = self.create([decision("search_bm25", sufficient=False, query="budget"),
                              decision("search_dense", sufficient=False, query="termination"),
                              decision("search_bm25", sufficient=False, query="fourth query"), answer()], tools).run()
        self.assertEqual(len(tools.calls), 3)
        self.assertEqual(result["agent"]["model_calls"], 4)
        self.assertEqual(result["agent"]["stop_reason"], "round_budget")

    def test_repeated_query_stops_without_running_tool_again(self):
        result = self.create([decision("search_dense", sufficient=False, query=" ORIGINAL question! "), answer()]).run()
        self.assertEqual(result["agent"]["stop_reason"], "repeated_query")
        self.assertEqual(len(self.tools.calls), 1)

    def test_no_new_evidence_stops_without_another_decision(self):
        tools = Tools([[unit(1)], [unit(1)]])
        result = self.create([decision("search_bm25", sufficient=False, query="budget"), answer()], tools).run()
        self.assertEqual(result["agent"]["stop_reason"], "no_new_evidence")
        self.assertEqual(len(self.backend.payloads), 2)

    def test_unallowed_tool_cannot_execute(self):
        bad = decision("execute_shell", sufficient=False, query="anything")
        agent = self.create([bad])
        with self.assertRaises(ValueError): agent.run()
        self.assertEqual(len(self.tools.calls), 1)
        self.assertEqual(agent.report["status"], "failed")
        self.assertFalse((agent.folder / "answer_result.json").exists())

    def test_unknown_evidence_id_is_rejected(self):
        agent = self.create([decision(ids=["C999"])])
        with self.assertRaises(ValueError): agent.run()
        self.assertEqual(len(self.backend.payloads), 1)

    def test_fabricated_quote_does_not_publish_an_answer(self):
        agent = self.create([decision(), answer(bad=True)])
        with self.assertRaises(ValueError): agent.run()
        self.assertFalse((agent.folder / "answer_result.json").exists())
        self.assertTrue((agent.folder / "calls/02_answer.response.json").is_file())

    def test_api_failure_is_not_retried_and_usage_is_unknown(self):
        agent = self.create([TimeoutError("offline injected timeout")])
        with self.assertRaises(TimeoutError): agent.run()
        report = json.loads((agent.folder / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(report["api_attempts"], 1)
        self.assertEqual(report["usage"]["unknown_usage_calls"], 1)
        self.assertFalse(report["usage"]["complete"])

    def test_index_change_during_model_call_stops(self):
        tools = Tools()
        def change():
            tools.changed = True
            return decision()
        agent = self.create([change], tools)
        with self.assertRaises(ValueError): agent.run()
        self.assertFalse((agent.folder / "answer_result.json").exists())

    def test_empty_retrieval_uses_no_model_calls(self):
        result = self.create([], Tools([[]])).run()
        self.assertEqual(result["answer"]["status"], "insufficient")
        self.assertEqual(result["agent"]["model_calls"], 0)

    def test_wrong_paper_result_is_rejected(self):
        tools = Tools()
        tools.canonical["chunk1"]["arxiv_id"] = "another-paper"
        tools.results = [[tools.canonical["chunk1"]]]
        agent = self.create([], tools)
        with self.assertRaises(ValueError): agent.run()
        self.assertEqual(len(self.backend.payloads), 0)

    def test_truncated_model_output_is_not_accepted(self):
        agent = self.create([('{"assessment":', 'length')])
        with self.assertRaises(ValueError): agent.run()
        self.assertEqual(len(self.backend.payloads), 1)


if __name__ == "__main__":
    unittest.main()
