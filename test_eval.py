"""Evaluation bookkeeping tests; all data are temporary, no network/API."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import run_eval as ev


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case = {"id": "Q01", "query": "中文实验问题？", "paper": "testv1"}
        self.session = self.root / "eval/results/test"
        self.before = ev.inventory(self.root)

    def put(self, relative, value):
        path = self.root / relative
        ev.save(path, value)
        return path

    def fixed(self, status="completed"):
        self.put("data/rag/qa_runs/new/run.json", {
            "query": self.case["query"], "paper_filter": "testv1",
            "candidate_count": 20, "status": status})
        if status == "completed":
            self.put("data/rag/qa_runs/new/answer_result.json", {
                "query": self.case["query"], "answer": {"status": "partial"}})
        self.put("data/rag/rerank_runs/new/raw_response.json", {
            "model": "fake", "usage": {"total_tokens": 100}})

    def collect(self, mode="fixed", code=0):
        return ev.collect(self.root, self.before, self.session, self.case, mode, code, 2.5)

    def test_fixed_adds_rerank_and_answer_without_double_counting(self):
        self.fixed()
        self.put("data/rag/answers/new/raw_response.json", {
            "model": "fake", "usage": {"total_tokens": 200}})
        self.put("data/rag/answers/new/answer_result.json", {"usage": {"total_tokens": 200}})
        result = self.collect()
        self.assertEqual(result["usage"]["known_total_tokens"], 300)
        self.assertEqual(result["api_attempts"], 2)
        self.assertTrue(result["usage"]["complete"])
        self.assertIsNone(result["human_quality"])

    def test_failed_fixed_keeps_known_usage_and_marks_total_unknown(self):
        self.fixed("failed")
        result = self.collect(code=1)
        self.assertEqual(result["usage"]["known_total_tokens"], 100)
        self.assertFalse(result["usage"]["complete"])
        self.assertIsNone(result["api_attempts"])
        self.assertIsNone(result["usage"]["unknown_usage_calls"])

    def test_post_response_validation_failure_has_complete_usage(self):
        self.fixed("failed")
        self.put("data/rag/answers/new/raw_response.json", {
            "model": "fake", "usage": {"total_tokens": 200}})
        result = self.collect(code=1)
        self.assertFalse(result["workflow_completed"])
        self.assertEqual(result["api_attempts"], 2)
        self.assertEqual(result["usage"], {"known_total_tokens": 300, "unknown_usage_calls": 0, "complete": True})

    def test_returned_response_with_missing_usage_keeps_count(self):
        self.fixed("failed")
        self.put("data/rag/answers/new/raw_response.json", {"model": "fake", "usage": None})
        result = self.collect(code=1)
        self.assertEqual(result["api_attempts"], 2)
        self.assertEqual(result["usage"]["unknown_usage_calls"], 1)
        self.assertFalse(result["usage"]["complete"])

    def test_select_modes_rejects_invalid_and_duplicate(self):
        self.assertEqual(ev.select_modes("agent_one,agent_multi"), ("agent_one", "agent_multi"))
        for value in ("", "fixed,fixed", "other"):
            with self.assertRaises(ValueError):
                ev.select_modes(value)

    def test_continuation_blocks_rerun_and_changed_business_code(self):
        source = self.put("source.json", {"v": 1})
        prior = self.put("old/manifest.json", {
            "tasks": [{"case_id": "Q01", "mode": "fixed", "status": "failed"},
                      {"case_id": "Q01", "mode": "agent_one", "status": "pending"}],
            "input_sha256": {"source.json": ev.digest(source), "run_eval.py": "old-harness-hash"}})
        ev.verify_continuation(self.root, prior, [self.case], ("agent_one",))
        with self.assertRaises(ValueError):
            ev.verify_continuation(self.root, prior, [self.case], ("fixed",))
        self.put("source.json", {"v": 2})
        with self.assertRaises(ValueError):
            ev.verify_continuation(self.root, prior, [self.case], ("agent_one",))

    def test_selected_modes_never_launch_fixed_and_use_six_call_cap(self):
        self.put("eval/pilot_v1/questions.json", {"version": "offline_test", "cases": [
            {**self.case, "pdf_path": "unused.pdf"}]})
        def fake(root, session, case, mode):
            return {"case_id": "Q01", "mode": mode, "pipeline_status": "completed",
                    "answer_self_status": "partial", "wall_seconds": 1, "api_attempts": 0,
                    "models": [], "workflow_completed": True,
                    "usage": {"known_total_tokens": 0, "complete": True}}
        with patch.object(ev, "ROOT", self.root), patch.object(ev, "check"), \
             patch.object(ev, "run_task", side_effect=fake) as run, \
             patch.object(ev.sys, "argv", ["run_eval.py", "--run", "--cases", "Q01",
                                           "--modes", "agent_one,agent_multi"]):
            self.assertEqual(ev.main(), 0)
        self.assertEqual([c.args[-1] for c in run.call_args_list], ["agent_one", "agent_multi"])
        manifest = ev.read(next((self.root / "eval/results").glob("*/manifest.json")))
        self.assertEqual(manifest["maximum_model_calls"], 6)

    def test_repeating_same_continuation_is_blocked(self):
        prior = self.put("old/manifest.json", {"input_sha256": {}, "tasks": [
            {"case_id": "Q01", "mode": "agent_one", "status": "pending"}]})
        self.put("eval/results/child/manifest.json", {
            "continuation": {"sha256": ev.digest(prior)},
            "tasks": [{"case_id": "Q01", "mode": "agent_one", "status": "failed"}]})
        with self.assertRaises(ValueError):
            ev.verify_continuation(self.root, prior, [self.case], ("agent_one",))

    def test_stale_answer_is_never_selected(self):
        self.fixed()
        self.before = ev.inventory(self.root)
        with self.assertRaises(ValueError):
            self.collect()

    def test_multiple_new_runs_are_rejected(self):
        self.fixed()
        self.put("data/rag/qa_runs/unrelated/run.json", {})
        with self.assertRaises(ValueError):
            self.collect()

    def test_wrong_question_is_rejected(self):
        self.fixed()
        self.put("data/rag/qa_runs/new/run.json", {"query": "another", "paper_filter": "testv1"})
        with self.assertRaises(ValueError):
            self.collect()

    def test_agent_usage_includes_decisions_and_unknown_attempt(self):
        self.put("data/rag/agent_runs/new/run.json", {
            "query": self.case["query"], "paper_filter": "testv1", "status": "failed",
            "api_attempts": 3, "model_calls": [
                {"model": "fake", "usage": {"total_tokens": 20}},
                {"model": "fake", "usage": {"total_tokens": 40}}, {"usage": None}],
            "trace": [{"kind": "retrieval"}, {"kind": "decision"}, {"kind": "retrieval"}]})
        result = self.collect("agent_multi", 1)
        self.assertEqual(result["usage"], {"known_total_tokens": 60, "unknown_usage_calls": 1, "complete": False})
        self.assertEqual(result["retrieval_rounds"], 2)
        self.assertEqual(result["api_attempts"], 3)

    def test_freeze_detects_changes(self):
        path = self.put("frozen.json", {"version": 1})
        frozen = {"frozen.json": ev.digest(path)}
        ev.assert_frozen(self.root, frozen)
        self.put("frozen.json", {"version": 2})
        with self.assertRaises(ValueError):
            ev.assert_frozen(self.root, frozen)

    def test_mode_budgets_and_gold_exclusion(self):
        one = ev.command(self.root, self.case, "agent_one")
        multi = ev.command(self.root, self.case, "agent_multi")
        self.assertEqual(one[-4:], ["--max-rounds", "1", "--max-calls", "2"])
        self.assertEqual(multi[-4:], ["--max-rounds", "3", "--max-calls", "4"])
        self.assertNotIn("reference_criteria.json", " ".join(one + multi))

    def test_utf8_subprocess_and_export(self):
        # Synthetic CLI exercises real subprocess/copy/ZIP behavior, without importing a model.
        script = '''import json,sys
from pathlib import Path
r=Path(__file__).resolve().parent
def save(name,value):
 p=r/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False),encoding="utf-8")
save("data/rag/qa_runs/fake/run.json",{"query":sys.argv[1],"paper_filter":sys.argv[3],"status":"insufficient_evidence","candidate_count":0})
print("中文日志，模拟零证据流程；没有 API")
'''
        (self.root / "run_qa.py").write_text(script, encoding="utf-8")
        result = ev.run_task(self.root, self.session, self.case, "fixed")
        self.assertEqual(result["api_attempts"], 0)
        self.assertTrue(result["workflow_completed"])
        log = self.session / "tasks/Q01_fixed/console.txt"
        self.assertIn("中文日志", log.read_text(encoding="utf-8"))
        ev.export(self.session)
        with ev.zipfile.ZipFile(self.session.with_suffix(".zip")) as z:
            self.assertTrue(any(n.endswith("metrics.json") for n in z.namelist()))

    def test_batch_stops_after_failure_and_exports_pending_tasks(self):
        self.put("eval/pilot_v1/questions.json", {"version": "offline_test", "cases": [
            {**self.case, "pdf_path": "unused.pdf"}]})
        failed = {"case_id": "Q01", "mode": "fixed", "pipeline_status": "failed",
                  "answer_self_status": None, "wall_seconds": 1, "api_attempts": None,
                  "models": [], "workflow_completed": False,
                  "usage": {"known_total_tokens": 0, "complete": False}}
        with patch.object(ev, "ROOT", self.root), patch.object(ev, "check"), \
             patch.object(ev, "run_task", return_value=failed) as run, \
             patch.object(ev.sys, "argv", ["run_eval.py", "--run", "--cases", "Q01"]):
            self.assertEqual(ev.main(), 1)
        self.assertEqual(run.call_count, 1)
        manifest = ev.read(next((self.root / "eval/results").glob("*/manifest.json")))
        self.assertEqual([t["status"] for t in manifest["tasks"]], ["failed", "pending", "pending"])
        self.assertTrue(list((self.root / "eval/results").glob("*.zip")))


if __name__ == "__main__":
    unittest.main()
