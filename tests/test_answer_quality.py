"""Offline regression cases for source-preserving quotations and coverage reports."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from answer_quality import review_agent_answer, extend_formula_quote, normalized
from tests.test_agent import Backend, Tools, answer, decision
from run_agent import AgentRun


def coverage(status="covered", numbers=None, limitation=""):
    return [{"aspect": "检索停止条件", "status": status,
             "point_numbers": [1] if numbers is None else numbers, "limitation": limitation}]


class AnswerQualityTests(unittest.TestCase):
    def fixture(self):
        p = Path(__file__).resolve().parents[1] / "eval/development_cases/RCER_agent_v1_20261020.json"
        if not p.exists():
            self.skipTest(f"missing fixture: {p}")
        return json.loads(p.read_text(encoding="utf-8"))

    def test_real_record_expands_two_formulas_without_mutating_input_or_inventing_coverage(self):
        record = self.fixture()
        original = deepcopy(record)
        reviewed, quality = review_agent_answer(record["answer"], record["evidence"])
        self.assertEqual(record, original)
        self.assertEqual(len(quality["quote_extensions"]), 2)
        self.assertEqual(quality["coverage_validation"], "unavailable")
        self.assertEqual(quality["coverage"], [])
        self.assertTrue(quality["needs_review"])
        for item in quality["quote_extensions"]:
            self.assertIn(normalized(item["expanded_quote"]), normalized(record["evidence"][item["evidence_id"]]["text"]))
            self.assertTrue(item["expanded_quote"].endswith(("(2)", "(4)")))
        again, audit = review_agent_answer(reviewed, record["evidence"])
        self.assertEqual(reviewed, again)
        self.assertFalse(audit["quote_extensions"])
        self.assertFalse(any(w["code"] == "quote_boundary_review" for w in audit["warnings"]))

    def test_ambiguous_occurrences_and_unfinished_or_overlong_equations_are_not_extended(self):
        quote = "The controller returns its relation analysis"
        for source in [quote + " M = [r, u] (2)",
                       quote + " M = " + "x" * 800 + ". (2)",
                       quote + " M = x. (2) " + quote + " M = y. (3)"]:
            with self.subTest(source=source[:80]):
                self.assertIsNone(extend_formula_quote(quote, source))

    def test_plain_prose_and_later_equations_are_not_guessed(self):
        quote = "The controller returns its relation analysis"
        source = quote + " for all collected evidence. Later the paper defines M = x. (2)"
        self.assertIsNone(extend_formula_quote(quote, source))

    def test_numbered_equation_with_period_after_number_is_supported(self):
        quote = "The controller returns its relation analysis"
        source = quote + " M = x (2). Next paragraph."
        self.assertEqual(extend_formula_quote(quote, source), quote + " M = x (2).")

    def test_unsupported_continuation_is_flagged_and_preserved(self):
        a = answer()
        a["coverage"] = coverage()
        a["points"][0]["citations"][0]["quote"] = "The controller returns its relation analysis"
        e = {"E1": {"text": "The controller returns its relation analysis but the next step"}}
        reviewed, quality = review_agent_answer(a, e)
        self.assertEqual(a, reviewed)
        self.assertTrue(quality["needs_review"])
        self.assertEqual(quality["warnings"][0]["code"], "quote_boundary_review")

    def test_fabricated_quote_still_fails_before_extension(self):
        with self.assertRaises(ValueError):
            review_agent_answer(answer(bad=True), {"E1": {"text": "Different evidence."}})

    def test_explicit_coverage_gap_downgrades_answered_and_keeps_gap(self):
        a = answer()
        a["coverage"] = coverage("partial", limitation="只说明预算，没有重复查询规则。")
        reviewed, quality = review_agent_answer(a, {"E1": Tools().canonical["chunk1"]})
        self.assertEqual(reviewed["status"], "partial")
        self.assertIn("重复查询规则", reviewed["limitations"][0])
        self.assertEqual(quality["model_answer_status"], "answered")
        self.assertEqual(quality["semantic_verification"], "not_performed")

    def test_invalid_self_review_cannot_be_shown_as_valid(self):
        for rows in [coverage(numbers=[2]), coverage(numbers=[True]), coverage(numbers=[1, 1]),
                     coverage("missing", numbers=[1], limitation="缺失"), coverage("partial"),
                     coverage("covered", numbers=[]), coverage() * 2]:
            with self.subTest(rows=rows):
                a = answer(); a["coverage"] = rows
                reviewed, quality = review_agent_answer(a, {"E1": Tools().canonical["chunk1"]})
                self.assertEqual(quality["coverage_validation"], "unavailable")
                self.assertFalse(quality["coverage"])
                self.assertTrue(quality["needs_review"])

    def test_successful_coverage_is_self_review_not_semantic_proof(self):
        a = answer(); a["coverage"] = coverage()
        reviewed, quality = review_agent_answer(a, {"E1": Tools().canonical["chunk1"]})
        self.assertEqual(reviewed, a)
        self.assertFalse(quality["needs_review"])
        self.assertEqual(quality["coverage_validation"], "valid_structure_only")
        self.assertEqual(quality["semantic_verification"], "not_performed")

    def test_agent_saves_review_and_raw_answer_without_extra_model_calls(self):
        a = answer(); a["coverage"] = coverage("missing", numbers=[], limitation="缺少停止规则。")
        with tempfile.TemporaryDirectory() as tmp:
            backend = Backend([decision(), a])
            runner = AgentRun(Path(tmp), "停止条件是什么？", "2610.19398v1", Tools(), backend)
            result = runner.run()
            self.assertEqual(len(backend.payloads), 2)
            self.assertEqual(result["answer"]["status"], "partial")
            self.assertEqual(result["answer_quality"]["model_answer_status"], "answered")
            raw = json.loads((runner.folder / "calls/02_answer.response.json").read_text(encoding="utf-8"))
            self.assertEqual(json.loads(raw["content"])["status"], "answered")
            preview = (runner.folder / "answer_preview.txt").read_text(encoding="utf-8")
            self.assertIn("回答检查", preview)
            self.assertIn("模型自评未覆盖", preview)
            self.assertFalse((Path(tmp) / "data/rag/answers").exists())


if __name__ == "__main__":
    unittest.main()
