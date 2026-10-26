"""Offline boundary tests for shared fixed/Agent schema handling."""
from copy import deepcopy
import unittest

from answer_schema import normalize_answer_schema
from answer_question import validate_answer
from answer_quality import review_agent_answer


class AnswerSchemaTests(unittest.TestCase):
    def setUp(self):
        self.quote = "The experiment uses a single generated token."
        self.evidence = {"E1": {"text": self.quote}}
        self.answer = {"status": "answered", "points": [{"text": "生成一个 token。", "citations": [
            {"id": "E1", "quote": self.quote}]}]}

    def test_missing_answered_field_is_audited_without_mutating_input(self):
        old = deepcopy(self.answer)
        fixed, changes = normalize_answer_schema(self.answer)
        validate_answer(fixed, self.evidence)
        self.assertEqual(self.answer, old)
        self.assertEqual(fixed["limitations"], [])
        self.assertEqual(changes[0]["original_presence"], False)
        self.assertEqual(fixed["points"], old["points"])

    def test_partial_and_insufficient_missing_gaps_still_fail(self):
        for status in ("partial", "insufficient"):
            answer = {**self.answer, "status": status}
            if status == "insufficient":
                answer["points"] = []
            fixed, changes = normalize_answer_schema(answer)
            self.assertFalse(changes)
            with self.assertRaises(ValueError):
                validate_answer(fixed, self.evidence)

    def test_explicit_invalid_limitations_are_not_erased(self):
        for value in (None, "", {}, [""]):
            fixed, changes = normalize_answer_schema({**self.answer, "limitations": value})
            self.assertFalse(changes)
            with self.assertRaises(ValueError):
                validate_answer(fixed, self.evidence)

    def test_existing_limitations_and_idempotence(self):
        answer = {**self.answer, "limitations": ["仍需要核对实验范围。"]}
        fixed, changes = normalize_answer_schema(answer)
        self.assertEqual(fixed, answer)
        self.assertFalse(changes)
        normalized, _ = normalize_answer_schema(self.answer)
        again, changes = normalize_answer_schema(normalized)
        self.assertEqual(again, normalized)
        self.assertFalse(changes)

    def test_fabricated_quote_still_rejected_in_both_paths(self):
        self.answer["points"][0]["citations"][0]["quote"] = "This statement does not occur in the source."
        fixed, _ = normalize_answer_schema(self.answer)
        with self.assertRaises(ValueError):
            validate_answer(fixed, self.evidence)
        with self.assertRaises(ValueError):
            review_agent_answer(self.answer, self.evidence)

    def test_agent_missing_coverage_remains_flagged_not_invented(self):
        reviewed, quality = review_agent_answer(self.answer, self.evidence)
        self.assertEqual(reviewed["limitations"], [])
        self.assertNotIn("coverage", reviewed)
        self.assertEqual(quality["coverage_validation"], "unavailable")
        self.assertTrue(quality["needs_review"])
        self.assertEqual(len(quality["schema_normalizations"]), 1)

    def test_real_coverage_gaps_still_downgrade_status(self):
        self.answer["coverage"] = [{"aspect": "预热", "status": "missing", "point_numbers": [],
                                    "limitation": "缺少预热证据"}]
        reviewed, quality = review_agent_answer(self.answer, self.evidence)
        self.assertEqual(reviewed["status"], "partial")
        self.assertTrue(reviewed["limitations"])
        self.assertEqual(quality["coverage_validation"], "valid_structure_only")


if __name__ == "__main__":
    unittest.main()
