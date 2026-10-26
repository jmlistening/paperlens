"""Partial failure must remain visible without hiding well-formed self-review rows."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from answer_quality import coverage_display_rows, check_coverage


def row(status="covered", numbers=None):
    return {"aspect": "停止条件", "status": status,
            "point_numbers": [1] if numbers is None else numbers,
            "limitation": "" if status == "covered" else "缺少具体内容。"}


class CoverageDisplayTests(unittest.TestCase):
    def test_real_record_shows_three_valid_rows_and_one_invalid_without_changing_claims(self):
        path = Path(__file__).resolve().parents[1] / 'eval/development_cases/RCER_agent_v11_20261020.json'
        if not path.exists():
            self.skipTest(f"missing fixture: {path}")
        record = json.loads(path.read_text(encoding='utf-8'))
        before = deepcopy(record)
        rows = coverage_display_rows(record['answer']['coverage'], len(record['answer']['points']))
        self.assertEqual(len(rows), 4)
        self.assertEqual([r['结构检查'] for r in rows], ['通过（仅结构）'] * 3 + ['待核对'])
        self.assertEqual(rows[3]['模型原始状态'], '自评部分覆盖')
        self.assertEqual(rows[3]['对应回答'], '无')
        self.assertIn('对应回答序号为空', rows[3]['核对说明'])
        self.assertEqual(record, before)
        with self.assertRaises(ValueError):
            check_coverage(record['answer']['coverage'], len(record['answer']['points']))

    def test_malformed_row_does_not_crash_or_hide_valid_row(self):
        for bad in (None, 'text', {}, row(numbers=[True]), row(numbers=[9])):
            rows = coverage_display_rows([row(), bad], 1)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]['结构检查'], '待核对')
            # Duplicated aspects are separately flagged, including the earlier row.
            if not isinstance(bad, dict) or bad.get('aspect') != '停止条件':
                self.assertEqual(rows[0]['结构检查'], '通过（仅结构）')

    def test_absent_or_oversized_list_does_not_invent_self_review(self):
        for value in (None, [], {}, [row()] * 7):
            self.assertEqual(coverage_display_rows(value, 1), [])

    def test_duplicate_aspects_flag_both_rows(self):
        rows = coverage_display_rows([row(), row()], 1)
        self.assertTrue(all(r['结构检查'] == '待核对' for r in rows))

    def test_missing_without_answer_is_valid_but_partial_without_answer_is_not(self):
        self.assertEqual(coverage_display_rows([row('missing', [])], 0)[0]['结构检查'], '通过（仅结构）')
        self.assertEqual(coverage_display_rows([row('partial', [])], 0)[0]['结构检查'], '待核对')


if __name__ == '__main__':
    unittest.main()
