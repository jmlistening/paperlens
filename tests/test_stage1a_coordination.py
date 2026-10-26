"""Offline guards for background continuation; does not run a task or any network."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
STAGE_DIR = ROOT/'docs/stage1a/20261022_history_prepare'
if not STAGE_DIR.exists():
    raise unittest.SkipTest(f"missing fixture: {STAGE_DIR}")
sys.path.insert(0,str(STAGE_DIR))
from continue_stage import readiness


class CoordinationTests(unittest.TestCase):
    def test_running_cannot_prepare_even_with_exit_fields(self):
        self.assertFalse(readiness({'status':'running','exit_code':0,'finished_at':'old'}))

    def test_unknown_or_incomplete_exit_cannot_prepare(self):
        for row in ({},{'status':'exited'},{'status':'exited','exit_code':False,'finished_at':'x'},
                    {'status':'exited','exit_code':0}):
            self.assertFalse(readiness(row))

    def test_completed_or_paused_exit_can_prepare_available_pdfs(self):
        for code in (0,2):
            self.assertTrue(readiness({'status':'exited','exit_code':code,'finished_at':'x'}))


if __name__ == '__main__':
    unittest.main()
