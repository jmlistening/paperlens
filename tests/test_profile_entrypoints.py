"""Entry points must resolve the profile without a legacy interests.json.

A fresh public clone contains only ``interests.example.json``; these tests
point each legacy entry point at a temporary root with no personal profile.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fetch_papers  # noqa: E402
import rank_papers  # noqa: E402
import run_pipeline  # noqa: E402

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"


class _StopPipeline(Exception):
    """Sentinel raised by the fake fetch stage."""


class _FakeSender:
    def dotenv_values(self, path, encoding=None):
        return {
            "MAIL_TO": "reader@example.invalid",
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_BASE_URL": "https://example.invalid",
            "DEEPSEEK_MODEL": "test-model",
        }

    def checked_address(self, value):
        return value


class LegacyEntrypointProfileTests(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.root = Path(self.tmp.name)
        (self.root / "interests.example.json").write_text(
            (ROOT / "interests.example.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        self._old_roots = (
            fetch_papers.ROOT,
            rank_papers.ROOT,
            run_pipeline.ROOT,
        )
        fetch_papers.ROOT = self.root
        rank_papers.ROOT = self.root
        run_pipeline.ROOT = self.root

    def tearDown(self):
        fetch_papers.ROOT = self._old_roots[0]
        rank_papers.ROOT = self._old_roots[1]
        run_pipeline.ROOT = self._old_roots[2]
        self.tmp.cleanup()

    def test_fetch_papers_main_uses_the_public_example(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            fetch_papers.main(dry_run=True)
        self.assertIn("[Example]", output.getvalue())
        self.assertFalse((self.root / "interests.json").exists())

    def test_rank_papers_main_reaches_env_check_without_legacy_profile(self):
        (self.root / "data").mkdir()
        (self.root / "data" / "papers.json").write_text(
            json.dumps([{"arxiv_id": "2601.00001v1"}]), encoding="utf-8"
        )
        with self.assertRaises(ValueError) as caught:
            rank_papers.main()
        self.assertIn(".env", str(caught.exception))
        self.assertFalse((self.root / "interests.json").exists())

    def test_run_pipeline_execute_resolves_profile_before_stages(self):
        data = self.root / "data"
        run_dir = data / "runs" / "test-run"
        run_dir.mkdir(parents=True)

        def stop_at_fetch():
            raise _StopPipeline()

        modules = {
            "send_digest": _FakeSender(),
            "fetch_papers": type(
                "FakeFetch", (), {"main": staticmethod(stop_at_fetch)}
            )(),
        }
        state = {"run_id": "test-run", "completed_stages": []}
        with self.assertRaises(_StopPipeline):
            run_pipeline.execute(modules, run_dir, state, send=False)
        self.assertFalse((self.root / "interests.json").exists())


if __name__ == "__main__":
    unittest.main()
