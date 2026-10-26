"""Offline end-to-end preview integration for the push pipeline (Task P10).

Fake arXiv results and a fake summarizer are injected through ``PushDeps``.
The test asserts that a preview run selects one paper, writes the digest and
``run.json``, and never touches SMTP. No fixture or test performs a network
call or a real model call.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from push_run import PushDeps, run_push  # noqa: E402

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "push"
NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)

ENV = {
    "DEEPSEEK_API_KEY": "integration-test-key",
    "DEEPSEEK_BASE_URL": "https://example.invalid",
    "DEEPSEEK_MODEL": "integration-test-model",
    "SMTP_HOST": "smtp.163.com",
    "SMTP_PORT": "465",
    "SMTP_SECURITY": "ssl",
    "SMTP_USER": "user@163.com",
    "SMTP_PASSWORD": "integration-app-code",
    "MAIL_TO": "user@163.com",
}


def load_json(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FixtureResult:
    """Duck-typed ``arxiv.Result`` built from a checked-in fixture."""

    def __init__(self, row: dict):
        self._arxiv_id = row["arxiv_id"]
        self.title = row["title"]
        self.summary = row["abstract"]
        self.authors = [SimpleNamespace(name=name) for name in row["authors"]]
        self.published = datetime.fromisoformat(row["published"])
        self.updated = datetime.fromisoformat(row["updated"])
        self.categories = list(row["categories"])

    def get_short_id(self) -> str:
        return self._arxiv_id


class FixtureArxivClient:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.calls = 0

    def results(self, request):
        self.calls += 1
        return [FixtureResult(row) for row in self.rows]


def build_root(tmp: str) -> Path:
    root = Path(tmp)
    (root / "interests.json").write_text(
        (ROOT / "interests.example.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    raw = json.loads((ROOT / "push_config.json").read_text(encoding="utf-8"))
    raw.update({"enabled": True, "paused": False, "quiet_mode": False})
    (root / "push_config.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root


class PushPreviewIntegrationTests(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.root = build_root(self.tmp.name)
        self.candidates = load_json("candidates.json")
        self.rank_rows = load_json("rank.json")
        self.summaries = load_json("summaries.json")
        self.client = FixtureArxivClient(self.candidates)

    def tearDown(self):
        self.tmp.cleanup()

    def fake_rank(self, papers, profile, feedback):
        by_id = {row["arxiv_id"]: row for row in self.rank_rows}
        evaluations = []
        for paper in papers:
            row = by_id[paper["arxiv_id"]]
            evaluations.append(
                {
                    "arxiv_id": paper["arxiv_id"],
                    "score": row["score"],
                    "reason": row["reason"],
                    "limitation": row.get("limitation", ""),
                    "feedback_refs": [],
                }
            )
        return {"evaluations": evaluations}

    def fake_summarize(self, papers, env):
        return {
            paper["arxiv_id"].split("v")[0]: self.summaries[paper["arxiv_id"]]
            for paper in papers
        }

    def test_preview_selects_one_paper_and_never_calls_smtp(self):
        smtp_send = mock.Mock(
            side_effect=AssertionError("preview must never call SMTP")
        )
        deps = PushDeps(
            root=self.root,
            env=ENV,
            arxiv_client=self.client,
            rank_fn=self.fake_rank,
            summarize_fn=self.fake_summarize,
            smtp_send=smtp_send,
        )
        result = run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=deps,
        )

        smtp_send.assert_not_called()
        self.assertEqual(result.status, "preview_ready")
        self.assertFalse(result.sent)
        self.assertEqual(result.candidate_count, 3)
        self.assertEqual(result.selected_count, 1)
        self.assertGreater(self.client.calls, 0)

        run_dir = self.root / "data" / "push" / "runs" / str(result.run_id)
        run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(run_json["status"], "preview_ready")
        self.assertEqual(run_json["counts"]["candidates"], 3)
        self.assertEqual(run_json["counts"]["selected"], 1)

        digest_html = (run_dir / "digest_preview.html").read_text(encoding="utf-8")
        meta = json.loads((run_dir / "digest_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(len(meta["paper_keys"]), 1)
        selected_title = next(
            row["title"]
            for row in self.candidates
            if row["arxiv_id"].split("v")[0] == meta["paper_keys"][0]
        )
        self.assertIn(selected_title, digest_html)
        self.assertTrue((run_dir / "pipeline.log").is_file())

    def test_second_preview_run_does_not_resend_or_crash(self):
        deps = PushDeps(
            root=self.root,
            env=ENV,
            arxiv_client=self.client,
            rank_fn=self.fake_rank,
            summarize_fn=self.fake_summarize,
            smtp_send=mock.Mock(),
        )
        first = run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=deps,
        )
        second = run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW + timedelta(hours=1),
            deps=deps,
        )
        self.assertEqual(first.status, "preview_ready")
        self.assertIn(second.status, {"preview_ready", "no_digest"})
        deps.smtp_send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
