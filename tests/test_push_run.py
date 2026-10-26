"""Offline tests for push orchestration (Task P7, preview slice)."""
from __future__ import annotations

import contextlib
import io
import json
import os
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

import push_run  # noqa: E402
import ui_state  # noqa: E402
from push_run import _FATAL_CLI_STATUSES, PushDeps, run_push  # noqa: E402
from push_state import PushState  # noqa: E402

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"
NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
ENV = {
    "DEEPSEEK_API_KEY": "test-key",
    "DEEPSEEK_BASE_URL": "https://example.invalid",
    "DEEPSEEK_MODEL": "test-model",
    "SMTP_HOST": "smtp.163.com",
    "SMTP_PORT": "465",
    "SMTP_SECURITY": "ssl",
    "SMTP_USER": "user@163.com",
    "SMTP_PASSWORD": "test-app-code",
    "MAIL_TO": "user@163.com",
}

PROFILE = {
    "research_goal": "test",
    "topics": ["RAG"],
    "preferred_methods": [],
    "less_interested": [],
    "top_k": 3,
    "min_score": 60,
    "search": {
        "lookback_days": 30,
        "max_candidates": 10,
        "per_query_limit": 10,
        "groups": [
            {
                "name": "RAG",
                "any_terms": ["retrieval augmented generation"],
                "require_any_terms": [],
            }
        ],
    },
}


class FakeResult:
    def __init__(self, arxiv_id: str, title: str, published: datetime):
        self._id = arxiv_id
        self.title = title
        self.summary = "A retrieval augmented generation paper."
        self.authors = [SimpleNamespace(name="A. Author")]
        self.published = published
        self.updated = published
        self.categories = ["cs.CL"]

    def get_short_id(self) -> str:
        return self._id


class FakeClient:
    def __init__(self, results=None, error=None):
        self._results = list(results or [])
        self._error = error
        self.calls = 0

    def results(self, request):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return list(self._results)


class ExplodingClient(FakeClient):
    def __init__(self):
        super().__init__()

    def results(self, request):
        raise AssertionError("network client must not be called")


def fake_rank(papers, profile, feedback):
    return {
        "evaluations": [
            {
                "arxiv_id": paper["arxiv_id"],
                "score": 90 if "RAG" in paper["title"] else 40,
                "reason": "matches the profile",
                "limitation": "",
                "feedback_refs": [],
            }
            for paper in papers
        ]
    }


def fake_summarize(papers, env):
    return {paper["arxiv_id"].split("v")[0]: "中文速览" for paper in papers}


def write_root(tmp: str, *, enabled=True, paused=False, quiet=False) -> Path:
    root = Path(tmp)
    (root / "interests.json").write_text(
        json.dumps(PROFILE, ensure_ascii=False), encoding="utf-8"
    )
    config = json.loads((ROOT / "push_config.json").read_text(encoding="utf-8"))
    config.update(enabled=enabled, paused=paused, quiet_mode=quiet)
    (root / "push_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root


class ProgressOutputTests(unittest.TestCase):
    def test_progress_is_opt_in(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PAPERLENS_PROGRESS", None)
            buffer = io.StringIO()
            with contextlib.redirect_stderr(buffer):
                push_run._progress("hidden")
            self.assertEqual(buffer.getvalue(), "")

        with mock.patch.dict(
            os.environ, {"PAPERLENS_PROGRESS": "1"}, clear=False
        ):
            buffer = io.StringIO()
            with contextlib.redirect_stderr(buffer):
                push_run._progress("正在抓取")
            self.assertIn("[PaperLens] 正在抓取", buffer.getvalue())


class PushRunPreviewTests(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.root = write_root(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def deps(self, client, **kwargs):
        return PushDeps(
            root=self.root,
            env=ENV,
            arxiv_client=client,
            rank_fn=fake_rank,
            summarize_fn=fake_summarize,
            **kwargs,
        )

    def test_disabled_exits_before_state_or_network(self):
        self.root = write_root(self.tmp.name, enabled=False)
        result = run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(ExplodingClient()),
        )
        self.assertEqual(result.status, "disabled")
        self.assertFalse((self.root / "data" / "push").exists())

    def test_paused_exits_before_network(self):
        self.root = write_root(self.tmp.name, paused=True)
        result = run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(ExplodingClient()),
        )
        self.assertEqual(result.status, "paused")

    def test_manual_bypass_is_send_only(self):
        self.root = write_root(self.tmp.name, enabled=False, paused=True)
        result = run_push(
            "scheduled",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(ExplodingClient()),
            manual=True,
        )
        self.assertEqual(result.status, "disabled")
        self.assertFalse(result.manual)

    def test_missing_smtp_env_does_not_block_preview(self):
        env = dict(ENV)
        env.pop("SMTP_PASSWORD")
        result = run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=PushDeps(
                root=self.root,
                env=env,
                arxiv_client=FakeClient(
                    [FakeResult("2601.00001v1", "A RAG paper", NOW)]
                ),
                rank_fn=fake_rank,
                summarize_fn=fake_summarize,
            ),
        )
        self.assertEqual(result.status, "preview_ready")
        self.assertFalse(result.sent)

        blocked = run_push(
            "send",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=PushDeps(root=self.root, env=env, arxiv_client=ExplodingClient()),
        )
        self.assertEqual(blocked.status, "config_error")
        self.assertTrue(blocked.errors)

    def test_preview_writes_digest_and_never_calls_smtp(self):
        results = [
            FakeResult("2601.00001v1", "A RAG paper", NOW),
            FakeResult("2601.00002v1", "An unrelated paper", NOW),
            FakeResult("2601.00003v1", "Another RAG paper", NOW),
        ]
        result = run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(
                FakeClient(results),
                smtp_send=lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("preview must not send")
                ),
            ),
        )
        self.assertEqual(result.status, "preview_ready")
        self.assertFalse(result.sent)
        self.assertEqual(result.candidate_count, 3)
        self.assertGreaterEqual(result.selected_count, 1)
        run_dir = self.root / "data" / "push" / "runs" / result.run_id
        self.assertTrue((run_dir / "run.json").is_file())
        self.assertTrue((run_dir / "digest_preview.html").is_file())
        html = (run_dir / "digest_preview.html").read_text(encoding="utf-8")
        self.assertIn("A RAG paper", html)
        preview_state = PushState(
            self.root / "data" / "push" / "preview.sqlite3"
        )
        preview_state.init_schema()
        self.assertGreaterEqual(preview_state.impression_count(), 1)

    def test_no_candidates_returns_no_digest(self):
        result = run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(FakeClient([])),
        )
        self.assertEqual(result.status, "no_digest")
        self.assertFalse(result.sent)
        self.assertIsNotNone(result.run_id)

    def test_fetch_error_is_recorded_without_send(self):
        result = run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(FakeClient(error=RuntimeError("boom"))),
        )
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.errors)
        self.assertFalse(result.sent)

    def test_quiet_mode_is_preview_only(self):
        self.root = write_root(self.tmp.name, quiet=True)
        result = run_push(
            "scheduled",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(
                FakeClient([FakeResult("2601.00001v1", "A RAG paper", NOW)]),
                smtp_send=lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("quiet mode must not send")
                ),
            ),
        )
        self.assertEqual(result.status, "quiet_preview")
        self.assertFalse(result.sent)

    def test_send_without_approval_is_refused(self):
        result = run_push(
            "send",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(FakeClient([])),
        )
        self.assertEqual(result.status, "send_not_approved")
        self.assertFalse(result.sent)

    def test_preview_does_not_touch_production_state(self):
        production = PushState(
            self.root / "data" / "push" / "frontier.sqlite3"
        )
        production.init_schema()
        production.record_paper(
            {"arxiv_id": "2601.99999v1", "title": "Delivered"}, "queued"
        )
        production.mark_status("2601.99999", "delivered")
        production.set_watermark("frontier", "2026-10-01T00:00:00+00:00")
        before_counts = production.status_counts()
        before_watermark = production.get_watermark("frontier")

        result = run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(
                FakeClient([FakeResult("2601.00001v1", "A RAG paper", NOW)])
            ),
        )
        self.assertEqual(result.status, "preview_ready")
        after_counts = production.status_counts()
        self.assertEqual(before_counts, after_counts)
        self.assertEqual(before_watermark, production.get_watermark("frontier"))
        self.assertTrue(
            (self.root / "data" / "push" / "preview.sqlite3").is_file()
        )

    def test_busy_slot_returns_busy(self):
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        state.init_schema()
        config = json.loads(
            (self.root / "push_config.json").read_text(encoding="utf-8")
        )
        slot_key = (
            f"{NOW.astimezone().date().isoformat()}"
            f"T{config['time_local']}"
        )
        self.assertTrue(
            state.claim_slot(
                slot_key, "scheduled", started_at=NOW.isoformat()
            )
        )
        result = run_push(
            "scheduled",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(ExplodingClient()),
        )
        self.assertEqual(result.status, "busy")

    def test_scheduled_with_an_unverified_ui_recipient_fails_before_network(self):
        ui_state.set_recipient(self.root, "alice@example.com")

        result = run_push(
            "scheduled",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(ExplodingClient()),
        )

        self.assertEqual(result.status, "recipient_unverified")
        self.assertFalse(result.sent)
        self.assertFalse(
            (self.root / "data" / "push" / "frontier.sqlite3").exists()
        )
        rendered = json.dumps(result.__dict__, ensure_ascii=False)
        self.assertNotIn("alice@example.com", rendered)

    def test_preview_with_an_unverified_ui_recipient_is_still_read_only(self):
        ui_state.set_recipient(self.root, "alice@example.com")

        result = run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(FakeClient([])),
        )

        self.assertEqual(result.status, "no_digest")

    def test_recipient_unverified_is_fatal_for_the_cli(self):
        self.assertIn("recipient_unverified", _FATAL_CLI_STATUSES)


class PushRunSendTests(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.root = write_root(self.tmp.name)
        self.sent = []
        self.client = FakeClient(
            [
                FakeResult("2601.00001v1", "A RAG paper", NOW),
                FakeResult("2601.00002v1", "Another RAG paper", NOW),
            ]
        )
        self.deps = PushDeps(
            root=self.root,
            env=ENV,
            arxiv_client=self.client,
            rank_fn=fake_rank,
            summarize_fn=fake_summarize,
            smtp_send=lambda settings, message: self.sent.append((settings, message)),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def approve(self):
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        state.init_schema()
        state.set_setting("first_send_approved_at", NOW.isoformat())
        return state

    def test_send_after_preview_reaches_fake_smtp_once(self):
        preview = run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps,
        )
        self.assertEqual(preview.status, "preview_ready")
        state = self.approve()
        result = run_push(
            "send",
            config_path=self.root / "push_config.json",
            now=NOW + timedelta(hours=1),
            deps=self.deps,
        )
        self.assertEqual(result.status, "sent")
        self.assertTrue(result.sent)
        self.assertEqual(len(self.sent), 1)
        self.assertIn("Subject", self.sent[0][1])
        self.assertEqual(state.status_counts().get("delivered"), preview.selected_count)

    def test_send_error_is_uncertain_and_autopauses(self):
        run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps,
        )
        state = self.approve()
        self.deps.smtp_send = lambda settings, message: (_ for _ in ()).throw(
            RuntimeError("smtp down")
        )
        result = run_push(
            "send",
            config_path=self.root / "push_config.json",
            now=NOW + timedelta(hours=1),
            deps=self.deps,
        )
        self.assertEqual(result.status, "uncertain")
        self.assertEqual(state.get_setting("auto_paused"), "true")
        blocked = run_push(
            "send",
            config_path=self.root / "push_config.json",
            now=NOW + timedelta(days=1),
            deps=self.deps,
        )
        self.assertEqual(blocked.status, "auto_paused")

    def test_second_send_does_not_resend_delivered_papers(self):
        run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps,
        )
        self.approve()
        first = run_push(
            "send",
            config_path=self.root / "push_config.json",
            now=NOW + timedelta(hours=1),
            deps=self.deps,
        )
        self.assertEqual(first.status, "sent")
        second = run_push(
            "send",
            config_path=self.root / "push_config.json",
            now=NOW + timedelta(days=1),
            deps=self.deps,
        )
        self.assertEqual(second.status, "no_digest")
        self.assertEqual(len(self.sent), 1)


if __name__ == "__main__":
    unittest.main()
