"""Offline tests for manual digest sends while auto-push is off (Task 5).

A manual send is an explicit user action: it may run with ``enabled=false`` and
``paused=true``, but it must still require a verified recipient, first-send
approval, and a non-auto-paused ledger. It never flips the auto-push switch,
ignores ``quiet_mode``, and reuses a fresh preview without fetching, ranking, or
summarizing again. Every test uses fake SMTP; no network or model call happens.
"""
from __future__ import annotations

import json
import sqlite3
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

import ui_state  # noqa: E402
from profile_config import profile_sha256  # noqa: E402
from push_run import PushDeps, run_push  # noqa: E402
from push_state import PushState  # noqa: E402

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"
NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
RECIPIENT = "alice@example.com"
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
NO_SMTP_ENV = {
    "DEEPSEEK_API_KEY": "test-key",
    "DEEPSEEK_BASE_URL": "https://example.invalid",
    "DEEPSEEK_MODEL": "test-model",
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
    def __init__(self, arxiv_id: str, title: str, published: datetime = NOW):
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
    def __init__(self, results=None):
        self._results = list(results or [])
        self.calls = 0

    def results(self, request):
        self.calls += 1
        return list(self._results)


class ExplodingClient(FakeClient):
    def results(self, request):
        raise AssertionError("fetch must not be called for a fresh preview")


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


def exploding_rank(*args, **kwargs):
    raise AssertionError("fresh preview reuse must not rank")


def exploding_summarize(*args, **kwargs):
    raise AssertionError("fresh preview reuse must not summarize")


def never_send(*args, **kwargs):
    raise AssertionError("SMTP must not be called")


def rag_results() -> list[FakeResult]:
    return [
        FakeResult("2601.00001v1", "A RAG paper"),
        FakeResult("2601.00002v1", "Another RAG paper"),
        FakeResult("2601.00003v1", "A third RAG paper"),
    ]


def write_root(
    tmp: str,
    *,
    enabled=False,
    paused=True,
    quiet=False,
    max_per_digest=3,
) -> Path:
    root = Path(tmp)
    (root / "interests.json").write_text(
        json.dumps(PROFILE, ensure_ascii=False), encoding="utf-8"
    )
    config = json.loads((ROOT / "push_config.json").read_text(encoding="utf-8"))
    config.update(
        enabled=enabled,
        paused=paused,
        quiet_mode=quiet,
        max_per_digest=max_per_digest,
    )
    (root / "push_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root


class ManualSendTests(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.root = write_root(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    # -- helpers ---------------------------------------------------------

    def deps(
        self,
        client,
        *,
        smtp_send=never_send,
        rank_fn=fake_rank,
        summarize_fn=fake_summarize,
        env=None,
    ):
        return PushDeps(
            root=self.root,
            env=env if env is not None else ENV,
            arxiv_client=client,
            rank_fn=rank_fn,
            summarize_fn=summarize_fn,
            smtp_send=smtp_send,
        )

    def verify(self, email: str = RECIPIENT):
        ui_state.set_recipient(self.root, email)
        ui_state.mark_verified(self.root, email, NOW.isoformat())

    def approve(self) -> PushState:
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        state.init_schema()
        state.set_setting("first_send_approved_at", NOW.isoformat())
        return state

    def preview(self, results=None, *, now: datetime = NOW, lookback_days=None):
        result = run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=now,
            deps=self.deps(FakeClient(results if results is not None else rag_results())),
            force=True,
            lookback_days=lookback_days,
        )
        self.assertEqual(result.status, "preview_ready")
        return result

    def write_profile(self, profile: dict) -> None:
        (self.root / "interests.json").write_text(
            json.dumps(profile, ensure_ascii=False), encoding="utf-8"
        )

    def reference(self, preview, *, count: int, **overrides) -> dict:
        reference = {
            "run_id": preview.run_id,
            "created_at": NOW.isoformat(),
            "profile_sha256": profile_sha256(PROFILE),
            "count": count,
            "recipient_hash": ui_state.recipient_hash(RECIPIENT),
            "local_date": NOW.astimezone().date().isoformat(),
        }
        reference.update(overrides)
        return ui_state.set_preview(self.root, reference)

    def manual(
        self,
        deps,
        *,
        now: datetime = NOW + timedelta(minutes=5),
        preview_run_id=None,
        lookback_days=None,
    ):
        return run_push(
            "send",
            config_path=self.root / "push_config.json",
            now=now,
            deps=deps,
            manual=True,
            preview_run_id=preview_run_id,
            lookback_days=lookback_days,
        )

    def run_json(self, result) -> dict:
        path = self.root / "data" / "push" / "runs" / str(result.run_id) / "run.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def read_config(self) -> dict:
        return json.loads(
            (self.root / "push_config.json").read_text(encoding="utf-8")
        )

    def run_slot(self, run_id: str) -> str:
        conn = sqlite3.connect(
            self.root / "data" / "push" / "frontier.sqlite3"
        )
        try:
            row = conn.execute(
                "SELECT slot_key FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(row)
        return row[0]

    # -- gate tests ------------------------------------------------------

    def test_manual_send_works_when_enabled_false_and_paused_true(self):
        self.verify()
        self.approve()
        before_config = self.read_config()
        sent = []

        result = self.manual(
            self.deps(
                FakeClient(rag_results()),
                smtp_send=lambda settings, message: sent.append((settings, message)),
            )
        )

        self.assertEqual(result.status, "sent")
        self.assertTrue(result.sent)
        self.assertTrue(result.manual)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][1]["To"], RECIPIENT)
        run_json = self.run_json(result)
        self.assertIs(run_json["manual"], True)
        self.assertEqual(run_json["mode"], "send")
        self.assertEqual(run_json["status"], "sent")
        self.assertTrue(self.run_slot(result.run_id).startswith("manual-send-"))
        # The explicit manual send must never flip the auto-push switch.
        self.assertEqual(self.read_config(), before_config)
        self.assertFalse(self.read_config()["enabled"])
        self.assertTrue(self.read_config()["paused"])

    def test_manual_send_ignores_quiet_mode(self):
        self.root = write_root(self.tmp.name, quiet=True)
        self.verify()
        self.approve()
        sent = []

        result = self.manual(
            self.deps(
                FakeClient(rag_results()),
                smtp_send=lambda settings, message: sent.append(message),
            )
        )

        self.assertEqual(result.status, "sent")
        self.assertEqual(len(sent), 1)
        self.assertEqual(self.run_json(result)["mode"], "send")

    def test_unverified_manual_send_fails_before_fetch_or_smtp(self):
        ui_state.set_recipient(self.root, RECIPIENT)  # saved but not verified

        result = self.manual(
            self.deps(ExplodingClient(), smtp_send=never_send)
        )

        self.assertEqual(result.status, "recipient_unverified")
        self.assertFalse(result.sent)
        self.assertTrue(result.manual)
        self.assertFalse(
            (self.root / "data" / "push" / "frontier.sqlite3").exists()
        )
        self.assertNotIn(RECIPIENT, json.dumps(result.__dict__, ensure_ascii=False))

    def test_unapproved_manual_send_fails_before_fetch_or_smtp(self):
        self.verify()

        result = self.manual(
            self.deps(ExplodingClient(), smtp_send=never_send)
        )

        self.assertEqual(result.status, "send_not_approved")
        self.assertFalse(result.sent)
        self.assertIs(self.run_json(result)["manual"], True)

    def test_auto_paused_manual_send_fails_before_fetch_or_smtp(self):
        self.verify()
        state = self.approve()
        state.set_setting("auto_paused", "true")

        result = self.manual(
            self.deps(ExplodingClient(), smtp_send=never_send)
        )

        self.assertEqual(result.status, "auto_paused")
        self.assertFalse(result.sent)
        self.assertIs(self.run_json(result)["manual"], True)

    def test_manual_send_with_missing_smtp_config_fails_before_network(self):
        self.verify()
        self.approve()

        result = self.manual(
            self.deps(
                ExplodingClient(),
                smtp_send=never_send,
                env=NO_SMTP_ENV,
            )
        )

        self.assertEqual(result.status, "config_error")
        self.assertFalse(result.sent)
        self.assertTrue(result.errors)
        self.assertFalse(
            (self.root / "data" / "push" / "runs").exists()
        )

    def test_manual_flag_is_ignored_for_preview_mode(self):
        result = run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(FakeClient(rag_results())),
            manual=True,
            force=True,
        )

        self.assertEqual(result.status, "preview_ready")
        self.assertFalse(result.manual)
        self.assertIs(self.run_json(result)["manual"], False)

    # -- preview reuse ---------------------------------------------------

    def test_fresh_preview_sends_exactly_its_papers_without_fetch_or_rank(self):
        self.verify()
        preview = self.preview(rag_results())
        self.approve()
        self.reference(preview, count=preview.selected_count)
        sent = []

        result = self.manual(
            self.deps(
                ExplodingClient(),
                smtp_send=lambda settings, message: sent.append(message),
                rank_fn=exploding_rank,
                summarize_fn=exploding_summarize,
            ),
            preview_run_id=preview.run_id,
        )

        self.assertEqual(result.status, "sent")
        self.assertEqual(len(sent), 1)
        self.assertIn("3 篇", sent[0]["Subject"])
        self.assertEqual(result.selected_count, 3)
        run_json = self.run_json(result)
        self.assertIs(run_json["manual"], True)
        self.assertEqual(run_json["counts"]["selected"], 3)
        self.assertEqual(run_json["counts"]["sent"], 3)
        preview_digest = json.loads(
            (
                self.root
                / "data"
                / "push"
                / "runs"
                / str(preview.run_id)
                / "digest.json"
            ).read_text(encoding="utf-8")
        )
        manual_digest = json.loads(
            (
                self.root
                / "data"
                / "push"
                / "runs"
                / str(result.run_id)
                / "digest.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            [paper["paper_key"] for paper in manual_digest["papers"]],
            [paper["paper_key"] for paper in preview_digest["papers"]],
        )
        self.assertEqual(
            run_json["paper_keys"],
            [paper["paper_key"] for paper in preview_digest["papers"]],
        )

    def test_blocked_delivery_is_absent_from_history_digest_and_email(self):
        """History must describe exactly the papers actually sent.

        ``_send_digest`` filters already-delivered papers and rebuilds the
        digest. The run files and history previously kept the pre-filter
        superset, so a blocked paper could appear in ``digest.json``,
        ``paper_keys``, and the "看邮件版" HTML even though it was not sent.
        """
        import push_history
        import send_digest as delivery

        self.verify()
        preview = self.preview(
            [
                FakeResult("2601.00001v1", "A RAG paper"),
                FakeResult("2601.00002v1", "Another RAG paper"),
            ]
        )
        self.approve()
        self.reference(preview, count=preview.selected_count)

        db = delivery.open_db(self.root)
        try:
            delivery.reserve(
                db,
                RECIPIENT,
                [{"arxiv_id": "2601.00002v1", "title": "Another RAG paper"}],
                imported=True,
            )
        finally:
            db.close()

        sent = []
        result = self.manual(
            self.deps(
                ExplodingClient(),
                smtp_send=lambda settings, message: sent.append(message),
                rank_fn=exploding_rank,
                summarize_fn=exploding_summarize,
            ),
            preview_run_id=preview.run_id,
        )

        self.assertEqual(result.status, "sent")
        self.assertEqual(len(sent), 1)
        body = sent[0].get_body().get_content()
        self.assertIn("A RAG paper", body)
        self.assertNotIn("Another RAG paper", body)
        self.assertIn("1 篇", sent[0]["Subject"])

        run_json = self.run_json(result)
        self.assertEqual(run_json["counts"]["sent"], 1)
        self.assertEqual(run_json["paper_keys"], ["2601.00001"])

        run_dir = (
            self.root / "data" / "push" / "runs" / str(result.run_id)
        )
        digest = json.loads(
            (run_dir / "digest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [paper["paper_key"] for paper in digest["papers"]],
            ["2601.00001"],
        )
        self.assertIn("1 篇", digest["subject"])
        meta = json.loads(
            (run_dir / "digest_meta.json").read_text(encoding="utf-8")
        )
        self.assertEqual(meta["paper_keys"], ["2601.00001"])
        self.assertNotIn("Another RAG paper", meta["subject"])

        html = (run_dir / "digest_preview.html").read_text(encoding="utf-8")
        text = (run_dir / "digest_preview.txt").read_text(encoding="utf-8")
        self.assertIn("A RAG paper", html)
        self.assertIn("A RAG paper", text)
        self.assertNotIn("Another RAG paper", html)
        self.assertNotIn("Another RAG paper", text)

        # The consumed preview set is still marked delivered in the production
        # frontier, including the paper blocked by the delivery ledger, so the
        # next run can fetch ahead instead of re-selecting it forever.
        frontier = PushState(
            self.root / "data" / "push" / "frontier.sqlite3"
        )
        frontier.init_schema()
        self.assertEqual(frontier.status_counts().get("delivered"), 2)
        self.assertNotIn("queued", frontier.status_counts())

        history = push_history.load_sent_history(self.root)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["run_id"], result.run_id)
        self.assertEqual(history[0]["paper_count"], 1)
        self.assertEqual(
            [paper["paper_key"] for paper in history[0]["papers"]],
            ["2601.00001"],
        )
        self.assertIn("1 篇", history[0]["subject"])

    def test_underfilled_fresh_preview_is_still_reused(self):
        self.verify()
        preview = self.preview(rag_results()[:2])
        self.assertEqual(preview.selected_count, 2)
        self.approve()
        self.reference(preview, count=2)
        sent = []

        result = self.manual(
            self.deps(
                ExplodingClient(),
                smtp_send=lambda settings, message: sent.append(message),
                rank_fn=exploding_rank,
                summarize_fn=exploding_summarize,
            ),
            preview_run_id=preview.run_id,
        )

        self.assertEqual(result.status, "sent")
        self.assertEqual(len(sent), 1)
        self.assertIn("2 篇", sent[0]["Subject"])
        self.assertEqual(result.selected_count, 2)

    def test_second_manual_send_of_the_same_preview_reports_already_sent(self):
        self.verify()
        preview = self.preview(rag_results()[:2])
        self.approve()
        self.reference(preview, count=2)
        sent = []
        deps = self.deps(
            ExplodingClient(),
            smtp_send=lambda settings, message: sent.append(message),
            rank_fn=exploding_rank,
            summarize_fn=exploding_summarize,
        )

        first = self.manual(
            deps, preview_run_id=preview.run_id, lookback_days=7
        )
        second = self.manual(
            deps,
            now=NOW + timedelta(minutes=30),
            preview_run_id=preview.run_id,
            lookback_days=7,
        )

        self.assertEqual(first.status, "sent")
        self.assertEqual(second.status, "already_sent")
        self.assertFalse(second.sent)
        self.assertEqual(second.selected_count, 2)
        self.assertEqual(second.__dict__["manual"], True)
        self.assertEqual(len(sent), 1)
        self.assertEqual(self.run_json(second)["counts"]["sent"], 0)
        self.assertIs(self.run_json(second)["manual"], True)

    def test_preview_send_advances_production_ledger_and_auto_push_resumes(self):
        """Preview reuse must not leave the frontier stuck on sent papers.

        ``_send_digest`` only UPDATEs existing frontier rows, so a paper that
        lived only in the preview snapshot would later be fetched/selected and
        report ``already_sent`` forever while the watermark stayed frozen.
        """
        self.verify()
        preview = self.preview(rag_results())
        self.approve()
        self.reference(preview, count=preview.selected_count)
        sent = []

        manual = self.manual(
            self.deps(
                ExplodingClient(),
                smtp_send=lambda settings, message: sent.append(message),
                rank_fn=exploding_rank,
                summarize_fn=exploding_summarize,
            ),
            preview_run_id=preview.run_id,
        )
        self.assertEqual(manual.status, "sent")
        self.assertEqual(len(sent), 1)

        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        state.init_schema()
        counts = state.status_counts()
        self.assertEqual(counts.get("delivered"), 3)
        self.assertNotIn("selected", counts)
        self.assertEqual(state.impression_count(), 3)

        raw = self.read_config()
        raw.update(enabled=True, paused=False)
        (self.root / "push_config.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        client = FakeClient(rag_results())
        deps = self.deps(
            client,
            smtp_send=lambda settings, message: sent.append(message),
        )

        first = run_push(
            "scheduled",
            config_path=self.root / "push_config.json",
            now=NOW + timedelta(hours=1),
            deps=deps,
        )
        # The previewed papers are already delivered. The fetch still runs and
        # finds nothing sendable, but they must not be left "selected".
        self.assertEqual(first.status, "no_digest")
        self.assertEqual(client.calls, 1)
        self.assertNotIn("selected", state.status_counts())

        # The following automatic run must fetch and send a genuinely new paper.
        client._results.append(FakeResult("2601.00004v1", "A brand new RAG paper"))
        # The daily scheduled slot is unique per local date, so use the next
        # day for the follow-up automatic run.
        second = run_push(
            "scheduled",
            config_path=self.root / "push_config.json",
            now=NOW + timedelta(days=1, hours=1),
            deps=deps,
        )
        self.assertEqual(second.status, "sent")
        self.assertEqual(client.calls, 2)
        self.assertEqual(len(sent), 2)
        self.assertIn("1 篇", sent[-1]["Subject"])
        self.assertEqual(state.status_counts().get("delivered"), 4)

    # -- stale preview fallbacks ----------------------------------------

    def test_stale_preview_after_profile_change_regenerates(self):
        self.verify()
        preview = self.preview([FakeResult("2601.00001v1", "A RAG paper")])
        self.approve()

        profile = json.loads(
            (self.root / "interests.json").read_text(encoding="utf-8")
        )
        profile["topics"] = ["RAG", "agents"]
        profile["top_k"] = 2
        (self.root / "interests.json").write_text(
            json.dumps(profile, ensure_ascii=False), encoding="utf-8"
        )

        client = FakeClient([FakeResult("2601.00002v1", "A new RAG paper")])
        sent = []
        result = self.manual(
            self.deps(
                client,
                smtp_send=lambda settings, message: sent.append(message),
            ),
            preview_run_id=preview.run_id,
        )

        self.assertEqual(result.status, "sent")
        self.assertEqual(client.calls, 1)
        self.assertEqual(len(sent), 1)
        manual_digest = json.loads(
            (
                self.root
                / "data"
                / "push"
                / "runs"
                / str(result.run_id)
                / "digest.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            [paper["paper_key"] for paper in manual_digest["papers"]],
            ["2601.00002"],
        )

    def test_stale_preview_after_count_change_regenerates(self):
        self.root = write_root(self.tmp.name, max_per_digest=2)
        self.verify()
        preview = self.preview(rag_results()[:2])
        self.assertEqual(preview.selected_count, 2)
        self.approve()

        raw = self.read_config()
        raw["max_per_digest"] = 1
        (self.root / "push_config.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        client = FakeClient(rag_results()[:2])
        sent = []
        result = self.manual(
            self.deps(
                client,
                smtp_send=lambda settings, message: sent.append(message),
            ),
            preview_run_id=preview.run_id,
        )

        self.assertEqual(result.status, "sent")
        self.assertEqual(client.calls, 1)
        self.assertEqual(len(sent), 1)
        self.assertEqual(result.selected_count, 1)
        self.assertIn("1 篇", sent[0]["Subject"])

    def test_missing_preview_reference_regenerates(self):
        """No ui_state reference means freshness cannot be proven: regenerate."""
        self.verify()
        preview = self.preview(rag_results())
        self.approve()

        client = FakeClient(rag_results())
        sent = []
        result = self.manual(
            self.deps(
                client,
                smtp_send=lambda settings, message: sent.append(message),
            ),
            preview_run_id=preview.run_id,
        )

        self.assertEqual(result.status, "sent")
        self.assertEqual(client.calls, 1)
        self.assertEqual(len(sent), 1)

    def test_preview_reference_for_another_recipient_regenerates(self):
        self.verify()
        preview = self.preview([FakeResult("2601.00001v1", "A RAG paper")])
        self.approve()
        self.reference(
            preview,
            count=preview.selected_count,
            recipient_hash=ui_state.recipient_hash("bob@example.com"),
        )

        client = FakeClient([FakeResult("2601.00002v1", "A new RAG paper")])
        sent = []
        result = self.manual(
            self.deps(
                client,
                smtp_send=lambda settings, message: sent.append(message),
            ),
            preview_run_id=preview.run_id,
        )

        self.assertEqual(result.status, "sent")
        self.assertEqual(client.calls, 1)
        self.assertEqual(len(sent), 1)

    def test_preview_digest_from_an_earlier_day_regenerates(self):
        """The digest's own created_at day is checked, not only the reference."""
        self.verify()
        preview = self.preview([FakeResult("2601.00001v1", "A RAG paper")])
        self.approve()
        old_created = NOW - timedelta(days=3)
        digest_path = (
            self.root
            / "data"
            / "push"
            / "runs"
            / str(preview.run_id)
            / "digest.json"
        )
        payload = json.loads(digest_path.read_text(encoding="utf-8"))
        payload["created_at"] = old_created.isoformat()
        digest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manual_now = NOW + timedelta(days=3)
        self.reference(
            preview,
            count=preview.selected_count,
            created_at=old_created.isoformat(),
            local_date=manual_now.astimezone().date().isoformat(),
        )

        client = FakeClient([FakeResult("2601.00002v1", "A new RAG paper")])
        sent = []
        result = self.manual(
            self.deps(
                client,
                smtp_send=lambda settings, message: sent.append(message),
            ),
            now=manual_now,
            preview_run_id=preview.run_id,
        )

        self.assertEqual(result.status, "sent")
        self.assertEqual(client.calls, 1)
        self.assertEqual(len(sent), 1)

    def test_stale_preview_reference_day_regenerates(self):
        self.verify()
        preview = self.preview([FakeResult("2601.00001v1", "A RAG paper")])
        self.approve()
        self.reference(
            preview,
            count=preview.selected_count,
            local_date="2000-01-01",
        )

        client = FakeClient([FakeResult("2601.00002v1", "A new RAG paper")])
        sent = []
        result = self.manual(
            self.deps(
                client,
                smtp_send=lambda settings, message: sent.append(message),
            ),
            preview_run_id=preview.run_id,
        )

        self.assertEqual(result.status, "sent")
        self.assertEqual(client.calls, 1)
        self.assertEqual(len(sent), 1)

    # -- same-day topic pivot -------------------------------------------

    def test_topic_pivot_requeues_a_paper_evaluated_under_the_old_profile(self):
        self.verify()
        self.approve()
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        state.init_schema()
        state.record_paper(
            {
                "arxiv_id": "2601.00001v1",
                "title": "An agentic retrieval paper",
                "abstract": "A retrieval augmented generation paper.",
                "published": NOW.isoformat(),
                "url": "https://arxiv.org/abs/2601.00001v1",
                "categories": ["cs.CL"],
            },
            "queued",
        )
        state.mark_status(
            "2601.00001", "evaluated", llm_score=10.0, final_score=10.0
        )
        profile_b = json.loads(json.dumps(PROFILE))
        profile_b["research_goal"] = "agents"
        profile_b["topics"] = ["agents"]
        profile_b["search"]["groups"] = [
            {
                "name": "Agents",
                "any_terms": ["language model agent"],
                "require_any_terms": [],
            }
        ]
        self.write_profile(profile_b)
        seen_profiles = []

        def rank_under_current_profile(papers, profile, feedback):
            seen_profiles.append(profile["search"]["groups"])
            return {
                "evaluations": [
                    {
                        "arxiv_id": paper["arxiv_id"],
                        "score": 92,
                        "reason": "matches the new topic",
                        "limitation": "",
                        "feedback_refs": [],
                    }
                    for paper in papers
                ]
            }

        sent = []
        result = self.manual(
            self.deps(
                FakeClient(
                    [FakeResult("2601.00001v1", "An agentic retrieval paper")]
                ),
                smtp_send=lambda settings, message: sent.append(message),
                rank_fn=rank_under_current_profile,
            ),
            lookback_days=7,
        )

        self.assertEqual(result.status, "sent")
        self.assertEqual(len(sent), 1)
        # The new topic, not the profile that produced the stale evaluation,
        # ranked the re-opened paper.
        self.assertEqual(seen_profiles[0][0]["name"], "Agents")
        self.assertEqual(state.status_counts().get("delivered"), 1)

    def test_second_same_day_manual_send_with_a_new_preview_sends_the_new_paper(
        self,
    ):
        self.verify()
        self.approve()
        sent = []
        preview_a = self.preview(
            [FakeResult("2601.00001v1", "A RAG paper")], lookback_days=7
        )
        self.reference(preview_a, count=preview_a.selected_count)
        first = self.manual(
            self.deps(
                ExplodingClient(),
                smtp_send=lambda settings, message: sent.append(message),
                rank_fn=exploding_rank,
                summarize_fn=exploding_summarize,
            ),
            preview_run_id=preview_a.run_id,
            lookback_days=7,
        )
        self.assertEqual(first.status, "sent")

        profile_b = json.loads(json.dumps(PROFILE))
        profile_b["research_goal"] = "agents"
        profile_b["topics"] = ["agents"]
        profile_b["search"]["groups"] = [
            {
                "name": "Agents",
                "any_terms": ["language model agent"],
                "require_any_terms": [],
            }
        ]
        self.write_profile(profile_b)
        preview_b = self.preview(
            [
                FakeResult("2601.00001v1", "A RAG paper"),
                FakeResult("2601.00002v1", "A RAG agent paper"),
            ],
            now=NOW + timedelta(minutes=10),
            lookback_days=7,
        )
        # The already-delivered paper stays delivered; only the new paper is
        # part of the second forecast.
        self.assertEqual(preview_b.selected_count, 1)
        self.reference(
            preview_b,
            count=preview_b.selected_count,
            created_at=(NOW + timedelta(minutes=10)).isoformat(),
            profile_sha256=profile_sha256(profile_b),
        )

        second = self.manual(
            self.deps(
                ExplodingClient(),
                smtp_send=lambda settings, message: sent.append(message),
                rank_fn=exploding_rank,
                summarize_fn=exploding_summarize,
            ),
            now=NOW + timedelta(minutes=20),
            preview_run_id=preview_b.run_id,
            lookback_days=7,
        )

        self.assertEqual(second.status, "sent")
        self.assertNotEqual(second.status, "already_sent")
        self.assertEqual(len(sent), 2)
        body = sent[-1].get_body().get_content()
        self.assertIn("A RAG agent paper", body)
        self.assertNotIn("A RAG paper", body)

    def test_lookback_prioritizes_freshly_fetched_papers_over_requeued_rows(self):
        """Requeued old rows must not fill the rank pool before new papers.

        A topic pivot re-opens every evaluated paper in the window; those rows
        keep their older ``first_seen_at``. With a plain oldest-first pool the
        rank limit can be consumed entirely by them, so the paper this fetch
        just returned never reaches ranking or selection.
        """
        self.verify()
        state = self.approve()
        old_published = NOW - timedelta(days=1)
        for index in range(12):
            state.record_paper(
                {
                    "arxiv_id": f"2601.810{index:02d}v1",
                    "title": f"Old evaluated paper {index}",
                    "abstract": "A retrieval augmented generation paper.",
                    "published": old_published.isoformat(),
                    "categories": ["cs.CL"],
                },
                "evaluated",
            )
        ranked_ids: list[str] = []

        def rank_fresh_first(papers, profile, feedback):
            ranked_ids.extend(paper["arxiv_id"] for paper in papers)
            return {
                "evaluations": [
                    {
                        "arxiv_id": paper["arxiv_id"],
                        "score": (
                            99
                            if paper["arxiv_id"].startswith("2601.99999")
                            else 90
                        ),
                        "reason": "matches the new topic",
                        "limitation": "",
                        "feedback_refs": [],
                    }
                    for paper in papers
                ]
            }

        sent: list = []
        result = self.manual(
            self.deps(
                FakeClient(
                    [FakeResult("2601.99999v1", "Fresh new topic paper")]
                ),
                smtp_send=lambda settings, message: sent.append(message),
                rank_fn=rank_fresh_first,
            ),
            lookback_days=7,
        )

        self.assertEqual(result.status, "sent")
        self.assertTrue(
            any(key.startswith("2601.99999") for key in ranked_ids),
            ranked_ids,
        )
        self.assertEqual(len(sent), 1)
        body = sent[0].get_body().get_content()
        self.assertIn("Fresh new topic paper", body)

    def test_invalid_lookback_days_is_rejected_before_claiming_a_slot(self):
        self.verify()
        self.approve()
        client = FakeClient([])
        for value in (0, -1, 366, "7", True):
            with self.subTest(value=value):
                result = run_push(
                    "send",
                    config_path=self.root / "push_config.json",
                    now=NOW,
                    deps=self.deps(client),
                    manual=True,
                    lookback_days=value,  # type: ignore[arg-type]
                )
                self.assertEqual(result.status, "config_error", result.errors)
                self.assertIsNone(result.run_id)
                self.assertIsNone(result.run_dir)
        self.assertEqual(client.calls, 0)
        conn = sqlite3.connect(
            self.root / "data" / "push" / "frontier.sqlite3"
        )
        try:
            count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(count, 0)

    def test_requeue_failure_finishes_the_claimed_run(self):
        self.verify()
        self.approve()
        with mock.patch.object(
            PushState,
            "requeue_for_lookback",
            side_effect=RuntimeError("lookback failed"),
        ):
            result = self.manual(self.deps(FakeClient([])), lookback_days=7)

        self.assertEqual(result.status, "failed")
        self.assertIsNotNone(result.run_id)
        conn = sqlite3.connect(
            self.root / "data" / "push" / "frontier.sqlite3"
        )
        try:
            row = conn.execute(
                "SELECT status FROM runs WHERE run_id=?", (result.run_id,)
            ).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "failed")

    def test_manual_send_reusing_a_preview_does_not_requeue_production(self):
        """A reused preview must not leave a rank backlog in production.

        The preview ranks only inside its snapshot. Requeueing production on
        the way to reusing that preview creates queued rows nothing will rank,
        and the next scheduled run spends its pool on them.
        """
        self.verify()
        self.approve()
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        old_published = NOW - timedelta(days=1)
        for index in range(25):
            state.record_paper(
                {
                    "arxiv_id": f"2601.820{index:02d}v1",
                    "title": f"Old evaluated paper {index}",
                    "abstract": "A retrieval augmented generation paper.",
                    "published": old_published.isoformat(),
                    "categories": ["cs.CL"],
                },
                "evaluated",
            )
        preview = self.preview(
            [FakeResult("2601.99998v1", "A RAG paper")], lookback_days=7
        )
        self.reference(preview, count=preview.selected_count)
        sent: list = []

        result = self.manual(
            self.deps(
                ExplodingClient(),
                smtp_send=lambda settings, message: sent.append(message),
            ),
            preview_run_id=preview.run_id,
            lookback_days=7,
        )

        self.assertEqual(result.status, "sent")
        counts = state.status_counts()
        self.assertEqual(counts.get("evaluated", 0), 25, counts)
        self.assertEqual(counts.get("queued", 0), 0, counts)
        self.assertEqual(counts.get("delivered", 0), 1, counts)

    def test_preview_reuse_clears_stale_selections_before_the_next_run(self):
        """A reused preview must not leave an old selected row to be sent.

        Skipping the full lookback requeue stops the backlog, but a
        ``selected`` row left by an interrupted run would otherwise survive and
        the next scheduled run would carry it over without ranking it under the
        current profile.
        """
        self.verify()
        state = self.approve()
        state.record_paper(
            {
                "arxiv_id": "2601.70001v1",
                "title": "Stale old selection",
                "abstract": "A retrieval augmented generation paper.",
                "published": (NOW - timedelta(days=1)).isoformat(),
                "categories": ["cs.CL"],
            },
            "queued",
        )
        state.mark_status(
            "2601.70001",
            "selected",
            '{"reason":"selected under the old profile"}',
            llm_score=95,
            final_score=95,
        )
        preview = self.preview(
            [FakeResult("2601.99997v1", "A RAG paper")], lookback_days=7
        )
        self.reference(preview, count=preview.selected_count)
        sent: list = []

        manual = self.manual(
            self.deps(
                ExplodingClient(),
                smtp_send=lambda settings, message: sent.append(message),
            ),
            preview_run_id=preview.run_id,
            lookback_days=7,
        )
        self.assertEqual(manual.status, "sent")
        self.assertEqual(state.selected(10), [])

        raw = self.read_config()
        raw.update(enabled=True, paused=False)
        (self.root / "push_config.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        ranked_ids: list[str] = []

        def rank_fresh_first(papers, profile, feedback):
            ranked_ids.extend(paper["arxiv_id"] for paper in papers)
            return {
                "evaluations": [
                    {
                        "arxiv_id": paper["arxiv_id"],
                        "score": (
                            99
                            if paper["arxiv_id"].startswith("2601.99996")
                            else 40
                        ),
                        "reason": "matches the topic",
                        "limitation": "",
                        "feedback_refs": [],
                    }
                    for paper in papers
                ]
            }

        result = run_push(
            "scheduled",
            config_path=self.root / "push_config.json",
            now=NOW + timedelta(hours=1),
            deps=self.deps(
                FakeClient(
                    [FakeResult("2601.99996v1", "Fresh scheduled paper")]
                ),
                smtp_send=lambda settings, message: sent.append(message),
                rank_fn=rank_fresh_first,
            ),
        )

        self.assertEqual(result.status, "sent")
        self.assertTrue(
            any(key.startswith("2601.99996") for key in ranked_ids),
            ranked_ids,
        )
        body = sent[-1].get_body().get_content()
        self.assertIn("Fresh scheduled paper", body)
        self.assertNotIn("Stale old selection", body)

    def test_scheduled_run_prioritizes_fresh_fetch_over_queued_backlog(self):
        """Leftover requeued rows must not hide a scheduled run's new paper."""
        self.verify()
        self.approve()
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        old_published = NOW - timedelta(days=1)
        for index in range(25):
            state.record_paper(
                {
                    "arxiv_id": f"2601.830{index:02d}v1",
                    "title": f"Queued backlog paper {index}",
                    "abstract": "A retrieval augmented generation paper.",
                    "published": old_published.isoformat(),
                    "categories": ["cs.CL"],
                },
                "queued",
            )
        raw = self.read_config()
        raw.update(enabled=True, paused=False)
        (self.root / "push_config.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        ranked_ids: list[str] = []

        def rank_fresh_first(papers, profile, feedback):
            ranked_ids.extend(paper["arxiv_id"] for paper in papers)
            return {
                "evaluations": [
                    {
                        "arxiv_id": paper["arxiv_id"],
                        "score": (
                            99
                            if paper["arxiv_id"].startswith("2601.99999")
                            else 40
                        ),
                        "reason": "matches the topic",
                        "limitation": "",
                        "feedback_refs": [],
                    }
                    for paper in papers
                ]
            }

        sent: list = []
        result = run_push(
            "scheduled",
            config_path=self.root / "push_config.json",
            now=NOW + timedelta(minutes=10),
            deps=self.deps(
                FakeClient(
                    [FakeResult("2601.99999v1", "Fresh incremental paper")]
                ),
                smtp_send=lambda settings, message: sent.append(message),
                rank_fn=rank_fresh_first,
            ),
        )

        self.assertEqual(result.status, "sent")
        self.assertTrue(
            any(key.startswith("2601.99999") for key in ranked_ids),
            ranked_ids,
        )
        body = sent[0].get_body().get_content()
        self.assertIn("Fresh incremental paper", body)

    def seed_scheduled_run(
        self, slot_key: str, outcome: str, *, sent: int = 0
    ) -> str:
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        state.init_schema()
        self.assertTrue(
            state.claim_slot(slot_key, "scheduled", started_at=NOW.isoformat())
        )
        run_id = state.last_run_id
        run_dir = self.root / "data" / "push" / "runs" / str(run_id)
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "mode": "scheduled",
                    "status": outcome,
                    "counts": {"sent": sent},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        table_status = (
            "failed" if outcome in {"failed", "uncertain"} else "completed"
        )
        state.finish_run(str(run_id), table_status, {"sent": sent})
        return str(run_id)

    def test_scheduled_run_retries_after_a_no_digest_slot(self):
        self.verify()
        self.approve()
        raw = self.read_config()
        raw.update(enabled=True, paused=False)
        (self.root / "push_config.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        slot_key = (
            f"{NOW.astimezone().date().isoformat()}T{raw['time_local']}"
        )
        old_run = self.seed_scheduled_run(slot_key, "no_digest")
        sent: list = []

        result = run_push(
            "scheduled",
            config_path=self.root / "push_config.json",
            now=NOW + timedelta(minutes=5),
            deps=self.deps(
                FakeClient([FakeResult("2601.99990v1", "A RAG paper")]),
                smtp_send=lambda settings, message: sent.append(message),
            ),
        )

        self.assertEqual(result.status, "sent")
        self.assertNotEqual(result.run_id, old_run)
        self.assertEqual(len(sent), 1)
        # The new run owns today's slot; the old attempt stays in history
        # under a unique archived slot key.
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        self.assertEqual(state.slot_run(slot_key)["run_id"], result.run_id)
        archived = state.slot_run(f"{slot_key}:attempt:{old_run}")
        self.assertIsNotNone(archived)
        self.assertEqual(archived["run_id"], old_run)

    def test_scheduled_run_keeps_the_slot_after_a_sent_run(self):
        self.verify()
        self.approve()
        raw = self.read_config()
        raw.update(enabled=True, paused=False)
        (self.root / "push_config.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        slot_key = (
            f"{NOW.astimezone().date().isoformat()}T{raw['time_local']}"
        )
        self.seed_scheduled_run(slot_key, "sent", sent=1)
        client = FakeClient([])

        result = run_push(
            "scheduled",
            config_path=self.root / "push_config.json",
            now=NOW + timedelta(minutes=5),
            deps=self.deps(client),
        )

        self.assertEqual(result.status, "busy")
        self.assertIsNone(result.run_id)
        self.assertEqual(client.calls, 0)

    def test_scheduled_time_change_creates_a_new_test_slot(self):
        """Changing the configured time is an explicit new same-day attempt.

        The day is not blocked by an earlier successful time; the delivery
        ledger still prevents sending the same paper twice.
        """
        self.verify()
        self.approve()
        raw = self.read_config()
        raw.update(enabled=True, paused=False)
        (self.root / "push_config.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        old_slot = (
            f"{NOW.astimezone().date().isoformat()}T{raw['time_local']}"
        )
        self.seed_scheduled_run(old_slot, "sent", sent=1)
        new_time = "01:05" if raw["time_local"] != "01:05" else "01:06"
        raw["time_local"] = new_time
        (self.root / "push_config.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        sent: list = []

        result = run_push(
            "scheduled",
            config_path=self.root / "push_config.json",
            now=NOW + timedelta(minutes=5),
            deps=self.deps(
                FakeClient([FakeResult("2601.99991v1", "A new RAG paper")]),
                smtp_send=lambda settings, message: sent.append(message),
            ),
        )

        self.assertEqual(result.status, "sent")
        self.assertEqual(len(sent), 1)
        new_slot = (
            f"{NOW.astimezone().date().isoformat()}T{new_time}"
        )
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        self.assertEqual(state.slot_run(new_slot)["run_id"], result.run_id)
        self.assertEqual(state.slot_run(old_slot)["status"], "completed")

    def test_scheduled_run_keeps_the_slot_after_an_uncertain_run(self):
        self.verify()
        self.approve()
        raw = self.read_config()
        raw.update(enabled=True, paused=False)
        (self.root / "push_config.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        slot_key = (
            f"{NOW.astimezone().date().isoformat()}T{raw['time_local']}"
        )
        self.seed_scheduled_run(slot_key, "uncertain")
        client = FakeClient([])

        result = run_push(
            "scheduled",
            config_path=self.root / "push_config.json",
            now=NOW + timedelta(minutes=5),
            deps=self.deps(client),
        )

        self.assertEqual(result.status, "busy")
        self.assertEqual(client.calls, 0)

    def test_scheduled_run_without_lookback_keeps_the_incremental_watermark(self):
        self.verify()
        self.approve()
        raw = self.read_config()
        raw.update(enabled=True, paused=False)
        (self.root / "push_config.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        state.init_schema()
        state.set_watermark(
            "frontier", (NOW - timedelta(days=1)).isoformat()
        )
        old_paper = FakeResult(
            "2601.00005v1",
            "A five-day-old RAG paper",
            published=NOW - timedelta(days=5),
        )
        client = FakeClient([old_paper])

        scheduled = run_push(
            "scheduled",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(client),
        )

        # The automatic path keeps the incremental window: a paper older than
        # watermark - overlap_hours is not fetched, so nothing advances.
        self.assertEqual(scheduled.status, "no_digest")
        self.assertEqual(client.calls, 1)
        self.assertFalse(state.is_seen("2601.00005"))

        sent = []
        manual = self.manual(
            self.deps(
                FakeClient([old_paper]),
                smtp_send=lambda settings, message: sent.append(message),
            ),
            now=NOW + timedelta(minutes=30),
            lookback_days=7,
        )

        self.assertEqual(manual.status, "sent")
        self.assertEqual(len(sent), 1)
        self.assertEqual(state.status_counts().get("delivered"), 1)


if __name__ == "__main__":
    unittest.main()
