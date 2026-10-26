"""Offline tests for forecast preview (Task 4).

The preview is a read-only forecast: it must run without SMTP credentials, must
never send, must snapshot the production frontier ledger instead of writing to
it, and must persist a structured ``digest.json`` with Chinese title/summary for
the history page. No network, SMTP, or model call is performed.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ui_state  # noqa: E402
from profile_config import profile_sha256  # noqa: E402
from push_config import load_push_config, validate_push_config  # noqa: E402
from push_run import PushDeps, run_push  # noqa: E402
from push_state import PushState  # noqa: E402

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"
NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)

# A UI-only user has no legacy SMTP credentials in ``.env`` at all.
NO_SMTP_ENV = {
    "DEEPSEEK_API_KEY": "test-key",
    "DEEPSEEK_BASE_URL": "https://example.invalid",
    "DEEPSEEK_MODEL": "test-model",
}

ENV = {
    **NO_SMTP_ENV,
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

SUMMARIES = {
    "2601.00001": (
        "中文标题：检索增强生成综述\n"
        "研究问题：如何用证据支撑文献综述？\n"
        "核心方法：检索增强生成。\n"
        "主要结果：结果优于基线。\n"
        "阅读全文时需要确认：实验规模。"
    ),
    "2601.00002": (
        "研究问题：没有中文标题时如何回退？\n"
        "核心方法：直接摘要。\n"
        "主要结果：回退为空字符串。\n"
        "阅读全文时需要确认：无。"
    ),
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
    return {
        paper["arxiv_id"].split("v")[0]: SUMMARIES.get(
            paper["arxiv_id"].split("v")[0], "研究问题：通用摘要"
        )
        for paper in papers
    }


def never_send(*args, **kwargs):
    raise AssertionError("preview must never send")


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


def production_fingerprint(root: Path) -> str:
    """Logical hash of every production frontier row the preview must not touch."""
    path = root / "data" / "push" / "frontier.sqlite3"
    conn = sqlite3.connect(path)
    try:
        data = {
            "papers": [
                list(row)
                for row in conn.execute("SELECT * FROM papers ORDER BY paper_key")
            ],
            "watermarks": [
                list(row)
                for row in conn.execute(
                    "SELECT name,value FROM watermarks ORDER BY name"
                )
            ],
            "impressions": [
                list(row)
                for row in conn.execute(
                    "SELECT run_id,paper_key,position,score,channel "
                    "FROM impressions ORDER BY id"
                )
            ],
            "settings": [
                list(row)
                for row in conn.execute("SELECT name,value FROM settings ORDER BY name")
            ],
            "runs": conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
        }
    finally:
        conn.close()
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ForecastPreviewTests(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.root = write_root(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def deps(self, client, env=None, **kwargs):
        return PushDeps(
            root=self.root,
            env=env if env is not None else ENV,
            arxiv_client=client,
            rank_fn=fake_rank,
            summarize_fn=fake_summarize,
            **kwargs,
        )

    def run_dir(self, result) -> Path:
        return self.root / "data" / "push" / "runs" / str(result.run_id)

    def test_force_preview_runs_when_disabled_and_paused_without_smtp(self):
        self.root = write_root(self.tmp.name, enabled=False, paused=True)

        with mock.patch(
            "mail_transport.resolve_smtp",
            side_effect=AssertionError("preview must not build SMTP settings"),
        ) as resolve_smtp:
            result = run_push(
                "preview",
                config_path=self.root / "push_config.json",
                now=NOW,
                deps=self.deps(
                    FakeClient([FakeResult("2601.00001v1", "A RAG paper")]),
                    env=NO_SMTP_ENV,
                    smtp_send=never_send,
                ),
                force=True,
            )
        resolve_smtp.assert_not_called()

        self.assertEqual(result.status, "preview_ready")
        self.assertFalse(result.sent)
        self.assertGreaterEqual(result.selected_count, 1)
        run_dir = self.run_dir(result)
        self.assertTrue((run_dir / "digest_preview.html").is_file())
        self.assertTrue((run_dir / "digest.json").is_file())

    def test_forced_quiet_preview_is_preview_only_never_send(self):
        self.root = write_root(self.tmp.name, quiet=True)

        result = run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(
                FakeClient([FakeResult("2601.00001v1", "A RAG paper")]),
                smtp_send=never_send,
            ),
            force=True,
        )

        # quiet_mode must not turn a preview into a quiet send path.
        self.assertEqual(result.status, "preview_ready")
        self.assertFalse(result.sent)
        run_json = json.loads(
            (self.run_dir(result) / "run.json").read_text(encoding="utf-8")
        )
        self.assertEqual(run_json["mode"], "preview")

    def test_force_is_only_honored_for_preview(self):
        self.root = write_root(self.tmp.name, enabled=False, paused=True)

        for mode in ("scheduled", "send"):
            with self.subTest(mode=mode):
                result = run_push(
                    mode,
                    config_path=self.root / "push_config.json",
                    now=NOW,
                    deps=self.deps(ExplodingClient()),
                    force=True,
                )
                self.assertEqual(result.status, "disabled")
                self.assertFalse(
                    (self.root / "data" / "push" / "frontier.sqlite3").exists()
                )

    def test_preview_needs_no_smtp_or_mail_to(self):
        config = load_push_config(self.root / "push_config.json")

        self.assertEqual(
            validate_push_config(config, NO_SMTP_ENV, require_smtp=False), []
        )
        strict = "\n".join(validate_push_config(config, NO_SMTP_ENV))
        self.assertIn("SMTP_PASSWORD", strict)
        self.assertIn("MAIL_TO", strict)

        result = run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(
                FakeClient([FakeResult("2601.00001v1", "A RAG paper")]),
                env=NO_SMTP_ENV,
                smtp_send=never_send,
            ),
        )

        self.assertEqual(result.status, "preview_ready")
        self.assertFalse(result.sent)

    def test_preview_with_only_an_unverified_ui_recipient_needs_no_mail_to(self):
        ui_state.set_recipient(self.root, "alice@example.com")

        result = run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(
                FakeClient([FakeResult("2601.00001v1", "A RAG paper")]),
                env=NO_SMTP_ENV,
                smtp_send=never_send,
            ),
            force=True,
        )

        self.assertEqual(result.status, "preview_ready")
        rendered = json.dumps(result.__dict__, ensure_ascii=False)
        self.assertNotIn("alice@example.com", rendered)

    def test_scheduled_and_send_keep_strict_smtp_validation(self):
        for mode in ("scheduled", "send"):
            with self.subTest(mode=mode):
                result = run_push(
                    mode,
                    config_path=self.root / "push_config.json",
                    now=NOW,
                    deps=self.deps(ExplodingClient(), env=NO_SMTP_ENV),
                )
                self.assertEqual(result.status, "config_error")
                self.assertTrue(result.errors)
                self.assertFalse(result.sent)

    def test_preview_snapshots_production_and_leaves_it_unchanged(self):
        production = PushState(
            self.root / "data" / "push" / "frontier.sqlite3"
        )
        production.init_schema()
        production.record_paper(
            {"arxiv_id": "2601.88888v1", "title": "Database indexing survey"},
            "queued",
        )
        production.record_paper(
            {"arxiv_id": "2601.77777v1", "title": "A delivered RAG paper"},
            "queued",
        )
        production.mark_status("2601.77777", "delivered")
        production.set_watermark("frontier", "2026-10-20T00:00:00+00:00")
        production.set_setting("first_send_approved_at", NOW.isoformat())
        production.record_impression(
            "production-run", "2601.77777v1", 1, 88.0, "email"
        )
        before = production_fingerprint(self.root)

        result = run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(
                FakeClient(
                    [
                        FakeResult("2601.00001v1", "A RAG paper"),
                        FakeResult("2601.00002v1", "Another RAG paper"),
                    ]
                )
            ),
        )

        self.assertEqual(result.status, "preview_ready")
        self.assertEqual(production_fingerprint(self.root), before)
        self.assertEqual(
            production.get_watermark("frontier"), "2026-10-20T00:00:00+00:00"
        )
        self.assertEqual(production.status_counts().get("delivered"), 1)
        self.assertNotIn("selected", production.status_counts())

        preview_path = self.root / "data" / "push" / "preview.sqlite3"
        self.assertTrue(preview_path.is_file())
        preview = PushState(preview_path)
        preview.init_schema()
        self.assertGreaterEqual(preview.status_counts().get("selected", 0), 1)
        self.assertGreaterEqual(preview.impression_count(), 1)
        # The snapshot starts from the production rows instead of an empty DB.
        self.assertTrue(preview.is_seen("2601.88888v1"))
        self.assertTrue(preview.is_seen("2601.77777v1"))
        # The preview advances only its own watermark copy.
        self.assertEqual(preview.get_watermark("frontier"), NOW.isoformat())

    def test_preview_starts_empty_when_production_is_absent(self):
        stale = PushState(self.root / "data" / "push" / "preview.sqlite3")
        stale.init_schema()
        stale.record_paper(
            {"arxiv_id": "2601.88888v1", "title": "Stale preview row"}, "queued"
        )

        result = run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(FakeClient([FakeResult("2601.00001v1", "A RAG paper")])),
        )

        self.assertEqual(result.status, "preview_ready")
        self.assertFalse(
            (self.root / "data" / "push" / "frontier.sqlite3").exists()
        )
        preview = PushState(self.root / "data" / "push" / "preview.sqlite3")
        preview.init_schema()
        self.assertFalse(preview.is_seen("2601.88888v1"))

    def test_digest_json_has_structured_chinese_fields(self):
        result = run_push(
            "preview",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(
                FakeClient(
                    [
                        FakeResult("2601.00001v1", "A RAG paper"),
                        FakeResult("2601.00002v1", "Another RAG paper"),
                    ]
                )
            ),
        )
        self.assertEqual(result.status, "preview_ready")
        self.assertEqual(result.selected_count, 2)

        run_dir = self.run_dir(result)
        raw = (run_dir / "digest.json").read_text(encoding="utf-8")
        payload = json.loads(raw)

        self.assertEqual(payload["run_id"], result.run_id)
        self.assertEqual(payload["mode"], "preview")
        self.assertEqual(payload["created_at"], NOW.isoformat())
        self.assertIn("篇", payload["subject"])
        self.assertEqual(payload["profile_sha256"], profile_sha256(PROFILE))
        self.assertEqual(payload["warnings"], [])
        self.assertEqual(len(payload["papers"]), 2)

        meta = json.loads(
            (run_dir / "digest_meta.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["subject"], meta["subject"])
        self.assertEqual(
            [paper["paper_key"] for paper in payload["papers"]],
            meta["paper_keys"],
        )

        by_key = {paper["paper_key"]: paper for paper in payload["papers"]}
        first = by_key["2601.00001"]
        self.assertEqual(
            set(first),
            {
                "arxiv_id",
                "paper_key",
                "title",
                "title_zh",
                "reason",
                "limitation",
                "summary",
                "url",
                "score",
                "categories",
            },
        )
        self.assertEqual(first["arxiv_id"], "2601.00001v1")
        self.assertEqual(first["title"], "A RAG paper")
        self.assertEqual(first["title_zh"], "检索增强生成综述")
        self.assertIn("研究问题：如何用证据支撑文献综述？", first["summary"])
        self.assertEqual(first["reason"], "matches the profile")
        self.assertEqual(first["limitation"], "")
        self.assertEqual(first["score"], 90)
        self.assertEqual(first["url"], "https://arxiv.org/abs/2601.00001v1")
        # Categories are not persisted in the frontier ledger, so the digest
        # falls back to an empty list exactly like the rendered email does.
        self.assertEqual(first["categories"], [])

        second = by_key["2601.00002"]
        self.assertEqual(second["title_zh"], "")
        self.assertIn("研究问题：没有中文标题时如何回退？", second["summary"])

        for leaked in ("user@163.com", "test-app-code", "alice@example.com"):
            self.assertNotIn(leaked, raw)

    def test_send_run_also_writes_digest_json(self):
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        state.init_schema()
        state.set_setting("first_send_approved_at", NOW.isoformat())
        sent = []

        result = run_push(
            "send",
            config_path=self.root / "push_config.json",
            now=NOW,
            deps=self.deps(
                FakeClient([FakeResult("2601.00001v1", "A RAG paper")]),
                smtp_send=lambda settings, message: sent.append(message),
            ),
        )

        self.assertEqual(result.status, "sent")
        self.assertEqual(len(sent), 1)
        run_dir = self.run_dir(result)
        raw = (run_dir / "digest.json").read_text(encoding="utf-8")
        payload = json.loads(raw)
        self.assertEqual(payload["mode"], "send")
        self.assertEqual(payload["papers"][0]["title_zh"], "检索增强生成综述")
        self.assertNotIn("user@163.com", raw)


if __name__ == "__main__":
    unittest.main()
