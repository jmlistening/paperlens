"""Offline tests for the advanced sent-push history (Task 7).

Only real sends (``status == "sent"`` with a send-like mode) may appear. New
runs are read from ``digest.json``; old runs fall back to the rendered email
HTML and are marked ``legacy``. The reader never touches ``.env``/SMTP or
recipient state, and the page is read-only: rendering it performs no network,
model, or mail call.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from push_history import load_sent_history  # noqa: E402

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"
PAGE = ROOT / "advanced" / "history.py"

ITEM_KEYS = {
    "run_id",
    "sent_at",
    "mode",
    "manual",
    "subject",
    "paper_count",
    "papers",
    "digest_html",
    "legacy",
}
PAPER_KEYS = {
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
}

TITLE_EN = "Retrieval with Evidence-Aware Scoring"
TITLE_ZH = "面向证据感知检索的新方法"
SUMMARY = (
    f"中文标题：{TITLE_ZH}\n\n"
    "研究问题：如何让检索分数反映证据质量。\n\n"
    "核心方法：加入证据感知的重排器。\n\n"
    "主要结果：在三个数据集上提升。"
)


def paper(**overrides):
    value = {
        "arxiv_id": "2610.00001",
        "paper_key": "2610.00001",
        "title": TITLE_EN,
        "title_zh": TITLE_ZH,
        "reason": "与你的研究兴趣高度相关。",
        "limitation": "只在英文数据上验证。",
        "summary": SUMMARY,
        "url": "https://arxiv.org/abs/2610.00001",
        "score": 88,
        "categories": ["cs.CL", "cs.IR"],
    }
    value.update(overrides)
    return value


def digest_payload(run_id, papers=None, *, subject="PaperLens｜前沿论文推荐", mode="send"):
    return {
        "run_id": run_id,
        "mode": mode,
        "created_at": "2026-10-20T10:00:00+00:00",
        "subject": subject,
        "profile_sha256": "a" * 64,
        "papers": [paper()] if papers is None else papers,
        "warnings": [],
    }


class HistoryFixture(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.root = Path(self.tmp.name)
        self.runs = self.root / "data" / "push" / "runs"

    def tearDown(self):
        self.tmp.cleanup()

    def write_run(
        self,
        run_id,
        *,
        status="sent",
        mode="send",
        finished_at="2026-10-20T10:05:00+00:00",
        started_at="2026-10-20T10:00:00+00:00",
        manual=False,
        subject="PaperLens｜前沿论文推荐（1 篇）",
        counts=None,
        paper_keys=None,
        digest=None,
        html="<html><body>邮件版</body></html>",
        extra=None,
        run_payload=None,
    ):
        run_dir = self.runs / run_id
        run_dir.mkdir(parents=True)
        if run_payload is None:
            payload = {
                "run_id": run_id,
                "mode": mode,
                "status": status,
                "started_at": started_at,
                "finished_at": finished_at,
                "errors": [],
                "counts": counts if counts is not None else {"selected": 1, "sent": 1},
                "paper_keys": paper_keys if paper_keys is not None else ["2610.00001"],
                "digest_subject": subject,
                "manual": manual,
            }
            if extra:
                payload.update(extra)
        else:
            payload = run_payload
        (run_dir / "run.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        if digest is not None:
            (run_dir / "digest.json").write_text(
                json.dumps(digest, ensure_ascii=False), encoding="utf-8"
            )
        if html is not None:
            (run_dir / "digest_preview.html").write_text(html, encoding="utf-8")
        return run_dir


class SentFilterTests(HistoryFixture):
    def test_includes_only_sent_send_like_modes(self):
        self.write_run("sent-send", status="sent", mode="send")
        self.write_run("sent-scheduled", status="sent", mode="scheduled")
        self.write_run("sent-manual", status="sent", mode="manual")
        self.write_run("preview", status="preview_ready", mode="preview")
        self.write_run("quiet", status="quiet_preview", mode="scheduled")
        self.write_run("failed", status="failed", mode="send")
        self.write_run("no-digest", status="no_digest", mode="send")
        self.write_run("already", status="already_sent", mode="send")
        self.write_run("send-preview-mode", status="sent", mode="preview")

        rows = load_sent_history(self.root)

        self.assertEqual(
            {row["run_id"] for row in rows},
            {"sent-send", "sent-scheduled", "sent-manual"},
        )

    def test_preview_and_failed_runs_do_not_leak_into_history(self):
        self.write_run(
            "preview",
            status="preview_ready",
            mode="preview",
            digest=digest_payload("preview"),
        )
        self.write_run("failed", status="failed", mode="send")

        self.assertEqual(load_sent_history(self.root), [])

    def test_missing_run_json_and_malformed_json_are_skipped(self):
        (self.runs / "empty").mkdir(parents=True)
        bad = self.runs / "bad"
        bad.mkdir(parents=True)
        (bad / "run.json").write_text("{not json", encoding="utf-8")
        listy = self.runs / "listy"
        listy.mkdir(parents=True)
        (listy / "run.json").write_text("[]", encoding="utf-8")
        self.write_run("good")

        rows = load_sent_history(self.root)

        self.assertEqual([row["run_id"] for row in rows], ["good"])

    def test_empty_or_missing_runs_directory_returns_empty_list(self):
        self.assertEqual(load_sent_history(self.root), [])
        self.assertEqual(load_sent_history(self.root / "does-not-exist"), [])


class OrderingAndLimitTests(HistoryFixture):
    def test_sorted_by_finished_at_descending(self):
        self.write_run("old", finished_at="2026-10-01T10:00:00+00:00")
        self.write_run("newest", finished_at="2026-10-25T10:00:00+00:00")
        self.write_run("middle", finished_at="2026-10-10T10:00:00+00:00")

        rows = load_sent_history(self.root)

        self.assertEqual([row["run_id"] for row in rows], ["newest", "middle", "old"])

    def test_falls_back_to_started_at_when_finished_at_is_missing(self):
        self.write_run(
            "no-finished",
            started_at="2026-10-24T10:00:00+00:00",
            run_payload={
                "run_id": "no-finished",
                "mode": "send",
                "status": "sent",
                "started_at": "2026-10-24T10:00:00+00:00",
                "counts": {"selected": 1},
                "paper_keys": ["2610.00001"],
                "digest_subject": "旧任务",
            },
        )
        self.write_run("finished", finished_at="2026-10-20T10:00:00+00:00")

        rows = load_sent_history(self.root)

        self.assertEqual([row["run_id"] for row in rows], ["no-finished", "finished"])
        self.assertEqual(rows[0]["sent_at"], "2026-10-24T10:00:00+00:00")

    def test_falls_back_to_started_at_when_finished_at_is_malformed(self):
        self.write_run(
            "bad-finished",
            finished_at="not-a-timestamp",
            started_at="2026-10-24T10:00:00+00:00",
        )
        self.write_run("valid", finished_at="2026-10-20T10:00:00+00:00")

        rows = load_sent_history(self.root)

        self.assertEqual([row["run_id"] for row in rows], ["bad-finished", "valid"])
        self.assertEqual(rows[0]["sent_at"], "2026-10-24T10:00:00+00:00")

    def test_timezone_offsets_are_compared_as_instants(self):
        # 2026-10-21 09:00 +08:00 == 2026-10-21 01:00 UTC, earlier than 02:00 UTC.
        self.write_run(
            "shanghai", finished_at="2026-10-21T09:00:00+08:00"
        )
        self.write_run("utc", finished_at="2026-10-21T02:00:00+00:00")

        rows = load_sent_history(self.root)

        self.assertEqual([row["run_id"] for row in rows], ["utc", "shanghai"])

    def test_limit_applies_after_sorting(self):
        for day in range(1, 6):
            self.write_run(
                f"run-{day}", finished_at=f"2026-10-0{day}T10:00:00+00:00"
            )

        rows = load_sent_history(self.root, limit=2)

        self.assertEqual([row["run_id"] for row in rows], ["run-5", "run-4"])
        self.assertEqual(len(load_sent_history(self.root)), 5)

    def test_non_positive_limit_returns_no_rows(self):
        self.write_run("run")

        self.assertEqual(load_sent_history(self.root, limit=0), [])


class ItemShapeTests(HistoryFixture):
    def test_structured_run_has_exact_item_keys(self):
        run_dir = self.write_run("structured", digest=digest_payload("structured"))

        rows = load_sent_history(self.root)

        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), ITEM_KEYS)
        self.assertEqual(rows[0]["run_id"], "structured")
        self.assertEqual(rows[0]["mode"], "send")
        self.assertFalse(rows[0]["manual"])
        self.assertEqual(rows[0]["subject"], "PaperLens｜前沿论文推荐（1 篇）")
        self.assertEqual(rows[0]["paper_count"], 1)
        self.assertFalse(rows[0]["legacy"])
        self.assertEqual(
            rows[0]["digest_html"], str(run_dir / "digest_preview.html")
        )

    def test_structured_papers_keep_chinese_fields(self):
        self.write_run("structured", digest=digest_payload("structured"))

        item = load_sent_history(self.root)[0]
        stored = item["papers"][0]

        self.assertEqual(set(stored), PAPER_KEYS)
        self.assertEqual(stored["title"], TITLE_EN)
        self.assertEqual(stored["title_zh"], TITLE_ZH)
        self.assertEqual(stored["reason"], "与你的研究兴趣高度相关。")
        self.assertEqual(stored["summary"], SUMMARY)
        self.assertEqual(stored["url"], "https://arxiv.org/abs/2610.00001")

    def test_subject_and_paper_count_fall_back_to_digest(self):
        self.write_run(
            "structured",
            subject=None,
            counts={},
            paper_keys=[],
            digest=digest_payload(
                "structured", papers=[paper(), paper(arxiv_id="2610.00002")]
            ),
            run_payload={
                "run_id": "structured",
                "mode": "send",
                "status": "sent",
                "started_at": "2026-10-20T10:00:00+00:00",
                "finished_at": "2026-10-20T10:05:00+00:00",
            },
        )

        item = load_sent_history(self.root)[0]

        self.assertEqual(item["subject"], "PaperLens｜前沿论文推荐")
        self.assertEqual(item["paper_count"], 2)
        self.assertEqual(len(item["papers"]), 2)

    def test_manual_flag_comes_from_run_json_or_manual_mode(self):
        self.write_run("manual-bool", manual=True)
        self.write_run("manual-mode", mode="manual")
        self.write_run("automatic", manual=False)

        rows = {row["run_id"]: row for row in load_sent_history(self.root)}

        self.assertTrue(rows["manual-bool"]["manual"])
        self.assertTrue(rows["manual-mode"]["manual"])
        self.assertFalse(rows["automatic"]["manual"])

    def test_structured_run_with_malformed_digest_falls_back_to_legacy(self):
        run_dir = self.write_run("broken-digest")
        (run_dir / "digest.json").write_text("{broken", encoding="utf-8")

        item = load_sent_history(self.root)[0]

        self.assertTrue(item["legacy"])
        self.assertEqual(item["papers"], [])
        self.assertTrue(item["digest_html"].endswith("digest_preview.html"))


class LegacyFallbackTests(HistoryFixture):
    def test_legacy_run_uses_email_html_and_run_counts(self):
        run_dir = self.write_run(
            "legacy",
            counts={"selected": 2, "sent": 2},
            paper_keys=["2610.00001", "2610.00002"],
        )

        item = load_sent_history(self.root)[0]

        self.assertTrue(item["legacy"])
        self.assertEqual(item["papers"], [])
        self.assertEqual(item["paper_count"], 2)
        self.assertEqual(item["sent_at"], "2026-10-20T10:05:00+00:00")
        self.assertEqual(item["subject"], "PaperLens｜前沿论文推荐（1 篇）")
        self.assertEqual(
            item["digest_html"], str(run_dir / "digest_preview.html")
        )
        self.assertTrue(Path(item["digest_html"]).is_file())

    def test_legacy_paper_count_falls_back_to_paper_keys_then_counts(self):
        self.write_run(
            "keys",
            counts={},
            paper_keys=["2610.00001", "2610.00002", "2610.00003"],
        )
        self.write_run("counts", counts={"selected": 4}, paper_keys=[])

        rows = {row["run_id"]: row for row in load_sent_history(self.root)}

        self.assertEqual(rows["keys"]["paper_count"], 3)
        self.assertEqual(rows["counts"]["paper_count"], 4)

    def test_legacy_without_html_still_returns_an_item(self):
        self.write_run("no-html", html=None)

        item = load_sent_history(self.root)[0]

        self.assertTrue(item["legacy"])
        self.assertTrue(item["digest_html"].endswith("digest_preview.html"))


class PrivacyTests(HistoryFixture):
    def test_never_returns_recipient_or_smtp_values(self):
        digest = digest_payload(
            "private",
            papers=[
                paper(
                    recipient="leak@example.com",
                    smtp_password="fake-smtp-secret",
                    mail_to="leak@example.com",
                )
            ],
        )
        self.write_run(
            "private",
            digest=digest,
            extra={
                "recipient": "leak@example.com",
                "mail_to": "leak@example.com",
                "smtp_password": "fake-smtp-secret",
                "smtp_user": "leak@example.com",
            },
        )
        (self.root / ".env").write_text(
            "MAIL_TO=leak@example.com\nSMTP_PASSWORD=fake-smtp-secret\n",
            encoding="utf-8",
        )

        item = load_sent_history(self.root)[0]
        serialized = json.dumps(item, ensure_ascii=False)

        self.assertEqual(set(item), ITEM_KEYS)
        self.assertEqual(set(item["papers"][0]), PAPER_KEYS)
        for forbidden in (
            "recipient",
            "mail_to",
            "smtp_password",
            "smtp_user",
            "leak@example.com",
            "fake-smtp-secret",
        ):
            self.assertNotIn(forbidden, serialized)


class HistoryPageSourceTests(unittest.TestCase):
    def setUp(self):
        if not PAGE.is_file():
            raise AssertionError(f"missing Streamlit page: {PAGE}")
        self.source = PAGE.read_text(encoding="utf-8")

    def test_page_does_not_set_page_config(self):
        self.assertNotIn("set_page_config", self.source)

    def test_page_is_read_only_and_has_no_mail_transport(self):
        self.assertIn("load_sent_history", self.source)
        self.assertIn("from push_history import", self.source)
        for forbidden in (
            "smtplib",
            "send_message",
            "resolve_smtp",
            "SMTP_PASSWORD",
            "run_push",
            "send_digest",
        ):
            self.assertNotIn(forbidden, self.source)

    def test_page_exposes_history_labels(self):
        for text in ("推送历史", "看邮件版", "手动", "自动", "arXiv"):
            self.assertIn(text, self.source)


class HistoryPageRenderTests(HistoryFixture):
    def setUp(self):
        super().setUp()
        self._old_env = os.environ.get("PAPERLENS_PUSH_ROOT")
        os.environ["PAPERLENS_PUSH_ROOT"] = str(self.root)

    def tearDown(self):
        if self._old_env is None:
            os.environ.pop("PAPERLENS_PUSH_ROOT", None)
        else:
            os.environ["PAPERLENS_PUSH_ROOT"] = self._old_env
        super().tearDown()

    def run_page(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PAGE))
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)
        return app

    def test_renders_structured_sent_run_with_chinese_fields_once(self):
        self.write_run(
            "manual-run", manual=True, digest=digest_payload("manual-run")
        )

        app = self.run_page()

        self.assertEqual(app.title[0].value, "推送历史")
        labels = [expander.label for expander in app.expander]
        self.assertTrue(any("手动" in label and "1 篇" in label for label in labels), labels)
        self.assertTrue(any("看邮件版" in label for label in labels), labels)

        rendered = "\n".join(node.value for node in app.markdown)
        self.assertIn(TITLE_EN, rendered)
        # The summary starts with the same 中文标题 line; the page must strip it.
        self.assertEqual(rendered.count(TITLE_ZH), 1)
        self.assertIn("https://arxiv.org/abs/2610.00001", rendered)
        self.assertIn("推荐理由", rendered)

    def test_renders_automatic_badge_and_legacy_email(self):
        self.write_run("legacy-run", manual=False, html="<html>旧邮件版</html>")

        app = self.run_page()

        labels = [expander.label for expander in app.expander]
        self.assertTrue(any("自动" in label for label in labels), labels)
        self.assertTrue(any("看邮件版" in label for label in labels), labels)
        self.assertTrue(any("旧运行" in caption.value for caption in app.caption))

    def test_renders_empty_state_without_sent_runs(self):
        app = self.run_page()

        self.assertEqual(app.title[0].value, "推送历史")
        self.assertTrue(any("还没有" in info.value for info in app.info), app.info)
        self.assertEqual(list(app.expander), [])

    def test_rendering_does_not_mutate_history_files(self):
        run_dir = self.write_run("structured", digest=digest_payload("structured"))
        before = {
            path: path.stat().st_mtime_ns
            for path in run_dir.iterdir()
            if path.is_file()
        }

        self.run_page()

        after = {
            path: path.stat().st_mtime_ns
            for path in run_dir.iterdir()
            if path.is_file()
        }
        self.assertEqual(before, after)


class SummaryPresentationTests(unittest.TestCase):
    def load_page_module(self):
        spec = importlib.util.spec_from_file_location("paperlens_advanced_history", PAGE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_summary_without_title_strips_leading_chinese_title(self):
        page = self.load_page_module()

        body = page._summary_without_title(SUMMARY)

        self.assertNotIn("中文标题", body)
        self.assertIn("研究问题", body)
        self.assertTrue(body.startswith("研究问题"))

    def test_title_zh_is_read_from_summary_when_digest_field_is_empty(self):
        page = self.load_page_module()

        title = page._display_title_zh({"title_zh": "", "summary": SUMMARY})

        self.assertEqual(title, TITLE_ZH)

    def test_existing_title_zh_is_not_duplicated(self):
        page = self.load_page_module()

        title = page._display_title_zh({"title_zh": TITLE_ZH, "summary": SUMMARY})

        self.assertEqual(title, TITLE_ZH)


if __name__ == "__main__":
    unittest.main()
