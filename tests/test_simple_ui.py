"""Offline tests for the simple push page's core logic (Task 8).

Every test uses a temporary root, fake arXiv/rank/summary/model dependencies,
and fake SMTP. Nothing touches the network, a real mailbox, or the project's
own ``data/`` directory. Addresses are synthetic ``example.com`` fixtures.
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

import ui_state  # noqa: E402
from profile_config import load_active_profile, profile_sha256  # noqa: E402
from push_run import PushDeps, RunResult  # noqa: E402
from push_state import PushState  # noqa: E402
from topic_queries import TopicQueryError  # noqa: E402

import simple_ui  # noqa: E402

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"
NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
RECIPIENT = "alice@example.com"
NEW_RECIPIENT = "bob@example.com"
VERIFIED_AT = "2026-10-25T10:00:00+00:00"
ENV = {
    "DEEPSEEK_API_KEY": "test-key",
    "DEEPSEEK_BASE_URL": "https://example.invalid",
    "DEEPSEEK_MODEL": "test-model",
    "SMTP_HOST": "smtp.163.com",
    "SMTP_PORT": "465",
    "SMTP_SECURITY": "ssl",
    "SMTP_USER": "sender@example.com",
    "SMTP_PASSWORD": "test-app-code",
    "MAIL_TO": "legacy@example.com",
}

PROFILE = {
    "research_goal": "RAG",
    "topics": ["RAG"],
    "preferred_methods": ["hybrid retrieval"],
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

NEW_GROUPS = [
    {
        "name": "Retrieval",
        "any_terms": ["retrieval augmented generation"],
        "require_any_terms": [],
    },
    {
        "name": "Agents",
        "any_terms": ["large language model agent"],
        "require_any_terms": [],
    },
]


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
        raise AssertionError("fetch must not be called")


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


def rag_results():
    return [
        FakeResult("2601.00001v1", "A RAG paper"),
        FakeResult("2601.00002v1", "Another RAG paper"),
        FakeResult("2601.00003v1", "A third RAG paper"),
    ]


def json_chat(payload, calls=None):
    def chat(messages):
        if calls is not None:
            calls.append(messages)
        return json.dumps(payload, ensure_ascii=False)

    return chat


def exploding_chat(messages):
    raise AssertionError("the topic generator must not be called")


def write_root(
    tmp: str,
    *,
    enabled: bool = False,
    paused: bool = False,
    quiet: bool = False,
    max_per_digest: int = 3,
) -> Path:
    root = Path(tmp)
    profile = json.loads(json.dumps(PROFILE))
    profile["top_k"] = max_per_digest
    (root / "interests.json").write_text(
        json.dumps(profile, ensure_ascii=False), encoding="utf-8"
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


class SimpleUiTestBase(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.root = write_root(self.tmp.name)
        self.config_path = self.root / "push_config.json"

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

    def verify(self, email: str = RECIPIENT) -> None:
        ui_state.set_recipient(self.root, email)
        ui_state.mark_verified(self.root, email, VERIFIED_AT)

    def record_test(self, email: str = RECIPIENT) -> None:
        ui_state.set_recipient(self.root, email)
        ui_state.record_test_email(self.root, email, VERIFIED_AT)

    def approve(self) -> PushState:
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        state.init_schema()
        state.set_setting("first_send_approved_at", NOW.isoformat())
        return state

    def read_config(self) -> dict:
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def write_config(self, **patch) -> dict:
        raw = self.read_config()
        raw.update(patch)
        self.config_path.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return raw

    def saved_profile(self) -> dict:
        return load_active_profile(self.root)

    def preview(self, results=None, *, now: datetime = NOW, recipient=True):
        if recipient:
            ui_state.set_recipient(self.root, RECIPIENT)
        return simple_ui.preview(
            self.root,
            self.config_path,
            ENV,
            deps=self.deps(
                FakeClient(results if results is not None else rag_results())
            ),
            now=now,
        )


class SaveSettingsTests(SimpleUiTestBase):
    def test_count_time_and_recipient_save_without_llm_when_topics_unchanged(self):
        with mock.patch(
            "simple_ui.push_ctl.register_task",
            side_effect=AssertionError("disabled push must not touch the task"),
        ) as register:
            result = simple_ui.save_settings(
                self.root,
                self.config_path,
                topics_text="RAG",
                email=RECIPIENT,
                count=5,
                time_local="09:15",
                chat_fn=exploding_chat,
            )

        self.assertFalse(result["topics_changed"])
        self.assertFalse(result["auto_disabled"])
        self.assertIsNone(result["task_error"])
        self.assertFalse(register.called)
        raw = self.read_config()
        self.assertEqual(raw["max_per_digest"], 5)
        self.assertEqual(raw["time_local"], "09:15")
        self.assertEqual(self.saved_profile()["top_k"], 5)
        self.assertEqual(
            ui_state.load_state(self.root)["recipient"], RECIPIENT
        )

    def test_legacy_profile_without_goal_saves_count_without_llm(self):
        profile = json.loads(json.dumps(PROFILE))
        profile["research_goal"] = ""
        (self.root / "interests.json").write_text(
            json.dumps(profile, ensure_ascii=False), encoding="utf-8"
        )
        ui_state.set_recipient(self.root, RECIPIENT)

        result = simple_ui.save_settings(
            self.root,
            self.config_path,
            topics_text="RAG",
            email="",
            count=4,
            time_local="08:30",
            chat_fn=exploding_chat,
        )

        self.assertFalse(result["topics_changed"])
        self.assertEqual(self.saved_profile()["top_k"], 4)
        # The original goal field is preserved instead of being rewritten.
        self.assertEqual(self.saved_profile()["research_goal"], "")

    def test_changed_topics_generate_and_save_groups(self):
        calls = []
        result = simple_ui.save_settings(
            self.root,
            self.config_path,
            topics_text="RAG\nAgents",
            email=RECIPIENT,
            count=4,
            time_local="08:30",
            chat_fn=json_chat({"groups": NEW_GROUPS}, calls),
        )

        self.assertTrue(result["topics_changed"])
        self.assertEqual(len(calls), 1)
        profile = self.saved_profile()
        self.assertEqual(profile["research_goal"], "RAG\nAgents")
        self.assertEqual(profile["topics"], ["RAG", "Agents"])
        self.assertEqual(profile["top_k"], 4)
        self.assertEqual(profile["search"]["groups"], NEW_GROUPS)
        # Advanced fields stay untouched, so precise edits are not lost.
        self.assertEqual(profile["preferred_methods"], ["hybrid retrieval"])
        self.assertEqual(profile["search"]["lookback_days"], 30)
        self.assertEqual(self.read_config()["max_per_digest"], 4)

    def test_topic_generation_failure_saves_no_new_topic(self):
        before_profile = self.saved_profile()
        before_config = self.read_config()

        with self.assertRaises(TopicQueryError):
            simple_ui.save_settings(
                self.root,
                self.config_path,
                topics_text="Quantum computing",
                email=RECIPIENT,
                count=5,
                time_local="09:15",
                chat_fn=json_chat({"not_groups": []}),
            )

        self.assertEqual(self.saved_profile(), before_profile)
        self.assertEqual(self.read_config(), before_config)
        self.assertFalse((self.root / ui_state.STATE_REL).exists())

    def test_empty_topics_and_bad_values_are_rejected(self):
        for kwargs in (
            {"topics_text": "  ", "count": 3, "time_local": "08:30"},
            {"topics_text": "RAG", "count": 11, "time_local": "08:30"},
            {"topics_text": "RAG", "count": 3, "time_local": "25:99"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(simple_ui.SimpleUiError):
                    simple_ui.save_settings(
                        self.root,
                        self.config_path,
                        email=RECIPIENT,
                        chat_fn=exploding_chat,
                        **kwargs,
                    )

    def test_email_change_while_enabled_disables_auto_and_clears_verification(
        self,
    ):
        self.write_config(enabled=True)
        self.verify(RECIPIENT)
        ui_state.set_preview(
            self.root,
            {
                "run_id": "preview-1",
                "created_at": NOW.isoformat(),
                "profile_sha256": profile_sha256(self.saved_profile()),
                "count": 3,
                "recipient_hash": ui_state.recipient_hash(RECIPIENT),
                "local_date": NOW.astimezone().date().isoformat(),
            },
        )

        result = simple_ui.save_settings(
            self.root,
            self.config_path,
            topics_text="RAG",
            email=NEW_RECIPIENT,
            count=3,
            time_local="08:30",
            chat_fn=exploding_chat,
        )

        self.assertTrue(result["auto_disabled"])
        self.assertTrue(result["recipient_changed"])
        self.assertFalse(self.read_config()["enabled"])
        state = ui_state.load_state(self.root)
        self.assertEqual(state["recipient"], NEW_RECIPIENT)
        self.assertIsNone(state["verified_at"])
        self.assertIsNone(state["preview"])

    def test_resaving_the_same_email_keeps_enabled_and_verification(self):
        self.write_config(enabled=True)
        self.verify(RECIPIENT)

        result = simple_ui.save_settings(
            self.root,
            self.config_path,
            topics_text="RAG",
            email=RECIPIENT,
            count=3,
            time_local="08:30",
            chat_fn=exploding_chat,
        )

        self.assertFalse(result["auto_disabled"])
        self.assertTrue(self.read_config()["enabled"])
        self.assertIsNotNone(ui_state.load_state(self.root)["verified_at"])

    def test_time_change_while_enabled_registers_task_and_surfaces_failure(self):
        self.write_config(enabled=True)
        self.verify(RECIPIENT)

        with mock.patch(
            "simple_ui.push_ctl.register_task",
            return_value={"installed": False, "returncode": 5, "output": ""},
        ) as register:
            result = simple_ui.save_settings(
                self.root,
                self.config_path,
                topics_text="RAG",
                email="",
                count=3,
                time_local="07:45",
                chat_fn=exploding_chat,
            )

        register.assert_called_once_with(self.root, "07:45")
        self.assertEqual(result["task_error"], "计划任务时间未更新")
        self.assertFalse(result["task_updated"])
        raw = self.read_config()
        self.assertEqual(raw["time_local"], "07:45")
        self.assertTrue(raw["enabled"])

    def test_time_change_while_enabled_reports_success(self):
        self.write_config(enabled=True)
        self.verify(RECIPIENT)

        with mock.patch(
            "simple_ui.push_ctl.register_task",
            return_value={"installed": True, "returncode": 0, "output": ""},
        ) as register:
            result = simple_ui.save_settings(
                self.root,
                self.config_path,
                topics_text="RAG",
                email="",
                count=3,
                time_local="06:10",
                chat_fn=exploding_chat,
            )

        register.assert_called_once_with(self.root, "06:10")
        self.assertTrue(result["task_updated"])
        self.assertIsNone(result["task_error"])

    def test_time_change_while_disabled_only_saves_the_value(self):
        ui_state.set_recipient(self.root, RECIPIENT)

        with mock.patch("simple_ui.push_ctl.register_task") as register:
            result = simple_ui.save_settings(
                self.root,
                self.config_path,
                topics_text="RAG",
                email="",
                count=3,
                time_local="06:10",
                chat_fn=exploding_chat,
            )

        register.assert_not_called()
        self.assertIsNone(result["task_error"])
        self.assertEqual(self.read_config()["time_local"], "06:10")

    def test_email_and_time_change_together_do_not_register_task(self):
        self.write_config(enabled=True)
        self.verify(RECIPIENT)

        with mock.patch("simple_ui.push_ctl.register_task") as register:
            result = simple_ui.save_settings(
                self.root,
                self.config_path,
                topics_text="RAG",
                email=NEW_RECIPIENT,
                count=3,
                time_local="06:10",
                chat_fn=exploding_chat,
            )

        register.assert_not_called()
        self.assertTrue(result["auto_disabled"])
        raw = self.read_config()
        self.assertFalse(raw["enabled"])
        self.assertEqual(raw["time_local"], "06:10")

    def test_register_task_error_is_a_visible_task_error(self):
        from push_ctl import PushCtlError

        self.write_config(enabled=True)
        self.verify(RECIPIENT)

        with mock.patch(
            "simple_ui.push_ctl.register_task", side_effect=PushCtlError("bad")
        ):
            result = simple_ui.save_settings(
                self.root,
                self.config_path,
                topics_text="RAG",
                email="",
                count=3,
                time_local="06:10",
                chat_fn=exploding_chat,
            )

        self.assertEqual(result["task_error"], "计划任务时间未更新")
        # The configured value is kept even though the task was not updated.
        self.assertEqual(self.read_config()["time_local"], "06:10")


class PreviewTests(SimpleUiTestBase):
    def test_preview_records_reference_and_never_sends(self):
        ui_state.set_recipient(self.root, RECIPIENT)

        result = self.preview()

        self.assertEqual(result.status, "preview_ready")
        self.assertFalse(result.sent)
        self.assertFalse(self.read_config()["enabled"])
        digest = json.loads(
            (
                self.root
                / "data"
                / "push"
                / "runs"
                / str(result.run_id)
                / "digest.json"
            ).read_text(encoding="utf-8")
        )
        reference = ui_state.get_preview(self.root)
        self.assertIsNotNone(reference)
        self.assertEqual(
            set(reference),
            {
                "run_id",
                "created_at",
                "profile_sha256",
                "count",
                "recipient_hash",
                "local_date",
            },
        )
        self.assertEqual(reference["run_id"], result.run_id)
        self.assertEqual(reference["created_at"], digest["created_at"])
        self.assertEqual(
            reference["profile_sha256"], profile_sha256(self.saved_profile())
        )
        self.assertEqual(reference["count"], result.selected_count)
        self.assertEqual(
            reference["recipient_hash"], ui_state.recipient_hash(RECIPIENT)
        )
        self.assertEqual(
            reference["local_date"], NOW.astimezone().date().isoformat()
        )
        rendered = json.dumps(reference, ensure_ascii=False)
        self.assertNotIn(RECIPIENT, rendered)
        self.assertNotIn("legacy@example.com", rendered)

    def test_empty_preview_reports_no_digest_and_records_no_reference(self):
        ui_state.set_recipient(self.root, RECIPIENT)

        result = self.preview(results=[])

        self.assertEqual(result.status, "no_digest")
        self.assertFalse(result.sent)
        self.assertIsNone(ui_state.get_preview(self.root))

    def test_preview_reference_count_matches_the_previewed_papers(self):
        ui_state.set_recipient(self.root, RECIPIENT)

        result = self.preview(results=rag_results()[:2])

        self.assertEqual(result.selected_count, 2)
        self.assertEqual(ui_state.get_preview(self.root)["count"], 2)


class LookbackWindowTests(SimpleUiTestBase):
    def empty_result(self) -> RunResult:
        return RunResult(
            run_id=None,
            status="no_digest",
            candidate_count=0,
            selected_count=0,
            sent=False,
            errors=[],
            run_dir=None,
        )

    def test_preview_uses_the_configured_catchup_window(self):
        self.write_config(catchup_days=5)
        ui_state.set_recipient(self.root, RECIPIENT)

        with mock.patch(
            "simple_ui.run_push", return_value=self.empty_result()
        ) as run:
            simple_ui.preview(self.root, self.config_path, ENV, now=NOW)

        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs["lookback_days"], 5)

    def test_manual_send_uses_the_configured_catchup_window(self):
        self.write_config(catchup_days=4)
        self.verify(RECIPIENT)
        self.approve()

        with mock.patch(
            "simple_ui.run_push", return_value=self.empty_result()
        ) as run:
            simple_ui.manual_send(
                self.root, self.config_path, ENV, now=NOW, confirm=True
            )

        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs["lookback_days"], 4)
        self.assertTrue(run.call_args.kwargs["manual"])


class ManualSendTests(SimpleUiTestBase):
    def manual(self, deps=None, *, now=NOW + timedelta(minutes=5), confirm=True):
        return simple_ui.manual_send(
            self.root,
            self.config_path,
            ENV,
            deps=deps
            or self.deps(
                ExplodingClient(),
                smtp_send=never_send,
                rank_fn=exploding_rank,
                summarize_fn=exploding_summarize,
            ),
            now=now,
            confirm=confirm,
        )

    def test_manual_send_requires_confirmation_verification_and_approval(self):
        # No recipient at all.
        with self.assertRaises(simple_ui.SimpleUiError):
            self.manual()

        # Saved but unverified.
        ui_state.set_recipient(self.root, RECIPIENT)
        with self.assertRaises(simple_ui.SimpleUiError):
            self.manual()

        # Verified but first send not approved.
        ui_state.mark_verified(self.root, RECIPIENT, VERIFIED_AT)
        with self.assertRaises(simple_ui.SimpleUiError):
            self.manual()

        # Everything approved but the confirmation checkbox is off.
        self.approve()
        with self.assertRaises(simple_ui.SimpleUiError):
            self.manual(confirm=False)
        self.assertIsNone(ui_state.load_state(self.root)["last_manual_send"])

    def test_manual_send_reuses_a_fresh_preview_and_keeps_enabled_off(self):
        ui_state.set_recipient(self.root, RECIPIENT)
        preview = self.preview()
        self.verify(RECIPIENT)
        self.approve()
        sent = []
        before_config = self.read_config()

        result = self.manual(
            self.deps(
                ExplodingClient(),
                smtp_send=lambda settings, message: sent.append(message),
                rank_fn=exploding_rank,
                summarize_fn=exploding_summarize,
            )
        )

        self.assertEqual(result.status, "sent")
        self.assertTrue(result.sent)
        self.assertTrue(result.manual)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["To"], RECIPIENT)
        self.assertEqual(self.read_config(), before_config)
        self.assertFalse(self.read_config()["enabled"])
        record = ui_state.load_state(self.root)["last_manual_send"]
        self.assertEqual(record["run_id"], result.run_id)
        self.assertEqual(record["count"], preview.selected_count)
        self.assertNotIn(RECIPIENT, json.dumps(record, ensure_ascii=False))

    def test_manual_send_regenerates_when_no_preview_reference_exists(self):
        self.verify(RECIPIENT)
        self.approve()
        client = FakeClient([FakeResult("2601.00004v1", "A new RAG paper")])
        sent = []

        result = self.manual(
            self.deps(
                client,
                smtp_send=lambda settings, message: sent.append(message),
            )
        )

        self.assertEqual(result.status, "sent")
        self.assertEqual(client.calls, 1)
        self.assertEqual(len(sent), 1)
        self.assertEqual(
            ui_state.load_state(self.root)["last_manual_send"]["count"], 1
        )

    def test_manual_send_never_changes_the_auto_push_switch(self):
        ui_state.set_recipient(self.root, RECIPIENT)
        self.preview()
        self.verify(RECIPIENT)
        self.approve()
        sent = []

        with mock.patch("simple_ui.push_ctl.set_auto_push") as set_auto:
            result = self.manual(
                self.deps(
                    ExplodingClient(),
                    smtp_send=lambda settings, message: sent.append(message),
                    rank_fn=exploding_rank,
                    summarize_fn=exploding_summarize,
                )
            )

        self.assertEqual(result.status, "sent")
        self.assertEqual(len(sent), 1)
        set_auto.assert_not_called()


class EmailVerificationTests(SimpleUiTestBase):
    def test_test_email_uses_the_saved_recipient_and_does_not_verify(self):
        ui_state.set_recipient(self.root, RECIPIENT)
        original_env = dict(ENV)

        with mock.patch(
            "simple_ui.push_ctl.test_email",
            return_value={"status": "sent", "host": "smtp.163.com", "port": 465},
        ) as send:
            result = simple_ui.test_email(self.root, self.config_path, ENV)

        self.assertEqual(result["status"], "sent")
        args, kwargs = send.call_args
        self.assertEqual(args[0], self.config_path)
        self.assertEqual(args[1]["MAIL_TO"], RECIPIENT)
        self.assertTrue(kwargs["send"])
        self.assertTrue(kwargs["confirm"])
        self.assertEqual(ENV, original_env)
        state = ui_state.load_state(self.root)
        self.assertIsNone(state["verified_at"])
        self.assertTrue(ui_state.test_email_confirmed(self.root, RECIPIENT))

    def test_failed_test_email_clears_any_previous_receipt(self):
        from push_ctl import PushCtlError

        self.record_test(RECIPIENT)
        self.assertTrue(ui_state.test_email_confirmed(self.root, RECIPIENT))

        with mock.patch(
            "simple_ui.push_ctl.test_email",
            side_effect=PushCtlError("SMTP authentication failed"),
        ):
            with self.assertRaises(PushCtlError):
                simple_ui.test_email(self.root, self.config_path, ENV)

        self.assertFalse(ui_state.test_email_confirmed(self.root, RECIPIENT))
        self.assertIsNone(ui_state.load_state(self.root)["verified_at"])

    def test_confirm_requires_a_successful_test_for_an_unverified_recipient(self):
        ui_state.set_recipient(self.root, RECIPIENT)  # saved, not verified
        calls = []

        def register(root, time_local):
            calls.append((root, time_local))
            return {"installed": True}

        result = simple_ui.confirm_test_and_enable(
            self.root,
            self.config_path,
            ENV,
            register_task_fn=register,
        )

        self.assertFalse(result["enabled"])
        self.assertIn("测试邮件", result["error"])
        self.assertEqual(calls, [])
        self.assertFalse(self.read_config()["enabled"])
        state = ui_state.load_state(self.root)
        self.assertIsNone(state["verified_at"])
        self.assertFalse(
            (self.root / "data" / "push" / "frontier.sqlite3").exists()
        )

    def test_successful_test_email_then_confirm_enables(self):
        ui_state.set_recipient(self.root, RECIPIENT)

        with mock.patch(
            "simple_ui.push_ctl.test_email",
            return_value={"status": "sent", "host": "smtp.163.com", "port": 465},
        ):
            outcome = simple_ui.test_email(self.root, self.config_path, ENV)

        self.assertEqual(outcome["status"], "sent")
        self.assertTrue(ui_state.test_email_confirmed(self.root, RECIPIENT))
        self.assertIsNone(ui_state.load_state(self.root)["verified_at"])

        result = simple_ui.confirm_test_and_enable(
            self.root,
            self.config_path,
            ENV,
            register_task_fn=lambda root, time_local: {"installed": True},
        )

        self.assertTrue(result["enabled"])
        self.assertTrue(self.read_config()["enabled"])
        self.assertIsNotNone(ui_state.load_state(self.root)["verified_at"])

    def test_test_email_seeds_private_state_for_a_legacy_env_recipient(self):
        self.assertFalse((self.root / ui_state.STATE_REL).exists())

        with mock.patch(
            "simple_ui.push_ctl.test_email",
            return_value={"status": "sent", "host": "smtp.163.com", "port": 465},
        ) as send:
            simple_ui.test_email(self.root, self.config_path, ENV)

        args, _kwargs = send.call_args
        self.assertEqual(args[1]["MAIL_TO"], "legacy@example.com")
        state = ui_state.load_state(self.root)
        self.assertEqual(state["recipient"], "legacy@example.com")
        self.assertIsNone(state["verified_at"])

    def test_confirm_test_and_enable_seeds_legacy_env_recipient(self):
        self.assertFalse((self.root / ui_state.STATE_REL).exists())

        result = simple_ui.confirm_test_and_enable(
            self.root,
            self.config_path,
            ENV,
            register_task_fn=lambda root, time_local: {"installed": True},
        )

        self.assertTrue(result["enabled"])
        state = ui_state.load_state(self.root)
        self.assertEqual(state["recipient"], "legacy@example.com")
        self.assertIsNotNone(state["verified_at"])
        self.assertTrue(self.read_config()["enabled"])

    def test_confirm_test_and_enable_verifies_approves_installs_and_enables(self):
        self.record_test(RECIPIENT)
        calls = []

        def register(root, time_local):
            calls.append((root, time_local))
            return {"installed": True, "returncode": 0, "output": ""}

        result = simple_ui.confirm_test_and_enable(
            self.root,
            self.config_path,
            ENV,
            register_task_fn=register,
        )

        self.assertTrue(result["enabled"])
        # The temp config is copied from the live one; assert the time it
        # actually carries instead of a historical default.
        self.assertEqual(calls, [(self.root, self.read_config()["time_local"])])
        self.assertTrue(self.read_config()["enabled"])
        self.assertIsNotNone(ui_state.load_state(self.root)["verified_at"])
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        state.init_schema()
        self.assertIsNotNone(state.get_setting("first_send_approved_at"))
        rendered = json.dumps(result, ensure_ascii=False)
        self.assertNotIn(RECIPIENT, rendered)

    def test_confirm_test_and_verify_requires_a_receipt_and_never_enables(self):
        ui_state.set_recipient(self.root, RECIPIENT)
        before_config = self.config_path.read_text(encoding="utf-8")

        with mock.patch(
            "simple_ui.push_ctl.register_task",
            side_effect=AssertionError("verify-only must not install a task"),
        ), mock.patch(
            "simple_ui.push_ctl.set_auto_push",
            side_effect=AssertionError("verify-only must not touch auto push"),
        ):
            refused = simple_ui.confirm_test_and_verify(
                self.root, self.config_path, ENV, now=NOW
            )

            self.assertFalse(refused["verified"])
            self.assertIn("测试邮件", refused["error"])
            self.assertFalse(
                (self.root / "data" / "push" / "frontier.sqlite3").exists()
            )
            self.assertIsNone(ui_state.load_state(self.root)["verified_at"])

            self.record_test(RECIPIENT)
            verified = simple_ui.confirm_test_and_verify(
                self.root, self.config_path, ENV, now=NOW
            )

        self.assertTrue(verified["verified"])
        self.assertTrue(verified["approved"])
        self.assertEqual(verified["recipient"], "a***@example.com")
        self.assertNotIn(RECIPIENT, json.dumps(verified, ensure_ascii=False))
        state = ui_state.load_state(self.root)
        self.assertEqual(
            state["verified_at"], NOW.isoformat(timespec="seconds")
        )
        frontier = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        frontier.init_schema()
        self.assertIsNotNone(frontier.get_setting("first_send_approved_at"))
        # Verification never writes `enabled`/`paused` or the config file.
        self.assertEqual(
            self.config_path.read_text(encoding="utf-8"), before_config
        )
        raw = self.read_config()
        self.assertFalse(raw["enabled"])
        self.assertFalse(raw["paused"])

    def test_confirm_test_and_verify_accepts_an_already_verified_recipient(self):
        self.verify(RECIPIENT)
        self.assertFalse(ui_state.test_email_confirmed(self.root, RECIPIENT))

        result = simple_ui.confirm_test_and_verify(
            self.root, self.config_path, ENV, now=NOW
        )

        self.assertTrue(result["verified"])
        self.assertTrue(result["approved"])
        frontier = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        frontier.init_schema()
        self.assertIsNotNone(frontier.get_setting("first_send_approved_at"))
        self.assertFalse(self.read_config()["enabled"])

    def test_confirm_test_and_verify_seeds_a_legacy_env_recipient(self):
        self.assertFalse((self.root / ui_state.STATE_REL).exists())

        result = simple_ui.confirm_test_and_verify(
            self.root, self.config_path, ENV, now=NOW
        )

        self.assertTrue(result["verified"])
        state = ui_state.load_state(self.root)
        self.assertEqual(state["recipient"], "legacy@example.com")
        self.assertIsNotNone(state["verified_at"])
        self.assertFalse(self.read_config()["enabled"])

    def test_confirm_failure_leaves_push_disabled(self):
        self.record_test(RECIPIENT)

        result = simple_ui.confirm_test_and_enable(
            self.root,
            self.config_path,
            ENV,
            register_task_fn=lambda root, time_local: {
                "installed": False,
                "returncode": 5,
                "output": "access denied",
            },
        )

        self.assertFalse(result["enabled"])
        self.assertEqual(result["task_error"], "计划任务安装失败，未开启自动推送")
        self.assertFalse(self.read_config()["enabled"])

    def test_register_error_leaves_push_disabled(self):
        from push_ctl import PushCtlError

        self.record_test(RECIPIENT)

        def register(root, time_local):
            raise PushCtlError("invalid time")

        result = simple_ui.confirm_test_and_enable(
            self.root,
            self.config_path,
            ENV,
            register_task_fn=register,
        )

        self.assertFalse(result["enabled"])
        self.assertTrue(result["task_error"])
        self.assertFalse(self.read_config()["enabled"])

    def test_auto_paused_state_stays_disabled(self):
        self.record_test(RECIPIENT)
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        state.init_schema()
        state.set_setting("auto_paused", "true")

        result = simple_ui.confirm_test_and_enable(
            self.root,
            self.config_path,
            ENV,
            register_task_fn=lambda root, time_local: {"installed": True},
        )

        self.assertFalse(result["enabled"])
        self.assertIn("auto-paused", result["error"])
        self.assertFalse(self.read_config()["enabled"])

    def test_quiet_mode_stays_disabled(self):
        self.write_config(quiet_mode=True)
        self.record_test(RECIPIENT)

        result = simple_ui.confirm_test_and_enable(
            self.root,
            self.config_path,
            ENV,
            register_task_fn=lambda root, time_local: {"installed": True},
        )

        self.assertFalse(result["enabled"])
        self.assertIn("quiet", result["error"])
        self.assertFalse(self.read_config()["enabled"])

    def test_verified_recipient_enables_in_one_step(self):
        self.verify(RECIPIENT)
        self.approve()

        result = simple_ui.enable(
            self.root,
            self.config_path,
            ENV,
            register_task_fn=lambda root, time_local: {"installed": True},
        )

        self.assertTrue(result["enabled"])
        self.assertTrue(self.read_config()["enabled"])

    def test_enable_requires_a_verified_recipient(self):
        ui_state.set_recipient(self.root, RECIPIENT)  # saved, not verified
        calls = []

        result = simple_ui.enable(
            self.root,
            self.config_path,
            ENV,
            register_task_fn=lambda root, time_local: calls.append(
                (root, time_local)
            )
            or {"installed": True},
        )

        self.assertFalse(result["enabled"])
        self.assertIn("验证", result["error"])
        self.assertEqual(calls, [])
        self.assertFalse(self.read_config()["enabled"])

    def test_enable_requires_first_send_approval(self):
        """A legacy env-verified recipient still needs the first-test gate."""
        calls = []

        def register(root, time_local):
            calls.append((root, time_local))
            return {"installed": True}

        refused = simple_ui.enable(
            self.root,
            self.config_path,
            ENV,
            register_task_fn=register,
        )

        self.assertFalse(refused["enabled"])
        self.assertIn("批准", refused["error"])
        self.assertEqual(calls, [])
        self.assertFalse(self.read_config()["enabled"])

        self.approve()
        allowed = simple_ui.enable(
            self.root,
            self.config_path,
            ENV,
            register_task_fn=register,
        )

        self.assertTrue(allowed["enabled"])
        self.assertEqual(len(calls), 1)
        self.assertTrue(self.read_config()["enabled"])

    def test_disable_keeps_safety_flags(self):
        self.write_config(enabled=True, quiet_mode=True)
        state = self.approve()
        state.set_setting("auto_paused", "true")

        result = simple_ui.disable(self.root, self.config_path, ENV)

        self.assertFalse(result["enabled"])
        raw = self.read_config()
        self.assertFalse(raw["enabled"])
        self.assertTrue(raw["quiet_mode"])
        self.assertEqual(
            PushState(
                self.root / "data" / "push" / "frontier.sqlite3"
            ).get_setting("auto_paused"),
            "true",
        )


class StatusAndActionTests(SimpleUiTestBase):
    def test_status_exposes_the_last_scheduled_attempt(self):
        run_dir = (
            self.root / "data" / "push" / "runs" / "scheduled-check"
        )
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "run_id": "scheduled-check",
                    "mode": "scheduled",
                    "status": "no_digest",
                    "started_at": NOW.isoformat(),
                    "finished_at": NOW.isoformat(),
                    "counts": {"candidates": 10, "selected": 0},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        report = simple_ui.status(
            self.root, self.config_path, ENV, now=NOW
        )

        self.assertIsNotNone(report["last_scheduled_run"])
        self.assertEqual(
            report["last_scheduled_run"]["run_id"], "scheduled-check"
        )
        self.assertEqual(
            report["last_scheduled_run"]["status"], "no_digest"
        )

    def test_status_masks_the_recipient_and_reports_flags_and_last_send(self):
        self.write_config(enabled=True)
        self.verify(RECIPIENT)
        self.approve()
        ui_state.record_manual_send(self.root, "run-1", NOW.isoformat(), 2)

        report = simple_ui.status(
            self.root, self.config_path, ENV, now=NOW
        )

        self.assertTrue(report["enabled"])
        self.assertFalse(report["paused"])
        self.assertFalse(report["auto_paused"])
        self.assertFalse(report["quiet"])
        self.assertEqual(report["mail_to"], "a***@example.com")
        self.assertTrue(report["recipient_verified"])
        self.assertTrue(report["first_send_approved"])
        self.assertIsNotNone(report["next_expected"])
        self.assertEqual(report["last_manual_send"]["count"], 2)
        self.assertEqual(report["count"], 3)
        self.assertEqual(
            report["time_local"], self.read_config()["time_local"]
        )
        rendered = json.dumps(report, ensure_ascii=False)
        for secret in (RECIPIENT, ENV["SMTP_PASSWORD"], "legacy@example.com"):
            self.assertNotIn(secret, rendered)

    def test_status_reports_the_test_email_receipt_for_the_current_recipient(
        self,
    ):
        ui_state.set_recipient(self.root, RECIPIENT)
        before = simple_ui.status(self.root, self.config_path, ENV)
        self.assertFalse(before["test_email_confirmed"])

        self.record_test(RECIPIENT)
        after = simple_ui.status(self.root, self.config_path, ENV)
        self.assertTrue(after["test_email_confirmed"])

        ui_state.set_recipient(self.root, NEW_RECIPIENT)
        changed = simple_ui.status(self.root, self.config_path, ENV)
        self.assertFalse(changed["test_email_confirmed"])

    def test_status_explains_paused_auto_paused_and_quiet(self):
        self.write_config(enabled=True, paused=True, quiet_mode=True)
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        state.init_schema()
        state.set_setting("auto_paused", "true")

        report = simple_ui.status(
            self.root, self.config_path, ENV, now=NOW
        )

        self.assertTrue(report["paused"])
        self.assertTrue(report["auto_paused"])
        self.assertTrue(report["quiet"])
        self.assertIsNone(report["next_expected"])

    def test_unsaved_email_disables_auto_and_manual_actions(self):
        self.verify(RECIPIENT)
        self.approve()

        saved = simple_ui.action_state(
            self.root, self.config_path, ENV, email_field=""
        )
        self.assertFalse(saved["email_pending"])
        self.assertTrue(saved["can_manual"])
        self.assertTrue(saved["can_enable"])

        same = simple_ui.action_state(
            self.root, self.config_path, ENV, email_field=RECIPIENT
        )
        self.assertFalse(same["email_pending"])

        pending = simple_ui.action_state(
            self.root, self.config_path, ENV, email_field=NEW_RECIPIENT
        )
        self.assertTrue(pending["email_pending"])
        self.assertFalse(pending["can_manual"])
        self.assertFalse(pending["can_enable"])

    def test_unsaved_email_is_pending_without_a_stored_recipient(self):
        # No private state: the legacy env recipient is what a push would use.
        self.assertFalse(simple_ui.email_pending(self.root, "", ENV))
        self.assertFalse(
            simple_ui.email_pending(self.root, "legacy@example.com", ENV)
        )
        self.assertTrue(
            simple_ui.email_pending(self.root, NEW_RECIPIENT, ENV)
        )
        # No recipient anywhere is still unsaved, not silently accepted.
        self.assertTrue(
            simple_ui.email_pending(self.root, NEW_RECIPIENT, {})
        )

        self.approve()  # env recipient is verified; approval completes the gate
        saved_actions = simple_ui.action_state(
            self.root, self.config_path, ENV, email_field=""
        )
        self.assertTrue(saved_actions["can_enable"])
        self.assertTrue(saved_actions["can_verify"])

        pending_actions = simple_ui.action_state(
            self.root, self.config_path, ENV, email_field=NEW_RECIPIENT
        )
        self.assertTrue(pending_actions["email_pending"])
        self.assertFalse(pending_actions["can_enable"])
        self.assertFalse(pending_actions["can_verify"])
        self.assertFalse(pending_actions["can_manual"])

    def test_unverified_or_unapproved_recipient_cannot_send_manually(self):
        ui_state.set_recipient(self.root, RECIPIENT)

        actions = simple_ui.action_state(
            self.root, self.config_path, ENV, email_field=""
        )

        self.assertFalse(actions["verified"])
        self.assertFalse(actions["can_manual"])
        self.assertFalse(actions["can_enable"])
        self.assertTrue(actions["can_verify"])

    def test_enable_actions_require_first_send_approval(self):
        self.verify(RECIPIENT)

        before = simple_ui.action_state(
            self.root, self.config_path, ENV, email_field=""
        )
        self.assertFalse(before["can_enable"])

        self.approve()
        after = simple_ui.action_state(
            self.root, self.config_path, ENV, email_field=""
        )
        self.assertTrue(after["can_enable"])


if __name__ == "__main__":
    unittest.main()
