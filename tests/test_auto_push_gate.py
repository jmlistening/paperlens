"""Offline tests for the shared auto-push gate, recipient status, and task wrappers.

Every address is a synthetic ``example.com`` fixture. No network, SMTP, model,
or real Task Scheduler call happens here: both the pipeline and the task
wrappers are exercised with fakes/mocks against temporary roots only.
"""
from __future__ import annotations

import json
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
from push_ctl import (  # noqa: E402
    TASK_NAME,
    PushCtlError,
    register_task,
    set_auto_push,
    status_json,
    task_installed,
    unregister_task,
)
from push_run import PushDeps, run_push  # noqa: E402
from push_state import PushState  # noqa: E402

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"
NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
VERIFIED_AT = "2026-10-25T10:00:00+00:00"
UI_EMAIL = "alice@example.com"
LEGACY_EMAIL = "legacy@example.com"
ENV = {
    "DEEPSEEK_API_KEY": "test-key",
    "DEEPSEEK_BASE_URL": "https://example.invalid",
    "DEEPSEEK_MODEL": "test-model",
    "SMTP_HOST": "smtp.163.com",
    "SMTP_PORT": "465",
    "SMTP_SECURITY": "ssl",
    "SMTP_USER": "sender@example.com",
    "SMTP_PASSWORD": "test-app-code",
    "MAIL_TO": LEGACY_EMAIL,
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
    return {paper["arxiv_id"].split("v")[0]: "中文速览" for paper in papers}


def write_root(
    tmp: str, *, enabled: bool = False, paused: bool = False, quiet: bool = False
) -> Path:
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


class AutoPushGateTests(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.root = write_root(self.tmp.name)
        self.config_path = self.root / "push_config.json"

    def tearDown(self):
        self.tmp.cleanup()

    def read_config(self) -> dict:
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def state(self) -> PushState:
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        state.init_schema()
        return state

    def verify_ui_recipient(self) -> None:
        ui_state.set_recipient(self.root, UI_EMAIL)
        ui_state.mark_verified(self.root, UI_EMAIL, VERIFIED_AT)

    def test_enabling_with_an_unverified_ui_recipient_is_refused(self):
        ui_state.set_recipient(self.root, UI_EMAIL)

        with self.assertRaises(PushCtlError) as ctx:
            set_auto_push(True, self.root, self.config_path, ENV)

        self.assertNotIn(UI_EMAIL, str(ctx.exception))
        self.assertFalse(self.read_config()["enabled"])

    def test_enabling_with_a_verified_ui_recipient_uses_the_ui_recipient(self):
        self.verify_ui_recipient()
        env = dict(ENV)
        env.pop("MAIL_TO")  # the UI recipient must satisfy validation

        result = set_auto_push(True, self.root, self.config_path, env)

        self.assertTrue(result["enabled"])
        self.assertTrue(result["recipient_verified"])
        self.assertEqual(result["recipient_source"], "ui_state")
        rendered = json.dumps(result, ensure_ascii=False)
        self.assertNotIn(UI_EMAIL, rendered)
        raw = self.read_config()
        self.assertTrue(raw["enabled"])
        self.assertFalse(raw["paused"])

    def test_enabling_with_legacy_env_succeeds_without_ui_state(self):
        self.assertFalse((self.root / "data" / "push" / "ui_state.json").exists())
        paused_root = write_root(self.tmp.name, enabled=False, paused=True)

        result = set_auto_push(True, paused_root, self.config_path, ENV)

        self.assertTrue(result["enabled"])
        self.assertTrue(result["recipient_verified"])
        self.assertEqual(result["recipient_source"], "env")
        raw = self.read_config()
        self.assertTrue(raw["enabled"])
        self.assertFalse(raw["paused"])

    def test_enabling_without_any_recipient_is_refused(self):
        env = dict(ENV)
        env.pop("MAIL_TO")

        with self.assertRaises(PushCtlError):
            set_auto_push(True, self.root, self.config_path, env)

        self.assertFalse(self.read_config()["enabled"])

    def test_enabling_is_refused_while_auto_paused(self):
        self.state().set_setting("auto_paused", "true")

        with self.assertRaises(PushCtlError):
            set_auto_push(True, self.root, self.config_path, ENV)

        self.assertFalse(self.read_config()["enabled"])
        self.assertEqual(
            self.state().get_setting("auto_paused"), "true"
        )

    def test_enabling_is_refused_in_quiet_mode(self):
        quiet_root = write_root(self.tmp.name, quiet=True)

        with self.assertRaises(PushCtlError):
            set_auto_push(True, quiet_root, self.config_path, ENV)

        raw = self.read_config()
        self.assertFalse(raw["enabled"])
        self.assertTrue(raw["quiet_mode"])

    def test_disabling_keeps_auto_pause_and_quiet_mode(self):
        enabled_root = write_root(self.tmp.name, enabled=True, quiet=True)
        state = self.state()
        state.set_setting("auto_paused", "true")

        result = set_auto_push(False, enabled_root, self.config_path, ENV)

        self.assertFalse(result["enabled"])
        raw = self.read_config()
        self.assertFalse(raw["enabled"])
        self.assertTrue(raw["quiet_mode"])
        self.assertEqual(
            self.state().get_setting("auto_paused"), "true"
        )


class RecipientStatusTests(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.root = write_root(self.tmp.name, enabled=True)
        self.config_path = self.root / "push_config.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_legacy_env_recipient_is_reported_as_verified_and_masked(self):
        status = status_json(self.root, self.config_path, ENV)

        self.assertEqual(status["mail_to"], "l***@example.com")
        self.assertTrue(status["recipient_verified"])
        self.assertEqual(status["recipient_source"], "env")
        rendered = json.dumps(status, ensure_ascii=False)
        self.assertNotIn(LEGACY_EMAIL, rendered)

    def test_verified_ui_recipient_wins_over_env_and_stays_masked(self):
        ui_state.set_recipient(self.root, UI_EMAIL)
        ui_state.mark_verified(self.root, UI_EMAIL, VERIFIED_AT)

        status = status_json(self.root, self.config_path, ENV)

        self.assertEqual(status["mail_to"], "a***@example.com")
        self.assertTrue(status["recipient_verified"])
        self.assertEqual(status["recipient_source"], "ui_state")
        rendered = json.dumps(status, ensure_ascii=False)
        self.assertNotIn(UI_EMAIL, rendered)
        self.assertNotIn(LEGACY_EMAIL, rendered)

    def test_unverified_ui_recipient_is_reported_unverified(self):
        ui_state.set_recipient(self.root, UI_EMAIL)

        status = status_json(self.root, self.config_path, ENV)

        self.assertEqual(status["mail_to"], "a***@example.com")
        self.assertFalse(status["recipient_verified"])
        self.assertEqual(status["recipient_source"], "ui_state")

    def test_verified_ui_recipient_satisfies_status_validation(self):
        ui_state.set_recipient(self.root, UI_EMAIL)
        ui_state.mark_verified(self.root, UI_EMAIL, VERIFIED_AT)
        env = dict(ENV)
        env.pop("MAIL_TO")

        status = status_json(self.root, self.config_path, env)

        self.assertEqual(status["mail_to"], "a***@example.com")
        self.assertEqual(status["config_problems"], [])

    def test_corrupt_ui_state_fails_closed_without_env_fallback(self):
        path = self.root / ui_state.STATE_REL
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not-json", encoding="utf-8")

        status = status_json(self.root, self.config_path, ENV)

        self.assertIsNone(status["mail_to"])
        self.assertFalse(status["recipient_verified"])
        self.assertEqual(status["recipient_source"], "ui_state")
        rendered = json.dumps(status, ensure_ascii=False)
        self.assertNotIn(LEGACY_EMAIL, rendered)


class ScheduledRecipientTests(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.root = write_root(self.tmp.name, enabled=True)
        self.config_path = self.root / "push_config.json"

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

    def test_scheduled_fails_closed_with_an_unverified_ui_recipient(self):
        ui_state.set_recipient(self.root, UI_EMAIL)

        result = run_push(
            "scheduled",
            config_path=self.config_path,
            now=NOW,
            deps=self.deps(
                ExplodingClient(),
                smtp_send=lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("unverified recipient must not send")
                ),
            ),
        )

        self.assertEqual(result.status, "recipient_unverified")
        self.assertFalse(result.sent)
        self.assertFalse(
            (self.root / "data" / "push" / "frontier.sqlite3").exists()
        )
        self.assertNotIn(UI_EMAIL, json.dumps(result.__dict__, ensure_ascii=False))

    def test_send_fails_closed_with_an_unverified_ui_recipient(self):
        ui_state.set_recipient(self.root, UI_EMAIL)

        result = run_push(
            "send",
            config_path=self.config_path,
            now=NOW,
            deps=self.deps(ExplodingClient()),
        )

        self.assertEqual(result.status, "recipient_unverified")
        self.assertFalse(result.sent)

    def test_preview_remains_available_before_verification(self):
        ui_state.set_recipient(self.root, UI_EMAIL)

        result = run_push(
            "preview",
            config_path=self.config_path,
            now=NOW,
            deps=self.deps(FakeClient([])),
        )

        self.assertNotEqual(result.status, "recipient_unverified")
        self.assertEqual(result.status, "no_digest")

    def test_scheduled_sends_to_the_verified_ui_recipient_not_env(self):
        ui_state.set_recipient(self.root, UI_EMAIL)
        ui_state.mark_verified(self.root, UI_EMAIL, VERIFIED_AT)
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        state.init_schema()
        state.set_setting("first_send_approved_at", NOW.isoformat())
        sent: list = []

        result = run_push(
            "scheduled",
            config_path=self.config_path,
            now=NOW,
            deps=self.deps(
                FakeClient([FakeResult("2601.00001v1", "A RAG paper", NOW)]),
                smtp_send=lambda settings, message: sent.append((settings, message)),
            ),
        )

        self.assertEqual(result.status, "sent")
        self.assertEqual(len(sent), 1)
        message = sent[0][1]
        self.assertEqual(message["To"], UI_EMAIL)
        self.assertNotEqual(message["To"], LEGACY_EMAIL)


class TaskWrapperTests(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_task_installed_queries_scheduled_tasks(self):
        with mock.patch("push_ctl.subprocess.run") as run:
            run.return_value = SimpleNamespace(
                returncode=0, stdout="TaskName: test", stderr=""
            )

            self.assertTrue(task_installed())

        command = run.call_args.args[0]
        self.assertEqual(command[0], "schtasks")
        self.assertIn("/Query", command)
        self.assertIn(TASK_NAME, command)

    def test_task_installed_is_false_on_error_or_scheduler_failure(self):
        with mock.patch("push_ctl.subprocess.run") as run:
            run.return_value = SimpleNamespace(returncode=1, stdout="", stderr="")
            self.assertFalse(task_installed())

        with mock.patch("push_ctl.subprocess.run", side_effect=OSError("boom")):
            self.assertFalse(task_installed())

    def test_register_task_calls_the_checked_in_script(self):
        with mock.patch("push_ctl.subprocess.run") as run:
            run.return_value = SimpleNamespace(
                returncode=0, stdout="registered", stderr=""
            )

            result = register_task(self.root, "07:45")

        self.assertTrue(result["installed"])
        self.assertEqual(result["returncode"], 0)
        command = run.call_args.args[0]
        self.assertEqual(command[0], "powershell.exe")
        script = command[command.index("-File") + 1]
        self.assertTrue(script.endswith("register_push_task.ps1"), script)
        self.assertEqual(command[command.index("-ProjectDir") + 1], str(self.root))
        self.assertEqual(command[command.index("-Time") + 1], "07:45")

    def test_register_task_rejects_an_invalid_time_before_any_subprocess(self):
        with mock.patch("push_ctl.subprocess.run") as run:
            with self.assertRaises(PushCtlError):
                register_task(self.root, "25:99")
            run.assert_not_called()

    def test_register_task_reports_a_failed_install(self):
        with mock.patch("push_ctl.subprocess.run") as run:
            run.return_value = SimpleNamespace(
                returncode=5, stdout="", stderr="access denied"
            )

            result = register_task(self.root, "07:45")

        self.assertFalse(result["installed"])
        self.assertEqual(result["returncode"], 5)
        self.assertIn("access denied", result["output"])

    def test_unregister_task_calls_the_checked_in_script(self):
        with mock.patch("push_ctl.subprocess.run") as run:
            run.return_value = SimpleNamespace(
                returncode=0, stdout="removed", stderr=""
            )

            result = unregister_task()

        self.assertTrue(result["removed"])
        command = run.call_args.args[0]
        script = command[command.index("-File") + 1]
        self.assertTrue(script.endswith("unregister_push_task.ps1"), script)

    def test_unregister_task_reports_failure(self):
        with mock.patch("push_ctl.subprocess.run") as run:
            run.return_value = SimpleNamespace(
                returncode=1, stdout="", stderr="failed"
            )

            result = unregister_task()

        self.assertFalse(result["removed"])
        self.assertEqual(result["returncode"], 1)

    def test_push_ctl_names_the_checked_in_scripts(self):
        source = (ROOT / "push_ctl.py").read_text(encoding="utf-8")
        for text in (
            "register_push_task.ps1",
            "unregister_push_task.ps1",
            "schtasks",
            TASK_NAME,
        ):
            self.assertIn(text, source)

    def test_ui_pages_use_the_shared_gate(self):
        settings = (ROOT / "advanced" / "push_settings.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("set_auto_push", settings)
        self.assertNotIn("set_enabled", settings)

        # The overview keeps status and scheduler controls only; enabling push
        # must go through the simple page and the shared gate, never a second
        # toggle of its own.
        overview = (ROOT / "advanced" / "overview.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("set_auto_push", overview)
        self.assertNotIn("set_enabled", overview)


if __name__ == "__main__":
    unittest.main()
