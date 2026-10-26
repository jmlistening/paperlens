"""Offline Streamlit tests for the simple subscription page (Task 8).

Rendering only reads local configuration and private state; no arXiv, model,
or SMTP call happens. Every temporary root is isolated from the project's own
``data/`` directory, and all addresses are synthetic ``example.com`` fixtures.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ui_state  # noqa: E402
from push_run import RunResult  # noqa: E402
from push_state import PushState  # noqa: E402

PAGE = ROOT / "simple_page.py"
ENTRY = ROOT / "app.py"
TMP_ROOT = Path(__file__).resolve().parent / ".tmp"
RECIPIENT = "alice@example.com"
NEW_RECIPIENT = "bob@example.com"

PROFILE = {
    "research_goal": "RAG",
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


class SimplePageSourceTests(unittest.TestCase):
    def setUp(self):
        if not PAGE.is_file():
            raise AssertionError(f"missing Streamlit page: {PAGE}")
        self.source = PAGE.read_text(encoding="utf-8")

    def test_page_does_not_set_page_config(self):
        self.assertNotIn("set_page_config", self.source)

    def test_page_exposes_the_required_labels(self):
        for text in (
            "我想关注什么",
            "推送到哪个邮箱",
            "每次推送几篇",
            "先预览这周会推什么",
            "现在推送今天的",
            "每天自动推送",
            "每天几点",
            "高级 / 研究模式",
            "仅验证邮箱（不开启自动推送）",
            "完成首次发送批准（不开启自动推送）",
        ):
            self.assertIn(text, self.source)

    def test_confirm_button_sits_at_the_page_bottom_right(self):
        confirm_at = self.source.index('key="simple_confirm_settings"')
        time_at = self.source.index('key="simple_time"')
        advanced_at = self.source.index("advanced/home.py")
        self.assertGreater(confirm_at, time_at)
        self.assertLess(confirm_at, advanced_at)
        self.assertIn("with confirm_column:", self.source)

    def test_page_has_no_research_or_secret_surfaces(self):
        lowered = self.source.casefold()
        for forbidden in (
            "evidence",
            "评测",
            "token",
            "trace",
            "baseline",
            "rerank",
            "schema",
            "quote",
            "smtplib",
            "send_message",
            "smtp_password",
        ):
            self.assertNotIn(forbidden, lowered)


class SimplePageRenderTests(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.root = Path(self.tmp.name)
        profile_dir = self.root / "data" / "push"
        profile_dir.mkdir(parents=True)
        (profile_dir / "profile.json").write_text(
            json.dumps(PROFILE, ensure_ascii=False), encoding="utf-8"
        )
        config = json.loads((ROOT / "push_config.json").read_text(encoding="utf-8"))
        config.update({"enabled": False, "paused": False, "quiet_mode": False})
        self.config_path = self.root / "push_config.json"
        self.config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._old_env = {
            key: os.environ.get(key)
            for key in ("PAPERLENS_PUSH_ROOT", "PAPERLENS_PUSH_CONFIG")
        }
        os.environ["PAPERLENS_PUSH_ROOT"] = str(self.root)
        os.environ["PAPERLENS_PUSH_CONFIG"] = str(self.config_path)

    def tearDown(self):
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmp.cleanup()

    def run_page(self):
        from streamlit.testing.v1 import AppTest

        # The simple page is the default destination of the application entry;
        # running through it also verifies the advanced page link resolves.
        app = AppTest.from_file(str(ENTRY))
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)
        return app

    def verify_and_approve(self) -> None:
        ui_state.set_recipient(self.root, RECIPIENT)
        ui_state.mark_verified(self.root, RECIPIENT, "2026-10-25T10:00:00+00:00")
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        state.init_schema()
        state.set_setting(
            "first_send_approved_at", "2026-10-25T10:00:00+00:00"
        )

    def record_test(self, recipient: str = RECIPIENT) -> None:
        ui_state.set_recipient(self.root, recipient)
        ui_state.record_test_email(
            self.root, recipient, "2026-10-25T10:00:00+00:00"
        )

    def first_send_approved(self) -> bool:
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        state.init_schema()
        return bool(state.get_setting("first_send_approved_at"))

    def write_config(self, **patch) -> str:
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        raw.update(patch)
        text = json.dumps(raw, ensure_ascii=False, indent=2) + "\n"
        self.config_path.write_text(text, encoding="utf-8")
        return text

    def write_profile(self, **patch) -> str:
        path = self.root / "data" / "push" / "profile.json"
        profile = json.loads(path.read_text(encoding="utf-8"))
        profile.update(patch)
        text = json.dumps(profile, ensure_ascii=False) + "\n"
        path.write_text(text, encoding="utf-8")
        return text

    def read_config_text(self) -> str:
        return self.config_path.read_text(encoding="utf-8")

    def read_profile_text(self) -> str:
        return (self.root / "data" / "push" / "profile.json").read_text(
            encoding="utf-8"
        )

    def test_external_auto_toggle_change_is_reconciled_without_a_write(self):
        """Streamlit ignores ``value=`` once a key exists: sync it explicitly.

        An external edit to ``push_config.json`` must be reflected on the next
        rerun, in both directions, without a render writing the stale session
        value back.
        """
        self.verify_and_approve()
        app = self.run_page()
        self.assertFalse(app.toggle(key="simple_auto").value)

        external = self.write_config(enabled=True)
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertTrue(app.toggle(key="simple_auto").value)
        self.assertEqual(self.read_config_text(), external)

        external = self.write_config(enabled=False)
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertFalse(app.toggle(key="simple_auto").value)
        self.assertEqual(self.read_config_text(), external)

    def test_external_count_time_and_topics_edits_are_not_reverted(self):
        """A plain rerun must follow external files, never overwrite them."""
        ui_state.set_recipient(self.root, RECIPIENT)
        app = self.run_page()
        self.assertEqual(app.slider(key="simple_count").value, 3)
        self.assertEqual(app.text_area(key="simple_topics").value, "RAG")

        external_config = self.write_config(
            max_per_digest=8, time_local="10:15"
        )
        external_profile = self.write_profile(
            research_goal="External new topic",
            topics=["External new topic"],
        )

        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertEqual(app.slider(key="simple_count").value, 8)
        self.assertEqual(
            app.time_input(key="simple_time").value, time(10, 15)
        )
        self.assertEqual(
            app.text_area(key="simple_topics").value, "External new topic"
        )
        self.assertEqual(self.read_config_text(), external_config)
        self.assertEqual(self.read_profile_text(), external_profile)

    def test_preview_after_an_external_profile_edit_uses_the_new_topics(self):
        """A later preview follows the file edit and never saves on its own."""
        ui_state.set_recipient(self.root, RECIPIENT)
        app = self.run_page()

        external_config = self.write_config(
            max_per_digest=8, time_local="10:15"
        )
        external_profile = self.write_profile(
            research_goal="External new topic",
            topics=["External new topic"],
        )

        no_digest = RunResult(
            run_id=None,
            status="no_digest",
            candidate_count=0,
            selected_count=0,
            sent=False,
            errors=[],
            run_dir=None,
        )
        with mock.patch(
            "simple_ui.save_settings",
            side_effect=AssertionError("preview must not save settings"),
        ) as save, mock.patch(
            "simple_ui.preview", return_value=no_digest
        ) as preview:
            app.button(key="simple_preview").click()
            app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertFalse(save.called)
        self.assertTrue(preview.called)
        self.assertEqual(self.read_config_text(), external_config)
        self.assertEqual(self.read_profile_text(), external_profile)

    def test_count_is_saved_only_after_confirmation(self):
        """Slider changes stay pending; 确认设置 writes, a rerun does not."""
        ui_state.set_recipient(self.root, RECIPIENT)
        app = self.run_page()

        app.slider(key="simple_count").set_value(6)
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertEqual(
            json.loads(self.read_config_text())["max_per_digest"], 3
        )

        app.button(key="simple_confirm_settings").click()
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertEqual(
            json.loads(self.read_config_text())["max_per_digest"], 6
        )

        before = self.read_config_text()
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertEqual(self.read_config_text(), before)

    def test_quick_count_buttons_update_the_pending_count(self):
        ui_state.set_recipient(self.root, RECIPIENT)
        app = self.run_page()

        app.button(key="simple_count_10").click()
        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertEqual(app.slider(key="simple_count").value, 10)
        self.assertEqual(
            json.loads(self.read_config_text())["max_per_digest"], 3
        )

        app.button(key="simple_confirm_settings").click()
        app.run(timeout=30)
        config = json.loads(self.read_config_text())
        self.assertEqual(config["max_per_digest"], 10)
        self.assertEqual(app.slider(key="simple_count").value, 10)

    def test_confirm_settings_keeps_values_visible_and_shows_saved_summary(self):
        ui_state.set_recipient(self.root, RECIPIENT)
        app = self.run_page()

        app.slider(key="simple_count").set_value(5)
        app.text_input(key="simple_email").input(RECIPIENT)
        app.run(timeout=30)
        self.assertFalse(app.button(key="simple_confirm_settings").disabled)

        app.button(key="simple_confirm_settings").click()
        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        # Nothing disappears or locks after saving.
        self.assertFalse(app.text_input(key="simple_email").disabled)
        self.assertFalse(app.slider(key="simple_count").disabled)
        self.assertEqual(app.text_input(key="simple_email").value, RECIPIENT)
        self.assertEqual(app.slider(key="simple_count").value, 5)
        infos = "\n".join(item.value for item in app.info)
        self.assertIn("5 篇/次", infos)
        self.assertIn("a***@example.com", infos)
        # The confirm control stays available for another save.
        self.assertFalse(app.button(key="simple_confirm_settings").disabled)

    def test_unsaved_topic_disables_actions_until_confirmed(self):
        ui_state.set_recipient(self.root, RECIPIENT)
        app = self.run_page()

        app.text_area(key="simple_topics").input("A new topic")
        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertTrue(app.button(key="simple_preview").disabled)
        self.assertTrue(app.button(key="simple_manual").disabled)
        warnings = "\n".join(item.value for item in app.warning)
        self.assertIn("请到页面最下方点「确认设置」", warnings)

    def test_time_change_requires_confirm_and_persists(self):
        self.verify_and_approve()
        before = json.loads(self.read_config_text())["time_local"]
        app = self.run_page()

        app.time_input(key="simple_time").set_value(time(9, 45))
        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertEqual(
            json.loads(self.read_config_text())["time_local"], before
        )
        self.assertFalse(app.button(key="simple_confirm_settings").disabled)
        self.assertTrue(app.button(key="simple_preview").disabled)
        self.assertTrue(app.toggle(key="simple_auto").disabled)

        app.button(key="simple_confirm_settings").click()
        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertEqual(
            json.loads(self.read_config_text())["time_local"], "09:45"
        )
        self.assertFalse(app.button(key="simple_confirm_settings").disabled)
        self.assertFalse(app.button(key="simple_preview").disabled)
        self.assertFalse(app.toggle(key="simple_auto").disabled)

    def test_page_renders_without_network_or_smtp(self):
        with mock.patch(
            "simple_ui.manual_send", side_effect=AssertionError("no send")
        ) as manual, mock.patch(
            "simple_ui.test_email", side_effect=AssertionError("no test mail")
        ) as test_mail, mock.patch(
            "simple_ui.enable", side_effect=AssertionError("no enable")
        ) as enable, mock.patch(
            "simple_ui.disable", side_effect=AssertionError("no disable")
        ) as disable:
            app = self.run_page()

        self.assertIn("PaperLens", app.title[0].value)
        self.assertFalse(manual.called)
        self.assertFalse(test_mail.called)
        self.assertFalse(enable.called)
        self.assertFalse(disable.called)

    def test_page_shows_subscription_controls(self):
        app = self.run_page()

        self.assertEqual(
            [area.label for area in app.text_area], ["我想关注什么"]
        )
        self.assertEqual(
            [field.label for field in app.text_input], ["推送到哪个邮箱"]
        )
        self.assertEqual(
            [slider.label for slider in app.slider], ["每次推送几篇"]
        )
        button_labels = [button.label for button in app.button]
        for label in (
            "确认设置",
            "先预览这周会推什么",
            "现在推送今天的",
        ):
            self.assertIn(label, button_labels)
        for quick in ("1", "3", "5", "10"):
            self.assertIn(quick, button_labels)
        self.assertEqual(
            [toggle.label for toggle in app.toggle], ["每天自动推送"]
        )
        self.assertEqual(
            [field.label for field in app.time_input], ["每天几点"]
        )
        self.assertIn(
            "高级 / 研究模式", [link.label for link in app.page_link]
        )

    def test_unsaved_email_disables_auto_push(self):
        self.verify_and_approve()
        app = self.run_page()
        self.assertFalse(app.toggle(key="simple_auto").disabled)

        app.text_input(key="simple_email").input(NEW_RECIPIENT)
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)

        self.assertTrue(app.toggle(key="simple_auto").disabled)
        self.assertTrue(app.button(key="simple_manual").disabled)

    def test_unsaved_email_disables_auto_push_on_a_legacy_env_install(self):
        # No private recipient: the legacy .env address is what a send uses.
        (self.root / ".env").write_text(
            "MAIL_TO=legacy@example.com\n", encoding="utf-8"
        )
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        state.init_schema()
        state.set_setting(
            "first_send_approved_at", "2026-10-25T10:00:00+00:00"
        )

        app = self.run_page()
        self.assertFalse(app.toggle(key="simple_auto").disabled)

        app.text_input(key="simple_email").input(NEW_RECIPIENT)
        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertTrue(app.toggle(key="simple_auto").disabled)
        self.assertTrue(app.button(key="simple_manual").disabled)
        warnings = "\n".join(item.value for item in app.warning)
        self.assertIn("请先保存收件邮箱", warnings)

    def test_confirm_without_a_recipient_explains_and_does_not_save(self):
        app = self.run_page()
        before = json.loads(self.config_path.read_text(encoding="utf-8"))

        app.slider(key="simple_count").set_value(5)
        app.run(timeout=30)
        app.button(key="simple_confirm_settings").click()
        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        errors = "\n".join(item.value for item in app.error)
        self.assertIn("请填写要推送的邮箱", errors)
        after = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            after["max_per_digest"], before["max_per_digest"]
        )
        # A failed confirm leaves the settings editable.
        self.assertFalse(app.button(key="simple_confirm_settings").disabled)

    def test_count_change_with_a_saved_recipient_persists_on_confirm(self):
        ui_state.set_recipient(self.root, RECIPIENT)
        app = self.run_page()

        app.slider(key="simple_count").set_value(5)
        app.run(timeout=30)
        app.button(key="simple_confirm_settings").click()
        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["max_per_digest"], 5)
        profile = json.loads(
            (self.root / "data" / "push" / "profile.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(profile["top_k"], 5)

    def test_enabled_unapproved_push_can_still_be_turned_off(self):
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        raw["enabled"] = True
        self.config_path.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.root / ".env").write_text(
            "MAIL_TO=legacy@example.com\n"
            "SMTP_USER=sender@example.com\n"
            "SMTP_PASSWORD=test-app-code\n",
            encoding="utf-8",
        )

        app = self.run_page()
        toggle = app.toggle(key="simple_auto")
        self.assertTrue(toggle.value)
        self.assertFalse(toggle.disabled)

        toggle.set_value(False)
        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertFalse(config["enabled"])

    def test_legacy_env_unapproved_recipient_can_complete_the_flow(self):
        (self.root / ".env").write_text(
            "MAIL_TO=legacy@example.com\n"
            "SMTP_USER=sender@example.com\n"
            "SMTP_PASSWORD=test-app-code\n",
            encoding="utf-8",
        )

        app = self.run_page()

        self.assertIn(
            "首次开启：先批准发送",
            [item.value for item in app.subheader],
        )
        labels = [button.label for button in app.button]
        self.assertIn("发送测试邮件", labels)
        self.assertIn("我收到了，开启推送", labels)
        self.assertTrue(app.button(key="simple_confirm_enable").disabled)
        # No successful test yet: the receipt checkbox is not usable.
        self.assertTrue(app.checkbox(key="simple_received").disabled)

        with mock.patch(
            "simple_ui.push_ctl.test_email",
            return_value={"status": "sent", "host": "smtp.163.com", "port": 465},
        ) as send_test:
            app.button(key="simple_test_email").click()
            app.run(timeout=30)
        send_test.assert_called_once()
        self.assertFalse(app.checkbox(key="simple_received").disabled)

        app.checkbox(key="simple_received").check()
        app.run(timeout=30)
        self.assertFalse(app.button(key="simple_confirm_enable").disabled)

        with mock.patch(
            "simple_ui.push_ctl.register_task",
            return_value={"installed": True, "returncode": 0, "output": ""},
        ) as register:
            app.button(key="simple_confirm_enable").click()
            app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        register.assert_called_once()
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertTrue(config["enabled"])
        state = ui_state.load_state(self.root)
        self.assertEqual(state["recipient"], "legacy@example.com")
        self.assertIsNotNone(state["verified_at"])
        frontier = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        frontier.init_schema()
        self.assertIsNotNone(frontier.get_setting("first_send_approved_at"))
        successes = "\n".join(item.value for item in app.success)
        self.assertIn("已开启", successes)

    def test_manual_send_requires_the_confirmation_checkbox(self):
        self.verify_and_approve()
        app = self.run_page()

        self.assertTrue(app.button(key="simple_manual").disabled)

        app.checkbox(key="simple_confirm_send").check()
        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertFalse(app.button(key="simple_manual").disabled)

    def test_unverified_recipient_shows_the_test_email_step(self):
        ui_state.set_recipient(self.root, RECIPIENT)

        app = self.run_page()

        labels = [button.label for button in app.button]
        self.assertIn("发送测试邮件", labels)
        self.assertIn("我收到了，开启推送", labels)
        self.assertTrue(app.button(key="simple_manual").disabled)

    def test_failed_test_email_keeps_receipt_and_confirm_disabled(self):
        ui_state.set_recipient(self.root, RECIPIENT)
        app = self.run_page()

        with mock.patch(
            "simple_ui.push_ctl.test_email",
            side_effect=RuntimeError("super-secret-smtp-detail"),
        ):
            app.button(key="simple_test_email").click()
            app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertTrue(app.checkbox(key="simple_received").disabled)
        self.assertTrue(app.button(key="simple_confirm_enable").disabled)
        errors = "\n".join(item.value for item in app.error)
        self.assertIn("RuntimeError", errors)
        self.assertNotIn("super-secret-smtp-detail", errors)
        config = json.loads(self.read_config_text())
        self.assertFalse(config["enabled"])

    def test_successful_test_email_unlocks_receipt_and_confirm(self):
        (self.root / ".env").write_text(
            "SMTP_HOST=smtp.163.com\n"
            "SMTP_PORT=465\n"
            "SMTP_SECURITY=ssl\n"
            "SMTP_USER=sender@example.com\n"
            "SMTP_PASSWORD=test-app-code\n",
            encoding="utf-8",
        )
        ui_state.set_recipient(self.root, RECIPIENT)
        app = self.run_page()

        with mock.patch(
            "simple_ui.push_ctl.test_email",
            return_value={"status": "sent", "host": "smtp.163.com", "port": 465},
        ):
            app.button(key="simple_test_email").click()
            app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertFalse(app.checkbox(key="simple_received").disabled)
        self.assertTrue(ui_state.test_email_confirmed(self.root, RECIPIENT))

        app.checkbox(key="simple_received").check()
        app.run(timeout=30)
        self.assertFalse(app.button(key="simple_confirm_enable").disabled)

        with mock.patch(
            "simple_ui.push_ctl.register_task",
            return_value={"installed": True, "returncode": 0, "output": ""},
        ):
            app.button(key="simple_confirm_enable").click()
            app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertTrue(json.loads(self.read_config_text())["enabled"])
        state = ui_state.load_state(self.root)
        self.assertIsNotNone(state["verified_at"])

    def test_verify_only_button_unlocks_manual_send_without_enabling_auto(self):
        self.record_test()
        before = self.read_config_text()

        app = self.run_page()

        labels = [button.label for button in app.button]
        self.assertIn("仅验证邮箱（不开启自动推送）", labels)
        self.assertTrue(app.button(key="simple_verify_only").disabled)
        self.assertTrue(app.button(key="simple_manual").disabled)

        app.checkbox(key="simple_received").check()
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertFalse(app.button(key="simple_verify_only").disabled)

        with mock.patch(
            "simple_ui.push_ctl.register_task",
            side_effect=AssertionError("verify-only must not install a task"),
        ) as register:
            app.button(key="simple_verify_only").click()
            app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertFalse(register.called)
        self.assertEqual(self.read_config_text(), before)
        config = json.loads(before)
        self.assertFalse(config["enabled"])
        self.assertFalse(config["paused"])
        state = ui_state.load_state(self.root)
        self.assertIsNotNone(state["verified_at"])
        self.assertTrue(self.first_send_approved())
        successes = "\n".join(item.value for item in app.success)
        self.assertIn("邮箱已验证，手动推送可用；自动推送开关未改变。", successes)

        app.checkbox(key="simple_confirm_send").check()
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertFalse(app.button(key="simple_manual").disabled)

    def test_failed_test_email_cannot_verify(self):
        ui_state.set_recipient(self.root, RECIPIENT)
        before = self.read_config_text()

        app = self.run_page()

        with mock.patch(
            "simple_ui.push_ctl.test_email",
            side_effect=RuntimeError("super-secret-smtp-detail"),
        ):
            app.button(key="simple_test_email").click()
            app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertTrue(app.checkbox(key="simple_received").disabled)
        self.assertTrue(app.button(key="simple_verify_only").disabled)
        self.assertIsNone(ui_state.load_state(self.root)["verified_at"])
        self.assertFalse(self.first_send_approved())
        self.assertEqual(self.read_config_text(), before)
        errors = "\n".join(item.value for item in app.error)
        self.assertIn("RuntimeError", errors)
        self.assertNotIn("super-secret-smtp-detail", errors)

    def test_already_verified_recipient_uses_the_approval_only_button(self):
        ui_state.set_recipient(self.root, RECIPIENT)
        ui_state.mark_verified(
            self.root, RECIPIENT, "2026-10-25T10:00:00+00:00"
        )
        ui_state.record_test_email(
            self.root, RECIPIENT, "2026-10-25T10:01:00+00:00"
        )
        before = self.read_config_text()

        app = self.run_page()

        labels = [button.label for button in app.button]
        self.assertIn("完成首次发送批准（不开启自动推送）", labels)
        self.assertNotIn("仅验证邮箱（不开启自动推送）", labels)
        self.assertNotIn("发送测试邮件", labels)
        self.assertFalse(app.button(key="simple_verify_only").disabled)
        captions = "\n".join(item.value for item in app.caption)
        self.assertIn("邮箱已验证", captions)
        self.assertIn("批准首次发送", captions)
        self.assertNotIn("测试邮件确认", captions)
        self.assertNotIn("发送测试邮件成功后", captions)

        app.button(key="simple_verify_only").click()
        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertEqual(self.read_config_text(), before)
        self.assertTrue(self.first_send_approved())
        successes = "\n".join(item.value for item in app.success)
        self.assertIn("邮箱已验证，手动推送可用；自动推送开关未改变。", successes)

    def test_manual_unavailable_caption_asks_for_a_saved_recipient_first(self):
        app = self.run_page()
        captions = "\n".join(item.value for item in app.caption)
        self.assertIn("手动推送暂不可用", captions)
        self.assertIn("请先保存收件邮箱", captions)
        self.assertNotIn("未验证邮箱", captions)

    def test_manual_unavailable_caption_stays_visible_when_auto_is_enabled(self):
        ui_state.set_recipient(self.root, RECIPIENT)
        config = json.loads(self.read_config_text())
        config["enabled"] = True
        self.config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        app = self.run_page()

        captions = "\n".join(item.value for item in app.caption)
        self.assertIn("手动推送暂不可用", captions)
        self.assertIn("未验证邮箱", captions)

    def test_manual_unavailable_caption_names_the_current_gate(self):
        ui_state.set_recipient(self.root, RECIPIENT)
        app = self.run_page()
        captions = "\n".join(item.value for item in app.caption)
        self.assertIn("手动推送暂不可用", captions)
        self.assertIn("未验证邮箱", captions)

        ui_state.mark_verified(
            self.root, RECIPIENT, "2026-10-25T10:00:00+00:00"
        )
        app.run(timeout=30)
        captions = "\n".join(item.value for item in app.caption)
        self.assertIn("未完成首次发送批准", captions)

        app.text_input(key="simple_email").input(NEW_RECIPIENT)
        app.run(timeout=30)
        captions = "\n".join(item.value for item in app.caption)
        self.assertIn("有待保存的邮箱", captions)

    def test_last_scheduled_attempt_is_visible(self):
        run_dir = self.root / "data" / "push" / "runs" / "scheduled-check"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "run_id": "scheduled-check",
                    "mode": "scheduled",
                    "status": "no_digest",
                    "started_at": "2026-10-25T16:05:03+00:00",
                    "finished_at": "2026-10-25T16:05:16+00:00",
                    "counts": {"candidates": 10, "ranked": 10, "selected": 0},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        app = self.run_page()

        captions = "\n".join(item.value for item in app.caption)
        self.assertIn("上次定时尝试", captions)
        self.assertIn("没有合格论文", captions)

    def test_empty_preview_shows_a_no_papers_state(self):
        with mock.patch(
            "simple_ui.save_settings",
            side_effect=AssertionError("preview must not save settings"),
        ) as save, mock.patch(
            "simple_ui.preview",
            return_value=RunResult(
                run_id=None,
                status="no_digest",
                candidate_count=0,
                selected_count=0,
                sent=False,
                errors=[],
                run_dir=None,
            ),
        ) as preview:
            app = self.run_page()
            app.button(key="simple_preview").click()
            app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertFalse(save.called)
        self.assertTrue(preview.called)
        rendered = "\n".join(info.value for info in app.info)
        self.assertIn("没有找到", rendered)

    def test_unexpected_preview_error_shows_only_the_type_name(self):
        ui_state.set_recipient(self.root, RECIPIENT)
        app = self.run_page()

        with mock.patch(
            "simple_ui.save_settings",
            return_value={"saved": True, "enabled": False},
        ), mock.patch(
            "simple_ui.preview",
            side_effect=RuntimeError("super-secret-preview-detail"),
        ):
            app.button(key="simple_preview").click()
            app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        errors = "\n".join(item.value for item in app.error)
        self.assertIn("RuntimeError", errors)
        self.assertNotIn("super-secret-preview-detail", errors)
        self.assertNotIn("Traceback", errors)

    def test_unexpected_manual_error_shows_only_the_type_name(self):
        self.verify_and_approve()
        app = self.run_page()
        app.checkbox(key="simple_confirm_send").check()
        app.run(timeout=30)

        with mock.patch(
            "simple_ui.manual_send",
            side_effect=OSError("super-secret-manual-detail"),
        ):
            app.button(key="simple_manual").click()
            app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        errors = "\n".join(item.value for item in app.error)
        self.assertIn("OSError", errors)
        self.assertNotIn("super-secret-manual-detail", errors)
        self.assertNotIn("Traceback", errors)

    def test_page_does_not_write_project_state(self):
        project_state = ROOT / "data" / "push" / "ui_state.json"
        state_before = (
            project_state.stat().st_mtime_ns if project_state.exists() else None
        )
        config_before = (ROOT / "push_config.json").stat().st_mtime_ns

        self.run_page()

        state_after = (
            project_state.stat().st_mtime_ns if project_state.exists() else None
        )
        self.assertEqual(state_after, state_before)
        self.assertEqual(
            (ROOT / "push_config.json").stat().st_mtime_ns, config_before
        )
        self.assertTrue(
            (self.root / "data" / "push" / "frontier.sqlite3").exists()
        )


if __name__ == "__main__":
    unittest.main()
