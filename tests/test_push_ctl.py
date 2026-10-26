"""Offline tests for the push control plane."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from push_ctl import (  # noqa: E402
    PushCtlError,
    approve_first_send,
    doctor,
    resume,
    set_enabled,
    set_paused,
    status_json,
    test_email,
)
from push_state import PushState  # noqa: E402

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"
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


def make_root(tmp: str) -> tuple[Path, Path]:
    root = Path(tmp)
    raw = json.loads((ROOT / "push_config.json").read_text(encoding="utf-8"))
    raw["enabled"] = True
    config_path = root / "push_config.json"
    config_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return root, config_path


class PushCtlTests(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.root, self.config_path = make_root(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_enable_requires_valid_env(self):
        result = set_enabled(True, self.config_path, ENV)
        self.assertTrue(result["enabled"])
        self.assertTrue(json.loads(self.config_path.read_text())["enabled"])
        with self.assertRaises(PushCtlError):
            set_enabled(True, self.config_path, {})

    def test_disable_does_not_require_env(self):
        result = set_enabled(False, self.config_path, {})
        self.assertFalse(result["enabled"])

    def test_approve_requires_confirm_and_is_visible_in_status(self):
        with self.assertRaises(PushCtlError):
            approve_first_send(self.root, confirm=False)
        result = approve_first_send(self.root, confirm=True)
        self.assertIn("first_send_approved_at", result)
        status = status_json(self.root, self.config_path, ENV)
        self.assertTrue(status["first_send_approved"])

    def test_status_masks_email_and_never_exposes_password(self):
        status = status_json(self.root, self.config_path, ENV)
        rendered = json.dumps(status, ensure_ascii=False)
        self.assertNotIn("test-app-code", rendered)
        self.assertNotIn("user@163.com", rendered)
        self.assertIn("u***@163.com", rendered)
        self.assertTrue(status["recipient_verified"])
        self.assertEqual(status["recipient_source"], "env")
        self.assertEqual(status["config_problems"], [])

    def test_status_prefers_the_ui_state_recipient(self):
        from ui_state import mark_verified, set_recipient

        set_recipient(self.root, "alice@example.com")
        mark_verified(self.root, "alice@example.com", "2026-10-25T08:30:00+00:00")

        status = status_json(self.root, self.config_path, ENV)

        self.assertEqual(status["mail_to"], "a***@example.com")
        self.assertTrue(status["recipient_verified"])
        self.assertEqual(status["recipient_source"], "ui_state")
        rendered = json.dumps(status, ensure_ascii=False)
        self.assertNotIn("alice@example.com", rendered)
        self.assertNotIn("user@163.com", rendered)

    def test_test_email_login_only_does_not_send(self):
        calls = []
        result = test_email(
            self.config_path, ENV, login_fn=lambda settings: calls.append(settings)
        )
        self.assertEqual(result["status"], "login_ok")
        self.assertEqual(len(calls), 1)

    def test_test_email_send_requires_confirm(self):
        with self.assertRaises(PushCtlError):
            test_email(self.config_path, ENV, send=True, confirm=False)
        sent = []
        result = test_email(
            self.config_path, ENV, send=True, confirm=True,
            send_fn=lambda settings, message: sent.append(message),
        )
        self.assertEqual(result["status"], "sent")
        self.assertEqual(len(sent), 1)

    def test_resume_clears_pause_and_auto_pause(self):
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        state.init_schema()
        state.set_setting("auto_paused", "true")
        set_paused(True, self.config_path)
        result = resume(self.root, self.config_path)
        self.assertFalse(result["paused"])
        self.assertFalse(result["auto_paused"])
        self.assertEqual(state.get_setting("auto_paused"), "false")
        self.assertFalse(json.loads(self.config_path.read_text())["paused"])

    def test_doctor_reports_missing_env_without_network(self):
        report = doctor(self.config_path, {})
        self.assertFalse(report["ok"])
        self.assertTrue(report["problems"])


if __name__ == "__main__":
    unittest.main()
