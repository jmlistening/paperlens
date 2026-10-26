"""Contract tests for the Streamlit push settings page (Task P8/P9).

The page must never perform network, model, or SMTP work just by being
rendered. The AppTest run below points the page at a temporary, disabled
configuration so no project state and no credential is touched.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from push_run import RunResult  # noqa: E402

PAGE = ROOT / "advanced" / "push_settings.py"
TMP_ROOT = Path(__file__).resolve().parent / ".tmp"


class PushPageSourceContractTests(unittest.TestCase):
    def setUp(self):
        if not PAGE.is_file():
            raise AssertionError(f"missing Streamlit page: {PAGE}")
        self.source = PAGE.read_text(encoding="utf-8")

    def test_exposes_all_required_controls(self):
        for label in (
            "163",
            "QQ",
            "预览",
            "测试邮箱登录",
            "批准首次发送",
            "暂停",
            "恢复",
            "最近运行",
        ):
            self.assertIn(label, self.source, f"page is missing control text: {label}")

    def test_uses_the_shared_status_and_control_helpers(self):
        self.assertIn("push_status", self.source)
        self.assertIn("push_ctl", self.source)

    def test_never_constructs_smtp_connections_itself(self):
        self.assertNotIn("smtplib", self.source)
        self.assertNotIn("send_message", self.source)
        self.assertNotIn("SMTP_SSL", self.source)

    def test_does_not_embed_secret_values(self):
        self.assertNotIn("SMTP_PASSWORD=", self.source)
        self.assertNotIn("test-app-code", self.source)

    def test_page_does_not_set_page_config(self):
        # The application entry owns the single page configuration.
        self.assertNotIn("set_page_config", self.source)


class PushPageAppTest(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        root = Path(self.tmp.name)
        self.root = root
        raw = json.loads((ROOT / "push_config.json").read_text(encoding="utf-8"))
        raw.update({"enabled": False, "paused": False, "quiet_mode": False})
        self.config_path = root / "push_config.json"
        self.config_path.write_text(
            json.dumps(raw, ensure_ascii=False), encoding="utf-8"
        )
        self._old_env = {
            key: os.environ.get(key)
            for key in ("PAPERLENS_PUSH_ROOT", "PAPERLENS_PUSH_CONFIG")
        }
        os.environ["PAPERLENS_PUSH_ROOT"] = str(root)
        os.environ["PAPERLENS_PUSH_CONFIG"] = str(self.config_path)

    def tearDown(self):
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmp.cleanup()

    def test_page_renders_without_network_or_smtp(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PAGE))
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertIn("推送设置", app.title[0].value)

    def test_page_offers_buttons_toggle_and_providers(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PAGE))
        app.run(timeout=30)

        labels = [button.label for button in app.button]
        self.assertTrue(any("预览" in label for label in labels), labels)
        self.assertTrue(any("测试邮箱登录" in label for label in labels), labels)
        self.assertTrue(any("批准首次发送" in label for label in labels), labels)

        self.assertTrue(any("启用" in toggle.label for toggle in app.toggle))

        options = [option for box in app.selectbox for option in box.options]
        self.assertIn("163", options)
        self.assertIn("QQ", options)

    def test_page_does_not_write_project_push_state(self):
        from streamlit.testing.v1 import AppTest

        project_db = ROOT / "data" / "push" / "frontier.sqlite3"
        before = project_db.stat().st_mtime_ns if project_db.exists() else None
        app = AppTest.from_file(str(PAGE))
        app.run(timeout=30)
        after = project_db.stat().st_mtime_ns if project_db.exists() else None
        self.assertEqual(before, after)

    def write_config(self, **patch) -> str:
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        raw.update(patch)
        text = json.dumps(raw, ensure_ascii=False, indent=2) + "\n"
        self.config_path.write_text(text, encoding="utf-8")
        return text

    def test_external_toggle_change_is_reconciled_without_a_write(self):
        """A keyed toggle ignores ``value=``; mirror an external edit instead."""
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PAGE))
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertFalse(app.toggle(key="push_enabled_toggle").value)

        external = self.write_config(enabled=True)
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertTrue(app.toggle(key="push_enabled_toggle").value)
        self.assertEqual(
            self.config_path.read_text(encoding="utf-8"), external
        )

        external = self.write_config(enabled=False)
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertFalse(app.toggle(key="push_enabled_toggle").value)
        self.assertEqual(
            self.config_path.read_text(encoding="utf-8"), external
        )

    def test_user_toggle_click_still_updates_both_directions(self):
        from streamlit.testing.v1 import AppTest

        (self.root / ".env").write_text(
            "MAIL_TO=user@example.com\n"
            "SMTP_USER=user@example.com\n"
            "SMTP_PASSWORD=test-app-code\n",
            encoding="utf-8",
        )
        app = AppTest.from_file(str(PAGE))
        app.run(timeout=30)
        app.toggle(key="push_enabled_toggle").set_value(True)
        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertTrue(config["enabled"])
        successes = "\n".join(item.value for item in app.success)
        self.assertIn("总开关已更新", successes)

        app.toggle(key="push_enabled_toggle").set_value(False)
        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertFalse(config["enabled"])

    def test_preview_button_forces_preview_and_never_writes_enabled_state(self):
        """The advanced preview must work while push is off or paused."""
        from streamlit.testing.v1 import AppTest

        before = self.write_config(enabled=False, paused=True)
        result = RunResult(
            run_id="preview-run",
            status="preview_ready",
            candidate_count=1,
            selected_count=1,
            sent=False,
            errors=[],
            run_dir=None,
        )

        with mock.patch(
            "mail_transport.send_message",
            side_effect=AssertionError("preview must not send SMTP"),
        ), mock.patch(
            "push_run.run_push", return_value=result
        ) as run:
            app = AppTest.from_file(str(PAGE))
            app.run(timeout=30)
            app.button(key="push_preview_button").click()
            app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        run.assert_called_once()
        self.assertIs(run.call_args.kwargs.get("force"), True)
        successes = "\n".join(item.value for item in app.success)
        self.assertIn("没有发送邮件", successes)
        self.assertEqual(
            self.config_path.read_text(encoding="utf-8"), before
        )


if __name__ == "__main__":
    unittest.main()
