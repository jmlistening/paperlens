"""Contract tests for the moved PaperLens overview dashboard (advanced mode).

The overview keeps its status, scheduler controls, and evaluation tables but
no longer owns an auto-push toggle: enabling or disabling push lives on the
simple default page. Rendering only reads the temporary configuration below.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAGE = ROOT / "advanced" / "overview.py"
TMP_ROOT = Path(__file__).resolve().parent / ".tmp"


class DashboardSourceTests(unittest.TestCase):
    def test_page_exposes_status_and_scheduler_controls(self):
        source = PAGE.read_text(encoding="utf-8")
        for text in (
            "自动推送",
            "安装每日计划任务",
            "移除计划任务",
            "60 题检索评测",
        ):
            self.assertIn(text, source)
        self.assertNotIn("send_message", source)
        self.assertNotIn("smtplib", source)

    def test_page_has_no_second_auto_push_enable_path(self):
        source = PAGE.read_text(encoding="utf-8")
        for forbidden in (
            "set_auto_push",
            "set_enabled",
            "dash_auto_push_toggle",
            "应用推送开关",
        ):
            self.assertNotIn(forbidden, source)

    def test_page_does_not_set_page_config(self):
        self.assertNotIn(
            "set_page_config", PAGE.read_text(encoding="utf-8")
        )


class DashboardAppTest(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        root = Path(self.tmp.name)
        config = json.loads((ROOT / "push_config.json").read_text(encoding="utf-8"))
        config.update({"enabled": False, "paused": False, "quiet_mode": False})
        self.config_path = root / "push_config.json"
        self.config_path.write_text(
            json.dumps(config, ensure_ascii=False), encoding="utf-8"
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

    def test_dashboard_renders_without_sending_or_writing_config(self):
        from streamlit.testing.v1 import AppTest

        before = self.config_path.read_bytes()
        app = AppTest.from_file(str(PAGE))
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertEqual([toggle.label for toggle in app.toggle], [])
        buttons = [button.label for button in app.button]
        self.assertTrue(any("安装每日计划任务" in label for label in buttons), buttons)
        self.assertTrue(any("移除计划任务" in label for label in buttons), buttons)
        self.assertNotIn("应用推送开关", buttons)
        self.assertEqual(self.config_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
