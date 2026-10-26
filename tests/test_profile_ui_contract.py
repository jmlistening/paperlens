"""Contract tests for the personalized interest-configuration page."""
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

PAGE = ROOT / "advanced" / "interests.py"
TMP_ROOT = Path(__file__).resolve().parent / ".tmp"


class ProfilePageSourceTests(unittest.TestCase):
    def test_page_contains_personalization_controls_and_no_hidden_topics(self):
        source = PAGE.read_text(encoding="utf-8")
        for text in (
            "兴趣配置",
            "查询组",
            "候选池",
            "保存兴趣配置",
            "不会联网",
            "收藏/已读/忽略",
        ):
            self.assertIn(text, source)
        self.assertNotIn("smtplib", source)
        self.assertNotIn("send_message", source)
        self.assertNotIn("set_page_config", source)


class ProfilePageAppTest(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        root = Path(self.tmp.name)
        profile = json.loads(
            (ROOT / "interests.example.json").read_text(encoding="utf-8")
        )
        config = json.loads((ROOT / "push_config.json").read_text(encoding="utf-8"))
        config.update({"enabled": False, "paused": False, "quiet_mode": False})
        self.profile_path = root / "interests.json"
        self.config_path = root / "push_config.json"
        self.profile_path.write_text(
            json.dumps(profile, ensure_ascii=False), encoding="utf-8"
        )
        self.config_path.write_text(
            json.dumps(config, ensure_ascii=False), encoding="utf-8"
        )
        self._old_env = {
            key: os.environ.get(key)
            for key in (
                "PAPERLENS_PUSH_ROOT",
                "PAPERLENS_PROFILE_PATH",
                "PAPERLENS_PUSH_CONFIG",
            )
        }
        os.environ["PAPERLENS_PUSH_ROOT"] = str(root)
        os.environ["PAPERLENS_PROFILE_PATH"] = str(self.profile_path)
        os.environ["PAPERLENS_PUSH_CONFIG"] = str(self.config_path)

    def tearDown(self):
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmp.cleanup()

    def test_page_renders_without_writing_project_files(self):
        from streamlit.testing.v1 import AppTest

        before = (ROOT / "interests.example.json").stat().st_mtime_ns
        app = AppTest.from_file(str(PAGE))
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertIn("兴趣配置", app.title[0].value)
        after = (ROOT / "interests.example.json").stat().st_mtime_ns
        self.assertEqual(before, after)

    def test_page_shows_query_preview_and_controls(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PAGE))
        app.run(timeout=30)
        labels = [button.label for button in app.button]
        self.assertTrue(any("保存兴趣配置" in label for label in labels), labels)
        self.assertTrue(
            any("候选池" in caption.value for caption in app.caption),
            [caption.value for caption in app.caption],
        )

    def test_form_submit_writes_active_profile_and_keeps_example(self):
        from streamlit.testing.v1 import AppTest

        root = Path(self.tmp.name) / "public"
        root.mkdir()
        example = root / "interests.example.json"
        example.write_bytes((ROOT / "interests.example.json").read_bytes())
        original = example.read_bytes()
        (root / "push_config.json").write_text(
            (ROOT / "push_config.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        os.environ["PAPERLENS_PUSH_ROOT"] = str(root)
        os.environ.pop("PAPERLENS_PROFILE_PATH", None)
        os.environ.pop("PAPERLENS_PUSH_CONFIG", None)

        app = AppTest.from_file(str(PAGE))
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)
        submit = next(
            button for button in app.button if "保存兴趣配置" in button.label
        )
        submit.click()
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)

        active = root / "data" / "push" / "profile.json"
        self.assertTrue(active.is_file())
        self.assertFalse((root / "interests.json").exists())
        self.assertEqual(example.read_bytes(), original)
        saved = json.loads(active.read_text(encoding="utf-8"))
        self.assertEqual(saved["search"]["groups"][0]["name"], "Example")
        self.assertTrue(app.success)


if __name__ == "__main__":
    unittest.main()
