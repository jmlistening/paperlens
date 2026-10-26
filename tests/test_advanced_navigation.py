"""Contract tests for the Task 9 advanced navigation and page migration.

The application entry owns the only ``st.set_page_config`` call: the simple
daily-push page is the default destination and the advanced research pages are
hidden pages reached through one advanced home page. Every render below is
offline: temporary roots and disabled configurations only.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENTRY = ROOT / "app.py"
HOME = ROOT / "advanced" / "home.py"
OVERVIEW = ROOT / "advanced" / "overview.py"
EVIDENCE_QA = ROOT / "advanced" / "evidence_qa.py"
LIBRARY = ROOT / "advanced" / "library.py"
DISCOVERY = ROOT / "advanced" / "discovery.py"
PUSH_SETTINGS = ROOT / "advanced" / "push_settings.py"
INTERESTS = ROOT / "advanced" / "interests.py"
HISTORY = ROOT / "advanced" / "history.py"
ADVANCED_DIR = ROOT / "advanced"
TMP_ROOT = Path(__file__).resolve().parent / ".tmp"

ADVANCED_PAGES = (
    HOME,
    OVERVIEW,
    EVIDENCE_QA,
    LIBRARY,
    DISCOVERY,
    PUSH_SETTINGS,
    INTERESTS,
    HISTORY,
)

HOME_LINK_TARGETS = (
    "advanced/overview.py",
    "advanced/evidence_qa.py",
    "advanced/push_settings.py",
    "advanced/interests.py",
    "advanced/library.py",
    "advanced/discovery.py",
    "advanced/history.py",
)

HOME_LINK_LABELS = (
    "总览与评测",
    "证据问答",
    "推送设置",
    "精确兴趣组",
    "论文库",
    "论文发现",
    "推送历史",
)

OLD_PAGES = (
    "pages/0_总览.py",
    "pages/1_论文发现.py",
    "pages/2_论文库.py",
    "pages/3_推送设置.py",
    "pages/4_兴趣配置.py",
)


class NavigationSourceContractTests(unittest.TestCase):
    def test_entry_configures_the_page_once_and_registers_default_and_home(self):
        source = ENTRY.read_text(encoding="utf-8")
        self.assertEqual(source.count("set_page_config"), 1)
        self.assertIn("st.navigation", source)
        self.assertIn('position="sidebar"', source)
        self.assertIn('st.Page("simple_page.py"', source)
        self.assertIn("default=True", source)
        self.assertIn('st.Page("advanced/home.py"', source)
        self.assertIn("每日推送", source)
        self.assertIn("高级 / 研究模式", source)

    def test_entry_registers_every_advanced_page_as_hidden(self):
        source = ENTRY.read_text(encoding="utf-8")
        for target in HOME_LINK_TARGETS:
            with self.subTest(target=target):
                self.assertIn(f'st.Page("{target}"', source)
        self.assertGreaterEqual(source.count('visibility="hidden"'), 7)

    def test_advanced_pages_do_not_set_page_config(self):
        for page in ADVANCED_PAGES:
            with self.subTest(page=page.name):
                self.assertTrue(page.is_file(), page)
                self.assertNotIn(
                    "set_page_config", page.read_text(encoding="utf-8")
                )

    def test_home_links_every_advanced_page(self):
        source = HOME.read_text(encoding="utf-8")
        for target, label in zip(HOME_LINK_TARGETS, HOME_LINK_LABELS):
            with self.subTest(target=target):
                self.assertIn(f'st.page_link("{target}"', source)
                self.assertIn(label, source)
        self.assertIn('st.page_link("simple_page.py"', source)

    def test_overview_has_no_independent_auto_push_toggle(self):
        source = OVERVIEW.read_text(encoding="utf-8")
        for forbidden in (
            "set_auto_push",
            "dash_auto_push_toggle",
            "应用推送开关",
            "st.toggle",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("自动推送", source)

    def test_old_pages_have_moved_out_of_the_pages_directory(self):
        for relative in OLD_PAGES:
            with self.subTest(relative=relative):
                self.assertFalse((ROOT / relative).exists(), relative)

    def test_evidence_qa_resolves_the_project_root(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "advanced_evidence_qa_contract", EVIDENCE_QA
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.ROOT, ROOT)
        self.assertEqual(module.RAG, ROOT / "data" / "rag")

    def test_moved_pages_calculate_paths_from_the_project_root(self):
        for page in (OVERVIEW, DISCOVERY, LIBRARY, PUSH_SETTINGS, INTERESTS):
            with self.subTest(page=page.name):
                source = page.read_text(encoding="utf-8")
                self.assertIn("Path(__file__).resolve().parents[1]", source)


class _TempPushRootMixin:
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.root = Path(self.tmp.name)
        config = json.loads(
            (ROOT / "push_config.json").read_text(encoding="utf-8")
        )
        config.update({"enabled": False, "paused": False, "quiet_mode": False})
        self.config_path = self.root / "push_config.json"
        self.config_path.write_text(
            json.dumps(config, ensure_ascii=False), encoding="utf-8"
        )
        self.profile_path = self.root / "interests.json"
        self.profile_path.write_text(
            (ROOT / "interests.example.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        self._old_env = {
            key: os.environ.get(key)
            for key in (
                "PAPERLENS_PUSH_ROOT",
                "PAPERLENS_PUSH_CONFIG",
                "PAPERLENS_PROFILE_PATH",
            )
        }
        os.environ["PAPERLENS_PUSH_ROOT"] = str(self.root)
        os.environ["PAPERLENS_PUSH_CONFIG"] = str(self.config_path)
        os.environ["PAPERLENS_PROFILE_PATH"] = str(self.profile_path)

    def tearDown(self):
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmp.cleanup()


class NavigationAppTests(_TempPushRootMixin, unittest.TestCase):
    def run_entry(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ENTRY))
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)
        return app

    def test_simple_daily_push_page_is_the_default(self):
        app = self.run_entry()
        self.assertIn("PaperLens", app.title[0].value)
        self.assertEqual(
            [slider.label for slider in app.slider], ["每次推送几篇"]
        )
        self.assertEqual(
            [toggle.label for toggle in app.toggle], ["每天自动推送"]
        )
        labels = [link.label for link in app.page_link]
        self.assertIn("高级 / 研究模式", labels)

    def test_advanced_home_links_all_hidden_pages(self):
        app = self.run_entry()
        app.switch_page("advanced/home.py")
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertIn("高级 / 研究模式", app.title[0].value)
        labels = [link.label for link in app.page_link]
        for label in HOME_LINK_LABELS:
            self.assertIn(label, labels)
        self.assertIn("返回每日推送", labels)


class AdvancedPageRenderTests(_TempPushRootMixin, unittest.TestCase):
    def run_page(self, page: Path):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(page))
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0, app.exception)
        return app

    def test_overview_renders_without_a_second_push_toggle(self):
        before = self.config_path.read_bytes()
        app = self.run_page(OVERVIEW)
        self.assertEqual([toggle.label for toggle in app.toggle], [])
        buttons = [button.label for button in app.button]
        self.assertIn("安装每日计划任务", buttons)
        self.assertNotIn("应用推送开关", buttons)
        self.assertEqual(self.config_path.read_bytes(), before)

    def test_evidence_qa_renders_read_only(self):
        app = self.run_page(EVIDENCE_QA)
        self.assertIn("PaperLens", app.title[0].value)
        self.assertTrue(
            any("论文问答" in header.value for header in app.sidebar.header),
            [header.value for header in app.sidebar.header],
        )

    def test_library_renders_without_running_ingest(self):
        with mock.patch(
            "subprocess.Popen", side_effect=AssertionError("no ingest")
        ) as popen:
            app = self.run_page(LIBRARY)
        self.assertFalse(popen.called)
        self.assertIn("本地论文库", app.title[0].value)

    def test_discovery_renders_without_touching_the_project_database(self):
        from streamlit.testing.v1 import AppTest

        project_db = ROOT / "data" / "reading.sqlite3"
        before = project_db.stat().st_mtime_ns if project_db.exists() else None
        temp_db = Path(self.tmp.name) / "reading.sqlite3"
        real_connect = sqlite3.connect

        def connect_to_temp(path, *args, **kwargs):
            return real_connect(temp_db, *args, **kwargs)

        with mock.patch("sqlite3.connect", side_effect=connect_to_temp):
            app = AppTest.from_file(str(DISCOVERY))
            app.run(timeout=30)

        self.assertEqual(len(app.exception), 0, app.exception)
        self.assertIn("论文发现与阅读管理", app.title[0].value)
        after = project_db.stat().st_mtime_ns if project_db.exists() else None
        self.assertEqual(after, before)
        self.assertTrue(temp_db.exists())


if __name__ == "__main__":
    unittest.main()
