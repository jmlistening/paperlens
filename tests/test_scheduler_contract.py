"""Contract tests for the Windows Task Scheduler scripts (Task P9).

These tests only read the checked-in scripts. They never register, start, or
remove a real scheduled task and never touch the network.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RUN_PUSH = SCRIPTS / "run_push.bat"
REGISTER = SCRIPTS / "register_push_task.ps1"
UNREGISTER = SCRIPTS / "unregister_push_task.ps1"
TMP_ROOT = Path(__file__).resolve().parent / ".tmp"

TASK_NAME = "PaperLens Frontier Push"


def read_script(path: Path) -> str:
    if not path.is_file():
        raise AssertionError(f"missing script: {path}")
    return path.read_text(encoding="utf-8-sig")


class RunPushBatContractTests(unittest.TestCase):
    def setUp(self):
        self.text = read_script(RUN_PUSH)

    def test_uses_a_configurable_interpreter(self):
        self.assertIn("PAPERLENS_PYTHON", self.text)
        self.assertIn('set "PAPERLENS_PYTHON=python"', self.text)

    def test_runs_push_run_scheduled_entrypoint(self):
        self.assertIn("push_run.py", self.text)
        self.assertIn("--scheduled", self.text)

    def test_changes_into_the_checkout_that_contains_the_batch(self):
        # A task registered from a worktree must run that worktree, not the
        # main checkout. The project directory is derived, never hardcoded.
        self.assertIn("%~dp0", self.text)
        self.assertIn("cd /d", self.text.lower())
        self.assertNotIn("D:/main-checkout", self.text)

    def test_propagates_the_python_exit_code(self):
        lowered = self.text.lower()
        self.assertTrue(
            "exit /b %errorlevel%" in lowered or "if errorlevel 1" in lowered,
            "run_push.bat must propagate a non-zero exit code to Task Scheduler",
        )

    def test_batch_file_contains_no_credentials(self):
        lowered = self.text.lower()
        self.assertNotIn("password", lowered)
        self.assertNotIn("smtp_password", lowered)
        self.assertNotIn("authorization code", lowered)

    def test_batch_prints_readable_progress_and_holds_the_window(self):
        for text in (
            "PAPERLENS_PROGRESS",
            "正在启动定时推送",
            "正在读取配置、检查邮箱并抓取论文",
            "运行结束",
            "timeout /t 6",
        ):
            with self.subTest(text=text):
                self.assertIn(text, self.text)


class RunPushBatResolutionTests(unittest.TestCase):
    """The batch must resolve the checkout it lives in, not a fixed path."""

    def setUp(self):
        if os.name != "nt":
            self.skipTest("Windows batch resolution test")
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.root = Path(self.tmp.name)
        (self.root / "scripts").mkdir()
        shutil.copyfile(RUN_PUSH, self.root / "scripts" / "run_push.bat")
        (self.root / "push_run.py").write_text(
            "import json, pathlib\n"
            "print(json.dumps({'project': str("
            "pathlib.Path(__file__).resolve().parent)}))\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_batch_runs_the_checkout_it_lives_in(self):
        env = dict(os.environ)
        env["PAPERLENS_NO_HOLD"] = "1"
        env["PAPERLENS_PYTHON"] = sys.executable
        completed = subprocess.run(
            ["cmd.exe", "/c", str(self.root / "scripts" / "run_push.bat")],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = None
        for line in reversed(completed.stdout.strip().splitlines()):
            if line.lstrip().startswith("{"):
                payload = json.loads(line)
                break
        self.assertIsNotNone(payload, completed.stdout)
        self.assertEqual(
            Path(payload["project"]).resolve(), self.root.resolve()
        )


class RegisterTaskContractTests(unittest.TestCase):
    def setUp(self):
        self.text = read_script(REGISTER)
        self.lowered = self.text.lower()

    def test_default_project_directory_is_derived_from_the_script(self):
        self.assertIn("$PSScriptRoot", self.text)
        self.assertNotIn("[string]$ProjectDir = 'D:/main-checkout'", self.text)

    def test_registers_the_expected_task_name(self):
        self.assertIn(TASK_NAME, self.text)

    def test_calls_the_run_push_batch_file(self):
        self.assertIn("run_push.bat", self.text)
        self.assertIn("Register-ScheduledTask", self.text)

    def test_uses_start_when_available(self):
        self.assertIn("StartWhenAvailable", self.text)

    def test_ignores_new_instances(self):
        self.assertIn("IgnoreNew", self.text)

    def test_restarts_twice_after_ten_minutes(self):
        self.assertIn("RestartCount 2", self.text)
        self.assertIn("RestartInterval", self.text)
        self.assertIn("New-TimeSpan -Minutes 10", self.text)

    def test_has_a_two_hour_execution_limit(self):
        self.assertIn("ExecutionTimeLimit", self.text)
        self.assertIn("New-TimeSpan -Hours 2", self.text)

    def test_supports_what_if(self):
        self.assertIn("SupportsShouldProcess", self.text)
        self.assertIn("ShouldProcess", self.text)
        self.assertIn("-WhatIf", self.text)

    def test_never_stores_or_requests_a_credential(self):
        self.assertNotIn("password", self.lowered)
        self.assertNotIn("smtp_password", self.lowered)
        self.assertNotIn("ConvertTo-SecureString", self.text)
        self.assertNotIn("-PSCredential", self.text)

    def test_uses_an_interactive_current_user_principal(self):
        self.assertIn("New-ScheduledTaskPrincipal", self.text)
        self.assertIn("Interactive", self.text)


class UnregisterTaskContractTests(unittest.TestCase):
    def setUp(self):
        self.text = read_script(UNREGISTER)
        self.lowered = self.text.lower()

    def test_removes_the_expected_task_name(self):
        self.assertIn(TASK_NAME, self.text)
        self.assertIn("Unregister-ScheduledTask", self.text)

    def test_supports_what_if(self):
        self.assertIn("SupportsShouldProcess", self.text)
        self.assertIn("ShouldProcess", self.text)
        self.assertIn("-WhatIf", self.text)

    def test_is_safe_when_the_task_is_absent(self):
        self.assertIn("Get-ScheduledTask", self.text)
        self.assertIn("SilentlyContinue", self.text)

    def test_contains_no_credentials(self):
        self.assertNotIn("password", self.lowered)
        self.assertNotIn("smtp_password", self.lowered)


if __name__ == "__main__":
    unittest.main()
