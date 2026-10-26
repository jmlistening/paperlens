"""M0 repository hygiene checks. No network, no paid API, no .env reads."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profile_config import validate_profile  # noqa: E402

SCAN_SCRIPT = ROOT / "scripts" / "check_public_repo.py"
ACTIVE_DOCS = (
    "README.md",
    "docs/PUSH.md",
)
OLD_PAGE_PATHS = (
    "pages/0_总览.py",
    "pages/1_论文发现.py",
    "pages/2_论文库.py",
    "pages/3_推送设置.py",
    "pages/4_兴趣配置.py",
)

# Private/local paths that must never be published again.
IGNORED_PRIVATE_PATHS = (
    ".env",
    "data/",
    "interests.json",
    "data/push/profile.json",
    "docs/handoff/ACCEPTANCE_REPORT.md",
    "docs/stage1a/",
    "eval/results/",
    "eval/reviews/",
    "paperlens_answer_notes.md",
    "review_notes/",
    "PROJECT_NOTES.md",
    "PROJECT_HANDOFF_NOTES.md",
    "dataset_targets.json",
    # Old Chinese handoff/acceptance notes must stay local.
    "使用说明_Agent.md",
    "验收记录_Agent.md",
    "接续记录.md",
    "README.txt",
)

OLD_TRACKED_PRIVATE_FILES = (
    "interests.json",
    "docs/handoff/ACCEPTANCE_REPORT.md",
    "docs/handoff/CURRENT_STATUS.md",
    "docs/handoff/NEXT_STAGE_PLAN.md",
    "docs/handoff/PROJECT_ROADMAP.md",
    "data/push/profile.json",
)


def load_scan_module():
    spec = importlib.util.spec_from_file_location(
        "check_public_repo", SCAN_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True
    )


class RepoHygieneTests(unittest.TestCase):
    def test_required_m0_files_exist(self):
        for relative in (
            ".gitignore",
            ".env.example",
            "requirements.txt",
            "README.md",
            "eval/CLAIMS.md",
            "eval/snapshot_baseline.py",
            "scripts/check_public_repo.py",
            "tests/__init__.py",
            "interests.example.json",
        ):
            with self.subTest(relative=relative):
                self.assertTrue((ROOT / relative).is_file(), relative)

    def test_git_ignores_secrets_and_data(self):
        if not (ROOT / ".git").exists():
            self.skipTest("git not initialized yet")
        for path in (".env", "data/"):
            with self.subTest(path=path):
                result = subprocess.run(
                    ["git", "check-ignore", "-q", path],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, f"{path} is not ignored")

    def test_git_ignores_private_profile_and_handoff(self):
        if not (ROOT / ".git").exists():
            self.skipTest("git not initialized yet")
        for path in IGNORED_PRIVATE_PATHS:
            with self.subTest(path=path):
                result = git("check-ignore", "-q", path)
                self.assertEqual(result.returncode, 0, f"{path} is not ignored")

    def test_private_profile_and_handoff_are_not_tracked(self):
        if not (ROOT / ".git").exists():
            self.skipTest("git not initialized yet")
        result = git("ls-files")
        self.assertEqual(result.returncode, 0, result.stderr)
        tracked = set(result.stdout.splitlines())
        for relative in OLD_TRACKED_PRIVATE_FILES:
            with self.subTest(relative=relative):
                self.assertNotIn(relative, tracked)

    def test_public_example_profile_validates_without_email(self):
        text = (ROOT / "interests.example.json").read_text(encoding="utf-8")
        validated = validate_profile(json.loads(text))
        self.assertGreaterEqual(len(validated["topics"]), 1)
        self.assertNotIn("@", text)

    def test_env_example_has_placeholders_not_secrets(self):
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("SMTP_PASSWORD=your_app_authorization_code", text)
        self.assertIn("DEEPSEEK_API_KEY=your_deepseek_api_key", text)
        self.assertNotIn("sk-", text)
        self.assertNotIn("smtp.163.com\r\nSMTP_PASSWORD=", text)

    def test_requirements_cover_runtime(self):
        text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        for package in (
            "streamlit==",
            "sentence-transformers==",
            "faiss-cpu==",
            "rank-bm25==",
            "PyMuPDF==",
        ):
            with self.subTest(package=package):
                self.assertIn(package, text)

    def test_snapshot_script_is_read_only(self):
        text = (ROOT / "eval/snapshot_baseline.py").read_text(encoding="utf-8")
        for forbidden in ("import run_qa", "from run_qa", "import run_agent", "from run_agent"):
            self.assertNotIn(forbidden, text)
        self.assertIn("mode=ro", text)

    def test_public_scan_is_clean_against_the_tracked_tree(self):
        result = subprocess.run(
            [sys.executable, str(SCAN_SCRIPT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("OK", result.stdout)

    def test_public_scan_detects_and_redacts_secrets(self):
        module = load_scan_module()
        # Build the fixture pieces separately so this test file stays clean
        # under the very scan it exercises.
        email = "person" + "@" + "some-company.cn"
        api_key = "sk-" + "0123456789abcdef" * 2
        smtp_password = "SMTP_PASSWORD" + "=" + "abcd1234efgh5678"
        text = f"contact {email}\nDEEPSEEK_API_KEY={api_key}\n{smtp_password}\n"
        findings = module.scan_text(text, "fixture.txt")
        kinds = {finding.kind for finding in findings}
        self.assertEqual(
            kinds, {"email", "api-key", "smtp-password"}
        )
        rendered = "\n".join(finding.render() for finding in findings)
        self.assertNotIn(email, rendered)
        self.assertNotIn("0123456789abcdef", rendered)
        self.assertNotIn("abcd1234efgh5678", rendered)
        self.assertIn("fixture.txt", rendered)

    def test_public_scan_reports_email_before_sentence_period(self):
        module = load_scan_module()
        # Build the fixture pieces separately so this test file stays clean
        # under the very scan it exercises.
        address = "reviewer" + "@" + "mail.corp"
        findings = module.scan_text("联系 " + address + ".\n", "fixture.txt")
        self.assertEqual([finding.kind for finding in findings], ["email"])
        # The sentence period must not be swallowed into the reported address.
        self.assertEqual(
            findings[0].detail, "r***@mail.corp (real-looking address)"
        )
        self.assertNotIn(address, findings[0].render())

    def test_public_scan_reports_quoted_smtp_passwords(self):
        module = load_scan_module()
        double_secret = "abcd1234efgh5678"
        single_secret = "wxyz9876mnop5432"
        # Spaces around "=" are allowed; both quote styles must be unwrapped.
        double_quoted = (
            "SMTP_PASSWORD" + " = " + '"' + double_secret + '"'
        )
        single_quoted = (
            "SMTP_PASSWORD" + "=" + "'" + single_secret + "'"
        )
        text = double_quoted + "\n" + single_quoted + "\n"
        findings = module.scan_text(text, "fixture.txt")
        self.assertEqual(
            [finding.kind for finding in findings],
            ["smtp-password", "smtp-password"],
        )
        rendered = "\n".join(finding.render() for finding in findings)
        self.assertNotIn(double_secret, rendered)
        self.assertNotIn(single_secret, rendered)

    def test_public_scan_still_reports_unquoted_smtp_passwords(self):
        module = load_scan_module()
        secret = "abcd1234efgh5678"
        text = "SMTP_PASSWORD" + "=" + secret + "\n"
        findings = module.scan_text(text, "fixture.txt")
        self.assertEqual([finding.kind for finding in findings], ["smtp-password"])
        self.assertNotIn(secret, findings[0].render())

    def test_public_scan_accepts_quoted_smtp_placeholders(self):
        module = load_scan_module()
        text = (
            "SMTP_PASSWORD" + '="' + "your_app_authorization_code" + '"' + "\n"
            "SMTP_PASSWORD" + "='" + "your_app_password" + "'" + "\n"
        )
        self.assertEqual(module.scan_text(text, "fixture.txt"), [])

    def test_public_scan_accepts_documented_placeholders(self):
        module = load_scan_module()
        text = (
            "SMTP_USER=your_account@163.com\n"
            "MAIL_TO=you@example.com\n"
            "SMTP_PASSWORD=your_app_authorization_code\n"
            "assertNotIn(\"SMTP_PASSWORD=\", source)\n"
            "token = 'sk-test-deepseek-secret'\n"
        )
        self.assertEqual(module.scan_text(text, "fixture.txt"), [])

    def test_public_scan_ignores_python_decorator_lines(self):
        module = load_scan_module()
        # A diff line can start with "+@decorator"; the "+" must not be
        # mistaken for an email local part. Both forms are Python decorators,
        # not addresses.
        text = (
            "@unittest.skipUnless(sys.platform == 'win32', 'Windows-only')\n"
            "+@unittest.skipUnless(sys.platform == 'win32', 'Windows-only')\n"
        )
        self.assertEqual(module.scan_text(text, "fixture.diff"), [])

    def test_public_scan_history_parser_reports_paths_without_content(self):
        module = load_scan_module()
        output = (
            "COMMIT\t2c8a835\tchore: freeze\n"
            "A\tinterests.json\n"
            "A\tdocs/handoff/CURRENT_STATUS.md\n"
            "\n"
            "COMMIT\tc82bb88\tchore: privatize\n"
            "D\tinterests.json\n"
        )
        findings = module.parse_history_name_status(output)
        rendered = "\n".join(finding.render() for finding in findings)
        self.assertIn("2c8a835", rendered)
        self.assertIn("c82bb88", rendered)
        self.assertIn("interests.json", rendered)
        self.assertIn("docs/handoff/CURRENT_STATUS.md", rendered)
        self.assertEqual(
            len(findings), 3
        )

    def test_active_docs_do_not_reference_removed_pages(self):
        for relative in ACTIVE_DOCS:
            text = (ROOT / relative).read_text(encoding="utf-8")
            for old_path in OLD_PAGE_PATHS:
                with self.subTest(doc=relative, path=old_path):
                    self.assertNotIn(old_path, text)
            with self.subTest(doc=relative):
                self.assertNotIn("手工编辑 `interests.json`", text)

    def test_active_docs_describe_the_simple_push_flow(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        push = (ROOT / "docs/PUSH.md").read_text(encoding="utf-8")
        for text in (readme, push):
            for phrase in ("每日推送", "测试邮件", "高级", "手动"):
                with self.subTest(phrase=phrase):
                    self.assertIn(phrase, text)
        self.assertIn("scripts/check_public_repo.py", readme)
        self.assertIn("interests.example.json", push)
        self.assertIn("data/push/profile.json", push)

    def test_stale_root_library_page_is_not_tracked(self):
        result = git("ls-files", "--", "2_论文库.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "")

    def test_advanced_pages_insert_the_project_root_for_direct_runs(self):
        for page in sorted((ROOT / "advanced").glob("*.py")):
            with self.subTest(page=page.name):
                source = page.read_text(encoding="utf-8")
                self.assertIn("sys.path.insert", source)
                self.assertIn(
                    "Path(__file__).resolve().parents[1]", source
                )


if __name__ == "__main__":
    unittest.main()
