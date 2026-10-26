"""Offline tests for the read-only, masked push status view (Task P8/P9)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ui_state  # noqa: E402
from push_state import PushState  # noqa: E402
from push_status import load_env, recent_runs, status  # noqa: E402

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"
ENV = {
    "DEEPSEEK_API_KEY": "sk-test-deepseek-secret",
    "DEEPSEEK_BASE_URL": "https://example.invalid",
    "DEEPSEEK_MODEL": "test-model",
    "SMTP_HOST": "smtp.163.com",
    "SMTP_PORT": "465",
    "SMTP_SECURITY": "ssl",
    "SMTP_USER": "user@163.com",
    "SMTP_PASSWORD": "test-app-code-secret",
    "MAIL_TO": "user@163.com",
}

REQUIRED_FIELDS = (
    "enabled",
    "paused",
    "quiet",
    "quiet_mode",
    "provider",
    "mail_to",
    "recipient_verified",
    "recipient_source",
    "first_send_approved",
    "auto_paused",
    "counts",
    "runs",
)


def make_root(tmp: str, **overrides) -> tuple[Path, Path]:
    root = Path(tmp)
    raw = json.loads((ROOT / "push_config.json").read_text(encoding="utf-8"))
    raw.update({"enabled": True, "paused": False, "quiet_mode": False})
    raw.update(overrides)
    config_path = root / "push_config.json"
    config_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return root, config_path


class PushStatusTests(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.root, self.config_path = make_root(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def state(self) -> PushState:
        state = PushState(self.root / "data" / "push" / "frontier.sqlite3")
        state.init_schema()
        return state

    def write_run(self, run_id: str, payload: dict) -> Path:
        run_dir = self.root / "data" / "push" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        return run_dir

    def test_status_contains_the_required_fields(self):
        report = status(self.root, self.config_path, ENV)
        for field in REQUIRED_FIELDS:
            self.assertIn(field, report)
        self.assertTrue(report["enabled"])
        self.assertEqual(report["provider"], "163")
        self.assertEqual(report["mail_to"], "u***@163.com")

    def test_status_never_exposes_secrets_or_the_full_address(self):
        rendered = json.dumps(status(self.root, self.config_path, ENV), ensure_ascii=False)
        self.assertNotIn("test-app-code-secret", rendered)
        self.assertNotIn("sk-test-deepseek-secret", rendered)
        self.assertNotIn("user@163.com", rendered)
        self.assertNotIn("SMTP_PASSWORD", rendered)
        self.assertNotIn("SMTP_USER", rendered)

    def test_status_reports_the_legacy_env_recipient_as_verified(self):
        report = status(self.root, self.config_path, ENV)
        self.assertTrue(report["recipient_verified"])
        self.assertEqual(report["recipient_source"], "env")

    def test_status_uses_a_verified_ui_recipient_and_keeps_it_masked(self):
        ui_state.set_recipient(self.root, "alice@example.com")
        ui_state.mark_verified(
            self.root, "alice@example.com", "2026-10-25T08:30:00+00:00"
        )

        report = status(self.root, self.config_path, ENV)

        self.assertEqual(report["mail_to"], "a***@example.com")
        self.assertTrue(report["recipient_verified"])
        self.assertEqual(report["recipient_source"], "ui_state")
        rendered = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("alice@example.com", rendered)
        self.assertNotIn("user@163.com", rendered)

    def test_status_reports_an_unverified_ui_recipient(self):
        ui_state.set_recipient(self.root, "alice@example.com")

        report = status(self.root, self.config_path, ENV)

        self.assertFalse(report["recipient_verified"])
        self.assertEqual(report["recipient_source"], "ui_state")
        self.assertEqual(report["mail_to"], "a***@example.com")

    def test_status_reports_auto_pause_and_ledger_counts(self):
        state = self.state()
        state.set_setting("auto_paused", "true")
        state.record_paper(
            {"arxiv_id": "2601.10001v1", "title": "T", "abstract": "A"}, "queued"
        )
        report = status(self.root, self.config_path, ENV)
        self.assertTrue(report["auto_paused"])
        self.assertEqual(report["counts"].get("queued"), 1)
        self.assertEqual(report["queue"], 1)

    def test_first_send_approval_is_visible(self):
        state = self.state()
        state.set_setting("first_send_approved_at", "2026-10-25T00:00:00+00:00")
        report = status(self.root, self.config_path, ENV)
        self.assertTrue(report["first_send_approved"])

    def test_last_success_and_runs_are_derived_from_history(self):
        self.write_run(
            "run-1",
            {
                "run_id": "run-1",
                "mode": "preview",
                "status": "preview_ready",
                "started_at": "2026-10-24T08:30:00+00:00",
                "finished_at": "2026-10-24T08:31:00+00:00",
                "counts": {"candidates": 3, "selected": 1, "sent": 0},
                "tokens": 123,
            },
        )
        report = status(self.root, self.config_path, ENV)
        self.assertEqual(report["last_success"], "2026-10-24T08:31:00+00:00")
        self.assertEqual(report["tokens"], 123)
        self.assertEqual(len(report["recent_runs"]), 1)

    def test_next_expected_is_none_when_disabled_or_paused(self):
        _, disabled = make_root(self.tmp.name, enabled=False)
        self.assertIsNone(status(self.root, disabled, ENV)["next_expected"])
        _, paused = make_root(self.tmp.name, paused=True)
        self.assertIsNone(status(self.root, paused, ENV)["next_expected"])

    def test_next_expected_uses_the_configured_local_time(self):
        report = status(
            self.root,
            self.config_path,
            ENV,
            now=datetime(2026, 10, 25, 7, 0, tzinfo=timezone.utc),
        )
        self.assertIsNotNone(report["next_expected"])
        configured = json.loads(
            (self.root / "push_config.json").read_text(encoding="utf-8")
        )["time_local"]
        self.assertIn(f"T{configured}:00", report["next_expected"])

    def test_recent_runs_redacts_secret_fields_and_addresses(self):
        self.write_run(
            "run-secret",
            {
                "run_id": "run-secret",
                "status": "failed",
                "started_at": "2026-10-24T08:30:00+00:00",
                "errors": ["boom"],
                "debug": {
                    "SMTP_PASSWORD": "test-app-code-secret",
                    "api_key": "sk-test-deepseek-secret",
                    "note": "contact user@163.com",
                },
            },
        )
        rows = recent_runs(self.root, limit=5, env=ENV)
        rendered = json.dumps(rows, ensure_ascii=False)
        self.assertEqual(len(rows), 1)
        self.assertNotIn("test-app-code-secret", rendered)
        self.assertNotIn("sk-test-deepseek-secret", rendered)
        self.assertNotIn("user@163.com", rendered)
        self.assertNotIn("SMTP_PASSWORD", rendered)
        self.assertIn("u***@163.com", rendered)

    def test_recent_runs_are_sorted_by_finish_time(self):
        self.write_run(
            "run-old",
            {
                "run_id": "run-old",
                "status": "preview_ready",
                "started_at": "2026-10-24T08:00:00+00:00",
                "finished_at": "2026-10-24T08:01:00+00:00",
            },
        )
        self.write_run(
            "run-new",
            {
                "run_id": "run-new",
                "status": "no_digest",
                "started_at": "2026-10-25T08:00:00+00:00",
                "finished_at": "2026-10-25T08:01:00+00:00",
            },
        )
        rows = recent_runs(self.root, limit=2, env=ENV)
        self.assertEqual([row["run_id"] for row in rows], ["run-new", "run-old"])

    def test_recent_runs_respects_the_limit(self):
        for index in range(3):
            self.write_run(
                f"run-{index}",
                {"run_id": f"run-{index}", "status": "preview_ready"},
            )
        self.assertEqual(len(recent_runs(self.root, limit=2, env=ENV)), 2)

    def test_load_env_reads_a_dotenv_file_without_printing_it(self):
        (self.root / ".env").write_text(
            "SMTP_PASSWORD=local-secret\nMAIL_TO=user@163.com\n", encoding="utf-8"
        )
        values = load_env(self.root)
        self.assertEqual(values["SMTP_PASSWORD"], "local-secret")
        self.assertEqual(values["MAIL_TO"], "user@163.com")


if __name__ == "__main__":
    unittest.main()
