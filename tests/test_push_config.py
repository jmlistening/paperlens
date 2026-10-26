"""Offline tests for push configuration and environment validation (Task P1).

No network, no SMTP, no paid API, and no reads of the real dotenv file. All
credentials in these tests are synthetic fixtures.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from push_config import (  # noqa: E402  (import after sys.path setup)
    DEFAULT_CONFIG_PATH,
    SMTP_PRESETS,
    PushConfig,
    PushConfigError,
    SmtpPreset,
    load_push_config,
    validate_push_config,
)

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"

VALID_163_ENV = {
    "SMTP_USER": "fixture.user@163.com",
    "SMTP_PASSWORD": "fixture-auth-code",
    "MAIL_TO": "fixture.user@163.com",
}


class PushConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self._tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT, prefix="push_config_")
        self.addCleanup(self._tmp.cleanup)
        self.tmp_dir = Path(self._tmp.name)

    def write_config(self, data: object, name: str = "push_config.json") -> Path:
        path = self.tmp_dir / name
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def default_data(self) -> dict:
        return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


class DefaultConfigTests(PushConfigTestCase):
    def test_default_config_is_disabled(self):
        # ``push_config.json`` is also the live settings file, so its
        # enabled/time values are user state; the dataclass keeps the defaults.
        self.assertIs(PushConfig().enabled, False)

    def test_default_file_matches_spec_7_1(self):
        config = load_push_config(DEFAULT_CONFIG_PATH)
        # enabled and time_local are user-editable live settings and are
        # intentionally not asserted here.
        self.assertEqual(config.provider, "163")
        self.assertEqual(config.frequency, "daily")
        self.assertEqual(config.window_hours, 72)
        self.assertEqual(config.overlap_hours, 24)
        self.assertEqual(config.catchup_days, 7)
        self.assertEqual(config.fetch_pool_limit, 40)
        self.assertEqual(config.rank_max_candidates, 20)
        self.assertEqual(config.max_per_digest, 3)
        self.assertEqual(config.min_new_papers, 1)
        self.assertEqual(config.auto_pause_after_failures, 3)
        self.assertIs(config.attach_pdfs, False)
        self.assertIs(config.paused, False)
        self.assertIs(config.quiet_mode, False)
        self.assertIsNone(config.search_categories)

    def test_default_config_path_points_at_repo_file(self):
        self.assertEqual(Path(DEFAULT_CONFIG_PATH).resolve(), ROOT / "push_config.json")
        self.assertTrue(Path(DEFAULT_CONFIG_PATH).is_file())

    def test_config_is_frozen(self):
        config = load_push_config()
        with self.assertRaises(FrozenInstanceError):
            config.enabled = True  # type: ignore[misc]

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_push_config(self.tmp_dir / "absent.json")

    def test_invalid_json_raises_config_error(self):
        path = self.tmp_dir / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(PushConfigError):
            load_push_config(path)

    def test_non_object_json_raises_config_error(self):
        path = self.write_config(["enabled", False])
        with self.assertRaises(PushConfigError):
            load_push_config(path)


class LoadConfigTests(PushConfigTestCase):
    def test_load_reads_json_without_mutating_it(self):
        data = self.default_data()
        data["enabled"] = True
        data["search_categories"] = ["cs.CL", "cs.AI"]
        path = self.write_config(data)
        before = path.read_bytes()

        first = load_push_config(path)
        second = load_push_config(path)

        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(first, second)
        self.assertEqual(first.search_categories, ("cs.CL", "cs.AI"))
        self.assertIsInstance(first.search_categories, tuple)

    def test_unknown_key_is_rejected_without_echoing_value(self):
        data = self.default_data()
        data["mystery_option"] = "do-not-echo-this-value"
        path = self.write_config(data)
        with self.assertRaises(PushConfigError) as ctx:
            load_push_config(path)
        message = str(ctx.exception)
        self.assertIn("mystery_option", message)
        self.assertNotIn("do-not-echo-this-value", message)

    def test_literal_password_key_is_rejected(self):
        data = self.default_data()
        data["password"] = "hunter2-should-never-be-echoed"
        path = self.write_config(data)
        with self.assertRaises(PushConfigError) as ctx:
            load_push_config(path)
        message = str(ctx.exception)
        self.assertIn("password", message)
        self.assertNotIn("hunter2-should-never-be-echoed", message)

    def test_nested_secret_key_is_rejected(self):
        data = self.default_data()
        data["smtp"] = {"password": "nested-secret-value"}
        path = self.write_config(data)
        with self.assertRaises(PushConfigError) as ctx:
            load_push_config(path)
        message = str(ctx.exception)
        self.assertIn("smtp.password", message)
        self.assertNotIn("nested-secret-value", message)

    def test_unknown_provider_is_rejected(self):
        data = self.default_data()
        data["provider"] = "protonmail"
        path = self.write_config(data)
        with self.assertRaises(PushConfigError) as ctx:
            load_push_config(path)
        self.assertIn("protonmail", str(ctx.exception))

    def test_wrong_types_are_rejected(self):
        cases = {
            "enabled": "false",
            "window_hours": "72",
            "max_per_digest": True,
            "search_categories": "cs.CL",
        }
        for key, value in cases.items():
            with self.subTest(key=key):
                data = self.default_data()
                data[key] = value
                with self.assertRaises(PushConfigError):
                    load_push_config(self.write_config(data, f"{key}.json"))

    def test_empty_search_categories_entry_is_rejected(self):
        data = self.default_data()
        data["search_categories"] = ["cs.CL", " "]
        with self.assertRaises(PushConfigError):
            load_push_config(self.write_config(data))


class SmtpPresetTests(unittest.TestCase):
    def test_163_preset(self):
        preset = SMTP_PRESETS["163"]
        self.assertIsInstance(preset, SmtpPreset)
        self.assertEqual(preset.host, "smtp.163.com")
        self.assertEqual(preset.port, 465)
        self.assertEqual(preset.security, "ssl")

    def test_qq_preset(self):
        preset = SMTP_PRESETS["qq"]
        self.assertEqual(preset.host, "smtp.qq.com")
        self.assertEqual(preset.port, 465)
        self.assertEqual(preset.security, "ssl")

    def test_preset_is_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            SMTP_PRESETS["163"].port = 587  # type: ignore[misc]


class ValidateStructuralTests(PushConfigTestCase):
    def test_disabled_default_config_has_no_problems(self):
        # The tracked file is also the live settings file; force the
        # disabled state this test is about.
        self.assertEqual(
            validate_push_config(
                replace(load_push_config(), enabled=False), {}
            ),
            [],
        )

    def test_max_per_digest_allows_1_through_10(self):
        config = replace(load_push_config(), enabled=False)
        for value in (1, 3, 10):
            with self.subTest(value=value):
                problems = validate_push_config(replace(config, max_per_digest=value), {})
                self.assertEqual(problems, [])

    def test_max_per_digest_out_of_range_is_reported(self):
        config = load_push_config()
        for value in (0, 11, 50):
            with self.subTest(value=value):
                problems = validate_push_config(replace(config, max_per_digest=value), {})
                self.assertTrue(
                    any("max_per_digest" in problem for problem in problems),
                    problems,
                )

    def test_unknown_provider_reported_by_validate(self):
        config = replace(load_push_config(), provider="protonmail")
        problems = validate_push_config(config, {})
        self.assertTrue(any("protonmail" in problem for problem in problems), problems)

    def test_bad_time_and_frequency_are_reported(self):
        config = replace(load_push_config(), time_local="8:30 am", frequency="hourly")
        joined = "\n".join(validate_push_config(config, {}))
        self.assertIn("time_local", joined)
        self.assertIn("frequency", joined)

    def test_inconsistent_limits_are_reported(self):
        config = replace(
            load_push_config(),
            fetch_pool_limit=5,
            rank_max_candidates=20,
            overlap_hours=-1,
            catchup_days=0,
            auto_pause_after_failures=0,
        )
        joined = "\n".join(validate_push_config(config, {}))
        for name in (
            "rank_max_candidates",
            "overlap_hours",
            "catchup_days",
            "auto_pause_after_failures",
        ):
            with self.subTest(name=name):
                self.assertIn(name, joined)

    def test_catchup_days_is_bounded(self):
        config = replace(load_push_config(), enabled=False)
        self.assertEqual(
            validate_push_config(replace(config, catchup_days=365), {}), []
        )
        problems = "\n".join(
            validate_push_config(replace(config, catchup_days=366), {})
        )
        self.assertIn("catchup_days", problems)

    def test_custom_provider_is_a_known_provider(self):
        config = replace(
            load_push_config(), enabled=False, provider="custom"
        )
        problems = validate_push_config(config, {})
        self.assertEqual(problems, [])

    def test_wrong_typed_fields_are_reported_not_raised(self):
        config = replace(load_push_config(), provider=163, frequency=None)  # type: ignore[arg-type]
        problems = validate_push_config(config, {})
        self.assertTrue(any("provider" in problem for problem in problems), problems)
        self.assertTrue(any("frequency" in problem for problem in problems), problems)


class ValidateEnvTests(PushConfigTestCase):
    def test_enabled_without_credentials_lists_required_names(self):
        config = replace(load_push_config(), enabled=True)
        joined = "\n".join(validate_push_config(config, {}))
        for name in ("SMTP_USER", "SMTP_PASSWORD", "MAIL_TO"):
            with self.subTest(name=name):
                self.assertIn(name, joined)

    def test_enabled_with_valid_163_env_passes(self):
        config = replace(load_push_config(), enabled=True)
        self.assertEqual(validate_push_config(config, dict(VALID_163_ENV)), [])

    def test_partial_env_reports_only_missing_password(self):
        config = replace(load_push_config(), enabled=True)
        env = {k: v for k, v in VALID_163_ENV.items() if k != "SMTP_PASSWORD"}
        problems = validate_push_config(config, env)
        joined = "\n".join(problems)
        self.assertIn("SMTP_PASSWORD", joined)
        self.assertNotIn("SMTP_USER", joined)
        self.assertNotIn("MAIL_TO", joined)

    def test_validate_never_echoes_env_values(self):
        config = replace(load_push_config(), enabled=True, provider="custom")
        env = {
            "SMTP_USER": "private.person@163.com",
            "SMTP_PASSWORD": "TOP-SECRET-AUTH-CODE",
            "MAIL_TO": "private.person@163.com",
        }
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            problems = validate_push_config(config, env)
        joined = "\n".join(problems)
        self.assertTrue(problems)
        self.assertIn("SMTP_HOST", joined)
        self.assertNotIn("TOP-SECRET-AUTH-CODE", joined)
        self.assertNotIn("private.person@163.com", joined)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(err.getvalue(), "")

    def test_custom_provider_requires_transport_env(self):
        config = replace(load_push_config(), enabled=True, provider="custom")
        joined = "\n".join(validate_push_config(config, dict(VALID_163_ENV)))
        for name in ("SMTP_HOST", "SMTP_PORT", "SMTP_SECURITY"):
            with self.subTest(name=name):
                self.assertIn(name, joined)

    def test_custom_provider_with_full_env_passes(self):
        config = replace(load_push_config(), enabled=True, provider="custom")
        env = dict(VALID_163_ENV)
        env.update(
            {
                "SMTP_HOST": "smtp.example.com",
                "SMTP_PORT": "587",
                "SMTP_SECURITY": "starttls",
            }
        )
        self.assertEqual(validate_push_config(config, env), [])

    def test_invalid_port_and_security_are_reported(self):
        config = replace(load_push_config(), enabled=True, provider="custom")
        env = dict(VALID_163_ENV)
        env.update({"SMTP_HOST": "smtp.example.com", "SMTP_PORT": "abc", "SMTP_SECURITY": "plain"})
        joined = "\n".join(validate_push_config(config, env))
        self.assertIn("SMTP_PORT", joined)
        self.assertIn("SMTP_SECURITY", joined)

    def test_missing_credentials_do_not_depend_on_disabled_state(self):
        # No env vars and enabled=False must not demand credentials.
        self.assertEqual(
            validate_push_config(
                replace(load_push_config(), enabled=False), {}
            ),
            [],
        )


class NoHiddenIoTest(unittest.TestCase):
    def test_module_never_reads_dotenv_or_interests_file(self):
        source = (ROOT / "push_config.py").read_text(encoding="utf-8")
        for forbidden in (
            "load_dotenv",
            "import dotenv",
            "from dotenv",
            '".env"',
            "'.env'",
            "interests.json",
            "open(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
