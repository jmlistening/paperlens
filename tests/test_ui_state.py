"""Offline tests for private UI recipient state and verification (Task 2).

All addresses are synthetic ``example.com`` fixtures. No network, SMTP, model,
or real ``data/`` access happens here; every root is a temporary directory.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mail_transport import mask_email  # noqa: E402
from ui_state import (  # noqa: E402
    STATE_REL,
    UiStateError,
    clear_test_email,
    get_preview,
    load_state,
    mail_env,
    mark_verified,
    recipient_hash,
    record_manual_send,
    record_test_email,
    resolve_recipient,
    save_state,
    set_preview,
    set_recipient,
    test_email_confirmed,
)

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"
FIRST = "alice@example.com"
SECOND = "bob@example.com"
LEGACY = "legacy@example.com"
VERIFIED_AT = "2026-10-25T08:30:00+00:00"
PREVIEW = {
    "run_id": "run-1",
    "created_at": VERIFIED_AT,
    "profile_sha256": "a" * 64,
    "count": 3,
    "recipient_hash": recipient_hash(FIRST),
    "local_date": "2026-10-25",
}


class UiStateTests(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def state_file(self) -> Path:
        return self.root / STATE_REL

    def read_state(self) -> dict:
        return json.loads(self.state_file().read_text(encoding="utf-8"))

    def test_state_path_constant(self):
        self.assertEqual(STATE_REL, Path("data/push/ui_state.json"))

    def test_save_is_atomic_and_roundtrips(self):
        real_replace = os.replace
        calls: list[tuple[Path, Path]] = []

        def recording_replace(src, dst):
            calls.append((Path(src), Path(dst)))
            self.assertTrue(Path(src).is_file())
            real_replace(src, dst)

        state = {
            "recipient": FIRST,
            "verified_at": VERIFIED_AT,
            "preview": None,
            "last_manual_send": None,
        }
        with patch("ui_state.os.replace", side_effect=recording_replace):
            saved = save_state(self.root, state)

        self.assertEqual(saved["recipient"], FIRST)
        self.assertEqual(self.read_state()["verified_at"], VERIFIED_AT)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], self.state_file())
        self.assertEqual(calls[0][0].name, "ui_state.json.tmp")
        self.assertEqual(list(self.state_file().parent.glob("*.tmp")), [])

    def test_test_email_receipt_is_separate_from_verification(self):
        set_recipient(self.root, FIRST)

        state = record_test_email(self.root, FIRST, VERIFIED_AT)

        self.assertEqual(state["test_email_sent_at"], VERIFIED_AT)
        self.assertEqual(
            state["test_email_recipient_hash"], recipient_hash(FIRST)
        )
        self.assertIsNone(state["verified_at"])
        self.assertTrue(test_email_confirmed(self.root, FIRST))
        self.assertFalse(test_email_confirmed(self.root, SECOND))

    def test_test_email_receipt_requires_the_same_recipient(self):
        set_recipient(self.root, FIRST)

        with self.assertRaises(UiStateError):
            record_test_email(self.root, SECOND, VERIFIED_AT)
        self.assertFalse(test_email_confirmed(self.root, FIRST))

    def test_clear_test_email_removes_the_receipt(self):
        set_recipient(self.root, FIRST)
        record_test_email(self.root, FIRST, VERIFIED_AT)

        state = clear_test_email(self.root)

        self.assertIsNone(state["test_email_sent_at"])
        self.assertIsNone(state["test_email_recipient_hash"])
        self.assertFalse(test_email_confirmed(self.root, FIRST))

    def test_changing_email_clears_the_test_email_receipt(self):
        set_recipient(self.root, FIRST)
        record_test_email(self.root, FIRST, VERIFIED_AT)

        state = set_recipient(self.root, SECOND)

        self.assertIsNone(state["test_email_sent_at"])
        self.assertIsNone(state["test_email_recipient_hash"])
        self.assertFalse(test_email_confirmed(self.root, SECOND))

    def test_test_email_receipt_is_false_without_state_or_with_bad_state(self):
        self.assertFalse(test_email_confirmed(self.root, FIRST))

        set_recipient(self.root, FIRST)
        self.state_file().write_text("not json", encoding="utf-8")

        self.assertFalse(test_email_confirmed(self.root, FIRST))

    def test_changing_email_clears_verification_and_preview(self):
        set_recipient(self.root, FIRST)
        mark_verified(self.root, FIRST, VERIFIED_AT)
        set_preview(self.root, PREVIEW)

        state = set_recipient(self.root, SECOND)

        self.assertEqual(state["recipient"], SECOND)
        self.assertIsNone(state["verified_at"])
        self.assertIsNone(state["preview"])
        resolved = resolve_recipient(self.root, {"MAIL_TO": FIRST})
        self.assertEqual(resolved["recipient"], SECOND)
        self.assertFalse(resolved["verified"])
        self.assertEqual(resolved["source"], "ui_state")
        self.assertTrue(resolved["state_exists"])
        self.assertIsNone(get_preview(self.root))

    def test_changing_back_to_an_old_email_still_clears(self):
        set_recipient(self.root, FIRST)
        mark_verified(self.root, FIRST, VERIFIED_AT)
        set_recipient(self.root, SECOND)
        mark_verified(self.root, SECOND, VERIFIED_AT)

        state = set_recipient(self.root, FIRST)

        self.assertEqual(state["recipient"], FIRST)
        self.assertIsNone(state["verified_at"])
        self.assertFalse(resolve_recipient(self.root, {})["verified"])

    def test_saving_the_same_email_preserves_verification_and_preview(self):
        set_recipient(self.root, FIRST)
        mark_verified(self.root, FIRST, VERIFIED_AT)
        set_preview(self.root, PREVIEW)

        state = set_recipient(self.root, f"  {FIRST.upper()}  ")

        self.assertEqual(state["recipient"], FIRST)
        self.assertEqual(state["verified_at"], VERIFIED_AT)
        self.assertEqual(state["preview"], PREVIEW)
        self.assertTrue(resolve_recipient(self.root, {})["verified"])

    def test_ui_state_takes_precedence_over_env(self):
        set_recipient(self.root, FIRST)
        env = {"MAIL_TO": LEGACY, "SMTP_USER": "sender@example.com"}

        resolved = resolve_recipient(self.root, env)
        self.assertEqual(
            resolved,
            {
                "recipient": FIRST,
                "verified": False,
                "source": "ui_state",
                "state_exists": True,
            },
        )

        mark_verified(self.root, FIRST, VERIFIED_AT)
        self.assertTrue(resolve_recipient(self.root, env)["verified"])
        self.assertEqual(mail_env(self.root, env)["MAIL_TO"], FIRST)

    def test_missing_state_uses_legacy_env(self):
        env = {"MAIL_TO": LEGACY, "SMTP_USER": "sender@example.com"}

        resolved = resolve_recipient(self.root, env)

        self.assertEqual(resolved["recipient"], LEGACY)
        self.assertTrue(resolved["verified"])
        self.assertEqual(resolved["source"], "env")
        self.assertFalse(resolved["state_exists"])
        self.assertEqual(mail_env(self.root, env)["MAIL_TO"], LEGACY)

    def test_mail_env_returns_a_copy_with_the_current_recipient(self):
        set_recipient(self.root, FIRST)
        env = {"MAIL_TO": LEGACY, "SMTP_USER": "sender@example.com"}

        merged = mail_env(self.root, env)

        self.assertEqual(merged["MAIL_TO"], FIRST)
        self.assertEqual(merged["SMTP_USER"], "sender@example.com")
        self.assertEqual(env["MAIL_TO"], LEGACY)

    def test_display_uses_the_shared_masking_helper(self):
        set_recipient(self.root, FIRST)

        resolved = resolve_recipient(self.root, {})

        self.assertEqual(mask_email(resolved["recipient"]), "a***@example.com")
        self.assertNotIn(FIRST, mask_email(resolved["recipient"]))

    def test_invalid_email_is_rejected_without_echoing_it(self):
        for value in ("not-an-email", "a@b", "a b@example.com"):
            with self.subTest(value=value):
                with self.assertRaises(UiStateError) as ctx:
                    set_recipient(self.root, value)
                self.assertNotIn(value, str(ctx.exception))

        self.assertFalse(self.state_file().exists())

    def test_preview_rejects_raw_email_fields(self):
        set_recipient(self.root, FIRST)

        with self.assertRaises(UiStateError) as ctx:
            set_preview(self.root, {"run_id": "run-1", "email": SECOND})

        self.assertNotIn(SECOND, str(ctx.exception))
        self.assertIsNone(get_preview(self.root))

    def test_preview_and_manual_send_records_roundtrip(self):
        set_recipient(self.root, FIRST)

        state = set_preview(self.root, PREVIEW)
        self.assertEqual(state["preview"], PREVIEW)
        self.assertEqual(get_preview(self.root), PREVIEW)

        state = record_manual_send(self.root, "manual-send-1", VERIFIED_AT, 2)
        self.assertEqual(
            state["last_manual_send"],
            {"run_id": "manual-send-1", "at": VERIFIED_AT, "count": 2},
        )
        self.assertEqual(self.read_state()["last_manual_send"]["count"], 2)
        self.assertEqual(get_preview(self.root), PREVIEW)

    def test_mark_verified_rejects_a_different_recipient(self):
        set_recipient(self.root, FIRST)

        with self.assertRaises(UiStateError) as ctx:
            mark_verified(self.root, SECOND, VERIFIED_AT)

        self.assertNotIn(SECOND, str(ctx.exception))
        self.assertFalse(resolve_recipient(self.root, {})["verified"])

    def test_corrupt_state_fails_closed_without_env_fallback(self):
        path = self.state_file()
        path.parent.mkdir(parents=True)
        path.write_text('{"recipient": "' + FIRST + '"', encoding="utf-8")

        resolved = resolve_recipient(self.root, {"MAIL_TO": LEGACY})

        self.assertEqual(resolved["recipient"], "")
        self.assertFalse(resolved["verified"])
        self.assertEqual(resolved["source"], "ui_state")
        self.assertTrue(resolved["state_exists"])
        with self.assertRaises(UiStateError) as ctx:
            load_state(self.root)
        self.assertNotIn(FIRST, str(ctx.exception))
        with self.assertRaises(UiStateError):
            mail_env(self.root, {"MAIL_TO": LEGACY})

        path.write_bytes(b'{"recipient": "' + FIRST.encode() + b'"}\xff')
        with self.assertRaises(UiStateError) as ctx:
            load_state(self.root)
        self.assertNotIn(FIRST, str(ctx.exception))
        self.assertFalse(resolve_recipient(self.root, {"MAIL_TO": LEGACY})["verified"])


if __name__ == "__main__":
    unittest.main()
