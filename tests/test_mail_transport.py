"""Offline tests for the 163/QQ SMTP transport. No network, no real credentials."""
from __future__ import annotations

import smtplib
import sys
import unittest
from email.message import EmailMessage
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mail_transport import (
    MailTransportError,
    SmtpSettings,
    resolve_smtp,
    send_message,
    test_login,
)

ENV = {
    "SMTP_USER": "user@163.com",
    "SMTP_PASSWORD": "secret-app-code",
    "MAIL_TO": "to@163.com",
}


class FakeSMTP:
    def __init__(self, host, port, **kwargs):
        self.host = host
        self.port = port
        self.kwargs = kwargs
        self.logged_in = False
        self.noop_called = False
        self.sent = []
        self.quit_called = False
        self.starttls_called = False

    def login(self, user, password):
        if not self.logged_in:
            self.logged_in = (user, password)

    def noop(self):
        self.noop_called = True

    def send_message(self, message, from_addr=None, to_addrs=None):
        self.sent.append((message, from_addr, to_addrs))
        return {}

    def ehlo(self):
        return None

    def starttls(self, context=None):
        self.starttls_called = True

    def quit(self):
        self.quit_called = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.quit()


class MailTransportTests(unittest.TestCase):
    def test_163_preset(self):
        settings = resolve_smtp("163", ENV)
        self.assertEqual(settings.host, "smtp.163.com")
        self.assertEqual(settings.port, 465)
        self.assertEqual(settings.security, "ssl")
        self.assertEqual(settings.mail_to, "to@163.com")

    def test_qq_preset(self):
        settings = resolve_smtp("qq", ENV)
        self.assertEqual(settings.host, "smtp.qq.com")
        self.assertEqual(settings.port, 465)
        self.assertEqual(settings.security, "ssl")

    def test_custom_587_uses_starttls(self):
        env = dict(ENV, SMTP_HOST="smtp.example.com", SMTP_PORT="587", SMTP_SECURITY="starttls")
        settings = resolve_smtp("custom", env)
        self.assertEqual(settings.port, 587)
        self.assertEqual(settings.security, "starttls")

    def test_plain_25_rejected_for_remote_host(self):
        env = dict(ENV, SMTP_HOST="smtp.example.com", SMTP_PORT="25", SMTP_SECURITY="none")
        with self.assertRaises(ValueError):
            resolve_smtp("custom", env)

    def test_repr_masks_password_and_addresses(self):
        settings = resolve_smtp("163", ENV)
        rendered = repr(settings)
        self.assertNotIn("secret-app-code", rendered)
        self.assertIn("u***@163.com", rendered)
        self.assertIn("t***@163.com", rendered)

    def test_login_only_no_send(self):
        fake = FakeSMTP("smtp.163.com", 465)
        with patch("mail_transport.smtplib.SMTP_SSL", return_value=fake):
            test_login(resolve_smtp("163", ENV))
        self.assertTrue(fake.logged_in)
        self.assertTrue(fake.noop_called)
        self.assertEqual(fake.sent, [])
        self.assertTrue(fake.quit_called)

    def test_send_message_once(self):
        fake = FakeSMTP("smtp.163.com", 465)
        message = EmailMessage()
        message["Subject"] = "test"
        message["From"] = "user@163.com"
        message["To"] = "to@163.com"
        with patch("mail_transport.smtplib.SMTP_SSL", return_value=fake):
            send_message(resolve_smtp("163", ENV), message)
        self.assertEqual(len(fake.sent), 1)
        self.assertEqual(fake.sent[0][1], "user@163.com")
        self.assertEqual(fake.sent[0][2], ["to@163.com"])
        self.assertTrue(fake.quit_called)

    def test_error_hides_password(self):
        fake = FakeSMTP("smtp.163.com", 465)

        def fail_login(user, password):
            raise smtplib.SMTPAuthenticationError(535, b"bad secret-app-code")

        fake.login = fail_login
        with patch("mail_transport.smtplib.SMTP_SSL", return_value=fake):
            with self.assertRaises(MailTransportError) as ctx:
                test_login(resolve_smtp("163", ENV))
        self.assertNotIn("secret-app-code", str(ctx.exception))
        self.assertIn("SMTPAuthenticationError", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
