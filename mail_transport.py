"""SMTP transport for PaperLens push, with 163/QQ presets.

Security rules:
- never include passwords or SMTP response bodies in exceptions/reprs;
- verify TLS certificates;
- allow `security=none` only for localhost.
"""
from __future__ import annotations

import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Mapping

PRESETS = {
    "163": {"host": "smtp.163.com", "port": 465, "security": "ssl"},
    "qq": {"host": "smtp.qq.com", "port": 465, "security": "ssl"},
}

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class MailTransportError(RuntimeError):
    """A safe-to-log SMTP failure (type + code only)."""


def mask_email(value: str) -> str:
    if "@" not in value:
        return "***"
    local, domain = value.split("@", 1)
    return f"{local[:1]}***@{domain}"


@dataclass(frozen=True, repr=False)
class SmtpSettings:
    host: str
    port: int
    security: str
    user: str
    password: str
    mail_to: str
    from_name: str = "PaperLens"

    def __repr__(self) -> str:
        return (
            "SmtpSettings("
            f"host={self.host!r}, port={self.port}, security={self.security!r}, "
            f"user={mask_email(self.user)!r}, mail_to={mask_email(self.mail_to)!r})"
        )


def _required(env: Mapping[str, str], key: str) -> str:
    value = (env.get(key) or "").strip()
    if not value:
        raise ValueError(f"missing {key}")
    return value


def resolve_smtp(provider: str, env: Mapping[str, str]) -> SmtpSettings:
    provider = (provider or "").strip().lower()
    if provider in PRESETS:
        preset = PRESETS[provider]
    elif provider == "custom":
        preset = {"host": "", "port": 587, "security": "starttls"}
    else:
        raise ValueError(f"unknown provider: {provider!r}")

    host = (env.get("SMTP_HOST") or preset["host"]).strip()
    if not host:
        raise ValueError("missing SMTP_HOST for custom provider")
    try:
        port = int((env.get("SMTP_PORT") or str(preset["port"])).strip())
    except ValueError as exc:
        raise ValueError("SMTP_PORT must be an integer") from exc
    security = (env.get("SMTP_SECURITY") or preset["security"]).strip().lower()
    if security not in {"ssl", "starttls", "none"}:
        raise ValueError("SMTP_SECURITY must be ssl, starttls, or none")
    if security == "none" and host.lower() not in LOCAL_HOSTS:
        raise ValueError("SMTP_SECURITY=none is only allowed for localhost")

    user = _required(env, "SMTP_USER")
    password = _required(env, "SMTP_PASSWORD")
    mail_to = _required(env, "MAIL_TO")
    for label, address in (("SMTP_USER", user), ("MAIL_TO", mail_to)):
        if "@" not in address:
            raise ValueError(f"{label} must be an email address")

    return SmtpSettings(
        host=host,
        port=port,
        security=security,
        user=user,
        password=password,
        mail_to=mail_to,
        from_name=(env.get("SMTP_FROM_NAME") or "PaperLens").strip(),
    )


def _open(settings: SmtpSettings) -> smtplib.SMTP:
    context = ssl.create_default_context()
    if settings.security == "ssl":
        return smtplib.SMTP_SSL(
            settings.host, settings.port, context=context, timeout=60
        )
    if settings.security == "starttls":
        server = smtplib.SMTP(settings.host, settings.port, timeout=60)
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        return server
    if settings.host.lower() not in LOCAL_HOSTS:
        raise ValueError("SMTP_SECURITY=none is only allowed for localhost")
    return smtplib.SMTP(settings.host, settings.port, timeout=60)


def _safe_error(exc: BaseException) -> MailTransportError:
    code = getattr(exc, "smtp_code", None)
    message = type(exc).__name__ + (f" (SMTP {code})" if code is not None else "")
    return MailTransportError(message)


def test_login(settings: SmtpSettings) -> None:
    """Connect, authenticate, and NOOP. Never sends a message."""
    server = None
    try:
        server = _open(settings)
        server.login(settings.user, settings.password)
        server.noop()
    except Exception as exc:
        raise _safe_error(exc) from None
    finally:
        if server is not None:
            try:
                server.quit()
            except Exception:
                pass


def send_message(settings: SmtpSettings, message: EmailMessage) -> None:
    """Send exactly one EmailMessage to settings.mail_to."""
    server = None
    try:
        server = _open(settings)
        server.login(settings.user, settings.password)
        server.send_message(
            message, from_addr=settings.user, to_addrs=[settings.mail_to]
        )
    except Exception as exc:
        raise _safe_error(exc) from None
    finally:
        if server is not None:
            try:
                server.quit()
            except Exception:
                pass
