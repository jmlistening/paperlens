"""Control plane for the optional PaperLens arXiv push."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import Any, Callable, Mapping

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ui_state  # noqa: E402
from mail_transport import mask_email, resolve_smtp, send_message, test_login  # noqa: E402
from push_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_push_config,
    parse_push_config,
    validate_push_config,
)
from push_run import PushDeps, run_push  # noqa: E402
from push_state import PushState  # noqa: E402

TASK_NAME = "PaperLens Frontier Push"
_TIME_LOCAL_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
_SUBPROCESS_TIMEOUT = 120
_TASK_QUERY_TIMEOUT = 10
_OUTPUT_LIMIT = 1200


class PushCtlError(RuntimeError):
    pass


def _env() -> dict[str, str]:
    values = dotenv_values(ROOT / ".env", encoding="utf-8-sig")
    return {key: str(value) for key, value in values.items() if value is not None}


def _read_raw(config_path: Path) -> dict[str, Any]:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def _write_raw(config_path: Path, data: Mapping[str, Any]) -> None:
    path = Path(config_path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def _state(root: Path) -> PushState:
    state = PushState(Path(root) / "data" / "push" / "frontier.sqlite3")
    state.init_schema()
    return state


def set_enabled(
    enabled: bool,
    config_path: Path = DEFAULT_CONFIG_PATH,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    raw = _read_raw(config_path)
    raw["enabled"] = bool(enabled)
    candidate = parse_push_config(raw)
    problems = validate_push_config(candidate, env if env is not None else _env())
    if enabled and problems:
        raise PushCtlError("; ".join(problems))
    _write_raw(config_path, raw)
    return {"enabled": bool(enabled), "problems": problems}


def set_paused(
    paused: bool,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    raw = _read_raw(config_path)
    raw["paused"] = bool(paused)
    _write_raw(config_path, raw)
    return {"paused": bool(paused)}


def _auto_paused(root: Path) -> bool:
    """Read the safety flag without creating a ledger when none exists."""
    path = Path(root) / "data" / "push" / "frontier.sqlite3"
    if not path.is_file():
        return False
    state = PushState(path)
    state.init_schema()
    return state.get_setting("auto_paused") == "true"


def set_auto_push(
    enabled: bool,
    root: Path = ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Shared UI gate for turning automatic push on or off.

    Enabling requires a verified recipient: an existing private UI state must
    have ``verified_at`` set, while a missing state keeps the legacy ``.env``
    ``MAIL_TO`` compatibility. Enabling is also refused while the ledger is
    auto-paused or ``quiet_mode`` is on, and it always clears the ordinary
    pause flag. Disabling only writes ``enabled=false`` and leaves the safety
    flags (``auto_paused``/``quiet_mode``) untouched.

    Never returns or logs the raw recipient; ``MAIL_TO`` validation uses the
    resolved recipient rather than a stale environment value.
    """
    values = dict(env) if env is not None else _env()
    raw = _read_raw(config_path)
    resolved = ui_state.resolve_recipient(Path(root), values)
    result: dict[str, Any] = {
        "enabled": bool(enabled),
        "recipient_verified": bool(resolved["verified"]),
        "recipient_source": resolved["source"],
        "problems": [],
    }

    if not enabled:
        raw["enabled"] = False
        _write_raw(config_path, raw)
        result.update(
            paused=bool(raw.get("paused", False)),
            quiet_mode=bool(raw.get("quiet_mode", False)),
        )
        return result

    if not resolved["verified"]:
        raise PushCtlError(
            "recipient is not verified; verify the current recipient before "
            "enabling auto-push"
        )
    if _auto_paused(root):
        raise PushCtlError(
            "push is auto-paused; resolve the last failure before enabling "
            "auto-push"
        )

    raw["enabled"] = True
    raw["paused"] = False
    candidate = parse_push_config(raw)
    if candidate.quiet_mode:
        raise PushCtlError(
            "quiet mode must be off before enabling auto-push"
        )

    effective = dict(values)
    effective["MAIL_TO"] = resolved["recipient"]
    problems = validate_push_config(candidate, effective)
    if problems:
        raise PushCtlError("; ".join(problems))

    _write_raw(config_path, raw)
    result.update(paused=False, quiet_mode=False)
    return result


def _run_task_script(
    script: Path, root: Path | None, time_local: str | None
) -> tuple[int, str]:
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "RemoteSigned",
        "-File",
        str(script),
    ]
    if root is not None:
        command.extend(["-ProjectDir", str(root)])
    if time_local is not None:
        command.extend(["-Time", time_local])
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, type(exc).__name__
    output = (completed.stdout or completed.stderr or "").strip()
    return int(completed.returncode), output[:_OUTPUT_LIMIT]


def task_installed() -> bool:
    """Return whether the Windows scheduled task is registered (read-only)."""
    try:
        completed = subprocess.run(
            ["schtasks", "/Query", "/TN", TASK_NAME, "/FO", "LIST"],
            capture_output=True,
            text=True,
            timeout=_TASK_QUERY_TIMEOUT,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def register_task(root: Path, time_local: str) -> dict[str, Any]:
    """Install/update the daily task through the checked-in PowerShell script."""
    if not isinstance(time_local, str) or not _TIME_LOCAL_PATTERN.match(
        time_local
    ):
        raise PushCtlError("time_local must be HH:MM in 24-hour local time")
    script = Path(root) / "scripts" / "register_push_task.ps1"
    code, output = _run_task_script(script, Path(root), time_local)
    return {"installed": code == 0, "returncode": code, "output": output}


def unregister_task() -> dict[str, Any]:
    """Remove the daily task through the checked-in PowerShell script."""
    script = ROOT / "scripts" / "unregister_push_task.ps1"
    code, output = _run_task_script(script, None, None)
    return {"removed": code == 0, "returncode": code, "output": output}


def approve_first_send(
    root: Path = ROOT, confirm: bool = False
) -> dict[str, Any]:
    if not confirm:
        raise PushCtlError("approve-first-send requires --confirm")
    state = _state(root)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state.set_setting("first_send_approved_at", timestamp)
    return {"first_send_approved_at": timestamp}


def resume(root: Path = ROOT, config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    set_paused(False, config_path)
    state = _state(root)
    state.set_setting("auto_paused", "false")
    return {"paused": False, "auto_paused": False}


def status_json(
    root: Path = ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    config = load_push_config(config_path)
    values = dict(env) if env is not None else _env()
    state = _state(root)
    resolved = ui_state.resolve_recipient(Path(root), values)
    mail_to = resolved["recipient"]
    # Validate against the recipient that would actually be used. A corrupt or
    # empty UI state must not silently fall back to a stale environment value.
    effective = dict(values)
    if mail_to:
        effective["MAIL_TO"] = mail_to
    else:
        effective.pop("MAIL_TO", None)
    return {
        "enabled": config.enabled,
        "paused": config.paused,
        "quiet_mode": config.quiet_mode,
        "provider": config.provider,
        "mail_to": mask_email(mail_to) if mail_to else None,
        "recipient_verified": bool(resolved["verified"]),
        "recipient_source": resolved["source"],
        "first_send_approved": bool(state.get_setting("first_send_approved_at")),
        "auto_paused": state.get_setting("auto_paused") == "true",
        "counts": state.status_counts(),
        "runs": state.run_status_counts(),
        "config_problems": validate_push_config(config, effective),
    }


def test_email(
    config_path: Path = DEFAULT_CONFIG_PATH,
    env: Mapping[str, str] | None = None,
    *,
    send: bool = False,
    confirm: bool = False,
    login_fn: Callable[[Any], None] | None = None,
    send_fn: Callable[[Any, EmailMessage], None] | None = None,
) -> dict[str, Any]:
    config = load_push_config(config_path)
    values = dict(env) if env is not None else _env()
    problems = validate_push_config(config, values)
    if problems:
        raise PushCtlError("; ".join(problems))
    settings = resolve_smtp(config.provider, values)
    if not send:
        (login_fn or test_login)(settings)
        return {"status": "login_ok", "host": settings.host, "port": settings.port}
    if not confirm:
        raise PushCtlError("test-email --send requires --confirm")
    message = EmailMessage()
    message["Subject"] = "PaperLens SMTP 测试"
    message["From"] = settings.user
    message["To"] = settings.mail_to
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid()
    message.set_content("这是一封 PaperLens SMTP 连通性测试邮件；没有推荐内容。")
    (send_fn or send_message)(settings, message)
    return {"status": "sent", "host": settings.host, "port": settings.port}


def history(root: Path = ROOT, limit: int = 10) -> list[dict[str, Any]]:
    runs_dir = Path(root) / "data" / "push" / "runs"
    if not runs_dir.is_dir():
        return []
    rows = []
    for run_dir in runs_dir.iterdir():
        run_json = run_dir / "run.json"
        if not run_json.is_file():
            continue
        try:
            payload = json.loads(run_json.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    # Run directory names are random ids, so sort by the actual finish time
    # (then start time) before applying the limit.
    rows.sort(
        key=lambda item: str(
            item.get("finished_at") or item.get("started_at") or ""
        ),
        reverse=True,
    )
    return rows[: int(limit)]


def doctor(
    config_path: Path = DEFAULT_CONFIG_PATH,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    config = load_push_config(config_path)
    values = dict(env) if env is not None else _env()
    problems = validate_push_config(config, values)
    return {
        "config_path": str(config_path),
        "enabled": config.enabled,
        "paused": config.paused,
        "provider": config.provider,
        "problems": problems,
        "ok": not problems,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PaperLens push control")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("enable")
    sub.add_parser("disable")
    sub.add_parser("pause")
    sub.add_parser("resume")
    approve = sub.add_parser("approve-first-send")
    approve.add_argument("--confirm", action="store_true")
    preview = sub.add_parser("preview")
    preview.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    send = sub.add_parser("send")
    send.add_argument("--confirm", action="store_true")
    sub.add_parser("doctor")
    status = sub.add_parser("status")
    status.add_argument("--json", action="store_true")
    sub.add_parser("history")
    test = sub.add_parser("test-email")
    test.add_argument("--send", action="store_true")
    test.add_argument("--confirm", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "enable":
            print(json.dumps(set_enabled(True, args.config), ensure_ascii=False))
        elif args.command == "disable":
            print(json.dumps(set_enabled(False, args.config), ensure_ascii=False))
        elif args.command == "pause":
            print(json.dumps(set_paused(True, args.config), ensure_ascii=False))
        elif args.command == "resume":
            print(json.dumps(resume(ROOT, args.config), ensure_ascii=False))
        elif args.command == "approve-first-send":
            print(
                json.dumps(
                    approve_first_send(ROOT, confirm=args.confirm),
                    ensure_ascii=False,
                )
            )
        elif args.command == "preview":
            result = run_push("preview", config_path=args.config, deps=PushDeps(root=ROOT))
            print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
        elif args.command == "send":
            if not args.confirm:
                raise PushCtlError("send requires --confirm")
            result = run_push("send", config_path=args.config, deps=PushDeps(root=ROOT))
            print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
        elif args.command == "doctor":
            print(json.dumps(doctor(args.config), ensure_ascii=False, indent=2))
        elif args.command == "status":
            print(json.dumps(status_json(ROOT, args.config), ensure_ascii=False, indent=2))
        elif args.command == "history":
            print(json.dumps(history(ROOT), ensure_ascii=False, indent=2))
        elif args.command == "test-email":
            print(
                json.dumps(
                    test_email(
                        args.config,
                        send=args.send,
                        confirm=args.confirm,
                    ),
                    ensure_ascii=False,
                )
            )
        return 0
    except (PushCtlError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
