#!/usr/bin/env python
"""Fail when private data is about to be published.

The default run scans only files tracked by git (``git ls-files``): it looks for
real-looking email addresses, ``sk-`` API keys, and ``SMTP_PASSWORD=`` values
that are not documented placeholders. That tracked-file check is the publish
gate.

``--history`` additionally audits git history for the same patterns and for the
old personal ``interests.json`` / ``docs/handoff/`` documents. History findings
are report-only: this script never rewrites history, creates a branch, or
pushes. The owner decides separately whether old history may be published.

Secrets are never printed. Findings contain a path (or commit hash), a line
number, and a masked value or pattern name only.

Exit codes: 0 clean, 1 findings, 2 the scan itself could not run.
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Placeholder policy
# ---------------------------------------------------------------------------
# Addresses used by the public example, docs, and offline tests. Every other
# address is treated as real. Keep this list explicit: adding a new fixture
# address should be a deliberate, reviewable change.
PLACEHOLDER_DOMAINS = frozenset(
    {
        "example.com",
        "example.net",
        "example.org",
        "example.invalid",
        "host.tld",
        "localhost",
        "test.invalid",
        "test.local",
    }
)
PLACEHOLDER_EMAILS = frozenset(
    {
        # Documented SMTP examples.
        "your_account@163.com",
        "your_account@qq.com",
        # Synthetic offline-test fixtures (see tests/).
        "user@163.com",
        "to@163.com",
        "fixture.user@163.com",
        "private.person@163.com",
    }
)
PLACEHOLDER_DOMAIN_SUFFIXES = (
    ".example",
    ".invalid",
    ".test",
    ".localhost",
)

# A real ``sk-`` key is a long random token. Shorter matches are false
# positives from titles such as "Task-Oriented". Fixture tokens carry an
# obvious placeholder word.
SK_PLACEHOLDER_MARKERS = (
    "test",
    "fake",
    "dummy",
    "example",
    "placeholder",
    "sample",
    "secret",
    "your",
)

SMTP_PLACEHOLDER_VALUES = frozenset(
    {
        "your_app_authorization_code",
        "your_app_password",
        "你的163客户端授权码",
        "16位QQ授权码",
        "fake-smtp-secret",
        "local-secret",
        "test-app-code",
    }
)
SMTP_PLACEHOLDER_MARKERS = (
    "your_",
    "your-",
    "placeholder",
    "fake",
    "dummy",
    "example",
    "test",
    "sample",
    "secret",
    "授权码",
)

# Labels are matched explicitly so a sentence-ending period is not swallowed
# and does not make the whole address unmatchable (``user@example.com.``).
EMAIL_RE = re.compile(
    # The lookahead requires at least one alphanumeric in the local part.
    # Without it, a diff marker plus a decorator (``+@unittest.skipUnless``)
    # looks like an address with local part "+".
    r"(?<![\w.%+-])(?=[A-Za-z0-9._%+-]*[A-Za-z0-9])"
    r"([A-Za-z0-9._%+-]+)@"
    r"([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})"
    r"(?![\w-])"
)
SK_RE = re.compile(r"(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{16,}")
# Accept an unquoted value or a double/single-quoted one, returning the inner
# value without the quotes. Placeholder policy is applied after unwrapping.
SMTP_RE = re.compile(
    r"""SMTP_PASSWORD\s*=\s*(?:"([^"\r\n]+)"|'([^'\r\n]+)'|([^\s"'`\\]+))"""
)

# Personal/handoff paths that must not be published again. The history audit
# reports them; the tracked-file gate fails if any of them is tracked.
PRIVATE_HISTORY_PATHSPECS = (
    "interests.json",
    "docs/handoff",
    "使用说明*.md",
    "验收记录*.md",
    "接续记录.md",
    "README.txt",
)
PRIVATE_TRACKED_PATHS = frozenset({"interests.json"})
PRIVATE_TRACKED_PREFIXES = ("docs/handoff/", "data/push/")
PRIVATE_TRACKED_GLOBS = (
    "使用说明*.md",
    "验收记录*.md",
    "接续记录.md",
    "README.txt",
)


class ScanError(RuntimeError):
    """The scan could not run (git missing, not a repository, ...)."""


@dataclass(frozen=True)
class Finding:
    kind: str
    source: str
    line: int | None
    detail: str

    def render(self) -> str:
        location = self.source if self.line is None else f"{self.source}:{self.line}"
        return f"[{self.kind}] {location} {self.detail}"


def mask_email(address: str) -> str:
    local, _, domain = address.partition("@")
    keep = local[:1] if local else ""
    return f"{keep}***@{domain}"


def is_placeholder_email(address: str) -> bool:
    lowered = address.lower()
    if lowered in PLACEHOLDER_EMAILS:
        return True
    _, _, domain = lowered.partition("@")
    if domain in PLACEHOLDER_DOMAINS:
        return True
    return any(
        domain.endswith(suffix) for suffix in PLACEHOLDER_DOMAIN_SUFFIXES
    )


def is_placeholder_sk(token: str) -> bool:
    lowered = token.lower()
    return any(marker in lowered for marker in SK_PLACEHOLDER_MARKERS)


def is_placeholder_password(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered:
        return True
    if lowered in SMTP_PLACEHOLDER_VALUES:
        return True
    return any(
        marker in lowered for marker in SMTP_PLACEHOLDER_MARKERS
    )


def _iter_lines(text: str) -> Iterable[tuple[int, str]]:
    yield from enumerate(text.splitlines(), start=1)


def scan_text(text: str, source: str) -> list[Finding]:
    """Return de-duplicated redacted findings for one text blob."""
    findings: dict[tuple[str, str], Finding] = {}
    for line_number, line in _iter_lines(text):
        for match in EMAIL_RE.finditer(line):
            address = match.group(0)
            if is_placeholder_email(address):
                continue
            key = ("email", address.lower())
            findings.setdefault(
                key,
                Finding(
                    kind="email",
                    source=source,
                    line=line_number,
                    detail=f"{mask_email(address)} (real-looking address)",
                ),
            )
        for match in SK_RE.finditer(line):
            token = match.group(0)
            if is_placeholder_sk(token):
                continue
            key = ("api-key", token)
            findings.setdefault(
                key,
                Finding(
                    kind="api-key",
                    source=source,
                    line=line_number,
                    detail=f"sk-*** (length {len(token)})",
                ),
            )
        for match in SMTP_RE.finditer(line):
            value = next(
                (group for group in match.groups() if group is not None),
                "",
            )
            if is_placeholder_password(value):
                continue
            key = ("smtp-password", value)
            findings.setdefault(
                key,
                Finding(
                    kind="smtp-password",
                    source=source,
                    line=line_number,
                    detail="SMTP_PASSWORD=<non-placeholder value>",
                ),
            )
    return sorted(
        findings.values(),
        key=lambda finding: (finding.source, finding.line or 0, finding.kind),
    )


def git(*args: str, binary: bool = False) -> subprocess.CompletedProcess:
    options: dict[str, object] = {
        "cwd": ROOT,
        "capture_output": True,
        "creationflags": (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if sys.platform == "win32"
            else 0
        ),
    }
    if binary:
        options["text"] = False
    else:
        options.update(text=True, encoding="utf-8", errors="replace")
    return subprocess.run(
        ["git", "-c", "core.quotepath=false", *args], **options
    )


def list_tracked_files() -> list[str]:
    result = git("ls-files", "-z", binary=True)
    if result.returncode != 0:
        raise ScanError(result.stderr.decode("utf-8", errors="replace"))
    return [
        item.decode("utf-8", errors="replace")
        for item in result.stdout.split(b"\0")
        if item
    ]


def _read_tracked_file(relative: str) -> str | None:
    path = ROOT / relative
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def private_tracked_path_findings(paths: Sequence[str]) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        name = Path(path).name
        if (
            path in PRIVATE_TRACKED_PATHS
            or path.startswith(PRIVATE_TRACKED_PREFIXES)
            or any(
                fnmatch.fnmatch(name, pattern)
                for pattern in PRIVATE_TRACKED_GLOBS
            )
        ):
            findings.append(
                Finding(
                    kind="tracked-private-path",
                    source=path,
                    line=None,
                    detail="private path must not be tracked",
                )
            )
    return findings


def scan_tracked_files(paths: Sequence[str] | None = None) -> list[Finding]:
    tracked = list(paths) if paths is not None else list_tracked_files()
    findings = private_tracked_path_findings(tracked)
    for relative in tracked:
        text = _read_tracked_file(relative)
        if text is None:
            continue
        findings.extend(scan_text(text, relative))
    return findings


# ---------------------------------------------------------------------------
# Read-only history audit
# ---------------------------------------------------------------------------
def parse_history_name_status(output: str) -> list[Finding]:
    """Parse ``git log --name-status`` output into redacted path findings."""
    findings: list[Finding] = []
    commit = ""
    seen: set[tuple[str, str, str]] = set()
    for line in output.splitlines():
        if not line.strip():
            continue
        if line.startswith("COMMIT\t"):
            parts = line.split("\t")
            commit = parts[1] if len(parts) > 1 else ""
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status, path = parts[0], parts[1]
        key = (commit, status, path)
        if key in seen:
            continue
        seen.add(key)
        findings.append(
            Finding(
                kind="history-private-path",
                source=f"commit {commit}" if commit else "history",
                line=None,
                detail=f"{status} {path}",
            )
        )
    return findings


def history_path_findings() -> list[Finding]:
    result = git(
        "log",
        "--all",
        "--name-status",
        "--format=COMMIT%x09%h%x09%s",
        "--",
        *PRIVATE_HISTORY_PATHSPECS,
    )
    if result.returncode != 0:
        raise ScanError(result.stderr)
    return parse_history_name_status(result.stdout)


def history_content_findings() -> list[Finding]:
    """Scan patch text per commit without printing any matched value."""
    result = git(
        "log",
        "--all",
        "--no-color",
        "-p",
        "--format=COMMIT%x09%h",
    )
    if result.returncode != 0:
        raise ScanError(result.stderr)
    findings: list[Finding] = []
    commit = ""
    chunk: list[str] = []

    def flush() -> None:
        if not commit or not chunk:
            return
        findings.extend(scan_text("\n".join(chunk), f"commit {commit}"))

    for line in result.stdout.splitlines():
        if line.startswith("COMMIT\t"):
            flush()
            parts = line.split("\t")
            commit = parts[1] if len(parts) > 1 else ""
            chunk = []
            continue
        # Diff metadata and context cannot add new secrets, but including it
        # keeps the scan simple and still redacted. Added/removed lines are the
        # only lines that could have introduced a value.
        if line.startswith(("+", "-")):
            chunk.append(line)
    flush()
    return findings


def _print_findings(title: str, findings: Sequence[Finding]) -> None:
    print(f"FAIL {title}: {len(findings)} finding(s)")
    for finding in findings:
        print(f"  {finding.render()}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan tracked files for real emails, sk- keys, and non-placeholder "
            "SMTP passwords. History is audited only with --history."
        )
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help=(
            "Also audit git history (read-only) for secrets and the old "
            "personal interests.json / handoff documents."
        ),
    )
    args = parser.parse_args(argv)

    try:
        tracked = list_tracked_files()
        findings = scan_tracked_files(tracked)
    except ScanError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Scanned {len(tracked)} tracked files.")
    if findings:
        _print_findings("tracked-file findings", findings)
        status = 1
    else:
        print(
            "OK: no real emails, sk- keys, or non-placeholder SMTP "
            "passwords in tracked files."
        )
        status = 0

    if args.history:
        print(
            "History audit is report-only: no rewrite, branch, or push is "
            "performed."
        )
        try:
            history = history_path_findings() + history_content_findings()
        except ScanError as exc:
            print(f"ERROR: history audit failed: {exc}", file=sys.stderr)
            return 2
        if history:
            _print_findings("history findings", history)
            status = 1
        else:
            print("OK: history audit found no private paths or secrets.")

    if status:
        print(
            "Do not publish or push this repository until the findings above "
            "are resolved or the owner explicitly accepts the history."
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
