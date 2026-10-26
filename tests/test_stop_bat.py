"""Contract and behavior tests for the Windows stop script."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAT = ROOT / "stop.bat"


class StopBatchContractTests(unittest.TestCase):
    def test_stopper_targets_the_launcher_port_and_streamlit(self):
        self.assertTrue(BAT.is_file(), "stop.bat is missing")
        text = BAT.read_text(encoding="utf-8")
        self.assertIn("8501", text)
        self.assertIn("Get-NetTCPConnection", text)
        self.assertIn("Stop-Process", text)
        self.assertIn("streamlit", text)
        self.assertIn("app.py", text)

    def test_stopper_does_not_contain_secrets(self):
        self.assertTrue(BAT.is_file(), "stop.bat is missing")
        text = BAT.read_text(encoding="utf-8")
        self.assertNotIn("SMTP_PASSWORD", text)
        self.assertNotIn("sk-", text)

    def test_batch_line_endings_are_pinned(self):
        attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("*.bat text eol=crlf", attrs)


@unittest.skipUnless(sys.platform == "win32", "stop.bat is Windows-only")
class StopBatchBehaviorTests(unittest.TestCase):
    def _free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _wait_for_listener(self, port: int, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.2)
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    return True
            time.sleep(0.1)
        return False

    def test_stops_a_listening_paperlens_like_process(self):
        port = self._free_port()
        code = (
            "import socket,sys,time;"
            "s=socket.socket();"
            "s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
            "s.bind(('127.0.0.1',int(sys.argv[1])));"
            "s.listen();"
            "time.sleep(120)"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code, str(port), "streamlit", "app.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            self.assertTrue(
                self._wait_for_listener(port),
                "fake server did not start listening",
            )
            env = {
                **os.environ,
                "PAPERLENS_PORT": str(port),
                "PAPERLENS_NO_HOLD": "1",
            }
            result = subprocess.run(
                ["cmd", "/c", str(BAT)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=30,
            )
            self.assertEqual(
                result.returncode,
                0,
                result.stdout + result.stderr,
            )
            self.assertIn("Stopped", result.stdout)
            proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
