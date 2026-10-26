"""Contract test for the one-click Windows launcher."""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAT = ROOT / "start.bat"


class StartBatchTests(unittest.TestCase):
    def test_launcher_uses_a_configurable_python_and_streamlit(self):
        text = BAT.read_text(encoding="utf-8")
        self.assertIn("PAPERLENS_PYTHON", text)
        self.assertIn('set "PAPERLENS_PYTHON=python"', text)
        self.assertIn("streamlit run", text)
        self.assertIn("app.py", text)
        self.assertIn("127.0.0.1", text)
        self.assertIn("8501", text)
        self.assertNotIn("SMTP_PASSWORD", text)
        self.assertNotIn("sk-", text)

    def test_batch_line_endings_are_pinned(self):
        attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("*.bat text eol=crlf", attrs)


if __name__ == "__main__":
    unittest.main()
