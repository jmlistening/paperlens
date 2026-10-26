"""Offline tests for links-first digest rendering."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from push_digest import build_digest


class Config:
    attach_pdfs = False
    max_per_digest = 3


class AttachConfig(Config):
    attach_pdfs = True


def paper(**overrides):
    value = {
        "arxiv_id": "2601.12345v1",
        "title": "A <script>alert(1)</script> paper",
        "url": "https://arxiv.org/abs/2601.12345v1",
        "categories": ["cs.CL"],
        "assessment": {
            "score": 88,
            "reason": "<b>matches RAG</b>",
            "limitation": "limited evaluation",
        },
        "note": "PRIVATE READING NOTE",
    }
    value.update(overrides)
    return value


class PushDigestTests(unittest.TestCase):
    def test_default_is_links_only_and_escapes_untrusted_text(self):
        digest = build_digest(
            [paper()],
            {"2601.12345": "中文摘要 <b>not html</b>"},
            Config(),
        )
        self.assertEqual(digest.paper_keys, ["2601.12345"])
        self.assertEqual(digest.attachments, [])
        self.assertIn("&lt;script&gt;", digest.html)
        self.assertNotIn("<script>", digest.html)
        self.assertNotIn("PRIVATE READING NOTE", digest.html)
        self.assertNotIn("PRIVATE READING NOTE", digest.text)
        self.assertIn("https://arxiv.org/abs/2601.12345v1", digest.html)
        self.assertNotIn("<img", digest.html.lower())
        self.assertIn("推送设置", digest.html)
        self.assertIn("1 篇", digest.subject)

    def test_summary_sections_are_separated_in_html_and_text(self):
        summary = (
            "中文标题：示例\n"
            "研究问题：它解决什么问题？\n"
            "核心方法：方法是什么？\n"
            "主要结果：结果是什么？\n"
            "阅读全文时需要确认：还需要看什么？"
        )
        digest = build_digest(
            [paper()], {"2601.12345": summary}, Config()
        )
        for label in ("研究问题：", "核心方法：", "主要结果：", "阅读全文时需要确认："):
            self.assertIn(f"<br><br>{label}", digest.html)
            self.assertIn(f"\n\n{label}", digest.text)

    def test_non_arxiv_url_is_rejected(self):
        with self.assertRaises(ValueError):
            build_digest(
                [paper(url="https://evil.example/x")],
                {"2601.12345": "s"},
                Config(),
            )

    def test_attachment_is_optional_and_small_pdf_is_kept(self):
        digest = build_digest(
            [paper()],
            {"2601.12345": "s"},
            AttachConfig(),
            pdf_loader=lambda p: ("2601.12345v1.pdf", b"%PDF-1.4 tiny"),
        )
        self.assertEqual(len(digest.attachments), 1)

    def test_oversized_attachment_degrades_to_link(self):
        digest = build_digest(
            [paper()],
            {"2601.12345": "s"},
            AttachConfig(),
            pdf_loader=lambda p: ("big.pdf", b"0" * (10 * 1024 * 1024 + 1)),
        )
        self.assertEqual(digest.attachments, [])
        self.assertTrue(digest.warnings)

    def test_too_many_papers_is_an_error(self):
        many = [paper(arxiv_id=f"2601.{i}v1", url=f"https://arxiv.org/abs/2601.{i}v1") for i in range(4)]
        with self.assertRaises(ValueError):
            build_digest(many, {p["arxiv_id"].split("v")[0]: "s" for p in many}, Config())


if __name__ == "__main__":
    unittest.main()
