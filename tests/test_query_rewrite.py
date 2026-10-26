"""Offline tests for page-blind Chinese query rewriting."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_rewrite import contains_chinese, rewrite_query  # noqa: E402

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"
ENV = {
    "DEEPSEEK_MODEL": "fake-model",
    "DEEPSEEK_BASE_URL": "https://example.invalid",
    "DEEPSEEK_API_KEY": "test-key",
}


class RewriteTests(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.cache = Path(self.tmp.name) / "cache"

    def tearDown(self):
        self.tmp.cleanup()

    def test_english_query_is_unchanged_without_calling_model(self):
        calls = []
        result = rewrite_query(
            "hybrid retrieval for question answering",
            chat_fn=lambda messages: calls.append(messages) or "{}",
            env=ENV,
            cache_dir=self.cache,
        )
        self.assertFalse(result.rewritten)
        self.assertEqual(result.retrieval_query, "hybrid retrieval for question answering")
        self.assertEqual(calls, [])

    def test_chinese_query_is_translated_and_cached(self):
        calls = []

        def chat(messages):
            calls.append(messages)
            return json.dumps({"query_en": "evidence coverage in RAG"})

        first = rewrite_query(
            "RAG 如何提高证据覆盖率？", chat_fn=chat, env=ENV, cache_dir=self.cache
        )
        second = rewrite_query(
            "RAG 如何提高证据覆盖率？", chat_fn=chat, env=ENV, cache_dir=self.cache
        )
        self.assertTrue(first.rewritten)
        self.assertEqual(first.retrieval_query, "evidence coverage in RAG")
        self.assertTrue(second.cached)
        self.assertEqual(len(calls), 1)

    def test_failure_falls_back_to_original(self):
        def broken(messages):
            raise RuntimeError("api down")

        result = rewrite_query(
            "中文问题", chat_fn=broken, env=ENV, cache_dir=self.cache
        )
        self.assertFalse(result.rewritten)
        self.assertEqual(result.retrieval_query, "中文问题")
        self.assertEqual(result.error, "RuntimeError")

    def test_env_switch_disables_rewrite(self):
        result = rewrite_query(
            "中文问题",
            chat_fn=lambda messages: "{}",
            env={**ENV, "PAPERLENS_QUERY_REWRITE": "0"},
            cache_dir=self.cache,
        )
        self.assertFalse(result.rewritten)

    def test_chinese_detection(self):
        self.assertTrue(contains_chinese("中文问题"))
        self.assertFalse(contains_chinese("english only"))


if __name__ == "__main__":
    unittest.main()
