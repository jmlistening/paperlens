"""Offline tests for LLM reranking of retrieval candidates."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.rerank import rerank_hits
from retrieval_core import Hit


class FakeChat:
    def __init__(self, response):
        self.response = response
        self.last_usage = {"total_tokens": 123}
        self.messages = None

    def __call__(self, messages):
        self.messages = messages
        return (
            self.response
            if isinstance(self.response, str)
            else json.dumps(self.response)
        )


def hits():
    return [
        Hit("c1", "p1", "2601.1v1", 0.9, "rrf", "first candidate", 1),
        Hit("c2", "p2", "2601.2v1", 0.8, "rrf", "second candidate", 2),
        Hit("c3", "p3", "2601.3v1", 0.7, "rrf", "third candidate", 3),
    ]


class RerankTests(unittest.TestCase):
    def test_orders_by_grade_and_returns_tokens(self):
        chat = FakeChat(
            {
                "ranking": [
                    {"id": "E1", "grade": 0, "reason": "not useful"},
                    {"id": "E2", "grade": 3, "reason": "direct support"},
                    {"id": "E3", "grade": 1, "reason": "background"},
                ]
            }
        )
        ordered, tokens = rerank_hits("question", hits(), chat, top_k=2)
        self.assertEqual([hit.parent_chunk_id for hit in ordered], ["p2", "p3"])
        self.assertEqual(tokens, 123)
        payload = json.loads(chat.messages[-1]["content"])
        self.assertEqual(payload["question"], "question")
        self.assertEqual(len(payload["candidates"]), 3)
        self.assertNotIn("gold", json.dumps(payload).lower())

    def test_incomplete_ranking_is_rejected(self):
        chat = FakeChat(
            {"ranking": [{"id": "E1", "grade": 2, "reason": "only one"}]}
        )
        with self.assertRaises(ValueError):
            rerank_hits("question", hits(), chat, top_k=2)

    def test_empty_candidates_do_not_call_model(self):
        chat = FakeChat({"ranking": []})
        ordered, tokens = rerank_hits("question", [], chat)
        self.assertEqual(ordered, [])
        self.assertEqual(tokens, 0)
        self.assertIsNone(chat.messages)


if __name__ == "__main__":
    unittest.main()
