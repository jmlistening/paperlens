"""Offline tests for topic-to-query generation (no provider calls)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topic_queries import (  # noqa: E402
    TopicQueryError,
    generate_query_groups,
    make_deepseek_chat_fn,
)


def group(name="Retrieval", any_terms=None, require_any_terms=None):
    payload = {"name": name, "any_terms": any_terms or ["retrieval"]}
    if require_any_terms is not None:
        payload["require_any_terms"] = require_any_terms
    return payload


def json_chat(payload, calls=None):
    """Return a fake chat_fn that answers with ``payload``."""

    def chat(messages):
        if calls is not None:
            calls.append(messages)
        return json.dumps(payload, ensure_ascii=False)

    return chat


class TopicQueryGenerationTests(unittest.TestCase):
    def test_chinese_topics_produce_validated_groups(self):
        calls = []
        payload = {
            "groups": [
                {
                    "name": "检索增强生成",
                    "any_terms": ["retrieval augmented generation", "RAG"],
                    "require_any_terms": [],
                }
            ]
        }

        groups = generate_query_groups(
            "我想关注检索增强生成和证据溯源。", json_chat(payload, calls)
        )

        self.assertEqual(
            groups,
            [
                {
                    "name": "检索增强生成",
                    "any_terms": ["retrieval augmented generation", "RAG"],
                    "require_any_terms": [],
                }
            ],
        )
        self.assertEqual(len(calls), 1)
        system = calls[0][0]["content"].lower()
        self.assertIn("translate", system)
        self.assertIn("do not add unrelated research directions", system)
        self.assertIn('"groups"', system)
        self.assertEqual(
            json.loads(calls[0][1]["content"])["topics"],
            "我想关注检索增强生成和证据溯源。",
        )

    def test_english_topics_are_supported(self):
        payload = {
            "groups": [
                group("Graph learning", ["graph neural networks"], ["node classification"])
            ]
        }
        groups = generate_query_groups(
            "graph learning\nnode classification", json_chat(payload)
        )
        self.assertEqual(groups[0]["name"], "Graph learning")
        self.assertEqual(groups[0]["require_any_terms"], ["node classification"])
        self.assertEqual(
            set(groups[0]), {"name", "any_terms", "require_any_terms"}
        )

    def test_up_to_three_groups_are_accepted(self):
        payload = {
            "groups": [
                group("One", ["retrieval"]),
                group("Two", ["ranking"]),
                group("Three", ["evaluation"]),
            ]
        }
        groups = generate_query_groups("topics", json_chat(payload))
        self.assertEqual([item["name"] for item in groups], ["One", "Two", "Three"])

    def test_max_groups_limits_the_result(self):
        calls = []
        payload = {
            "groups": [group("One", ["retrieval"]), group("Two", ["ranking"])]
        }
        self.assertEqual(
            len(
                generate_query_groups(
                    "topics", json_chat(payload, calls), max_groups=2
                )
            ),
            2,
        )
        self.assertIn("between 1 and 2 groups", calls[0][0]["content"])
        with self.assertRaises(TopicQueryError):
            generate_query_groups("topics", json_chat(payload), max_groups=1)

    def test_generated_groups_fit_the_saved_profile_schema(self):
        from profile_config import validate_profile

        payload = {
            "groups": [
                group("RAG", ["retrieval augmented generation", "RAG"])
            ]
        }
        groups = generate_query_groups("RAG", json_chat(payload))
        profile = json.loads(
            (ROOT / "interests.example.json").read_text(encoding="utf-8")
        )
        profile["search"]["groups"] = groups
        validated = validate_profile(profile)
        self.assertEqual(validated["search"]["groups"], groups)

    def test_invalid_max_groups_is_rejected(self):
        payload = {"groups": [group()]}
        for value in (0, 4, True):
            with self.subTest(max_groups=value):
                with self.assertRaises(TopicQueryError):
                    generate_query_groups(
                        "topics", json_chat(payload), max_groups=value
                    )

    def test_missing_require_any_terms_defaults_to_empty(self):
        payload = {"groups": [group()]}
        groups = generate_query_groups("topics", json_chat(payload))
        self.assertEqual(groups[0]["require_any_terms"], [])

    def test_duplicate_terms_are_stripped_case_insensitively(self):
        payload = {
            "groups": [
                group("RAG", ["RAG", "rag", " retrieval  augmented   generation "])
            ]
        }
        groups = generate_query_groups("topics", json_chat(payload))
        self.assertEqual(
            groups[0]["any_terms"],
            ["RAG", "retrieval augmented generation"],
        )

    def test_five_terms_and_five_required_terms_are_accepted(self):
        payload = {
            "groups": [
                group(
                    "Many",
                    ["one", "two", "three", "four", "five"],
                    ["alpha", "beta", "gamma", "delta", "epsilon"],
                )
            ]
        }
        groups = generate_query_groups("topics", json_chat(payload))
        self.assertEqual(len(groups[0]["any_terms"]), 5)
        self.assertEqual(len(groups[0]["require_any_terms"]), 5)

    def test_empty_topics_raise_without_calling_the_model(self):
        calls = []

        def chat(messages):  # pragma: no cover - must not run
            calls.append(messages)
            return "{}"

        for value in ("", "   \n\t"):
            with self.subTest(value=value):
                with self.assertRaises(TopicQueryError):
                    generate_query_groups(value, chat)
        self.assertEqual(calls, [])


class TopicQueryFailureTests(unittest.TestCase):
    def assert_rejected(self, payload, *, max_groups=3):
        chat = (
            json_chat(payload)
            if not isinstance(payload, str)
            else lambda messages: payload
        )
        with self.assertRaises(TopicQueryError):
            generate_query_groups("topics", chat, max_groups=max_groups)

    def test_malformed_json_is_rejected(self):
        self.assert_rejected("not json at all")
        self.assert_rejected('{"groups": [}')

    def test_wrong_top_level_shape_is_rejected(self):
        self.assert_rejected("[]")
        self.assert_rejected(json.dumps({"groups": "nope"}))
        self.assert_rejected(json.dumps({"queries": [group()]}))
        self.assert_rejected(json.dumps({"groups": [group()], "notes": "extra"}))

    def test_empty_groups_are_rejected(self):
        self.assert_rejected({"groups": []})

    def test_too_many_groups_are_rejected(self):
        self.assert_rejected(
            {
                "groups": [
                    group("One"),
                    group("Two"),
                    group("Three"),
                    group("Four"),
                ]
            }
        )

    def test_empty_and_missing_terms_are_rejected(self):
        self.assert_rejected({"groups": [{"name": "Empty", "any_terms": []}]})
        self.assert_rejected({"groups": [{"name": "Missing"}]})
        self.assert_rejected(
            {"groups": [{"name": "None", "any_terms": None}]}
        )
        self.assert_rejected(
            {"groups": [{"name": "String", "any_terms": "retrieval"}]}
        )

    def test_too_many_terms_are_rejected(self):
        self.assert_rejected(
            {
                "groups": [
                    group(
                        "Six",
                        ["one", "two", "three", "four", "five", "six"],
                    )
                ]
            }
        )
        self.assert_rejected(
            {
                "groups": [
                    group(
                        "Required",
                        ["retrieval"],
                        ["a", "b", "c", "d", "e", "f"],
                    )
                ]
            }
        )

    def test_invalid_terms_are_rejected(self):
        for terms in (
            ["retrieval", ""],
            ["retrieval", "   "],
            ["retrieval", 3],
            ["retrieval", None],
            ["cat:cs.CL AND retrieval"],
            ["检索增强生成"],
            ["retrieval (RAG)"],
            ["x" * 81],
        ):
            with self.subTest(terms=terms):
                self.assert_rejected(
                    {"groups": [group("Bad", terms)]}
                )

    def test_invalid_names_are_rejected(self):
        for name in ("", "   ", None, 3, "x" * 61):
            with self.subTest(name=name):
                self.assert_rejected(
                    {"groups": [group(name, ["retrieval"])]}
                )
        self.assert_rejected(
            {
                "groups": [
                    group("Same", ["retrieval"]),
                    group("Same", ["ranking"]),
                ]
            }
        )

    def test_unknown_group_fields_are_rejected(self):
        self.assert_rejected(
            {"groups": [{**group(), "query": "cat:cs.CL"}]}
        )
        self.assert_rejected(
            {"groups": [{**group(), "notes": "extra"}]}
        )

    def test_group_must_be_an_object(self):
        self.assert_rejected({"groups": [["retrieval"]]})
        self.assert_rejected({"groups": ["retrieval"]})

    def test_model_errors_are_sanitized(self):
        def broken(messages):
            raise RuntimeError("provider said: api key sk-secret-123")

        with self.assertRaises(TopicQueryError) as caught:
            generate_query_groups("topics", broken)
        message = str(caught.exception)
        self.assertNotIn("sk-secret-123", message)
        self.assertNotIn("provider said", message)
        self.assertNotIn("RuntimeError", message)

    def test_non_string_model_output_is_rejected(self):
        with self.assertRaises(TopicQueryError):
            generate_query_groups("topics", lambda messages: {"groups": []})


class DeepSeekFactoryTests(unittest.TestCase):
    def test_factory_is_lazy_and_rejects_missing_config_without_network(self):
        chat_fn = make_deepseek_chat_fn(env={})
        self.assertTrue(callable(chat_fn))
        with self.assertRaises(TopicQueryError) as caught:
            chat_fn([{"role": "user", "content": "topics"}])
        message = str(caught.exception).lower()
        self.assertIn("deepseek", message)
        self.assertNotIn("sk-", message)


if __name__ == "__main__":
    unittest.main()
