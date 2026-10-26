"""Offline tests for the personalized discovery profile editor."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profile_config import (  # noqa: E402
    ProfileError,
    load_profile,
    profile_queries,
    save_profile,
    split_terms,
    validate_categories,
)

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"


def base_profile() -> dict:
    return json.loads(
        (ROOT / "interests.example.json").read_text(encoding="utf-8")
    )


class ProfileConfigTests(unittest.TestCase):
    def test_roundtrip_and_atomic_save(self):
        TMP_ROOT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as tmp:
            path = Path(tmp) / "interests.json"
            profile = base_profile()
            profile["top_k"] = 5
            saved = save_profile(profile, path)
            self.assertEqual(saved["top_k"], 5)
            loaded = load_profile(path)
            self.assertEqual(loaded["top_k"], 5)
            self.assertEqual(loaded["search"]["groups"][0]["name"], "Example")

    def test_rejects_invalid_top_k(self):
        profile = base_profile()
        profile["top_k"] = 0
        with self.assertRaises(ProfileError):
            save_profile(profile, Path("unused.json"))

    def test_rejects_empty_group_terms(self):
        profile = base_profile()
        profile["search"]["groups"][0]["any_terms"] = []
        with self.assertRaises(ProfileError):
            save_profile(profile, Path("unused.json"))

    def test_categories_are_validated(self):
        self.assertEqual(validate_categories(["cs.CL", "cs.IR"]), ["cs.CL", "cs.IR"])
        with self.assertRaises(ProfileError):
            validate_categories(["not a category!"])

    def test_queries_are_built_from_user_terms_and_categories(self):
        profile = base_profile()
        queries = profile_queries(profile, ["cs.CL"])
        self.assertTrue(queries)
        self.assertEqual(queries[0]["name"], "Example")
        self.assertIn("cat:cs.CL", queries[0]["query"])
        self.assertIn("machine learning", queries[0]["query"])

    def test_term_parsing_accepts_commas_and_newlines(self):
        self.assertEqual(
            split_terms("rag, retrieval\nagents, rag"),
            ["rag", "retrieval", "agents"],
        )


if __name__ == "__main__":
    unittest.main()
