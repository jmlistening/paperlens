"""Offline tests for the private active profile resolver (Task 1)."""
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
    ACTIVE_PROFILE_REL,
    EXAMPLE_PROFILE,
    LEGACY_PROFILE,
    ProfileError,
    load_active_profile,
    profile_path,
    profile_sha256,
    save_active_profile,
    validate_profile,
)

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"

# Generic fixture only: no personal research goal, topics, or email address.
PROFILE = {
    "research_goal": "study generic machine learning methods",
    "topics": ["machine learning"],
    "preferred_methods": ["supervised learning"],
    "less_interested": [],
    "top_k": 3,
    "min_score": 60,
    "search": {
        "lookback_days": 30,
        "max_candidates": 12,
        "per_query_limit": 12,
        "groups": [
            {
                "name": "Example",
                "any_terms": ["machine learning"],
                "require_any_terms": [],
            }
        ],
    },
}


class ActiveProfileTests(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_json(self, path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_path_constants(self):
        self.assertEqual(ACTIVE_PROFILE_REL, Path("data/push/profile.json"))
        self.assertEqual(LEGACY_PROFILE, ROOT / "interests.json")
        self.assertEqual(EXAMPLE_PROFILE, ROOT / "interests.example.json")

    def test_active_profile_precedence(self):
        self.write_json(self.root / ACTIVE_PROFILE_REL, PROFILE)
        self.write_json(
            self.root / "interests.json", {**PROFILE, "top_k": 9}
        )
        self.assertEqual(
            profile_path(self.root), self.root / ACTIVE_PROFILE_REL
        )
        self.assertEqual(load_active_profile(self.root)["top_k"], 3)

    def test_legacy_profile_wins_over_example(self):
        self.write_json(self.root / "interests.json", {**PROFILE, "top_k": 9})
        self.write_json(
            self.root / "interests.example.json", {**PROFILE, "top_k": 1}
        )
        self.assertEqual(profile_path(self.root), self.root / "interests.json")
        self.assertEqual(load_active_profile(self.root)["top_k"], 9)

    def test_example_is_the_last_resort(self):
        self.write_json(
            self.root / "interests.example.json", {**PROFILE, "top_k": 1}
        )
        self.assertEqual(
            profile_path(self.root), self.root / "interests.example.json"
        )
        self.assertEqual(load_active_profile(self.root)["top_k"], 1)

    def test_save_writes_active_profile(self):
        saved = save_active_profile({**PROFILE, "top_k": 5}, self.root)
        active = self.root / ACTIVE_PROFILE_REL
        self.assertTrue(active.is_file())
        self.assertEqual(saved["top_k"], 5)
        self.assertEqual(load_active_profile(self.root)["top_k"], 5)
        self.assertFalse((self.root / "interests.json").exists())

    def test_save_rejects_invalid_profile(self):
        with self.assertRaises(ProfileError):
            save_active_profile({**PROFILE, "top_k": 0}, self.root)
        self.assertFalse((self.root / ACTIVE_PROFILE_REL).exists())

    def test_changing_active_does_not_touch_legacy(self):
        legacy = self.root / "interests.json"
        self.write_json(legacy, {**PROFILE, "top_k": 9})
        before = legacy.read_bytes()
        save_active_profile({**PROFILE, "top_k": 2}, self.root)
        self.assertEqual(legacy.read_bytes(), before)
        self.assertEqual(load_active_profile(self.root)["top_k"], 2)

    def test_public_example_profile_validates(self):
        profile = json.loads(EXAMPLE_PROFILE.read_text(encoding="utf-8"))
        validated = validate_profile(profile)
        self.assertEqual(validated["top_k"], 3)
        self.assertNotIn("@", EXAMPLE_PROFILE.read_text(encoding="utf-8"))

    def test_profile_sha256_is_canonical_and_content_sensitive(self):
        digest = profile_sha256(PROFILE)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        reordered = {key: PROFILE[key] for key in reversed(list(PROFILE))}
        self.assertEqual(digest, profile_sha256(reordered))
        self.assertNotEqual(digest, profile_sha256({**PROFILE, "top_k": 4}))


if __name__ == "__main__":
    unittest.main()
