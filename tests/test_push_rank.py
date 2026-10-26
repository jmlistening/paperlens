"""Offline tests for push ranking and digest selection (Task P4).

A fake ``rank_fn`` replaces the LLM and no embedding model is loaded. The
deterministic lexical fallback is exercised explicitly.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from push_rank import (  # noqa: E402
    cold_start,
    favorite_similarity,
    lexical_similarity,
    rank_batch,
    rank_cache_key,
    select_digest,
)

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"


def config(**overrides):
    values = {
        "min_score": 60,
        "max_per_digest": 3,
        "max_per_primary_category": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def paper(index: int, *, title=None, abstract="retrieval abstract", categories=("cs.CL",)):
    arxiv_id = f"2601.{index:05d}v1"
    return {
        "arxiv_id": arxiv_id,
        "title": title or f"Paper {index} about retrieval augmented generation",
        "abstract": abstract,
        "categories": list(categories),
        "url": f"https://arxiv.org/abs/{arxiv_id}",
    }


def ranked_item(index: int, *, llm_score, final_score=None, categories=("cs.CL",)):
    value = paper(index, categories=categories)
    value.update(
        {
            "llm_score": llm_score,
            "favorite_similarity": 0.0,
            "final_score": llm_score if final_score is None else final_score,
            "reason": f"reason {index}",
            "limitation": f"limitation {index}",
            "feedback_refs": [],
        }
    )
    return value


def make_rank_fn(scores, feedback_refs=None):
    calls = []

    def rank_fn(papers, profile, feedback):
        calls.append([p["arxiv_id"] for p in papers])
        rows = []
        for item in papers:
            arxiv_id = item["arxiv_id"]
            rows.append(
                {
                    "arxiv_id": arxiv_id,
                    "score": scores[arxiv_id],
                    "reason": f"reason for {arxiv_id}",
                    "limitation": "needs full text",
                    "feedback_refs": (feedback_refs or {}).get(arxiv_id, []),
                }
            )
        return rows

    return rank_fn, calls


class PushRankTestCase(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self._tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT, prefix="push_rank_")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.cache = self.tmp / "cache"

    @property
    def feedback(self):
        return {"favorites": [], "ignored_keys": []}


class ColdStartTests(PushRankTestCase):
    def test_cold_start_without_favorites(self):
        self.assertTrue(cold_start({}))
        self.assertTrue(cold_start(None))
        self.assertTrue(cold_start({"favorites": [], "ignored_keys": []}))

    def test_not_cold_start_with_favorites(self):
        self.assertFalse(
            cold_start({"favorites": [{"arxiv_id": "2601.00001v1"}], "ignored_keys": []})
        )


class RankBatchTests(PushRankTestCase):
    def test_cold_start_final_score_equals_llm_score(self):
        papers = [paper(1), paper(2)]
        rank_fn, calls = make_rank_fn(
            {"2601.00001v1": 88, "2601.00002v1": 42}
        )
        ranked = rank_batch(
            papers, {"min_score": 60}, self.feedback,
            {"model": "fake"}, self.cache, rank_fn=rank_fn,
        )
        self.assertEqual(len(ranked), 2)
        self.assertEqual(len(calls), 1)
        for item in ranked:
            self.assertEqual(item["final_score"], item["llm_score"])
            self.assertEqual(item["favorite_similarity"], 0.0)

    def test_ranked_items_have_required_fields_and_assessment(self):
        rank_fn, _ = make_rank_fn({"2601.00001v1": 77})
        ranked = rank_batch(
            [paper(1)], {}, self.feedback, {"model": "fake"}, self.cache,
            rank_fn=rank_fn,
        )
        item = ranked[0]
        for field in ("llm_score", "favorite_similarity", "final_score", "reason", "feedback_refs"):
            self.assertIn(field, item)
        self.assertEqual(item["llm_score"], 77)
        self.assertEqual(item["assessment"]["score"], 77)
        self.assertEqual(item["assessment"]["reason"], item["reason"])
        self.assertEqual(item["assessment"]["limitation"], item["limitation"])
        self.assertEqual(item["title"], paper(1)["title"])

    def test_dict_response_shape_is_accepted(self):
        def rank_fn(papers, profile, feedback):
            return {
                "evaluations": [
                    {
                        "arxiv_id": p["arxiv_id"],
                        "score": 66,
                        "reason": "r",
                        "limitation": "l",
                        "feedback_refs": [],
                    }
                    for p in papers
                ]
            }

        ranked = rank_batch(
            [paper(1)], {}, self.feedback, {"model": "fake"}, self.cache,
            rank_fn=rank_fn,
        )
        self.assertEqual(ranked[0]["llm_score"], 66)

    def test_favorite_weight_zero_equals_llm_only(self):
        favorite = {
            "arxiv_id": "2601.09999v1",
            "title": "Paper 1 about retrieval augmented generation",
            "abstract": "retrieval abstract",
        }
        feedback = {"favorites": [favorite], "ignored_keys": []}
        rank_fn, _ = make_rank_fn({"2601.00001v1": 73})
        ranked = rank_batch(
            [paper(1)], {"favorite_weight": 0.0}, feedback,
            {"model": "fake"}, self.cache, rank_fn=rank_fn,
            similarity_fn=lambda left, right: 1.0,
        )
        self.assertEqual(ranked[0]["final_score"], 73)
        self.assertEqual(ranked[0]["favorite_similarity"], 1.0)

    def test_favorite_weight_boosts_final_score(self):
        favorite = {"arxiv_id": "2601.09999v1", "title": "f", "abstract": "a"}
        feedback = {"favorites": [favorite], "ignored_keys": []}
        rank_fn, _ = make_rank_fn({"2601.00001v1": 70})
        ranked = rank_batch(
            [paper(1)], {"favorite_weight": 0.2}, feedback,
            {"model": "fake"}, self.cache, rank_fn=rank_fn,
            similarity_fn=lambda left, right: 0.5,
        )
        self.assertAlmostEqual(ranked[0]["final_score"], 80.0)

    def test_ignored_keys_are_not_sent_to_the_ranker(self):
        feedback = {"favorites": [], "ignored_keys": ["2601.00001"]}
        rank_fn, calls = make_rank_fn(
            {"2601.00001v1": 90, "2601.00002v1": 70}
        )
        ranked = rank_batch(
            [paper(1), paper(2)], {}, feedback, {"model": "fake"}, self.cache,
            rank_fn=rank_fn,
        )
        self.assertEqual([item["arxiv_id"] for item in ranked], ["2601.00002v1"])
        self.assertEqual(calls, [["2601.00002v1"]])

    def test_feedback_refs_are_preserved(self):
        favorite_id = "2601.09999v1"
        feedback = {
            "favorites": [{"arxiv_id": favorite_id, "title": "t", "abstract": "a"}],
            "ignored_keys": [],
        }
        rank_fn, _ = make_rank_fn(
            {"2601.00001v1": 80}, {"2601.00001v1": [favorite_id]}
        )
        ranked = rank_batch(
            [paper(1)], {"favorite_weight": 0.0}, feedback,
            {"model": "fake"}, self.cache, rank_fn=rank_fn,
            similarity_fn=lambda left, right: 0.0,
        )
        self.assertEqual(ranked[0]["feedback_refs"], [favorite_id])

    def test_cache_is_reused_and_invalidated_by_model_change(self):
        papers = [paper(1)]
        rank_fn, calls = make_rank_fn({"2601.00001v1": 81})
        first = rank_batch(
            papers, {"min_score": 60}, self.feedback,
            {"model": "model-a"}, self.cache, rank_fn=rank_fn,
        )
        second = rank_batch(
            papers, {"min_score": 60}, self.feedback,
            {"model": "model-a"}, self.cache, rank_fn=rank_fn,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(list(self.cache.glob("*.json"))), 1)

        third = rank_batch(
            papers, {"min_score": 60}, self.feedback,
            {"model": "model-b"}, self.cache, rank_fn=rank_fn,
        )
        self.assertEqual(third, first)
        self.assertEqual(len(calls), 2)

    def test_rank_cache_key_includes_profile_model_and_prompt(self):
        papers = [paper(1)]
        base = rank_cache_key(
            papers, {"min_score": 60}, {"model": "m"}, prompt_hash_value="prompt-a"
        )
        self.assertRegex(base, r"^[0-9a-f]{64}$")
        self.assertEqual(
            base,
            rank_cache_key(
                papers, {"min_score": 60}, {"model": "m"}, prompt_hash_value="prompt-a"
            ),
        )
        self.assertNotEqual(
            base,
            rank_cache_key(
                papers, {"min_score": 61}, {"model": "m"}, prompt_hash_value="prompt-a"
            ),
        )
        self.assertNotEqual(
            base,
            rank_cache_key(
                papers, {"min_score": 60}, {"model": "other"}, prompt_hash_value="prompt-a"
            ),
        )
        self.assertNotEqual(
            base,
            rank_cache_key(
                papers, {"min_score": 60}, {"model": "m"}, prompt_hash_value="prompt-b"
            ),
        )

    def test_default_prompt_hash_is_part_of_the_key(self):
        papers = [paper(1)]
        default = rank_cache_key(papers, {}, {"model": "m"})
        self.assertRegex(default, r"^[0-9a-f]{64}$")
        self.assertNotEqual(
            default, rank_cache_key(papers, {}, {"model": "m"}, prompt_hash_value="x")
        )


class SimilarityTests(PushRankTestCase):
    def test_lexical_similarity_is_deterministic(self):
        text = "retrieval augmented generation hybrid retrieval"
        self.assertEqual(lexical_similarity(text, text), 1.0)
        self.assertEqual(lexical_similarity("alpha beta", "gamma delta"), 0.0)
        self.assertEqual(
            lexical_similarity(text, "hybrid retrieval"), lexical_similarity(text, "hybrid retrieval")
        )

    def test_favorite_similarity_uses_max_and_excludes_self(self):
        target = paper(1)
        favorites = [
            # Same family as the candidate: must not be used as evidence.
            {"arxiv_id": "2601.00001v2", "title": target["title"], "abstract": target["abstract"]},
            {"arxiv_id": "2601.09999v1", "title": "unrelated words", "abstract": "nothing shared"},
        ]
        self.assertEqual(favorite_similarity(target, favorites), 0.0)

    def test_missing_embedding_model_falls_back_to_lexical_deterministically(self):
        target = paper(1, title="retrieval augmented generation", abstract="hybrid retrieval")
        favorite = {
            "arxiv_id": "2601.09999v1",
            "title": "retrieval augmented generation",
            "abstract": "hybrid retrieval",
        }
        model_config = {"embedding_model_path": str(self.tmp / "does-not-exist")}
        with self.assertWarns(RuntimeWarning):
            first = favorite_similarity(target, [favorite], model_config=model_config)
            second = favorite_similarity(target, [favorite], model_config=model_config)
        self.assertEqual(first, 1.0)
        self.assertEqual(first, second)
        self.assertEqual(
            first,
            lexical_similarity(
                f"{target['title']} {target['abstract']}",
                f"{favorite['title']} {favorite['abstract']}",
            ),
        )


class SelectDigestTests(PushRankTestCase):
    def test_min_score_applies_to_llm_score_not_final_score(self):
        items = [
            ranked_item(1, llm_score=50, final_score=95),
            ranked_item(2, llm_score=70, final_score=40),
        ]
        selected = select_digest(items, config(min_score=60), set())
        self.assertEqual([item["arxiv_id"] for item in selected], ["2601.00002v1"])

    def test_ignored_keys_are_hard_excluded(self):
        items = [ranked_item(1, llm_score=90), ranked_item(2, llm_score=80)]
        selected = select_digest(items, config(), {"2601.00001"})
        self.assertEqual([item["arxiv_id"] for item in selected], ["2601.00002v1"])

    def test_category_diversity_caps_per_primary_category(self):
        items = [ranked_item(i, llm_score=90 - i) for i in range(1, 6)]
        selected = select_digest(
            items, config(max_per_digest=5, max_per_primary_category=2), set()
        )
        self.assertEqual(len(selected), 2)

    def test_diversity_cap_is_per_category_not_global(self):
        items = [
            ranked_item(1, llm_score=90, categories=("cs.CL",)),
            ranked_item(2, llm_score=80, categories=("cs.CL",)),
            ranked_item(3, llm_score=70, categories=("cs.CL",)),
            ranked_item(4, llm_score=60, categories=("cs.LG",)),
            ranked_item(5, llm_score=50, categories=("cs.LG",)),
        ]
        selected = select_digest(
            items,
            config(min_score=0, max_per_digest=5, max_per_primary_category=2),
            set(),
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual(
            [item["arxiv_id"] for item in selected],
            ["2601.00001v1", "2601.00002v1", "2601.00004v1", "2601.00005v1"],
        )

    def test_selected_items_are_sorted_and_limited(self):
        items = [
            ranked_item(1, llm_score=80),
            ranked_item(2, llm_score=95),
            ranked_item(3, llm_score=70),
        ]
        selected = select_digest(items, config(max_per_digest=2), set())
        self.assertEqual([item["llm_score"] for item in selected], [95, 80])

    def test_selection_does_not_mutate_input(self):
        items = [ranked_item(1, llm_score=90)]
        before = [dict(item) for item in items]
        select_digest(items, config(), set())
        self.assertEqual(items, before)


if __name__ == "__main__":
    unittest.main()
