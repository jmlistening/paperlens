"""Tests for Task E1: held-out corpus selection.

These tests use synthetic catalogs only. They never read or modify ``data/``.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.benchmark_v2.select_corpus import (  # noqa: E402
    DEV_FAMILIES,
    SelectionError,
    build_catalog,
    build_splits,
    normalized_title,
    render_corpus_card,
    select_test_papers,
    title_jaccard,
    topic_score,
)

RECENT_N = 24
HISTORY_N = 16


def make_paper(
    family: str,
    *,
    arxiv_id: str | None = None,
    title: str | None = None,
    year: int = 2026,
    relevance: str = "core",
    source_batch: str = "dataset_v1",
    flags: list[str] | None = None,
    status: str = "prepared",
    exclude_from_test: bool = False,
    screening_status: str = "screened_pending_review",
    screening_missing: bool = False,
    screening_topics: list[str] | None = None,
    pdf_exists: bool = True,
) -> dict:
    return {
        "arxiv_id": arxiv_id or f"{family}v1",
        "family": family,
        "family_id": family,
        "title": title or f"Synthetic paper {family}",
        "year": year,
        "published": f"{year}-01-01T00:00:00Z",
        "relevance": relevance,
        "source_batch": source_batch,
        "flags": list(flags or []),
        "quality_flags": list(flags or []),
        "status": status,
        "exclude_from_test": exclude_from_test,
        "screening_status": screening_status,
        "screening_missing": screening_missing,
        "screening_topics": list(screening_topics or []),
        "pdf_exists": pdf_exists,
        "pdf_path": f"data/{source_batch}/pdfs/{arxiv_id or family + 'v1'}.pdf",
        "prepared_path": f"data/{source_batch}/prepared_v1/pages/{family}.jsonl",
        "license_status": "not_verified",
    }


def base_catalog() -> list[dict]:
    """A large synthetic catalog with enough eligible rows for 24 + 16."""
    papers: list[dict] = []

    # The 8 real dev families must never leak into test.
    for index, family in enumerate(sorted(DEV_FAMILIES)):
        papers.append(
            make_paper(
                family,
                title=f"Development-only paper number {index}",
                exclude_from_test=True,
            )
        )

    # 30 recent core + 20 recent adjacent.
    for index in range(30):
        papers.append(
            make_paper(
                f"2601.{index:05d}",
                title=f"Recent core retrieval method number {index}",
            )
        )
    for index in range(20):
        papers.append(
            make_paper(
                f"2602.{index:05d}",
                title=f"Adjacent survey topic number {index}",
                relevance="adjacent",
            )
        )

    # 60 historical papers: 10 for each year 2020-2025, all unscreened.
    for year in range(2020, 2026):
        for index in range(10):
            family = f"{year % 100:02d}01.{index:05d}"
            papers.append(
                make_paper(
                    family,
                    title=f"Historical paper from {year} number {index}",
                    year=year,
                    relevance="unscreened",
                    source_batch="dataset_history_v1",
                    screening_status="missing",
                    screening_missing=True,
                )
            )

    # Disallowed recent records.
    papers.extend(
        [
            make_paper("2603.90001", title="Out of scope paper", relevance="out_of_scope"),
            make_paper("2603.90002", title="Uncertain paper", relevance="uncertain"),
            make_paper(
                "2603.90003",
                title="Flagged low text paper",
                flags=["some_low_text_pages"],
            ),
            make_paper(
                "2603.90004",
                title="Unavailable source paper",
                status="source_unavailable",
            ),
            make_paper(
                "2603.90005",
                title="Recently unscreened paper",
                relevance="missing",
                screening_status="missing",
                screening_missing=True,
            ),
            make_paper(
                "2603.90006",
                title="Invalid screening response paper",
                relevance="missing",
                screening_status="invalid_response_pending_review",
                screening_missing=True,
            ),
            make_paper("2603.90007", title="Missing PDF paper", pdf_exists=False),
            make_paper("1912.00001", title="Too old for the split", year=2019),
        ]
    )
    return papers


class ExclusionTests(unittest.TestCase):
    def test_excludes_dev_families_and_disallowed_records(self) -> None:
        catalog = base_catalog()
        selected = select_test_papers(catalog, DEV_FAMILIES, RECENT_N, HISTORY_N)
        families = {row["family"] for row in selected}
        ids = {row["arxiv_id"] for row in selected}

        self.assertTrue(families.isdisjoint(DEV_FAMILIES))
        for bad_id in (
            "2603.90001v1",  # out_of_scope
            "2603.90002v1",  # uncertain
            "2603.90003v1",  # quality flag
            "2603.90004v1",  # source unavailable
            "2603.90005v1",  # recent screening missing
            "2603.90006v1",  # invalid screening response
            "2603.90007v1",  # missing PDF
            "1912.00001v1",  # year outside 2020-2026
        ):
            self.assertNotIn(bad_id, ids)

    def test_selected_records_have_no_exclusion_reason(self) -> None:
        selected = select_test_papers(base_catalog(), DEV_FAMILIES, RECENT_N, HISTORY_N)
        for row in selected:
            self.assertIsNone(row.get("exclusion_reason"))
            self.assertEqual(row.get("split"), "test")

    def test_chinese_and_punctuation_titles_normalize(self) -> None:
        self.assertEqual(normalized_title("RAG: A Survey!"), "rag a survey")
        self.assertEqual(title_jaccard("RAG: A Survey!", "RAG a survey"), 1.0)
        self.assertEqual(title_jaccard("", ""), 1.0)
        self.assertEqual(title_jaccard("abc", ""), 0.0)


class CountTests(unittest.TestCase):
    def test_selects_exactly_24_recent_and_16_historical(self) -> None:
        selected = select_test_papers(base_catalog(), DEV_FAMILIES, RECENT_N, HISTORY_N)
        recent = [row for row in selected if row["year"] == 2026]
        historical = [row for row in selected if 2020 <= row["year"] <= 2025]
        self.assertEqual(len(selected), RECENT_N + HISTORY_N)
        self.assertEqual(len(recent), RECENT_N)
        self.assertEqual(len(historical), HISTORY_N)

    def test_families_are_unique(self) -> None:
        selected = select_test_papers(base_catalog(), DEV_FAMILIES, RECENT_N, HISTORY_N)
        families = [row["family"] for row in selected]
        self.assertEqual(len(families), len(set(families)))

    def test_recent_selection_prefers_core_three_to_two(self) -> None:
        selected = select_test_papers(base_catalog(), DEV_FAMILIES, RECENT_N, HISTORY_N)
        recent = [row for row in selected if row["year"] == 2026]
        core = sum(1 for row in recent if row["relevance"] == "core")
        adjacent = sum(1 for row in recent if row["relevance"] == "adjacent")
        self.assertEqual(core + adjacent, RECENT_N)
        self.assertGreaterEqual(core, adjacent)
        # Target is roughly 3:2, allowing one row of slack.
        self.assertLessEqual(abs(core - round(RECENT_N * 0.6)), 1)

    def test_historical_selection_spans_multiple_years(self) -> None:
        selected = select_test_papers(base_catalog(), DEV_FAMILIES, RECENT_N, HISTORY_N)
        years = {row["year"] for row in selected if row["year"] != 2026}
        self.assertEqual(years, {2020, 2021, 2022, 2023, 2024, 2025})

    def test_deterministic_for_same_seed(self) -> None:
        first = select_test_papers(base_catalog(), DEV_FAMILIES, RECENT_N, HISTORY_N, seed=7)
        second = select_test_papers(base_catalog(), DEV_FAMILIES, RECENT_N, HISTORY_N, seed=7)
        self.assertEqual(
            [row["arxiv_id"] for row in first],
            [row["arxiv_id"] for row in second],
        )

    def test_pool_too_small_raises_and_does_not_shrink(self) -> None:
        small = [
            make_paper(f"2604.{index:05d}", title=f"Only recent paper {index}")
            for index in range(10)
        ] + [
            make_paper(
                f"2001.{index:05d}",
                title=f"Only historical paper {index}",
                year=2020,
                relevance="unscreened",
                source_batch="dataset_history_v1",
                screening_status="missing",
                screening_missing=True,
            )
            for index in range(10)
        ]
        with self.assertRaises(SelectionError) as ctx:
            select_test_papers(small, DEV_FAMILIES, RECENT_N, HISTORY_N)
        message = str(ctx.exception).lower()
        self.assertIn("insufficient", message)
        self.assertIn(str(RECENT_N), message)


class NearDuplicateTests(unittest.TestCase):
    def test_near_duplicate_title_is_rejected_and_recorded(self) -> None:
        papers = []
        duplicate_title = "A Study of Retrieval Augmented Generation for Question Answering"
        for index in range(23):
            papers.append(
                make_paper(
                    f"2605.{index:05d}",
                    title=f"Unrelated subject {index} about {index * 7} widgets",
                )
            )
        papers.append(make_paper("2605.90001", title=duplicate_title))
        papers.append(
            make_paper("2605.90002", title=duplicate_title + " Systems")
        )
        for index in range(20):
            papers.append(
                make_paper(
                    f"2002.{index:05d}",
                    title=f"Historical alternate paper {index}",
                    year=2020,
                    relevance="unscreened",
                    source_batch="dataset_history_v1",
                    screening_status="missing",
                    screening_missing=True,
                )
            )

        collisions: list[dict] = []
        selected = select_test_papers(
            papers,
            DEV_FAMILIES,
            RECENT_N,
            HISTORY_N,
            collisions=collisions,
        )
        self.assertEqual(len(selected), RECENT_N + HISTORY_N)
        selected_titles = [row["title"] for row in selected]
        self.assertFalse(
            duplicate_title in selected_titles
            and (duplicate_title + " Systems") in selected_titles
        )
        for left in range(len(selected)):
            for right in range(left + 1, len(selected)):
                self.assertLess(
                    title_jaccard(selected[left]["title"], selected[right]["title"]),
                    0.8,
                )
        duplicate_events = [
            item for item in collisions if item["reason"] == "near_duplicate_title"
        ]
        self.assertTrue(duplicate_events)
        self.assertIn("jaccard", duplicate_events[0])
        self.assertGreaterEqual(duplicate_events[0]["jaccard"], 0.8)


class TopicScoreTests(unittest.TestCase):
    def test_rag_topics_outrank_domain_only_papers(self):
        rag = make_paper(
            "2601.10001", relevance="adjacent",
            screening_topics=["rag", "domain_application"],
        )
        domain = make_paper(
            "2601.10002", relevance="adjacent",
            screening_topics=["domain_application", "llm_agent"],
        )
        self.assertGreater(topic_score(rag), topic_score(domain))

    def test_unscreened_has_no_topic_boost(self):
        unscreened = make_paper(
            "2601.10003", relevance="unscreened",
            screening_status="missing", screening_missing=True,
        )
        rag = make_paper("2601.10004", screening_topics=["rag"])
        self.assertLess(topic_score(unscreened), topic_score(rag))


class ScreeningMissingTests(unittest.TestCase):
    def test_unscreened_history_is_selected_but_marked_review_required(self) -> None:
        selected = select_test_papers(base_catalog(), DEV_FAMILIES, RECENT_N, HISTORY_N)
        historical = [row for row in selected if row["year"] != 2026]
        self.assertTrue(historical)
        for row in historical:
            self.assertEqual(row["relevance"], "unscreened")
            self.assertTrue(row["screening_missing"])
            self.assertTrue(row["review_required"])
        for row in selected:
            if row["year"] == 2026:
                self.assertFalse(row["screening_missing"])
                self.assertIn(row["relevance"], {"core", "adjacent"})


class BuildSplitsTests(unittest.TestCase):
    def test_splits_have_required_sections_and_hash(self) -> None:
        splits = build_splits(base_catalog(), DEV_FAMILIES, RECENT_N, HISTORY_N, seed=20261025)
        self.assertEqual(splits["status"], "DRAFT_PENDING_HUMAN_REVIEW")
        self.assertEqual(len(splits["test"]), RECENT_N + HISTORY_N)
        self.assertEqual(len(splits["dev"]), 8)
        self.assertGreater(len(splits["reserve"]), 0)
        self.assertGreater(len(splits["excluded"]), 0)
        self.assertEqual(len(splits["manifest_hash"]), 64)
        self.assertEqual(splits["counts"]["total"], RECENT_N + HISTORY_N)
        self.assertEqual(splits["counts"]["recent"], RECENT_N)
        self.assertEqual(splits["counts"]["history"], HISTORY_N)
        self.assertTrue(splits["blockers"])
        self.assertEqual(splits["distributions"]["year"]["2026"], RECENT_N)
        # Every excluded/reserve row must explain why it is not in test.
        for row in splits["excluded"] + splits["reserve"]:
            self.assertTrue(row["exclusion_reason"])

    def test_manifest_hash_is_deterministic(self) -> None:
        first = build_splits(base_catalog(), DEV_FAMILIES, RECENT_N, HISTORY_N, seed=11)
        second = build_splits(base_catalog(), DEV_FAMILIES, RECENT_N, HISTORY_N, seed=11)
        self.assertEqual(first["manifest_hash"], second["manifest_hash"])

    def test_dev_section_lists_all_eight_families(self) -> None:
        splits = build_splits(base_catalog(), DEV_FAMILIES, RECENT_N, HISTORY_N)
        self.assertEqual(
            {row["family"] for row in splits["dev"]},
            set(DEV_FAMILIES),
        )

    def test_corpus_card_is_draft_and_marks_missing_screening(self) -> None:
        splits = build_splits(base_catalog(), DEV_FAMILIES, RECENT_N, HISTORY_N)
        card = render_corpus_card(splits)
        self.assertIn("DRAFT_PENDING_HUMAN_REVIEW", card)
        for row in splits["test"]:
            self.assertIn(row["arxiv_id"], card)
        self.assertEqual(card.count("| MISSING |"), HISTORY_N)
        self.assertIn("rename `splits.draft.json`", card)


class BuildCatalogTests(unittest.TestCase):
    def test_build_catalog_reads_manifests_and_screening_read_only(self) -> None:
        tmp_base = Path(__file__).resolve().parent / ".tmp"
        tmp_base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_base) as tmp:
            root = Path(tmp)
            v1 = root / "data" / "dataset_v1"
            hist = root / "data" / "dataset_history_v1"
            for directory in (
                v1 / "prepared_v1" / "screening_v1",
                v1 / "pdfs",
                v1 / "prepared_v1" / "pages",
                hist / "pdfs",
                hist / "prepared_v1" / "pages",
                root / "docs/stage1c/20261023_screen_remaining_v3/output",
            ):
                directory.mkdir(parents=True, exist_ok=True)

            def manifest_paper(family, arxiv_id, year, flags, exclude, status="prepared"):
                return {
                    "family_id": family,
                    "arxiv_id": arxiv_id,
                    "title": f"Fixture paper {family}",
                    "abstract": "Fixture abstract.",
                    "published": f"{year}-10-20T15:26:15Z",
                    "flags": flags,
                    "exclude_from_test": exclude,
                    "status": status,
                    "pages_file": f"pages/{arxiv_id}.jsonl",
                    "license_status": "not_verified",
                    "query_matches": ["rag"],
                }

            good = manifest_paper("2601.00001", "2601.00001v1", 2026, [], False)
            bad = manifest_paper("2601.00002", "2601.00002v1", 2026, [], False)
            flagged = manifest_paper("2601.00003", "2601.00003v1", 2026, ["some_low_text_pages"], False)
            dev = manifest_paper("2610.19819", "2610.19819v1", 2026, [], True)
            historical = manifest_paper("2101.00001", "2101.00001v1", 2021, [], False)

            (v1 / "prepared_v1" / "manifest.json").write_text(
                json.dumps({"summary": {}, "papers": [good, bad, flagged, dev]}),
                encoding="utf-8",
            )
            (hist / "prepared_v1" / "manifest.json").write_text(
                json.dumps({"summary": {}, "papers": [historical]}),
                encoding="utf-8",
            )
            screening = {
                "settings": {},
                "results": {
                    "2601.00001v1": {
                        "paper_id": "2601.00001v1",
                        "status": "screened_pending_review",
                        "suggestion": {"relevance": "core", "topics": [], "reason": "", "evidence": {}},
                    },
                    "2601.00002v1": {
                        "paper_id": "2601.00002v1",
                        "status": "screened_pending_review",
                        "suggestion": {"relevance": "out_of_scope", "topics": [], "reason": "", "evidence": {}},
                    },
                    "2610.19819v1": {
                        "paper_id": "2610.19819v1",
                        "status": "screened_pending_review",
                        "suggestion": {"relevance": "core", "topics": [], "reason": "", "evidence": {}},
                    },
                },
            }
            (v1 / "prepared_v1" / "screening_v1" / "state.json").write_text(
                json.dumps(screening), encoding="utf-8"
            )
            history_screening = {
                "results": {
                    "2101.00001v1": {
                        "paper_id": "2101.00001v1",
                        "classification_status": "valid",
                        "suggestion": {
                            "relevance": "core",
                            "topics": ["rag"],
                            "reason": "",
                            "evidence": {},
                        },
                    }
                }
            }
            history_state_path = (
                root
                / "docs/stage1c/20261023_screen_remaining_v3/output/state.json"
            )
            history_state_path.write_text(
                json.dumps(history_screening), encoding="utf-8"
            )
            for relative in (
                "data/dataset_v1/pdfs/2601.00001v1.pdf",
                "data/dataset_v1/pdfs/2601.00002v1.pdf",
                "data/dataset_v1/pdfs/2601.00003v1.pdf",
                "data/dataset_v1/pdfs/2610.19819v1.pdf",
                "data/dataset_history_v1/pdfs/2101.00001v1.pdf",
            ):
                (root / relative).write_bytes(b"")

            catalog = build_catalog(root)
            by_id = {row["arxiv_id"]: row for row in catalog}
            self.assertEqual(len(catalog), 5)

            self.assertEqual(by_id["2601.00001v1"]["relevance"], "core")
            self.assertIsNone(by_id["2601.00001v1"]["exclusion_reason"])
            self.assertEqual(by_id["2601.00001v1"]["source_batch"], "dataset_v1")
            self.assertEqual(
                by_id["2601.00001v1"]["pdf_path"],
                "data/dataset_v1/pdfs/2601.00001v1.pdf",
            )
            self.assertEqual(
                by_id["2601.00001v1"]["prepared_path"],
                "data/dataset_v1/prepared_v1/pages/2601.00001v1.jsonl",
            )

            self.assertEqual(
                by_id["2601.00002v1"]["exclusion_reason"], "screening_out_of_scope"
            )
            self.assertTrue(
                by_id["2601.00003v1"]["exclusion_reason"].startswith("quality_flag")
            )
            self.assertEqual(by_id["2610.19819v1"]["exclusion_reason"], "dev_family")

            historical_row = by_id["2101.00001v1"]
            self.assertEqual(historical_row["relevance"], "core")
            self.assertFalse(historical_row["screening_missing"])
            self.assertFalse(historical_row["review_required"])
            self.assertEqual(historical_row["screening_topics"], ["rag"])
            self.assertIsNone(historical_row["exclusion_reason"])

            # If the stage1c history output is absent, the record must remain
            # explicitly unscreened rather than being guessed.
            history_state_path.unlink()
            catalog_without_history_screening = {
                row["arxiv_id"]: row for row in build_catalog(root)
            }
            fallback = catalog_without_history_screening["2101.00001v1"]
            self.assertEqual(fallback["relevance"], "unscreened")
            self.assertTrue(fallback["screening_missing"])
            self.assertTrue(fallback["review_required"])


if __name__ == "__main__":
    unittest.main()
