"""Offline tests for the incremental push fetch window (Task P3).

Everything here runs against fake arXiv results: no network, no real sleeps,
no paid API. Cooldown state is written only inside a temporary directory.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fetch_papers import CooldownActive  # noqa: E402
from push_fetch import (  # noqa: E402
    RateLimited,
    build_queries,
    compute_window,
    fetch_candidates,
)
from push_state import PushState  # noqa: E402

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"
NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)

PROFILE = {
    "search": {
        "lookback_days": 30,
        "max_candidates": 10,
        "per_query_limit": 10,
        "groups": [
            {
                "name": "RAG",
                "any_terms": ["retrieval augmented generation"],
                "require_any_terms": [],
            },
            {
                "name": "Agent",
                "any_terms": ["agent", "agents"],
                "require_any_terms": ["large language model", "LLM"],
            },
        ],
    }
}


def config(**overrides):
    values = {
        "window_hours": 72,
        "overlap_hours": 24,
        "catchup_days": 7,
        "fetch_pool_limit": 40,
        "search_categories": (),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeHttpError(Exception):
    def __init__(self, status=429):
        super().__init__("fake arXiv HTTP 429")
        self.status = status


class FakeResult:
    def __init__(
        self,
        arxiv_id: str,
        published: datetime,
        *,
        title: str | None = None,
        summary: str | None = None,
        categories=("cs.CL",),
    ):
        self._id = arxiv_id
        self.title = title or f"Title {arxiv_id}"
        self.summary = summary or f"Abstract {arxiv_id}"
        self.authors = [SimpleNamespace(name="Fixture Author")]
        self.categories = list(categories)
        self.published = published
        self.updated = published

    def get_short_id(self) -> str:
        return self._id


class FakeClient:
    """Minimal stand-in for arxiv.Client used by fetch_candidates.

    Every query receives the same configured result list. The request objects
    are recorded so tests can inspect the query text and result limit.
    """

    def __init__(self, results=None, error: Exception | None = None):
        self._results = list(results or [])
        self._error = error
        self.requests = []

    def results(self, request):
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return iter(self._results)


class FetchTestCase(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self._tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT, prefix="push_fetch_")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data_dir = self.tmp / "data"
        self.data_dir.mkdir()
        self.state = PushState(self.tmp / "frontier.sqlite3")
        self.state.init_schema()

    def result(self, arxiv_id, offset_hours=0):
        return FakeResult(arxiv_id, NOW + timedelta(hours=offset_hours))


class ComputeWindowTests(FetchTestCase):
    def test_first_run_uses_initial_lookback_days(self):
        start, end = compute_window(
            NOW, None, config(initial_lookback_days=10)
        )
        self.assertEqual(end, NOW)
        self.assertEqual(start, NOW - timedelta(days=10))

    def test_first_run_without_field_defaults_to_at_least_seven_days(self):
        start, end = compute_window(NOW, None, config(window_hours=72))
        self.assertEqual(end, NOW)
        self.assertEqual(start, NOW - timedelta(days=7))

    def test_later_run_uses_later_of_overlap_and_catchup(self):
        # watermark 3 days ago, overlap 24h -> now-4d; catchup 7d -> now-7d.
        watermark = (NOW - timedelta(days=3)).isoformat()
        start, end = compute_window(NOW, watermark, config())
        self.assertEqual(end, NOW)
        self.assertEqual(start, NOW - timedelta(days=4))

    def test_later_run_never_reaches_farther_than_catchup(self):
        watermark = (NOW - timedelta(days=30)).isoformat()
        start, _ = compute_window(NOW, watermark, config(catchup_days=7))
        self.assertEqual(start, NOW - timedelta(days=7))

    def test_naive_watermark_is_treated_as_utc(self):
        watermark = (NOW - timedelta(days=3)).replace(tzinfo=None).isoformat()
        start, _ = compute_window(NOW, watermark, config())
        self.assertEqual(start, NOW - timedelta(days=4))

    def test_invalid_watermark_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_window(NOW, "not-a-timestamp", config())


class BuildQueriesTests(FetchTestCase):
    def test_queries_reuse_term_clauses_and_add_categories(self):
        queries = build_queries(PROFILE, ["cs.CL", "cs.AI"])
        self.assertEqual([q["name"] for q in queries], ["RAG", "Agent"])
        rag = queries[0]["query"]
        self.assertIn('ti:"retrieval augmented generation"', rag)
        self.assertIn("cat:cs.CL", rag)
        self.assertIn("cat:cs.AI", rag)
        self.assertIn(" OR ", rag)
        agent = queries[1]["query"]
        self.assertIn('ti:"large language model"', agent)
        self.assertIn('abs:"LLM"', agent)

    def test_no_categories_means_no_cat_clause(self):
        queries = build_queries(PROFILE, [])
        for query in queries:
            self.assertNotIn("cat:", query["query"])

    def test_invalid_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            build_queries({"search": {}}, [])


class FetchCandidatesTests(FetchTestCase):
    def test_filters_results_to_the_requested_window(self):
        client = FakeClient(
            [
                self.result("2601.00001v1", offset_hours=-20),
                self.result("2601.00002v1", offset_hours=-1),
                self.result("2601.00003v1", offset_hours=5),
            ]
        )
        start = NOW - timedelta(hours=24)
        papers = fetch_candidates(
            PROFILE, start, NOW, config(), client=client, now=NOW,
            data_dir=self.data_dir,
        )
        self.assertEqual(
            [p["arxiv_id"] for p in papers], ["2601.00001v1", "2601.00002v1"]
        )
        self.assertEqual(len(client.requests), 2)
        self.assertIn("submittedDate:[", client.requests[0].query)
        self.assertEqual(client.requests[0].max_results, 10)

    def test_categories_from_config_are_sent_to_arxiv(self):
        client = FakeClient([self.result("2601.00001v1")])
        fetch_candidates(
            PROFILE, NOW - timedelta(hours=24), NOW,
            config(search_categories=("cs.CL", "cs.LG")),
            client=client, now=NOW, data_dir=self.data_dir,
        )
        self.assertIn("cat:cs.CL", client.requests[0].query)
        self.assertIn("cat:cs.LG", client.requests[0].query)

    def test_fetch_pool_limit_truncates_and_queues_overflow(self):
        results = [self.result(f"2601.{i:05d}v1") for i in range(1, 6)]
        client = FakeClient(results)
        overflow = []
        papers = fetch_candidates(
            PROFILE, NOW - timedelta(hours=24), NOW,
            config(fetch_pool_limit=3),
            client=client, state=self.state, now=NOW,
            data_dir=self.data_dir, overflow_out=overflow,
        )
        self.assertEqual(len(papers), 3)
        self.assertEqual(len(overflow), 2)
        eligible = self.state.eligible(10)
        self.assertEqual(len(eligible), 2)
        self.assertTrue(all(row["status"] == "queued" for row in eligible))
        self.assertEqual(
            self.state.get_watermark("frontier"), NOW.isoformat()
        )

    def test_rate_limit_raises_typed_error_without_advancing_watermark(self):
        old = (NOW - timedelta(days=2)).isoformat()
        self.state.set_watermark("frontier", old)
        client = FakeClient(error=FakeHttpError(429))
        with self.assertRaises(RateLimited) as ctx:
            fetch_candidates(
                PROFILE, NOW - timedelta(hours=24), NOW, config(),
                client=client, state=self.state, now=NOW,
                data_dir=self.data_dir,
            )
        self.assertEqual(ctx.exception.status, 429)
        self.assertEqual(self.state.get_watermark("frontier"), old)
        self.assertTrue((self.data_dir / "arxiv_cooldown.json").is_file())

    def test_active_cooldown_prevents_any_client_call(self):
        (self.data_dir / "arxiv_cooldown.json").write_text(
            '{"retry_not_before_utc": "'
            + (NOW + timedelta(minutes=10)).isoformat()
            + '"}',
            encoding="utf-8",
        )
        client = FakeClient([self.result("2601.00001v1")])
        with self.assertRaises(CooldownActive):
            fetch_candidates(
                PROFILE, NOW - timedelta(hours=24), NOW, config(),
                client=client, now=NOW, data_dir=self.data_dir,
            )
        self.assertEqual(client.requests, [])

    def test_success_advances_watermark_only_when_state_is_given(self):
        client = FakeClient([self.result("2601.00001v1")])
        papers = fetch_candidates(
            PROFILE, NOW - timedelta(hours=24), NOW, config(),
            client=client, now=NOW, data_dir=self.data_dir,
        )
        self.assertEqual(len(papers), 1)
        self.assertIsNone(self.state.get_watermark("frontier"))

    def test_empty_results_keep_the_existing_watermark(self):
        """A zero-result fetch must not skip the gap to wall-clock now.

        arXiv can lag behind the local clock (weekends, indexing delay). If an
        empty window advanced the frontier to ``now``, the next run would skip
        every paper arXiv later publishes with an earlier submission date.
        """
        old = (NOW - timedelta(days=2)).isoformat()
        self.state.set_watermark("frontier", old)
        client = FakeClient([])
        papers = fetch_candidates(
            PROFILE, NOW - timedelta(hours=24), NOW, config(),
            client=client, state=self.state, now=NOW,
            data_dir=self.data_dir,
        )
        self.assertEqual(papers, [])
        self.assertEqual(self.state.get_watermark("frontier"), old)

    def test_watermark_follows_the_latest_publication(self):
        """Advance to the newest fetched paper, not to the local clock."""
        self.state.set_watermark(
            "frontier", (NOW - timedelta(days=2)).isoformat()
        )
        client = FakeClient(
            [
                self.result("2601.00001v1", offset_hours=-10),
                self.result("2601.00002v1", offset_hours=-3),
            ]
        )
        fetch_candidates(
            PROFILE, NOW - timedelta(hours=24), NOW, config(),
            client=client, state=self.state, now=NOW,
            data_dir=self.data_dir,
        )
        self.assertEqual(
            self.state.get_watermark("frontier"),
            (NOW - timedelta(hours=3)).isoformat(),
        )

    def test_duplicate_families_are_merged_across_groups(self):
        client = FakeClient(
            [
                FakeResult("2601.00001v1", NOW - timedelta(hours=1)),
                FakeResult("2601.00001v2", NOW - timedelta(hours=1)),
            ]
        )
        papers = fetch_candidates(
            PROFILE, NOW - timedelta(hours=24), NOW, config(),
            client=client, now=NOW, data_dir=self.data_dir,
        )
        self.assertEqual([p["arxiv_id"] for p in papers], ["2601.00001v2"])

    def test_inverted_window_is_rejected(self):
        with self.assertRaises(ValueError):
            fetch_candidates(
                PROFILE, NOW, NOW - timedelta(hours=1), config(),
                client=FakeClient(), now=NOW, data_dir=self.data_dir,
            )


class DefaultCooldownPathTests(unittest.TestCase):
    def test_default_cooldown_uses_shared_data_dir(self):
        from push_fetch import DEFAULT_COOLDOWN_DIR

        self.assertEqual(DEFAULT_COOLDOWN_DIR, ROOT / "data")


if __name__ == "__main__":
    unittest.main()
