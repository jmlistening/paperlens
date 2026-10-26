"""Offline tests for the frontier push state ledger."""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from push_state import PushState, paper_key


class PushStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = PushState(Path(self.tmp.name) / "frontier.sqlite3")
        self.state.init_schema()

    def tearDown(self):
        self.tmp.cleanup()

    def test_paper_key_strips_version(self):
        self.assertEqual(paper_key("2601.12345v2"), "2601.12345")
        self.assertEqual(paper_key("2601.12345"), "2601.12345")
        with self.assertRaises(ValueError):
            paper_key("")

    def test_claim_slot_is_unique(self):
        self.assertTrue(self.state.claim_slot("2026-10-25", "scheduled"))
        first = self.state.last_run_id
        self.assertTrue(first)
        self.assertFalse(self.state.claim_slot("2026-10-25", "scheduled"))
        self.assertEqual(self.state.last_run_id, first)

    def test_record_paper_dedups_versions_without_downgrading_status(self):
        self.state.record_paper(
            {"arxiv_id": "2601.12345v1", "title": "A", "abstract": "x",
             "published_at": "2026-01-01T00:00:00+00:00"}, "queued")
        self.state.record_paper(
            {"arxiv_id": "2601.12345v2", "title": "B", "abstract": "y",
             "published_at": "2026-01-02T00:00:00+00:00"}, "new")
        self.assertTrue(self.state.is_seen("2601.12345"))
        rows = self.state.eligible(10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "B")
        self.assertEqual(rows[0]["status"], "queued")

    def test_delivered_and_ignored_are_not_eligible(self):
        self.state.record_paper({"arxiv_id": "2601.1v1", "title": "A"}, "queued")
        self.state.record_paper({"arxiv_id": "2601.2v1", "title": "B"}, "queued")
        self.state.mark_status("2601.1", "delivered")
        self.state.mark_status("2601.2", "ignored")
        self.assertEqual(self.state.eligible(10), [])
        counts = self.state.status_counts()
        self.assertEqual(counts.get("delivered"), 1)
        self.assertEqual(counts.get("ignored"), 1)

    def test_eligible_prioritizing_puts_fresh_keys_first(self):
        for key in ("2601.1", "2601.2", "2601.3"):
            self.state.record_paper(
                {"arxiv_id": f"{key}v1", "title": key}, "queued"
            )
        rows = self.state.eligible_prioritizing(
            ["2601.3", "2601.3", "2601.1"], 2
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            sorted(row["paper_key"] for row in rows), ["2601.1", "2601.3"]
        )

    def test_eligible_prioritizing_keeps_ineligible_fresh_keys_out(self):
        for key in ("2601.1", "2601.2", "2601.3"):
            self.state.record_paper(
                {"arxiv_id": f"{key}v1", "title": key}, "queued"
            )
        self.state.mark_status("2601.1", "delivered")
        self.state.mark_status("2601.2", "ignored")
        rows = self.state.eligible_prioritizing(["2601.1", "2601.2"], 10)
        self.assertEqual([row["paper_key"] for row in rows], ["2601.3"])

    def test_eligible_prioritizing_without_keys_matches_eligible(self):
        for key in ("2601.1", "2601.2"):
            self.state.record_paper(
                {"arxiv_id": f"{key}v1", "title": key}, "queued"
            )
        self.assertEqual(
            [
                row["paper_key"]
                for row in self.state.eligible_prioritizing([], 10)
            ],
            [row["paper_key"] for row in self.state.eligible(10)],
        )

    def test_watermark_is_explicit(self):
        self.assertIsNone(self.state.get_watermark("frontier"))
        self.state.set_watermark("frontier", "2026-10-25T00:00:00+00:00")
        self.assertEqual(
            self.state.get_watermark("frontier"), "2026-10-25T00:00:00+00:00"
        )

    def test_selected_rows_keep_assessment_and_scores(self):
        self.state.record_paper({"arxiv_id": "2601.1v1", "title": "A"}, "queued")
        self.state.mark_status(
            "2601.1", "selected", '{"reason":"ok"}',
            llm_score=88.0, final_score=90.0,
        )
        rows = self.state.selected(10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["detail"], '{"reason":"ok"}')
        self.assertEqual(rows[0]["llm_score"], 88.0)
        self.assertEqual(rows[0]["final_score"], 90.0)

    def test_finish_run_records_status_and_counts(self):
        self.state.claim_slot("2026-10-25", "preview")
        self.state.finish_run(
            self.state.last_run_id, "completed", {"selected": 3}
        )
        self.assertEqual(self.state.run_status_counts().get("completed"), 1)

    def test_settings_are_explicit(self):
        self.assertIsNone(self.state.get_setting("first_send_approved_at"))
        self.state.set_setting("first_send_approved_at", "2026-10-25T00:00:00+00:00")
        self.assertEqual(
            self.state.get_setting("first_send_approved_at"),
            "2026-10-25T00:00:00+00:00",
        )

    def test_impressions_persist(self):
        self.state.claim_slot("2026-10-25", "preview")
        run_id = self.state.last_run_id
        self.state.record_paper({"arxiv_id": "2601.1v1", "title": "A"}, "evaluated")
        self.state.record_impression(run_id, "2601.1", 1, 88.5, "preview")
        self.assertEqual(self.state.impression_count(), 1)

    def test_reap_stale_running_runs(self):
        self.state.claim_slot(
            "2026-10-25", "scheduled",
            started_at="2026-10-25T00:00:00+00:00",
        )
        now = datetime(2026, 10, 25, 4, 0, tzinfo=timezone.utc)
        self.assertEqual(self.state.reap_stale(now, max_age_hours=2), 1)
        counts = self.state.run_status_counts()
        self.assertEqual(counts.get("interrupted"), 1)

    # -- lookback requeue -------------------------------------------------

    def _paper_rows(self) -> dict[str, dict]:
        conn = sqlite3.connect(self.state.path)
        conn.row_factory = sqlite3.Row
        try:
            return {
                row["paper_key"]: dict(row)
                for row in conn.execute("SELECT * FROM papers")
            }
        finally:
            conn.close()

    def _seed(self, arxiv_id: str, status: str, published: str) -> None:
        self.state.record_paper(
            {"arxiv_id": f"{arxiv_id}v1", "title": arxiv_id,
             "published": published},
            "queued",
        )
        self.state.mark_status(arxiv_id, status)

    def test_requeue_for_lookback_only_reopens_recent_evaluated_skipped_selected(
        self,
    ):
        now = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
        recent = (now - timedelta(days=1)).isoformat()
        old = (now - timedelta(days=10)).isoformat()
        statuses = {
            "2601.00001": ("evaluated", recent),
            "2601.00002": ("skipped", recent),
            "2601.00003": ("selected", recent),
            "2601.00004": ("delivered", recent),
            "2601.00005": ("queued", recent),
            "2601.00006": ("new", recent),
            "2601.00007": ("ignored", recent),
            "2601.00008": ("evaluated", old),
        }
        for arxiv_id, (status, published) in statuses.items():
            self._seed(arxiv_id, status, published)
        # A selected row carries a detail payload; requeueing must clear it.
        self.state.mark_status(
            "2601.00003", "selected", '{"reason":"old topic"}',
            llm_score=91.0, final_score=93.0,
        )

        changed = self.state.requeue_for_lookback(now - timedelta(days=7))

        self.assertEqual(changed, 3)
        rows = self._paper_rows()
        for arxiv_id in ("2601.00001", "2601.00002", "2601.00003"):
            self.assertEqual(rows[arxiv_id]["status"], "queued", arxiv_id)
            self.assertEqual(rows[arxiv_id]["detail"], "", arxiv_id)
        for arxiv_id, status in (
            ("2601.00004", "delivered"),
            ("2601.00005", "queued"),
            ("2601.00006", "new"),
            ("2601.00007", "ignored"),
            ("2601.00008", "evaluated"),
        ):
            self.assertEqual(rows[arxiv_id]["status"], status, arxiv_id)

    def test_requeue_for_lookback_without_since_reopens_every_matching_row(self):
        now = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
        old = (now - timedelta(days=90)).isoformat()
        self._seed("2601.00001", "evaluated", old)
        self._seed("2601.00002", "selected", old)
        self._seed("2601.00003", "delivered", old)

        changed = self.state.requeue_for_lookback()

        self.assertEqual(changed, 2)
        rows = self._paper_rows()
        self.assertEqual(rows["2601.00001"]["status"], "queued")
        self.assertEqual(rows["2601.00002"]["status"], "queued")
        self.assertEqual(rows["2601.00003"]["status"], "delivered")

    def test_record_paper_persists_the_normalized_published_field(self):
        published = "2026-10-24T09:30:00+00:00"
        self.state.record_paper(
            {"arxiv_id": "2601.00001v1", "title": "A", "published": published},
            "queued",
        )

        row = self.state.eligible(10)[0]

        self.assertEqual(row["published_at"], published)

    def test_record_paper_prefers_an_explicit_published_at(self):
        explicit = "2026-10-23T00:00:00+00:00"
        self.state.record_paper(
            {
                "arxiv_id": "2601.00001v1",
                "title": "A",
                "published_at": explicit,
                "published": "2026-10-24T00:00:00+00:00",
            },
            "queued",
        )

        self.assertEqual(self.state.eligible(10)[0]["published_at"], explicit)


if __name__ == "__main__":
    unittest.main()
