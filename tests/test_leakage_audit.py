"""Tests for Task E3: leakage audit for the held-out split and test index.

Synthetic inputs only. Nothing here reads or modifies ``data/`` and no model or
network call is made.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.benchmark_v2.leakage_audit import (  # noqa: E402
    audit_leakage,
    main as leakage_main,
)

TMP_BASE = Path(__file__).resolve().parent / ".tmp"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def paper(
    arxiv_id: str,
    *,
    family: str | None = None,
    title: str | None = None,
    abstract: str | None = None,
    pdf_exists: bool = True,
    pdf_path: str | None = None,
    exclusion_reason: str | None = None,
    **extra,
) -> dict:
    family = family or arxiv_id.split("v")[0]
    record = {
        "arxiv_id": arxiv_id,
        "family": family,
        "family_id": family,
        "title": title if title is not None else f"Synthetic title {family}",
        "pdf_exists": pdf_exists,
        "pdf_path": pdf_path if pdf_path is not None else f"pdfs/{arxiv_id}.pdf",
        "exclusion_reason": exclusion_reason,
    }
    if abstract is not None:
        record["abstract"] = abstract
    record.update(extra)
    return record


def splits(
    *,
    dev: list[dict] | None = None,
    test: list[dict] | None = None,
    reserve: list[dict] | None = None,
    excluded: list[dict] | None = None,
) -> dict:
    return {
        "dev": list(dev or []),
        "test": list(test or []),
        "reserve": list(reserve or []),
        "excluded": list(excluded or []),
    }


def index_manifest(papers: list[dict]) -> dict:
    return {
        "paper_count": len(papers),
        "paper_ids": [item["arxiv_id"] for item in papers],
        "families": [item["family"] for item in papers],
        "papers": [
            {
                "paper_id": item["arxiv_id"],
                "family": item["family"],
                "title": item.get("title"),
            }
            for item in papers
        ],
    }


def codes(report: dict) -> set[str]:
    return {entry["code"] for entry in report["review_required"]}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class FamilyOverlapTests(unittest.TestCase):
    def test_injected_dev_test_family_overlap_is_detected(self) -> None:
        test_paper = paper(
            "1111.11111v2",
            family="1111.11111",
            title="A Completely Different Title About Sparse Retrieval",
        )
        report = audit_leakage(
            splits(
                dev=[paper("1111.11111v1", title="Alpha Study of Dense Retrieval")],
                test=[test_paper],
            ),
            corpus=[
                paper("1111.11111v1", title="Alpha Study of Dense Retrieval"),
                test_paper,
            ],
            index_manifest=index_manifest([test_paper]),
        )

        self.assertEqual(report["dev_test_family_overlap"], ["1111.11111"])
        self.assertIn("dev_test_family_overlap", codes(report))
        self.assertIn("dev_test_family_overlap", report["blockers"])
        self.assertEqual(report["status"], "blockers")

    def test_family_overlap_is_version_stripped(self) -> None:
        dev_paper = paper("2222.22222v3", family="2222.22222")
        test_paper = paper(
            "2222.22222v9",
            family="2222.22222",
            title="Independent Held Out Title",
        )
        report = audit_leakage(
            splits(dev=[dev_paper], test=[test_paper]),
            corpus=[dev_paper, test_paper],
            index_manifest=index_manifest([test_paper]),
        )

        self.assertEqual(report["dev_test_family_overlap"], ["2222.22222"])


class NearDuplicateTests(unittest.TestCase):
    def test_near_duplicate_title_is_flagged(self) -> None:
        dev_paper = paper(
            "3333.33333v1",
            title="Retrieval Augmented Generation for Scientific Document Question Answering",
        )
        test_paper = paper(
            "4444.44444v1",
            title="Retrieval-Augmented Generation for Scientific Document Question-Answering",
        )
        report = audit_leakage(
            splits(dev=[dev_paper], test=[test_paper]),
            corpus=[dev_paper, test_paper],
            index_manifest=index_manifest([test_paper]),
        )

        self.assertEqual(len(report["near_duplicate_title_pairs"]), 1)
        pair = report["near_duplicate_title_pairs"][0]
        self.assertGreaterEqual(pair["jaccard"], 0.8)
        self.assertEqual(
            sorted(pair["families"]), ["3333.33333", "4444.44444"]
        )
        self.assertEqual(pair["scope"], "dev_test")
        self.assertTrue(pair["cross_split"])
        self.assertIn("near_duplicate_title", codes(report))

    def test_near_duplicate_abstract_across_dev_test_is_flagged(self) -> None:
        shared_abstract = (
            "Retrieval augmented generation combines a dense retriever with a "
            "language model. We evaluate citation integrity and answer quality "
            "on scientific documents using held out questions."
        )
        dev_paper = paper(
            "5555.55555v1",
            title="Alpha Framework for Evidence Grounding",
            abstract=shared_abstract,
        )
        test_paper = paper(
            "6666.66666v1",
            title="Beta Benchmark for Citation Faithfulness",
            abstract=shared_abstract,
        )
        report = audit_leakage(
            splits(dev=[dev_paper], test=[test_paper]),
            corpus=[dev_paper, test_paper],
            index_manifest=index_manifest([test_paper]),
        )

        self.assertEqual(len(report["near_duplicate_abstract_pairs"]), 1)
        pair = report["near_duplicate_abstract_pairs"][0]
        self.assertGreaterEqual(pair["jaccard"], 0.75)
        self.assertEqual(pair["scope"], "dev_test")
        self.assertIn("near_duplicate_abstract", codes(report))

    def test_records_without_titles_are_not_flagged(self) -> None:
        dev_paper = paper("9090.11111v1", title="")
        test_paper = paper("9090.22222v1", title="")
        report = audit_leakage(
            splits(dev=[dev_paper], test=[test_paper]),
            corpus=[dev_paper, test_paper],
            index_manifest=index_manifest([test_paper]),
        )

        self.assertEqual(report["near_duplicate_title_pairs"], [])

    def test_same_family_repeated_in_corpus_is_not_a_duplicate(self) -> None:
        dev_paper = paper(
            "7777.77777v1",
            title="Identical Title Across Versions of the Same Paper",
        )
        test_paper = paper(
            "8888.88888v1",
            title="A Totally Different Held Out Paper",
        )
        report = audit_leakage(
            splits(dev=[dev_paper], test=[test_paper]),
            corpus=[
                paper(
                    "7777.77777v2",
                    family="7777.77777",
                    title="Identical Title Across Versions of the Same Paper",
                ),
                test_paper,
            ],
            index_manifest=index_manifest([test_paper]),
        )

        self.assertEqual(report["near_duplicate_title_pairs"], [])


class CleanSplitTests(unittest.TestCase):
    def test_clean_split_returns_zero_overlaps(self) -> None:
        dev_paper = paper("9999.99999v1", title="Dense Retrieval on Scientific PDFs")
        test_alpha = paper(
            "1010.10101v1", title="Sparse Lexical Ranking for Question Answering"
        )
        test_beta = paper(
            "1010.20202v1", title="Citation Integrity in Generated Answers"
        )
        report = audit_leakage(
            splits(
                dev=[dev_paper],
                test=[test_alpha, test_beta],
                reserve=[
                    paper(
                        "1010.30303v1",
                        title="Reserve Paper on Multimodal Tables",
                    )
                ],
            ),
            corpus=[
                dev_paper,
                test_alpha,
                test_beta,
                paper(
                    "1010.30303v1",
                    title="Reserve Paper on Multimodal Tables",
                ),
            ],
            index_manifest=index_manifest([test_alpha, test_beta]),
        )

        self.assertEqual(report["dev_test_family_overlap"], [])
        self.assertEqual(report["near_duplicate_title_pairs"], [])
        self.assertEqual(report["near_duplicate_abstract_pairs"], [])
        self.assertTrue(report["test_index_namespace_ok"])
        self.assertEqual(report["missing_pdf_or_index_ids"], [])
        self.assertEqual(report["review_required"], [])
        self.assertEqual(report["blockers"], [])
        self.assertEqual(report["status"], "clean")


class MissingIdTests(unittest.TestCase):
    def test_missing_index_id_is_flagged(self) -> None:
        alpha = paper("2020.11111v1", title="Indexed Alpha Paper")
        beta = paper("2020.22222v1", title="Unindexed Beta Paper")
        report = audit_leakage(
            splits(test=[alpha, beta]),
            corpus=[alpha, beta],
            index_manifest=index_manifest([alpha]),
        )

        self.assertIn("2020.22222v1", report["missing_pdf_or_index_ids"])
        self.assertIn("missing_index_id", codes(report))
        self.assertFalse(report["test_index_namespace_ok"])
        self.assertEqual(
            report["missing_details"]["2020.22222v1"], ["index"]
        )

    def test_missing_pdf_is_flagged(self) -> None:
        alpha = paper("2020.33333v1", title="Paper With A Missing PDF", pdf_exists=False)
        report = audit_leakage(
            splits(test=[alpha]),
            corpus=[alpha],
            index_manifest=index_manifest([alpha]),
        )

        self.assertIn("2020.33333v1", report["missing_pdf_or_index_ids"])
        self.assertIn("missing_pdf", codes(report))
        self.assertEqual(report["missing_details"]["2020.33333v1"], ["pdf"])

    def test_pdf_is_checked_on_disk_when_root_is_given(self) -> None:
        alpha = paper(
            "2020.44444v1",
            title="Paper With An Unwritten File",
            pdf_path="pdfs/2020.44444v1.pdf",
        )
        with tempfile.TemporaryDirectory(dir=TMP_BASE) as raw:
            root = Path(raw)
            report = audit_leakage(
                splits(test=[alpha]),
                corpus=[alpha],
                index_manifest=index_manifest([alpha]),
                root=root,
            )

            self.assertIn("2020.44444v1", report["missing_pdf_or_index_ids"])
            (root / "pdfs").mkdir()
            (root / "pdfs" / "2020.44444v1.pdf").write_bytes(b"%PDF-1.4\n")
            report = audit_leakage(
                splits(test=[alpha]),
                corpus=[alpha],
                index_manifest=index_manifest([alpha]),
                root=root,
            )

        self.assertNotIn("2020.44444v1", report["missing_pdf_or_index_ids"])


class NamespaceTests(unittest.TestCase):
    def test_foreign_family_in_index_breaks_namespace(self) -> None:
        dev_paper = paper("3030.11111v1", title="Dev Only Paper")
        test_paper = paper("3030.22222v1", title="Held Out Paper")
        report = audit_leakage(
            splits(dev=[dev_paper], test=[test_paper]),
            corpus=[dev_paper, test_paper],
            index_manifest=index_manifest([test_paper, dev_paper]),
        )

        self.assertFalse(report["test_index_namespace_ok"])
        self.assertIn("test_index_namespace", codes(report))
        self.assertIn(
            "3030.11111", report["index_namespace"]["foreign_families"]
        )

    def test_chunk_ids_outside_test_namespace_are_flagged(self) -> None:
        test_paper = paper("4040.11111v1", title="Held Out Chunk Paper")
        chunks = [
            {
                "chunk_id": "4040.11111v1:abcdef:p1:g1:c1",
                "arxiv_id": "4040.11111v1",
                "title": "Held Out Chunk Paper",
            },
            {
                "chunk_id": "4040.99999v1:abcdef:p1:g1:c1",
                "arxiv_id": "4040.99999v1",
                "title": "Foreign Chunk Paper",
            },
        ]
        report = audit_leakage(
            splits(test=[test_paper]),
            corpus=chunks,
            index_manifest=index_manifest([test_paper]),
        )

        self.assertFalse(report["test_index_namespace_ok"])
        self.assertIn("test_index_chunk_namespace", codes(report))
        self.assertEqual(
            report["index_namespace"]["foreign_chunk_paper_ids"],
            ["4040.99999v1"],
        )


class ExcludedReasonTests(unittest.TestCase):
    def test_excluded_reasons_are_counted(self) -> None:
        excluded = [
            paper("5050.11111v1", exclusion_reason="screening_out_of_scope"),
            paper("5050.22222v1", exclusion_reason="screening_out_of_scope"),
            paper("5050.33333v1", exclusion_reason="dev_family"),
            paper("5050.44444v1", exclusion_reason="quality_flag:some_low_text_pages"),
        ]
        test_paper = paper("5050.55555v1", title="Held Out Paper")
        report = audit_leakage(
            splits(test=[test_paper], excluded=excluded),
            corpus=[test_paper],
            index_manifest=index_manifest([test_paper]),
        )

        self.assertEqual(report["excluded_total"], 4)
        self.assertEqual(
            report["excluded_reasons"],
            {
                "quality_flag:some_low_text_pages": 1,
                "screening_out_of_scope": 2,
                "dev_family": 1,
            },
        )

    def test_test_item_carrying_an_exclusion_reason_is_flagged(self) -> None:
        leaked = paper(
            "6060.11111v1",
            title="Held Out Paper",
            exclusion_reason="quality_flag:some_low_text_pages",
        )
        report = audit_leakage(
            splits(test=[leaked]),
            corpus=[leaked],
            index_manifest=index_manifest([leaked]),
        )

        self.assertIn("test_item_excluded", codes(report))
        self.assertEqual(
            report["test_exclusion_leaks"][0]["arxiv_id"], "6060.11111v1"
        )

    def test_test_item_marked_exclude_from_test_is_flagged(self) -> None:
        leaked = paper("6060.22222v1", title="Held Out Paper", exclude_from_test=True)
        report = audit_leakage(
            splits(test=[leaked]),
            corpus=[leaked],
            index_manifest=index_manifest([leaked]),
        )

        self.assertIn("test_item_exclude_from_test", codes(report))


class CliTests(unittest.TestCase):
    def test_cli_writes_report_from_synthetic_inputs(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_BASE) as raw:
            root = Path(raw)
            (root / "pdfs").mkdir()
            test_paper = paper(
                "7070.11111v1",
                title="Held Out CLI Paper",
                pdf_path="pdfs/7070.11111v1.pdf",
            )
            (root / "pdfs" / "7070.11111v1.pdf").write_bytes(b"%PDF-1.4\n")

            splits_path = root / "splits.json"
            index_path = root / "index_manifest.json"
            manifest_a = root / "manifest_a.json"
            manifest_b = root / "manifest_b.json"
            out_path = root / "leakage_report.json"

            splits_path.write_text(
                json.dumps(splits(test=[test_paper])), encoding="utf-8"
            )
            index_path.write_text(
                json.dumps(index_manifest([test_paper])), encoding="utf-8"
            )
            manifest_a.write_text(
                json.dumps({"papers": [test_paper]}), encoding="utf-8"
            )
            manifest_b.write_text(json.dumps({"papers": []}), encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                code = leakage_main(
                    [
                        "--root",
                        str(root),
                        "--splits",
                        str(splits_path),
                        "--index-manifest",
                        str(index_path),
                        "--manifest",
                        str(manifest_a),
                        "--manifest",
                        str(manifest_b),
                        "--out",
                        str(out_path),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertTrue(out_path.exists())
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(report["dev_test_family_overlap"], [])
            self.assertEqual(report["status"], "clean")
            self.assertEqual(report["summary"]["test_papers"], 1)
            self.assertIn("sha256", report["inputs"]["splits"])
            self.assertEqual(report["inputs"]["manifests"][1]["paper_count"], 0)

    def test_cli_returns_nonzero_for_a_blocking_leak(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_BASE) as raw:
            root = Path(raw)
            (root / "pdfs").mkdir()
            test_paper = paper(
                "8080.11111v1",
                family="8080.11111",
                title="Held Out CLI Paper",
                pdf_path="pdfs/8080.11111v1.pdf",
            )
            dev_paper = paper(
                "8080.11111v2",
                family="8080.11111",
                title="Dev CLI Paper",
                pdf_path="pdfs/8080.11111v2.pdf",
            )
            (root / "pdfs" / "8080.11111v1.pdf").write_bytes(b"%PDF-1.4\n")
            (root / "pdfs" / "8080.11111v2.pdf").write_bytes(b"%PDF-1.4\n")

            splits_path = root / "splits.json"
            index_path = root / "index_manifest.json"
            manifest_path = root / "manifest.json"
            out_path = root / "leakage_report.json"

            splits_path.write_text(
                json.dumps(splits(dev=[dev_paper], test=[test_paper])),
                encoding="utf-8",
            )
            index_path.write_text(
                json.dumps(index_manifest([test_paper])), encoding="utf-8"
            )
            manifest_path.write_text(
                json.dumps({"papers": [dev_paper, test_paper]}), encoding="utf-8"
            )

            with contextlib.redirect_stdout(io.StringIO()):
                code = leakage_main(
                    [
                        "--root",
                        str(root),
                        "--splits",
                        str(splits_path),
                        "--index-manifest",
                        str(index_path),
                        "--manifest",
                        str(manifest_path),
                        "--out",
                        str(out_path),
                    ]
                )

            self.assertEqual(code, 1)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(report["dev_test_family_overlap"], ["8080.11111"])
            self.assertEqual(report["status"], "blockers")


if __name__ == "__main__":
    unittest.main()
