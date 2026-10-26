"""Offline tests for Task E6: experiment runner and report.

Everything here uses synthetic questions/qrels and a fake system. No model,
network, ``data/`` or ``eval/benchmark_v2/index`` access happens in these tests.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.analysis.figures import text_bar  # noqa: E402
from eval.analysis.make_report import bootstrap_ci, make_report  # noqa: E402
from eval.run_experiment import (  # noqa: E402
    ExperimentDeps,
    load_benchmark,
    load_matrix,
    run_experiment,
)

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"

QUESTIONS = [
    {
        "question_id": "q1",
        "question_zh": "问题一",
        "query_en": "alpha retrieval augmented generation",
        "type": "factual",
        "answerability": "answerable",
    },
    {
        "question_id": "q2",
        "question_zh": "问题二",
        "query_en": "numeric evidence budget",
        "type": "numeric",
        "answerability": "answerable",
    },
    {
        "question_id": "q3",
        "question_zh": "问题三",
        "query_en": "mechanism of reranking",
        "type": "mechanism",
        "answerability": "answerable",
    },
]

QRELS = [
    {"question_id": "q1", "gold_parent_ids": ["p1", "p2"], "answerability": "answerable"},
    {"question_id": "q2", "gold_parent_ids": ["p4"], "answerability": "answerable"},
    {"question_id": "q3", "gold_parent_ids": ["p5"], "answerability": "answerable"},
]

# q1: full recall, q2: full recall, q3: miss. Mean Recall@3 = 2/3.
DEFAULT_OUTPUTS = {
    "q1": {"retrieved_parent_ids": ["p1", "p2", "p3"]},
    "q2": {"retrieved_parent_ids": ["p4"]},
    "q3": {"retrieved_parent_ids": ["p6"]},
}

GOLD_KEYS = ("gold_parent_ids", "gold_parent_chunk_ids", "gold_chunks", "gold_points")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_benchmark(base: Path, questions=None, qrels=None) -> Path:
    bench = base / "benchmark"
    bench.mkdir(parents=True, exist_ok=True)
    write_jsonl(bench / "questions.jsonl", questions or QUESTIONS)
    write_jsonl(bench / "qrels.jsonl", qrels or QRELS)
    return bench


def make_matrix(systems: dict | None = None) -> dict:
    return {
        "version": 1,
        "systems": systems or {"fake_retrieval": {"kind": "retrieval", "top_k": 6}},
    }


class FakeDeps:
    """Strictly offline ExperimentDeps implementation for tests."""

    def __init__(self, outputs: dict | None = None, tokens: int = 4) -> None:
        self.outputs = DEFAULT_OUTPUTS if outputs is None else outputs
        self.tokens = tokens
        self.calls: list[tuple[str, str]] = []
        self.seen_questions: list[dict] = []

    def run_system(self, system: str, question: dict) -> dict:
        qid = question["question_id"]
        self.calls.append((system, qid))
        self.seen_questions.append(dict(question))
        if any(key in question for key in GOLD_KEYS):
            raise AssertionError("gold data leaked into the system question")
        value = self.outputs.get(qid, DEFAULT_OUTPUTS.get(qid, {"retrieved_parent_ids": []}))
        if isinstance(value, BaseException):
            raise value
        if callable(value):
            value = value(system, question)
        record = dict(value)
        record.setdefault("tokens", self.tokens)
        return record


class ExperimentTestCase(unittest.TestCase):
    def setUp(self) -> None:
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="e6_", dir=str(TMP_ROOT)))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.bench = make_benchmark(self.tmp)
        self.out = self.tmp / "run"

    def read_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))


class RunnerTests(ExperimentTestCase):
    def test_three_questions_write_three_checkpoints(self) -> None:
        deps = FakeDeps()
        result = run_experiment(make_matrix(), self.bench, self.out, deps)

        self.assertEqual(result["status"], "completed")
        self.assertEqual([qid for _, qid in deps.calls], ["q1", "q2", "q3"])
        for qid in ("q1", "q2", "q3"):
            checkpoint = self.read_json(self.out / "fake_retrieval" / f"{qid}.json")
            self.assertEqual(checkpoint["question_id"], qid)
            self.assertEqual(checkpoint["system"], "fake_retrieval")
            self.assertEqual(checkpoint["status"], "ok")
            self.assertIn("retrieved_parent_ids", checkpoint)

        summary = result["systems"]["fake_retrieval"]
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["n_failed"], 0)
        self.assertAlmostEqual(summary["metrics"]["recall@3"], 2 / 3, places=12)
        self.assertAlmostEqual(summary["metrics"]["all_gold@6"], 2 / 3, places=12)
        self.assertEqual(summary["tokens"], 12)

        system_run = self.read_json(self.out / "fake_retrieval" / "run.json")
        self.assertEqual(system_run["status"], "completed")
        self.assertEqual(system_run["n"], 3)

        run_json = self.read_json(self.out / "run.json")
        self.assertEqual(run_json["status"], "completed")
        self.assertEqual(len(run_json["results"]), 1)
        self.assertEqual(run_json["budget"]["tokens_used"], 12)
        self.assertFalse(run_json["budget"]["stopped"])

        self.assertEqual(len(deps.seen_questions), 3)
        self.assertEqual(deps.seen_questions[0]["query_en"], QUESTIONS[0]["query_en"])

    def test_failure_counts_as_zero_and_is_never_dropped(self) -> None:
        deps = FakeDeps({"q2": RuntimeError("boom")})
        result = run_experiment(make_matrix(), self.bench, self.out, deps)

        summary = result["systems"]["fake_retrieval"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["n_failed"], 1)
        self.assertEqual(summary["failures"], ["q2"])
        self.assertAlmostEqual(summary["metrics"]["recall@3"], 1 / 3, places=12)
        self.assertAlmostEqual(summary["metrics"]["schema_success"], 2 / 3, places=12)

        failed = self.read_json(self.out / "fake_retrieval" / "q2.json")
        self.assertEqual(failed["status"], "failed")
        self.assertIn("boom", failed["error"])
        for metric, value in failed["metrics"].items():
            if value is not None:
                self.assertEqual(value, 0.0, metric)

        self.assertTrue((self.out / "fake_retrieval" / "q3.json").is_file())

    def test_invalid_record_counts_as_failure(self) -> None:
        deps = FakeDeps({"q2": {"retrieved_parent_ids": "p4"}})
        result = run_experiment(make_matrix(), self.bench, self.out, deps)

        summary = result["systems"]["fake_retrieval"]
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["n_failed"], 1)
        self.assertAlmostEqual(summary["metrics"]["recall@3"], 1 / 3, places=12)
        invalid = self.read_json(self.out / "fake_retrieval" / "q2.json")
        self.assertEqual(invalid["status"], "invalid")
        self.assertTrue(invalid["error"])

    def test_resume_skips_completed_checkpoints(self) -> None:
        first_deps = FakeDeps()
        first = run_experiment(make_matrix(), self.bench, self.out, first_deps, limit=1)

        self.assertEqual([qid for _, qid in first_deps.calls], ["q1"])
        self.assertEqual(first["status"], "incomplete")
        self.assertEqual(first["stop_reason"], "limit")
        self.assertEqual(first["systems"]["fake_retrieval"]["n"], 1)
        self.assertFalse((self.out / "fake_retrieval" / "q2.json").exists())

        second_deps = FakeDeps()
        second = run_experiment(make_matrix(), self.bench, self.out, second_deps)
        self.assertEqual([qid for _, qid in second_deps.calls], ["q2", "q3"])
        self.assertEqual(second["status"], "completed")
        self.assertEqual(second["systems"]["fake_retrieval"]["n"], 3)
        self.assertAlmostEqual(second["systems"]["fake_retrieval"]["metrics"]["recall@3"], 2 / 3)

        third_deps = FakeDeps()
        third = run_experiment(make_matrix(), self.bench, self.out, third_deps)
        self.assertEqual(third_deps.calls, [])
        self.assertEqual(third["status"], "completed")
        self.assertAlmostEqual(third["systems"]["fake_retrieval"]["metrics"]["recall@3"], 2 / 3)

    def test_failed_checkpoint_is_not_retried_unless_asked(self) -> None:
        first = run_experiment(
            make_matrix(), self.bench, self.out, FakeDeps({"q2": RuntimeError("boom")})
        )
        self.assertEqual(first["systems"]["fake_retrieval"]["n_failed"], 1)

        skipped_deps = FakeDeps()
        second = run_experiment(make_matrix(), self.bench, self.out, skipped_deps)
        self.assertEqual(skipped_deps.calls, [])
        self.assertEqual(second["systems"]["fake_retrieval"]["n_failed"], 1)

        retry_deps = FakeDeps()
        third = run_experiment(
            make_matrix(), self.bench, self.out, retry_deps, retry_failed=True
        )
        self.assertEqual([qid for _, qid in retry_deps.calls], ["q2"])
        self.assertEqual(third["systems"]["fake_retrieval"]["n_failed"], 0)
        self.assertAlmostEqual(third["systems"]["fake_retrieval"]["metrics"]["recall@3"], 2 / 3)
        retried = self.read_json(self.out / "fake_retrieval" / "q2.json")
        self.assertEqual(retried["attempt"], 2)

    def test_corrupt_checkpoint_is_rerun(self) -> None:
        checkpoint = self.out / "fake_retrieval" / "q1.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("{not valid json", encoding="utf-8")

        deps = FakeDeps()
        result = run_experiment(make_matrix(), self.bench, self.out, deps)

        self.assertEqual([qid for _, qid in deps.calls], ["q1", "q2", "q3"])
        self.assertEqual(result["status"], "completed")
        repaired = self.read_json(checkpoint)
        self.assertEqual(repaired["status"], "ok")

    def test_budget_stops_cleanly_and_records_the_stop(self) -> None:
        deps = FakeDeps(tokens=4)
        result = run_experiment(make_matrix(), self.bench, self.out, deps, budget=5)

        self.assertEqual([qid for _, qid in deps.calls], ["q1", "q2"])
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["stop_reason"], "budget_exhausted")
        self.assertTrue(result["budget"]["stopped"])
        self.assertEqual(result["budget"]["max_tokens"], 5)
        self.assertEqual(result["budget"]["tokens_used"], 8)
        self.assertEqual(result["budget"]["remaining_question_ids"], ["q3"])
        self.assertFalse((self.out / "fake_retrieval" / "q3.json").exists())

        system_run = self.read_json(self.out / "fake_retrieval" / "run.json")
        self.assertEqual(system_run["status"], "budget_exhausted")
        run_json = self.read_json(self.out / "run.json")
        self.assertTrue(run_json["budget"]["stopped"])
        self.assertEqual(run_json["stop_reason"], "budget_exhausted")

        resume_deps = FakeDeps(tokens=4)
        resumed = run_experiment(make_matrix(), self.bench, self.out, resume_deps)
        self.assertEqual([qid for _, qid in resume_deps.calls], ["q3"])
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(resumed["budget"]["tokens_used"], 12)
        self.assertFalse(resumed["budget"]["stopped"])

    def test_unanswerable_question_is_scored_for_schema_but_not_retrieval(self) -> None:
        questions = QUESTIONS + [
            {
                "question_id": "q4",
                "question_zh": "问题四",
                "query_en": "not in corpus",
                "type": "unanswerable",
                "answerability": "unanswerable",
            }
        ]
        qrels = QRELS + [
            {"question_id": "q4", "gold_parent_ids": [], "answerability": "unanswerable"}
        ]
        bench = make_benchmark(self.tmp, questions, qrels)
        deps = FakeDeps({"q4": {"retrieved_parent_ids": []}})
        result = run_experiment(make_matrix(), bench, self.out, deps)

        summary = result["systems"]["fake_retrieval"]
        self.assertEqual(summary["n"], 4)
        self.assertEqual(summary["n_retrieval"], 3)
        self.assertAlmostEqual(summary["metrics"]["recall@3"], 2 / 3, places=12)
        self.assertAlmostEqual(summary["metrics"]["schema_success"], 1.0, places=12)

    def test_stub_systems_are_skipped_without_calling_deps(self) -> None:
        matrix = make_matrix(
            {
                "fixed_rag_matched": {"kind": "answer", "implemented": False},
                "agent_single_round": {"kind": "answer", "implemented": False},
            }
        )
        deps = FakeDeps()
        result = run_experiment(matrix, self.bench, self.out, deps)

        self.assertEqual(deps.calls, [])
        self.assertEqual(result["status"], "completed")
        for name in ("fixed_rag_matched", "agent_single_round"):
            self.assertEqual(result["systems"][name]["status"], "skipped")
            self.assertEqual(result["systems"][name]["n"], 0)
            self.assertEqual(result["systems"][name]["metrics"], {})
            self.assertEqual(
                result["systems"][name]["skipped_reason"], "not_implemented"
            )

    def test_unknown_system_filter_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_experiment(
                make_matrix(), self.bench, self.out, FakeDeps(), systems=["nope"]
            )


class BenchmarkAndMatrixTests(ExperimentTestCase):
    def test_question_id_separator_cannot_collide(self) -> None:
        questions = [
            {"question_id": "a/b", "query_en": "alpha"},
            {"question_id": "a_b", "query_en": "beta"},
        ]
        qrels = [
            {"question_id": "a/b", "gold_parent_ids": ["p1"]},
            {"question_id": "a_b", "gold_parent_ids": ["p2"]},
        ]
        bench = make_benchmark(self.tmp, questions, qrels)
        with self.assertRaises(ValueError):
            run_experiment(make_matrix(), bench, self.out, FakeDeps())

    def test_load_benchmark_reads_jsonl_and_rejects_duplicates(self) -> None:
        questions, qrels = load_benchmark(self.bench)
        self.assertEqual([row["question_id"] for row in questions], ["q1", "q2", "q3"])
        self.assertEqual(qrels["q1"]["gold_parent_ids"], ["p1", "p2"])

        duplicate_bench = make_benchmark(
            self.tmp / "dup",
            QUESTIONS + [dict(QUESTIONS[0])],
            QRELS,
        )
        with self.assertRaises(ValueError):
            load_benchmark(duplicate_bench)

    def test_matrix_yaml_declares_the_required_systems(self) -> None:
        matrix = load_matrix(ROOT / "eval" / "experiments" / "matrix.yaml")
        systems = matrix["systems"]
        required = {
            "bm25_top6",
            "dense_top6",
            "hybrid_rrf_top6",
            "hybrid_rrf_rerank_top6",
            "fixed_rag_matched",
            "agent_single_round",
        }
        self.assertTrue(required.issubset(set(systems)))
        self.assertEqual(set(systems) - required, {"dense_zh_top6"})

        for name in ("bm25_top6", "dense_top6", "hybrid_rrf_top6"):
            self.assertEqual(systems[name]["kind"], "retrieval")
            self.assertEqual(systems[name]["top_k"], 6)
            self.assertTrue(systems[name].get("implemented", True))
            self.assertFalse(systems[name].get("requires_llm", False))
        self.assertEqual(systems["dense_top6"]["query"], "query_en")
        self.assertEqual(systems["dense_zh_top6"]["query"], "query_zh")

        for name in ("hybrid_rrf_rerank_top6", "fixed_rag_matched", "agent_single_round"):
            self.assertFalse(systems[name]["implemented"])
            self.assertTrue(systems[name]["requires_llm"])


class ReportTests(ExperimentTestCase):
    def test_report_contains_n_and_ci_without_significance_claims(self) -> None:
        questions = [
            {
                "question_id": f"q{index:02d}",
                "question_zh": f"问题{index}",
                "query_en": f"query {index}",
                "type": "factual",
                "answerability": "answerable",
            }
            for index in range(40)
        ]
        qrels = [
            {"question_id": row["question_id"], "gold_parent_ids": [f"gold{index}"]}
            for index, row in enumerate(questions)
        ]
        outputs = {
            row["question_id"]: {
                "retrieved_parent_ids": (
                    [f"gold{index}", "other"] if index % 2 == 0 else ["other"]
                )
            }
            for index, row in enumerate(questions)
        }
        bench = make_benchmark(self.tmp, questions, qrels)
        result = run_experiment(make_matrix(), bench, self.out, FakeDeps(outputs))

        markdown = make_report(result["results"], self.out)
        written = (self.out / "report.md").read_text(encoding="utf-8")
        self.assertEqual(markdown, written)
        self.assertIn("95% CI", written)
        self.assertIn("40", written)
        self.assertIn("fake_retrieval", written)
        self.assertIn("no significance testing", written.lower())

        payload = self.read_json(self.out / "results.json")
        self.assertFalse(payload["significance_testing"])
        entry = payload["systems"][0]
        self.assertEqual(entry["n"], 40)
        metric = entry["metrics"]["recall@3"]
        self.assertAlmostEqual(metric["mean"], 0.5, places=12)
        self.assertEqual(metric["n"], 40)
        self.assertLessEqual(metric["ci_low"], metric["mean"])
        self.assertGreaterEqual(metric["ci_high"], metric["mean"])
        self.assertGreaterEqual(metric["ci_low"], 0.0)
        self.assertLessEqual(metric["ci_high"], 1.0)

        lowered = written.lower()
        for phrase in ("statistically significant", "p-value", "p <", "p=", "mcnemar"):
            self.assertNotIn(phrase, lowered)

    def test_report_lists_failures_and_skipped_systems_honestly(self) -> None:
        matrix = make_matrix(
            {
                "fake_retrieval": {"kind": "retrieval", "top_k": 6},
                "fixed_rag_matched": {"kind": "answer", "implemented": False},
            }
        )
        result = run_experiment(
            matrix, self.bench, self.out, FakeDeps({"q2": RuntimeError("boom")})
        )
        markdown = make_report(result["results"], self.out)

        self.assertIn("fake_retrieval", markdown)
        self.assertIn("fixed_rag_matched", markdown)
        self.assertIn("skipped", markdown)
        self.assertIn("q2", markdown)
        self.assertIn("1", markdown)
        payload = self.read_json(self.out / "results.json")
        self.assertEqual(payload["n_systems"], 2)

    def test_bootstrap_ci_is_deterministic_and_bounded(self) -> None:
        values = [0.0, 1.0, 1.0, 0.0, 1.0]
        first = bootstrap_ci(values, resamples=200, seed=7)
        second = bootstrap_ci(values, resamples=200, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean"], 0.6, places=12)
        self.assertLessEqual(first["ci_low"], first["ci_high"])
        self.assertTrue(all(math.isfinite(value) for value in first.values()))

        single = bootstrap_ci([0.25])
        self.assertEqual(single["ci_low"], 0.25)
        self.assertEqual(single["ci_high"], 0.25)
        self.assertIsNone(bootstrap_ci([]))

    def test_text_bar_is_clamped(self) -> None:
        self.assertEqual(text_bar(None), "")
        self.assertEqual(text_bar(1.0, width=10), "#" * 10)
        self.assertEqual(text_bar(0.0, width=10), "." * 10)
        self.assertEqual(text_bar(2.0, width=4), "####")
        self.assertEqual(text_bar(-1.0, width=4), "....")


class DepsProtocolTests(unittest.TestCase):
    def test_experiment_deps_exposes_run_system(self) -> None:
        self.assertTrue(hasattr(ExperimentDeps, "run_system"))
        deps = ExperimentDeps()
        with self.assertRaises(NotImplementedError):
            deps.run_system("x", {"question_id": "q"})


if __name__ == "__main__":
    unittest.main()
