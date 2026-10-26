"""Tests for Task E5b: question generation and candidate validation.

Everything here is offline. ``chat_fn`` is always a fake callable; no real
DeepSeek/OpenAI request is ever made, and no file under ``data/`` is written.
Temporary fixtures live under ``tests/.tmp/``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.benchmark_v2.generate_questions import (  # noqa: E402
    generate_candidate,
    generate_for_split,
    select_context_pages,
    translate_query,
)
from eval.benchmark_v2.validate_questions import (  # noqa: E402
    validate_candidate,
    validate_question_set,
)

TMP_BASE = Path(__file__).resolve().parent / ".tmp"
TMP_BASE.mkdir(parents=True, exist_ok=True)

PAPER_ID = "2607.00001v1"
ABSTRACT = (
    "We study sparse retrieval. The sparse retrieval module achieves "
    "12.5% gains over baselines."
)


def make_pages(page_count: int = 9, body_size: int = 0) -> list[dict]:
    """Build deterministic page records shaped like prepared_v1 pages."""
    pages: list[dict] = []
    for number in range(1, page_count + 1):
        if number == 1:
            text = (
                "Test Paper Title\n"
                "Abstract " + ABSTRACT + "\n"
                "1 Introduction\n"
                "Body introduction text."
            )
        elif number == 2:
            text = (
                "Our method uses a two-stage reranker. It processes "
                "8 tokens per query with 40% lower latency. "
                "Table 2 reports results."
            )
        elif number == 3:
            text = "The sparse retrieval module selects evidence from the corpus."
        elif number == 8:
            text = "PAGE_EIGHT_MARKER final discussion and conclusion."
        elif number == 9:
            text = "PAGE_NINE_MARKER appendix beyond the context window."
        else:
            text = f"Page {number} filler text about retrieval and evidence."
        if body_size and 2 <= number <= 8:
            text = text + " " + ("filler evidence sentence " * (body_size // 24))
        pages.append(
            {
                "paper_id": PAPER_ID,
                "pdf_page": number,
                "printed_page_label": str(number),
                "extracted_text": text,
                "normalized_text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    return pages


def make_paper(arxiv_id: str = PAPER_ID, title: str = "Test Paper Title") -> dict:
    return {
        "arxiv_id": arxiv_id,
        "family": arxiv_id.split("v")[0],
        "title": title,
        "year": 2026,
        "categories": ["cs.CL"],
    }


def make_candidate(**overrides) -> dict:
    base = {
        "question_id": "q001",
        "paper_ids": [PAPER_ID],
        "question_zh": "该方法使用了什么重排策略？",
        "query_en": "What reranking strategy does the method use?",
        "type": "mechanism",
        "answerability": "answerable",
        "gold_points": [
            {
                "requirement": "name the reranker",
                "quote": "two-stage reranker",
                "pdf_page": 2,
                "section": "Method",
            }
        ],
        "missing_aspects": [],
        "source_title": "Test Paper Title",
    }
    base.update(overrides)
    return base


class FakeChat:
    """Fake chat_fn with the list-of-messages signature used by the generator."""

    model = "fake-model"

    def __init__(self, response) -> None:
        self.response = response
        self.calls: list[list[dict]] = []
        self.last_usage: dict | None = None

    def __call__(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        self.last_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        if isinstance(self.response, str):
            return self.response
        return json.dumps(self.response, ensure_ascii=False)


class DynamicFakeChat(FakeChat):
    """Returns a different, valid question for each call."""

    def __init__(self, quote: str = "two-stage reranker", pdf_page: int = 2) -> None:
        super().__init__({})
        self.quote = quote
        self.pdf_page = pdf_page
        self.counter = 0

    def __call__(self, messages: list[dict]) -> str:
        self.counter += 1
        payload = {
            "question_zh": f"第 {self.counter} 题：该方法使用了什么重排策略？",
            "query_en": f"Question {self.counter}: what reranking strategy is used?",
            "type": "mechanism",
            "answerability": "answerable",
            "gold_points": [
                {
                    "requirement": "name the reranker",
                    "quote": self.quote,
                    "pdf_page": self.pdf_page,
                    "section": "Method",
                }
            ],
            "missing_aspects": [],
            "source_title": "Test Paper Title",
        }
        self.response = payload
        return super().__call__(messages)


class RepairingFakeChat(FakeChat):
    """Returns an invalid quote once, then a valid correction."""

    def __init__(self) -> None:
        super().__init__({})
        self.counter = 0

    def __call__(self, messages: list[dict]) -> str:
        self.counter += 1
        quote = "not present anywhere" if self.counter == 1 else "two-stage reranker"
        payload = {
            "question_zh": "该方法使用了什么重排策略？",
            "query_en": "What reranking strategy does the method use?",
            "type": "mechanism",
            "answerability": "answerable",
            "gold_points": [
                {
                    "requirement": "name the reranker",
                    "quote": quote,
                    "pdf_page": 2,
                    "section": "Method",
                }
            ],
            "missing_aspects": [],
            "source_title": "Test Paper Title",
        }
        self.response = payload
        return super().__call__(messages)


class TranslateQueryTests(unittest.TestCase):
    def test_translation_sees_only_the_question(self) -> None:
        captured: list[list[dict]] = []

        def chat(messages):
            captured.append(messages)
            return json.dumps({"query_en": "How many documents?"}, ensure_ascii=False)

        result = translate_query("包含多少篇文档？", chat, model="fake")
        self.assertEqual(result, "How many documents?")
        user_content = captured[0][-1]["content"]
        self.assertEqual(
            user_content, '{"question_zh": "包含多少篇文档？"}'
        )
        self.assertNotIn("page 2", user_content.lower())
        self.assertNotIn("pages", user_content.lower())


class GenerateCandidateTests(unittest.TestCase):
    def test_valid_response_is_enriched_with_generator_metadata(self) -> None:
        response = {
            "question_id": "q001",
            "question_zh": "该方法使用了什么重排策略？",
            "query_en": "What reranking strategy does the method use?",
            "type": "mechanism",
            "answerability": "answerable",
            "gold_points": [
                {
                    "requirement": "name the reranker",
                    "quote": "two-stage reranker",
                    "pdf_page": 2,
                    "section": "Method",
                }
            ],
            "missing_aspects": [],
            "source_title": "Test Paper Title",
        }
        chat = FakeChat(response)
        candidate = generate_candidate(
            make_paper(), make_pages(), chat, model="fake-model"
        )

        self.assertEqual(candidate["question_id"], "q001")
        self.assertEqual(candidate["paper_ids"], [PAPER_ID])
        self.assertEqual(candidate["gold_points"][0]["quote"], "two-stage reranker")
        generator = candidate["generator"]
        self.assertEqual(generator["model"], "fake-model")
        self.assertRegex(generator["prompt_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(generator["created_at"])
        self.assertEqual(
            candidate["review"],
            {"status": "pending", "reviewer": None, "timestamp": None},
        )
        self.assertEqual(len(chat.calls), 1)

    def test_variant_instruction_is_appended_to_system_prompt(self) -> None:
        chat = FakeChat(make_candidate())
        generate_candidate(
            make_paper(),
            make_pages(),
            chat,
            model="fake",
            extra_instruction="VARIANT_MARK",
        )
        self.assertIn("VARIANT_MARK", chat.calls[0][0]["content"])

    def test_context_uses_pages_two_through_eight(self) -> None:
        response = {
            "question_id": "q001",
            "question_zh": "该方法使用了什么重排策略？",
            "query_en": "What reranking strategy does the method use?",
            "type": "mechanism",
            "answerability": "answerable",
            "gold_points": [
                {
                    "requirement": "name the reranker",
                    "quote": "two-stage reranker",
                    "pdf_page": 2,
                    "section": "Method",
                }
            ],
            "missing_aspects": [],
            "source_title": "Test Paper Title",
        }
        chat = FakeChat(response)
        generate_candidate(make_paper(), make_pages(), chat, model="fake-model")

        prompt_text = json.dumps(chat.calls[0], ensure_ascii=False)
        self.assertIn("two-stage reranker", prompt_text)
        self.assertIn("PAGE_EIGHT_MARKER", prompt_text)
        self.assertNotIn("PAGE_NINE_MARKER", prompt_text)
        self.assertNotIn("Abstract " + ABSTRACT, prompt_text)
        self.assertNotIn(ABSTRACT, prompt_text)

    def test_context_text_is_capped_near_twelve_thousand_chars(self) -> None:
        selected, text = select_context_pages(make_pages(body_size=6000))
        self.assertTrue(selected)
        self.assertTrue(all(2 <= page["pdf_page"] <= 8 for page in selected))
        self.assertLessEqual(len(text), 12000)
        self.assertGreater(len(text), 5000)

    def test_invalid_json_raises_value_error(self) -> None:
        chat = FakeChat("not json at all")
        with self.assertRaises(ValueError):
            generate_candidate(make_paper(), make_pages(), chat, model="fake-model")


class ValidateCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pages = make_pages()

    def test_valid_candidate_has_no_errors(self) -> None:
        self.assertEqual(
            validate_candidate(make_candidate(), self.pages, ABSTRACT), []
        )

    def test_minimal_candidate_runs_content_rules_without_schema_noise(self) -> None:
        minimal = {
            "gold_points": [
                {
                    "requirement": "name the reranker",
                    "quote": "two-stage reranker",
                    "pdf_page": 2,
                    "section": "Method",
                }
            ]
        }
        self.assertEqual(validate_candidate(minimal, self.pages, ABSTRACT), [])

    def test_strict_schema_requires_metadata(self) -> None:
        minimal = {"gold_points": []}
        errors = validate_candidate(
            minimal, self.pages, ABSTRACT, strict_schema=True
        )
        self.assertTrue(any("question_id" in error for error in errors), errors)

    def test_quote_must_appear_on_the_given_page(self) -> None:
        candidate = make_candidate(
            gold_points=[
                {
                    "requirement": "bad quote",
                    "quote": "this phrase does not exist anywhere",
                    "pdf_page": 2,
                    "section": "Method",
                }
            ]
        )
        errors = validate_candidate(candidate, self.pages, ABSTRACT)
        self.assertTrue(any("substring" in error for error in errors), errors)

    def test_missing_page_is_rejected(self) -> None:
        candidate = make_candidate(
            gold_points=[
                {
                    "requirement": "bad page",
                    "quote": "two-stage reranker",
                    "pdf_page": 99,
                    "section": "Method",
                }
            ]
        )
        errors = validate_candidate(candidate, self.pages, ABSTRACT)
        self.assertTrue(any("page 99" in error for error in errors), errors)

    def test_abstract_only_quote_is_rejected(self) -> None:
        candidate = make_candidate(
            gold_points=[
                {
                    "requirement": "abstract quote",
                    "quote": (
                        "The sparse retrieval module achieves 12.5% gains "
                        "over baselines."
                    ),
                    "pdf_page": 1,
                    "section": "Abstract",
                }
            ]
        )
        errors = validate_candidate(candidate, self.pages, ABSTRACT)
        self.assertTrue(any("abstract" in error.lower() for error in errors), errors)

    def test_question_cannot_copy_the_gold_quote(self) -> None:
        quote = "The sparse retrieval module selects evidence from the corpus."
        candidate = make_candidate(
            type="factual",
            question_zh=quote,
            query_en=quote,
            gold_points=[
                {
                    "requirement": "describe the module",
                    "quote": quote,
                    "pdf_page": 3,
                    "section": "Method",
                }
            ],
        )
        problems = validate_candidate(candidate, make_pages(), ABSTRACT)
        self.assertTrue(
            any("copies the gold quote" in problem for problem in problems),
            problems,
        )

    def test_line_end_hyphenation_is_joined_for_quote_matching(self) -> None:
        pages = make_pages()
        pages[1]["extracted_text"] = (
            "The primary corpus for large-scale domain align-\nment "
            "is the SSH subset."
        )
        pages[1]["normalized_text"] = pages[1]["extracted_text"]
        candidate = make_candidate(
            type="factual",
            question_zh="大规模领域对齐的主要语料是什么？",
            query_en="What is the primary corpus for large-scale domain alignment?",
            gold_points=[
                {
                    "requirement": "name the corpus",
                    "quote": (
                        "The primary corpus for large-scale domain alignment "
                        "is the SSH subset."
                    ),
                    "pdf_page": 2,
                    "section": "Data",
                }
            ],
        )
        problems = validate_candidate(candidate, pages, ABSTRACT)
        self.assertFalse(
            any("not a substring" in problem for problem in problems), problems
        )

    def test_quote_that_also_appears_in_body_is_not_abstract_only(self) -> None:
        candidate = make_candidate(
            gold_points=[
                {
                    "requirement": "body repeats an abstract phrase",
                    "quote": "sparse retrieval",
                    "pdf_page": 3,
                    "section": "Method",
                }
            ]
        )
        self.assertEqual(validate_candidate(candidate, self.pages, ABSTRACT), [])

    def test_numeric_question_requires_digit_and_unit(self) -> None:
        candidate = make_candidate(
            type="numeric",
            question_zh="第 2 张表报告了多少结果？",
            query_en="What does Table 2 report?",
            gold_points=[
                {
                    "requirement": "table number without a unit",
                    "quote": "Table 2 reports results.",
                    "pdf_page": 2,
                    "section": "Results",
                }
            ],
        )
        errors = validate_candidate(candidate, self.pages, ABSTRACT)
        self.assertTrue(any("unit" in error.lower() for error in errors), errors)

    def test_numeric_table_cell_may_take_unit_from_question(self) -> None:
        candidate = make_candidate(
            type="numeric",
            question_zh="该方法每个 query 处理多少 token？",
            query_en="How many tokens are processed per query?",
            gold_points=[
                {
                    "requirement": "report the token count",
                    "quote": "8",
                    "pdf_page": 2,
                    "section": "Method",
                }
            ],
        )
        problems = validate_candidate(candidate, make_pages(), ABSTRACT)
        self.assertFalse(
            any("number and a unit" in problem for problem in problems), problems
        )

    def test_numeric_question_allows_auxiliary_context_point(self) -> None:
        candidate = make_candidate(
            type="numeric",
            question_zh="每个 query 处理多少 token，表 2 报告了什么？",
            query_en="How many tokens per query, and what does Table 2 report?",
            gold_points=[
                {
                    "requirement": "report the token count",
                    "quote": "8",
                    "pdf_page": 2,
                    "section": "Method",
                },
                {
                    "requirement": "mention Table 2",
                    "quote": "Table 2 reports results.",
                    "pdf_page": 2,
                    "section": "Results",
                },
            ],
        )
        problems = validate_candidate(candidate, make_pages(), ABSTRACT)
        self.assertFalse(
            any("number and a unit" in problem for problem in problems), problems
        )

    def test_numeric_question_without_any_unit_is_rejected(self) -> None:
        candidate = make_candidate(
            type="numeric",
            question_zh="这个数字是多少？",
            query_en="What is this number?",
            gold_points=[
                {
                    "requirement": "report the number",
                    "quote": "8",
                    "pdf_page": 2,
                    "section": "Method",
                }
            ],
        )
        problems = validate_candidate(candidate, make_pages(), ABSTRACT)
        self.assertTrue(
            any("number and a unit" in problem for problem in problems), problems
        )

    def test_numeric_question_with_unit_is_accepted(self) -> None:
        candidate = make_candidate(
            type="numeric",
            question_zh="每次查询处理多少 token？",
            query_en="How many tokens are processed per query?",
            gold_points=[
                {
                    "requirement": "token count",
                    "quote": "8 tokens per query",
                    "pdf_page": 2,
                    "section": "Method",
                }
            ],
        )
        self.assertEqual(validate_candidate(candidate, self.pages, ABSTRACT), [])

    def test_duplicate_question_text_against_existing_is_rejected(self) -> None:
        candidate = make_candidate(question_zh="该方法使用了什么重排策略？")
        errors = validate_candidate(
            candidate,
            self.pages,
            ABSTRACT,
            existing_question_texts=["该方法使用了什么重排策略？"],
        )
        self.assertTrue(any("duplicate" in error.lower() for error in errors), errors)

    def test_answerable_question_needs_at_least_one_gold_point(self) -> None:
        errors = validate_candidate(
            make_candidate(gold_points=[]), self.pages, ABSTRACT
        )
        self.assertTrue(any("gold" in error.lower() for error in errors), errors)

    def test_unanswerable_question_may_have_no_gold_points(self) -> None:
        candidate = make_candidate(
            type="unanswerable",
            answerability="unanswerable",
            question_zh="论文是否报告了 ImageNet 准确率？",
            query_en="Does the paper report ImageNet accuracy?",
            gold_points=[],
            missing_aspects=["no ImageNet experiment"],
        )
        self.assertEqual(validate_candidate(candidate, self.pages, ABSTRACT), [])


class ValidateQuestionSetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pages = make_pages()
        self.pages_by_paper = {PAPER_ID: self.pages}
        self.abstracts = {PAPER_ID: ABSTRACT}

    def test_duplicate_question_text_is_rejected_at_set_level(self) -> None:
        first = make_candidate(question_id="q001")
        second = make_candidate(
            question_id="q002",
            question_zh="  该方法使用了什么重排策略？ ",
            query_en="What reranking strategy is used?",
        )
        result = validate_question_set(
            [first, second], self.pages_by_paper, self.abstracts
        )
        self.assertFalse(result["ok"])
        self.assertIn("q002", result["duplicate_question_ids"])
        self.assertTrue(any("duplicate" in error.lower() for error in result["errors"]))

    def test_non_abstract_ratio_below_seventy_percent_fails(self) -> None:
        candidates = [
            make_candidate(
                question_id="q001",
                question_zh="问题一",
                query_en="Question one",
                gold_points=[
                    {
                        "requirement": "reranker",
                        "quote": "two-stage reranker",
                        "pdf_page": 2,
                        "section": "Method",
                    }
                ],
            ),
            make_candidate(
                question_id="q002",
                question_zh="问题二",
                query_en="Question two",
                gold_points=[
                    {
                        "requirement": "abstract phrase repeated in body",
                        "quote": "sparse retrieval",
                        "pdf_page": 3,
                        "section": "Method",
                    }
                ],
            ),
            make_candidate(
                question_id="q003",
                question_zh="问题三",
                query_en="Question three",
                gold_points=[
                    {
                        "requirement": "latency",
                        "quote": "40% lower latency",
                        "pdf_page": 2,
                        "section": "Results",
                    }
                ],
            ),
        ]
        result = validate_question_set(
            candidates, self.pages_by_paper, self.abstracts
        )
        self.assertEqual(result["answerable_quote_count"], 3)
        self.assertAlmostEqual(result["non_abstract_ratio"], 2 / 3, places=12)
        self.assertFalse(result["ok"])
        self.assertTrue(
            any("non-abstract" in error.lower() for error in result["errors"]),
            result["errors"],
        )

    def test_non_abstract_ratio_at_seventy_percent_passes(self) -> None:
        quotes = [
            ("two-stage reranker", 2),
            ("sparse retrieval", 3),
            ("40% lower latency", 2),
            ("8 tokens per query", 2),
        ]
        candidates = [
            make_candidate(
                question_id=f"q{index:03d}",
                question_zh=f"问题 {index}",
                query_en=f"Question {index}",
                gold_points=[
                    {
                        "requirement": f"point {index}",
                        "quote": quote,
                        "pdf_page": page,
                        "section": "Method",
                    }
                ],
            )
            for index, (quote, page) in enumerate(quotes, 1)
        ]
        result = validate_question_set(
            candidates, self.pages_by_paper, self.abstracts
        )
        self.assertEqual(result["answerable_quote_count"], 4)
        self.assertAlmostEqual(result["non_abstract_ratio"], 3 / 4, places=12)
        self.assertFalse(
            any("non-abstract" in error.lower() for error in result["errors"]),
            result["errors"],
        )


class GenerateForSplitTests(unittest.TestCase):
    def _write_pages(self, folder: Path, paper_id: str) -> Path:
        path = folder / f"{paper_id}.jsonl"
        rows = [
            {**page, "paper_id": paper_id}
            for page in make_pages(page_count=5)
        ]
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def test_writes_jsonl_and_usage_with_limit(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_BASE) as tmp:
            folder = Path(tmp)
            splits_path = folder / "splits.json"
            page_paths = [
                self._write_pages(folder, f"2607.0000{index}v1")
                for index in (1, 2)
            ]
            splits = {
                "test": [
                    {
                        "arxiv_id": f"2607.0000{index}v1",
                        "title": f"Paper {index}",
                        "prepared_path": str(path),
                    }
                    for index, path in zip((1, 2), page_paths)
                ]
            }
            splits_path.write_text(json.dumps(splits), encoding="utf-8")
            out_path = folder / "questions.draft.jsonl"
            chat = DynamicFakeChat()

            summary = generate_for_split(splits_path, out_path, chat, limit=2)

            self.assertEqual(summary["generated"], 2)
            self.assertEqual(summary["rejected"], 0)
            rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([row["question_id"] for row in rows], ["q001", "q002"])
            self.assertEqual(rows[0]["paper_ids"], ["2607.00001v1"])
            self.assertEqual(rows[1]["paper_ids"], ["2607.00002v1"])

            usage_path = out_path.parent / "generation_usage.json"
            self.assertTrue(usage_path.exists())
            usage = json.loads(usage_path.read_text(encoding="utf-8"))
            self.assertEqual(usage["model"], "fake-model")
            self.assertEqual(usage["calls"], 2)
            self.assertEqual(usage["totals"]["total_tokens"], 300)
            self.assertEqual(len(usage["questions"]), 2)
            self.assertRegex(
                usage["questions"][0]["prompt_sha256"], r"^[0-9a-f]{64}$"
            )

    def test_invalid_quote_is_repaired_on_second_call(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_BASE) as tmp:
            folder = Path(tmp)
            page_path = self._write_pages(folder, PAPER_ID)
            splits_path = folder / "splits.json"
            splits_path.write_text(
                json.dumps(
                    {
                        "test": [
                            {
                                "arxiv_id": PAPER_ID,
                                "title": "Paper 1",
                                "prepared_path": str(page_path),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out_path = folder / "questions.draft.jsonl"
            chat = RepairingFakeChat()

            summary = generate_for_split(splits_path, out_path, chat, limit=1)

            self.assertEqual(summary["generated"], 1)
            self.assertEqual(summary["rejected"], 0)
            self.assertEqual(chat.counter, 2)
            rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(rows[0]["gold_points"][0]["quote"], "two-stage reranker")
            usage = json.loads(
                (out_path.parent / "generation_usage.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(usage["calls"], 2)
            self.assertEqual(usage["question_count"], 1)

    def test_limit_one_only_processes_one_paper(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_BASE) as tmp:
            folder = Path(tmp)
            page_path = self._write_pages(folder, PAPER_ID)
            splits_path = folder / "splits.json"
            splits_path.write_text(
                json.dumps(
                    {
                        "test": [
                            {
                                "arxiv_id": PAPER_ID,
                                "title": "Paper 1",
                                "prepared_path": str(page_path),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out_path = folder / "questions.jsonl"
            summary = generate_for_split(
                splits_path, out_path, DynamicFakeChat(), limit=1
            )
            self.assertEqual(summary["generated"], 1)
            self.assertEqual(len(out_path.read_text(encoding="utf-8").splitlines()), 1)

    def test_invalid_candidate_is_rejected_and_not_written(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_BASE) as tmp:
            folder = Path(tmp)
            page_path = self._write_pages(folder, PAPER_ID)
            splits_path = folder / "splits.json"
            splits_path.write_text(
                json.dumps(
                    {
                        "test": [
                            {
                                "arxiv_id": PAPER_ID,
                                "title": "Paper 1",
                                "prepared_path": str(page_path),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out_path = folder / "questions.jsonl"
            chat = DynamicFakeChat(quote="quote that is not on any page")
            summary = generate_for_split(splits_path, out_path, chat)
            self.assertEqual(summary["generated"], 0)
            self.assertEqual(summary["rejected"], 1)
            self.assertEqual(
                out_path.read_text(encoding="utf-8").strip(), ""
            )
            self.assertTrue(summary["errors"])


if __name__ == "__main__":
    unittest.main()
