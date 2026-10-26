"""Task E5b: programmatic validation for generated benchmark questions.

The validator is deliberately deterministic and offline. It enforces the rules
that make held-out questions usable as retrieval gold:

* every gold quote is a whitespace-normalized substring of the page it cites;
* the cited page exists in the extracted page set;
* a quote that exists only in the abstract is not acceptable evidence;
* at set level at least 70% of answerable gold quotes must be non-abstract;
* numeric questions quote a number together with a unit;
* normalized duplicate question text is rejected.

``validate_candidate`` checks one candidate. ``validate_question_set`` adds the
set-level rules (duplicates and the non-abstract ratio), because neither can be
decided from one candidate alone.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]

QUESTION_TYPES = (
    "factual",
    "numeric",
    "mechanism",
    "multi_hop",
    "partial",
    "unanswerable",
)
ANSWERABILITY_VALUES = ("answerable", "partial", "unanswerable")
ANSWERABLE_TYPES = ("factual", "numeric", "mechanism", "multi_hop")
MIN_NON_ABSTRACT_RATIO = 0.70
ABSTRACT_MAX_CHARS = 3000

_ABSTRACT_HEADING = re.compile(r"(?i)\babstract\b")
_ABSTRACT_STOP = re.compile(
    r"(?im)^\s*(?:1\.?\s*)?(?:introduction|keywords|index terms)\b"
)
_NUMBER = r"\d[\d,]*(?:\.\d+)?"
_SYMBOL_UNIT = r"(?:%|‰|°(?:C|F)?|×)"
_WORD_UNITS = (
    r"percent(?:age)?|pp|points?|tokens?|parameters?|params?|"
    r"gb|mb|kb|tb|ms|s|secs?|seconds?|mins?|minutes?|"
    r"h|hrs?|hours?|days?|weeks?|months?|years?|k|m|b|"
    r"x|thousand|million|billion|"
    r"kilograms?|kg|grams?|g|milligrams?|mg|"
    r"kilomet(?:er|re)s?|km|met(?:er|re)s?|centimet(?:er|re)s?|cm|"
    r"millimet(?:er|re)s?|mm|lit(?:er|re)s?|millilit(?:er|re)s?|ml|"
    r"hz|khz|mhz|ghz|watts?|w|kilowatts?|kw|volts?|"
    r"f1|bleu|rouge(?:-[0-9]+)?|map|mrr|ndcg|recall|precision|accuracy"
)
_NUMBER_UNIT = re.compile(
    rf"(?i){_NUMBER}\s*(?:{_SYMBOL_UNIT}|(?:{_WORD_UNITS})\b)"
)
_NUMBER_ONLY = re.compile(rf"(?i){_NUMBER}")
# A table cell may contain only the number; the unit can live in the question
# (e.g. "how many million tokens?"). Keep this list stricter than _WORD_UNITS
# to avoid matching single letters inside model names.
_QUESTION_UNIT = re.compile(
    r"(?i)(?:%|percent(?:age)?|million|billion|thousand|tokens?|parameters?|"
    r"params?|points?|samples?|documents?|pages?|ms|seconds?|minutes?|hours?|"
    r"百万|万|亿|千|百|倍|页|篇|条|词元)"
)


# ---------------------------------------------------------------------------
# Small text/page helpers shared with the generator
# ---------------------------------------------------------------------------


def normalize_ws(value: object) -> str:
    """Collapse every run of whitespace to a single space and strip ends."""
    if value is None:
        return ""
    return " ".join(str(value).split())


def page_number(page: Mapping) -> int:
    """Return a positive page number from the page-record variants, else 0."""
    for key in ("pdf_page", "page_number", "page"):
        if key not in page:
            continue
        value = page.get(key)
        if isinstance(value, bool):
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return 0


_HYPHEN_BREAK = re.compile(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])")


def page_text(page: Mapping) -> str:
    """Return the best available body text for a page record.

    Line-end hyphenation is joined (``align-\\nment`` -> ``alignment``) so the
    prompt and the validator use the same spelling. This mirrors the cleaned
    corpus used to build the test index.
    """
    for key in ("extracted_text", "normalized_text", "text", "content"):
        value = page.get(key)
        if isinstance(value, str) and value.strip():
            return _HYPHEN_BREAK.sub("", value)
    return ""


def load_pages(path: Path) -> list[dict]:
    """Load prepared pages from JSONL or a JSON list/``{"pages": [...]}`` file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []

    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return [row for row in parsed if isinstance(row, dict)]
    if isinstance(parsed, dict):
        for key in ("pages", "page_records"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if any(key in parsed for key in ("extracted_text", "pdf_page", "page_number")):
            return [parsed]
        return []

    rows: list[dict] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON on line {line_number} of {path}: {exc}"
            ) from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def extract_abstract(pages: Sequence[Mapping], max_chars: int = ABSTRACT_MAX_CHARS) -> str:
    """Extract the abstract from the first page, without the introduction."""
    ordered = sorted(pages, key=page_number)
    if not ordered:
        return ""
    first = next((page for page in ordered if page_number(page) == 1), ordered[0])
    text = page_text(first)
    match = _ABSTRACT_HEADING.search(text)
    if not match:
        return normalize_ws(text[:max_chars])

    after = text[match.end() :]
    stop = _ABSTRACT_STOP.search(after)
    fragment = after[: stop.start()] if stop else after[:max_chars]
    return fragment.strip()


def quote_in_abstract(quote: str, abstract: str) -> bool:
    """True when the quote is a normalized substring of the abstract."""
    quote_norm = normalize_ws(quote)
    abstract_norm = normalize_ws(abstract)
    return bool(quote_norm and abstract_norm and quote_norm in abstract_norm)


def _body_text(pages: Sequence[Mapping], abstract: str) -> str:
    """All page text variants with the abstract span removed, normalized."""
    variants: list[str] = []
    for page in pages:
        for key in ("extracted_text", "normalized_text", "text", "content"):
            value = page.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            normalized = normalize_ws(_HYPHEN_BREAK.sub("", value))
            if normalized and normalized not in variants:
                variants.append(normalized)
    body = " ".join(variants)
    abstract_norm = normalize_ws(abstract)
    if abstract_norm:
        body = body.replace(abstract_norm, " ")
    return body


def has_number_and_unit(text: str) -> bool:
    """True when the text contains a number adjacent to a recognizable unit."""
    return bool(_NUMBER_UNIT.search(text or ""))


def has_number(text: str) -> bool:
    return bool(_NUMBER_ONLY.search(text or ""))


def has_question_unit(text: str) -> bool:
    return bool(_QUESTION_UNIT.search(text or ""))


def _overlap_tokens(text: str) -> set[str]:
    latin = re.findall(r"[a-z0-9]+", (text or "").lower())
    chinese = re.findall(r"[㐀-鿿]", text or "")
    return set(latin) | set(chinese)


def question_copies_quote(question: str, quote: str) -> bool:
    """Reject a question that is essentially the gold quote restated."""
    question_norm = normalize_ws(question)
    quote_norm = normalize_ws(quote)
    if not question_norm or not quote_norm:
        return False
    if question_norm in quote_norm or quote_norm in question_norm:
        return True
    question_tokens = _overlap_tokens(question_norm)
    quote_tokens = _overlap_tokens(quote_norm)
    if not question_tokens or not quote_tokens:
        return False
    return (
        len(question_tokens & quote_tokens) / len(question_tokens | quote_tokens)
        >= 0.8
    )


def _page_map(pages: Sequence[Mapping]) -> dict[int, list[str]]:
    """Map each page number to every distinct normalized text variant."""
    result: dict[int, list[str]] = {}
    for page in pages:
        number = page_number(page)
        if not number:
            continue
        variants = result.setdefault(number, [])
        for key in ("extracted_text", "normalized_text", "text", "content"):
            value = page.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            normalized = normalize_ws(_HYPHEN_BREAK.sub("", value))
            if normalized and normalized not in variants:
                variants.append(normalized)
    return result


# ---------------------------------------------------------------------------
# Candidate validation
# ---------------------------------------------------------------------------


def validate_candidate(
    candidate: dict,
    pages: list[dict],
    abstract: str,
    *,
    existing_question_texts: Iterable[str] = (),
    strict_schema: bool = False,
) -> list[str]:
    """Return every problem found in one candidate (empty list means valid).

    The content rules (quote/page/abstract/numeric/duplicate) always run.
    ``strict_schema=True`` additionally requires every schema field, which the
    generator uses for model output. The default is lenient about missing
    metadata so the content rules can be tested on a focused fixture.
    """
    if not isinstance(candidate, dict):
        return ["candidate must be a dict"]

    errors: list[str] = []
    page_list = list(pages or [])
    if not isinstance(pages, (list, tuple)):
        errors.append("pages must be a list of page records")
        page_list = []

    question_id = candidate.get("question_id")
    if question_id is None:
        if strict_schema:
            errors.append("question_id must be a non-empty string")
    elif not isinstance(question_id, str) or not question_id.strip():
        errors.append("question_id must be a non-empty string")

    paper_ids = candidate.get("paper_ids")
    if paper_ids is None:
        if strict_schema:
            errors.append("paper_ids must be a non-empty list of paper ids")
    elif (
        not isinstance(paper_ids, list)
        or not paper_ids
        or not all(isinstance(item, str) and item.strip() for item in paper_ids)
    ):
        errors.append("paper_ids must be a non-empty list of paper ids")

    question_zh = candidate.get("question_zh")
    query_en = candidate.get("query_en")
    if question_zh is None:
        if strict_schema:
            errors.append("question_zh must be a non-empty string")
    elif not isinstance(question_zh, str) or not question_zh.strip():
        errors.append("question_zh must be a non-empty string")
    if query_en is None:
        if strict_schema:
            errors.append("query_en must be a non-empty string")
    elif not isinstance(query_en, str) or not query_en.strip():
        errors.append("query_en must be a non-empty string")

    question_type = candidate.get("type")
    if question_type is None:
        if strict_schema:
            errors.append("type must be one of: " + ", ".join(QUESTION_TYPES))
    elif question_type not in QUESTION_TYPES:
        errors.append(
            "type must be one of: " + ", ".join(QUESTION_TYPES)
        )

    answerability = candidate.get("answerability")
    if answerability is None:
        if strict_schema:
            errors.append(
                "answerability must be one of: "
                + ", ".join(ANSWERABILITY_VALUES)
            )
    elif answerability not in ANSWERABILITY_VALUES:
        errors.append(
            "answerability must be one of: " + ", ".join(ANSWERABILITY_VALUES)
        )

    if (
        question_type in ANSWERABLE_TYPES
        and answerability is not None
        and answerability != "answerable"
    ):
        errors.append(
            f"type {question_type!r} requires answerability 'answerable'"
        )
    elif (
        question_type == "partial"
        and answerability is not None
        and answerability != "partial"
    ):
        errors.append("type 'partial' requires answerability 'partial'")
    elif (
        question_type == "unanswerable"
        and answerability is not None
        and answerability != "unanswerable"
    ):
        errors.append(
            "type 'unanswerable' requires answerability 'unanswerable'"
        )

    missing_aspects = candidate.get("missing_aspects", [])
    if missing_aspects is None:
        missing_aspects = []
    if not isinstance(missing_aspects, list) or not all(
        isinstance(item, str) for item in missing_aspects
    ):
        errors.append("missing_aspects must be a list of strings")

    gold_points = candidate.get("gold_points")
    if gold_points is None:
        if strict_schema or answerability in ("answerable", "partial"):
            errors.append("gold_points must be a list")
        gold_points = []
    elif not isinstance(gold_points, list):
        errors.append("gold_points must be a list")
        gold_points = []
    if answerability in ("answerable", "partial") and not gold_points:
        errors.append(
            "answerable and partial questions require at least one gold point"
        )

    page_lookup = _page_map(page_list)
    abstract_norm = normalize_ws(abstract)
    body_norm = _body_text(page_list, abstract)
    numeric_point_ok = False

    for index, gold in enumerate(gold_points):
        prefix = f"gold_points[{index}]"
        if not isinstance(gold, dict):
            errors.append(f"{prefix} must be an object")
            continue

        requirement = gold.get("requirement")
        if not isinstance(requirement, str) or not requirement.strip():
            errors.append(f"{prefix}.requirement must be a non-empty string")

        section = gold.get("section", "")
        if section is not None and not isinstance(section, str):
            errors.append(f"{prefix}.section must be a string")

        quote = gold.get("quote")
        if not isinstance(quote, str) or not quote.strip():
            errors.append(f"{prefix}.quote must be a non-empty string")
            continue

        cited_page = gold.get("pdf_page")
        if (
            isinstance(cited_page, bool)
            or not isinstance(cited_page, int)
            or cited_page <= 0
        ):
            errors.append(f"{prefix}.pdf_page must be a positive integer")
            continue

        page_variants = page_lookup.get(cited_page)
        if page_variants is None:
            errors.append(f"{prefix}.pdf_page {cited_page} does not exist")
            continue

        quote_norm = normalize_ws(quote)
        if not any(quote_norm in text for text in page_variants):
            errors.append(
                f"{prefix}.quote is not a substring of page {cited_page}"
            )
            continue

        if (
            abstract_norm
            and quote_norm in abstract_norm
            and quote_norm not in body_norm
        ):
            errors.append(f"{prefix}.quote appears only in the abstract")

        if question_copies_quote(question_zh, quote):
            errors.append(
                f"{prefix}.question copies the gold quote too closely"
            )

        if question_type == "numeric":
            # A table cell may quote only the value; accept the unit from the
            # question/query in that case. At least one gold point must carry
            # the numeric evidence; auxiliary format/context points may not.
            unit_elsewhere = has_question_unit(question_zh) or has_question_unit(
                query_en
            )
            if has_number(quote) and (
                has_number_and_unit(quote) or unit_elsewhere
            ):
                numeric_point_ok = True

    if question_type == "numeric" and gold_points and not numeric_point_ok:
        errors.append(
            "numeric question requires at least one gold quote with a number "
            "and a unit (the unit may be in the question for a table cell)"
        )

    if isinstance(question_zh, str):
        existing_norm = {
            normalize_ws(text)
            for text in existing_question_texts
            if isinstance(text, str) and normalize_ws(text)
        }
        if normalize_ws(question_zh) in existing_norm:
            errors.append("duplicate question text")

    return errors


def validate_question_set(
    candidates: Sequence[dict],
    pages_by_paper: Mapping[str, Sequence[Mapping]] | None = None,
    abstracts_by_paper: Mapping[str, str] | None = None,
    *,
    min_non_abstract_ratio: float = MIN_NON_ABSTRACT_RATIO,
) -> dict:
    """Validate a whole candidate set, including duplicates and abstract ratio.

    ``pages_by_paper`` / ``abstracts_by_paper`` are keyed by the first id in
    each candidate's ``paper_ids``. When an abstract is not supplied it is
    extracted from that paper's first page.
    """
    page_lookup = pages_by_paper or {}
    abstract_lookup = abstracts_by_paper or {}
    rows = list(candidates or [])

    errors: list[str] = []
    per_question: dict[str, list[str]] = {}
    duplicate_question_ids: list[str] = []
    seen_texts: dict[str, str] = {}
    answerable_quote_count = 0
    non_abstract_quote_count = 0

    for index, candidate in enumerate(rows):
        if not isinstance(candidate, dict):
            errors.append(f"candidate[{index}] must be a dict")
            continue

        question_id = str(candidate.get("question_id") or f"candidate[{index}]")
        paper_ids = candidate.get("paper_ids")
        primary_id = ""
        if isinstance(paper_ids, list) and paper_ids:
            primary_id = str(paper_ids[0])

        pages = list(page_lookup.get(primary_id, ()))
        if primary_id in abstract_lookup:
            abstract = abstract_lookup[primary_id]
        else:
            abstract = extract_abstract(pages)

        existing = list(seen_texts.keys())
        question_errors = validate_candidate(
            candidate, pages, abstract, existing_question_texts=existing
        )

        question_norm = normalize_ws(candidate.get("question_zh"))
        if question_norm:
            if question_norm in seen_texts:
                duplicate_question_ids.append(question_id)
                duplicate_message = (
                    f"duplicate question text (also used by "
                    f"{seen_texts[question_norm]})"
                )
                if not any("duplicate question text" in item for item in question_errors):
                    question_errors.append(duplicate_message)
            else:
                seen_texts[question_norm] = question_id

        per_question[question_id] = question_errors
        errors.extend(f"{question_id}: {item}" for item in question_errors)

        if candidate.get("answerability") == "answerable":
            for gold in candidate.get("gold_points") or []:
                if not isinstance(gold, dict):
                    continue
                quote = gold.get("quote")
                quote_norm = normalize_ws(quote)
                if not quote_norm:
                    continue
                answerable_quote_count += 1
                if not quote_in_abstract(quote_norm, abstract):
                    non_abstract_quote_count += 1

    ratio = (
        non_abstract_quote_count / answerable_quote_count
        if answerable_quote_count
        else None
    )
    if ratio is not None and ratio + 1e-12 < min_non_abstract_ratio:
        errors.append(
            "non-abstract gold quote ratio "
            f"{ratio:.3f} is below the required {min_non_abstract_ratio:.2f}"
        )

    return {
        "ok": not errors,
        "errors": errors,
        "per_question": per_question,
        "duplicate_question_ids": duplicate_question_ids,
        "answerable_quote_count": answerable_quote_count,
        "non_abstract_quote_count": non_abstract_quote_count,
        "non_abstract_ratio": ratio,
        "min_non_abstract_ratio": min_non_abstract_ratio,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_questions(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON on line {line_number} of {path}: {exc}"
            ) from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate generated PaperLens benchmark questions"
    )
    parser.add_argument("--questions", required=True, help="questions JSONL")
    parser.add_argument("--splits", required=True, help="splits JSON")
    parser.add_argument(
        "--min-non-abstract-ratio", type=float, default=MIN_NON_ABSTRACT_RATIO
    )
    parser.add_argument("--json-out", help="optional report path")
    args = parser.parse_args(argv)

    questions = _load_questions(Path(args.questions))
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    papers = splits.get("test") if isinstance(splits, dict) else splits
    test_papers = {
        str(paper.get("arxiv_id")): paper
        for paper in (papers or [])
        if isinstance(paper, dict) and paper.get("arxiv_id")
    }

    pages_by_paper: dict[str, list[dict]] = {}
    abstracts_by_paper: dict[str, str] = {}
    for paper_id, paper in test_papers.items():
        prepared = paper.get("prepared_path")
        if not prepared:
            continue
        path = Path(prepared)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            continue
        pages = load_pages(path)
        pages_by_paper[paper_id] = pages
        abstracts_by_paper[paper_id] = (
            paper.get("abstract") or extract_abstract(pages)
        )

    report = validate_question_set(
        questions,
        pages_by_paper,
        abstracts_by_paper,
        min_non_abstract_ratio=args.min_non_abstract_ratio,
    )
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
