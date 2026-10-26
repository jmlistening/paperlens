"""Task E5b: build parent-chunk qrels from validated questions.

Each gold quote is mapped to every parent chunk in ``corpus_clean.json`` whose
text contains that quote after whitespace normalization. Unanswerable questions
always receive an empty gold list, even if a draft accidentally carries a gold
point. Quotes that match no chunk are reported in the summary instead of being
silently dropped.

The output is JSONL with exactly:
``question_id``, ``gold_parent_ids``, ``answerability``, ``missing_aspects``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]

QRELS_FIELDS = (
    "question_id",
    "gold_parent_ids",
    "answerability",
    "missing_aspects",
)


def normalize_ws(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _refuse_data_write(path: Path) -> None:
    resolved = Path(path).resolve()
    data_root = (PROJECT_ROOT / "data").resolve()
    if resolved == data_root or data_root in resolved.parents:
        raise ValueError(f"Refusing to write qrels under data/: {path}")


def load_questions(path: Path) -> list[dict]:
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
        if not isinstance(row, dict):
            raise ValueError(
                f"line {line_number} of {path} must be a JSON object"
            )
        rows.append(row)
    return rows


def _load_chunks(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        chunks = data
    elif isinstance(data, dict):
        chunks = data.get("chunks")
        if not isinstance(chunks, list):
            chunks = []
            for paper in data.get("papers") or []:
                if isinstance(paper, dict) and isinstance(paper.get("chunks"), list):
                    chunks.extend(paper["chunks"])
    else:
        raise ValueError(f"unsupported corpus format in {path}")

    result: list[dict] = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            result.append(chunk)
    return result


def _chunk_id(chunk: Mapping) -> str:
    for key in ("chunk_id", "parent_chunk_id", "id"):
        value = chunk.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _missing_aspects(question: Mapping) -> list[str]:
    value = question.get("missing_aspects")
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def build_qrels(
    questions_path: Path,
    clean_corpus_path: Path,
    out_path: Path,
) -> dict:
    """Map every gold quote to parent chunks and write qrels JSONL."""
    questions_path = Path(questions_path)
    clean_corpus_path = Path(clean_corpus_path)
    out_path = Path(out_path)
    _refuse_data_write(out_path)

    questions = load_questions(questions_path)
    chunks = _load_chunks(clean_corpus_path)
    normalized_chunks = [
        (_chunk_id(chunk), normalize_ws(chunk.get("text")))
        for chunk in chunks
        if _chunk_id(chunk)
    ]

    records: list[dict] = []
    unmatched: list[dict] = []
    with_gold = 0
    unanswerable_count = 0

    for question in questions:
        question_id = question.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("every question needs a non-empty question_id")

        raw_answerability = question.get("answerability")
        if raw_answerability:
            answerability = str(raw_answerability)
        elif question.get("type") == "unanswerable":
            answerability = "unanswerable"
        else:
            answerability = "answerable"
        gold_ids: list[str] = []

        if answerability == "unanswerable":
            unanswerable_count += 1
        else:
            for gold in question.get("gold_points") or []:
                if not isinstance(gold, dict):
                    continue
                quote = normalize_ws(gold.get("quote"))
                if not quote:
                    continue
                matches = [
                    chunk_id
                    for chunk_id, text in normalized_chunks
                    if quote in text
                ]
                if not matches:
                    unmatched.append(
                        {
                            "question_id": question_id,
                            "quote": str(gold.get("quote") or "")[:160],
                            "pdf_page": gold.get("pdf_page"),
                            "paper_ids": question.get("paper_ids") or [],
                        }
                    )
                gold_ids.extend(matches)

        unique_ids = sorted(set(gold_ids))
        if unique_ids:
            with_gold += 1
        records.append(
            {
                "question_id": question_id,
                "gold_parent_ids": unique_ids,
                "answerability": answerability,
                "missing_aspects": _missing_aspects(question),
            }
        )

    _atomic_write_text(
        out_path,
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records
        ),
    )

    return {
        "out_path": str(out_path),
        "count": len(records),
        "question_count": len(records),
        "with_gold": with_gold,
        "without_gold": len(records) - with_gold,
        "unanswerable_count": unanswerable_count,
        "gold_parent_id_count": sum(
            len(record["gold_parent_ids"]) for record in records
        ),
        "unmatched_quotes": unmatched,
        "records": records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build parent-chunk qrels for PaperLens benchmark questions"
    )
    parser.add_argument("--questions", required=True)
    parser.add_argument(
        "--clean-corpus",
        default=str(
            PROJECT_ROOT / "eval" / "benchmark_v2" / "index" / "corpus_clean.json"
        ),
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "eval" / "benchmark_v2" / "qrels.jsonl"),
    )
    args = parser.parse_args(argv)

    summary = build_qrels(
        Path(args.questions), Path(args.clean_corpus), Path(args.out)
    )
    printable = {key: value for key, value in summary.items() if key != "records"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
