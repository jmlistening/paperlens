"""Task E5b: freeze the benchmark artifacts into a hashed manifest.

Freezing records SHA-256 hashes and counts for the questions and qrels (plus
optional split/index artifacts). The manifest is keyed by a ``freeze_id`` that
is derived only from artifact hashes, so refreezing unchanged inputs is
idempotent. Changing an artifact requires an explicit ``overwrite=True``.

Nothing here touches the network, models, or ``data/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCHEMA_VERSION = 1


class FreezeError(RuntimeError):
    """Raised when a frozen manifest would be replaced without permission."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _require_file(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _refuse_data_write(path: Path) -> None:
    resolved = Path(path).resolve()
    data_root = (PROJECT_ROOT / "data").resolve()
    if resolved == data_root or data_root in resolved.parents:
        raise ValueError(f"Refusing to write a freeze manifest under data/: {path}")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
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
            raise ValueError(f"line {line_number} of {path} must be an object")
        rows.append(row)
    return rows


def _questions_summary(rows: Sequence[Mapping]) -> dict:
    ids = [
        str(row.get("question_id"))
        for row in rows
        if row.get("question_id") is not None
    ]
    types = Counter(str(row.get("type") or "missing") for row in rows)
    answerability = Counter(
        str(row.get("answerability") or "missing") for row in rows
    )
    review_status = Counter()
    gold_point_count = 0
    for row in rows:
        review = row.get("review")
        if isinstance(review, dict):
            review_status[str(review.get("status") or "missing")] += 1
        else:
            review_status["missing"] += 1
        gold_points = row.get("gold_points")
        if isinstance(gold_points, list):
            gold_point_count += len(gold_points)
    return {
        "count": len(rows),
        "question_ids": ids,
        "types": dict(sorted(types.items())),
        "answerability": dict(sorted(answerability.items())),
        "review_status": dict(sorted(review_status.items())),
        "gold_point_count": gold_point_count,
    }


def _qrels_summary(rows: Sequence[Mapping]) -> dict:
    with_gold = 0
    gold_id_count = 0
    answerability = Counter()
    for row in rows:
        answerability[str(row.get("answerability") or "missing")] += 1
        gold_ids = row.get("gold_parent_ids")
        if isinstance(gold_ids, list):
            gold_id_count += len(gold_ids)
            if gold_ids:
                with_gold += 1
    return {
        "count": len(rows),
        "with_gold": with_gold,
        "without_gold": len(rows) - with_gold,
        "gold_parent_id_count": gold_id_count,
        "answerability": dict(sorted(answerability.items())),
    }


def _artifact(path: Path) -> dict:
    path = Path(path)
    if path.is_dir():
        digest = hashlib.sha256()
        file_count = 0
        for child in sorted(path.rglob("*")):
            if not child.is_file():
                continue
            relative = child.relative_to(path).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(sha256_file(child).encode("ascii"))
            digest.update(b"\0")
            file_count += 1
        return {
            "path": str(path),
            "sha256": digest.hexdigest(),
            "kind": "directory",
            "file_count": file_count,
        }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "kind": "file",
        "size_bytes": path.stat().st_size,
    }


def _require_artifact(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file() and not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _freeze_id(artifacts: Mapping[str, Mapping]) -> str:
    fingerprint = {
        name: artifact.get("sha256")
        for name, artifact in sorted(artifacts.items())
        if isinstance(artifact, Mapping)
    }
    payload = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze_benchmark(
    questions_path: Path,
    qrels_path: Path,
    out_path: Path,
    *,
    overwrite: bool = False,
    extra_artifacts: Mapping[str, Path] | None = None,
    splits_path: Path | None = None,
    index_manifest_path: Path | None = None,
) -> dict:
    """Hash questions/qrels plus optional artifacts and write ``manifest_v1``."""
    questions_path = _require_file(Path(questions_path), "questions file")
    qrels_path = _require_file(Path(qrels_path), "qrels file")
    out_path = Path(out_path)
    _refuse_data_write(out_path)

    question_rows = _read_jsonl(questions_path)
    qrels_rows = _read_jsonl(qrels_path)

    artifacts: dict[str, dict] = {
        "questions": {
            **_artifact(questions_path),
            **_questions_summary(question_rows),
        },
        "qrels": {
            **_artifact(qrels_path),
            **_qrels_summary(qrels_rows),
        },
    }

    optional: list[tuple[str, Path]] = []
    if splits_path is not None:
        optional.append(("splits", Path(splits_path)))
    if index_manifest_path is not None:
        optional.append(("index_manifest", Path(index_manifest_path)))
    for name, path in (extra_artifacts or {}).items():
        optional.append((str(name), Path(path)))

    for name, path in optional:
        path = _require_artifact(path, f"artifact {name!r}")
        artifacts[name] = _artifact(path)

    freeze_id = _freeze_id(artifacts)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "created_at": _utc_now(),
        "freeze_id": freeze_id,
        "manifest_hash": freeze_id,
        "questions_sha256": artifacts["questions"]["sha256"],
        "qrels_sha256": artifacts["qrels"]["sha256"],
        "artifacts": artifacts,
    }

    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and existing.get("freeze_id") == freeze_id:
            result = dict(existing)
            result["already_frozen"] = True
            return result
        if not overwrite:
            raise FreezeError(
                f"{out_path} already exists with a different freeze_id; "
                "pass overwrite=True to replace it"
            )

    _atomic_write_text(
        out_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return dict(manifest)


def _parse_extra(values: Sequence[str]) -> dict[str, Path]:
    extras: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--extra must be NAME=PATH, got: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"--extra needs a non-empty name: {value}")
        extras[name] = Path(raw_path)
    return extras


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze PaperLens benchmark questions and qrels"
    )
    parser.add_argument("--questions", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "eval" / "benchmark_v2" / "manifest_v1.json"),
    )
    parser.add_argument("--splits", default=None)
    parser.add_argument("--index-manifest", default=None)
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="additional NAME=PATH artifact to hash",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    manifest = freeze_benchmark(
        Path(args.questions),
        Path(args.qrels),
        Path(args.out),
        overwrite=args.overwrite,
        extra_artifacts=_parse_extra(args.extra),
        splits_path=Path(args.splits) if args.splits else None,
        index_manifest_path=(
            Path(args.index_manifest) if args.index_manifest else None
        ),
    )
    printable = {
        key: manifest.get(key)
        for key in (
            "status",
            "freeze_id",
            "created_at",
            "questions_sha256",
            "qrels_sha256",
            "already_frozen",
        )
        if key in manifest
    }
    printable["out_path"] = str(args.out)
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
