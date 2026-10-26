#!/usr/bin/env python
"""Read-only baseline snapshot for PaperLens.

This script intentionally does not import project business modules and does not
read .env. It only reads persisted manifests/states and SQLite table counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "eval" / "baseline_snapshot_20261025.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count_files(pattern: str) -> int:
    return sum(1 for _ in ROOT.glob(pattern))


def sqlite_table_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    uri = f"file:{path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        return {
            table: conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in tables
        }


def vector_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as store:
        metadata = json.loads(str(store["metadata"].item()))
    keys = [
        "model_name",
        "model_revision",
        "dimension",
        "index_type",
        "parent_chunk_count",
        "vector_count",
    ]
    return {key: metadata.get(key) for key in keys}


def build_snapshot() -> dict[str, Any]:
    manifest_path = ROOT / "data" / "rag" / "library_manifest.json"
    corpus_clean_path = ROOT / "data" / "rag" / "corpus_clean.json"
    vector_path = ROOT / "data" / "rag" / "vector_store.npz"
    dataset_v1_path = ROOT / "data" / "dataset_v1" / "state.json"
    history_v1_path = ROOT / "data" / "dataset_history_v1" / "state.json"

    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    corpus_clean = read_json(corpus_clean_path) if corpus_clean_path.exists() else {}
    dataset_v1 = read_json(dataset_v1_path) if dataset_v1_path.exists() else {}
    history_v1 = read_json(history_v1_path) if history_v1_path.exists() else {}

    hashes = {}
    for label, path in {
        "library_manifest": manifest_path,
        "corpus_clean": corpus_clean_path,
        "vector_store": vector_path,
        "dataset_v1_state": dataset_v1_path,
        "dataset_history_v1_state": history_v1_path,
    }.items():
        hashes[label] = sha256_file(path) if path.exists() else None

    return {
        "snapshot_schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "dev_index": {
            "paper_count": corpus_clean.get("paper_count"),
            "chunk_count": corpus_clean.get("chunk_count"),
            "manifest_source_papers": len(manifest.get("source_hashes", {})),
            "manifest_outputs": len(manifest.get("output_hashes", {})),
            "vector": vector_metadata(vector_path),
        },
        "qa_history": {
            "answer_result_files": count_files("data/rag/answers/*/answer_result.json"),
            "qa_run_files": count_files("data/rag/qa_runs/*/run.json"),
            "agent_run_files": count_files("data/rag/agent_runs/*/run.json"),
        },
        "datasets": {
            "dataset_v1": {
                "papers": len(dataset_v1.get("papers", {})),
                "queries": len(dataset_v1.get("queries", [])),
                "duplicate_hits": dataset_v1.get("duplicate_hits"),
            },
            "dataset_history_v1": {
                "papers": len(history_v1.get("papers", {})),
                "queries": len(history_v1.get("queries", [])),
                "duplicate_hits": history_v1.get("duplicate_hits"),
                "excluded_hits": history_v1.get("excluded_hits"),
            },
        },
        "sqlite_counts": {
            "reading": sqlite_table_counts(ROOT / "data" / "reading.sqlite3"),
            "delivery": sqlite_table_counts(ROOT / "data" / "delivery.sqlite3"),
        },
        "hashes": hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    snapshot = build_snapshot()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
