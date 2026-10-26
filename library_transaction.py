"""Publish a validated library snapshot, with backups and crash recovery.

Call publish/recover only while holding run_pipeline.pipeline_lock(root).
Readers check the journal; the six files are not one filesystem-atomic object.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

from prepare_corpus import atomic_write, sha256

NAMES = ("corpus.json", "corpus_clean.json", "vector_store.npz",
         "extraction_preview.txt", "cleaning_preview.txt", "library_manifest.json")
JOURNAL = ".library_transaction.json"


def digest(path):
    return sha256(path.read_bytes()) if path.is_file() else None


def state(root):
    folder = Path(root) / "data" / "rag"
    return {name: digest(folder / name) for name in NAMES}


def check_available(root):
    if (Path(root) / "data" / "rag" / JOURNAL).exists():
        raise RuntimeError("论文库更新中或上次发布中断。等待任务结束；若已中断，运行 "
                           "python ingest_library.py --recover 后重试。")


def copy_atomic(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as inp:
            shutil.copyfileobj(inp, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def recover(root):
    root = Path(root).resolve()
    folder = root / "data" / "rag"
    marker = folder / JOURNAL
    if not marker.exists():
        return False
    record = json.loads(marker.read_text(encoding="utf-8"))
    run = (root / record["run"]).resolve()
    if not run.is_relative_to((root / "data" / "library_runs").resolve()):
        raise ValueError("恢复目录不在 library_runs 内；未改动现有文件。")
    old, new = record["old"], record["new"]
    if set(old) != set(NAMES) or set(new) != set(NAMES):
        raise ValueError("恢复清单不完整；请保留运行目录。")
    # Validate every backup and destination before performing any restoration.
    for name in NAMES:
        if old[name] is not None and digest(run / "previous" / name) != old[name]:
            raise ValueError(f"备份校验失败：{name}；未开始恢复。")
        if digest(folder / name) not in (old[name], new[name]):
            raise ValueError(f"{name} 被其他程序修改，停止自动恢复以保护文件。")
    for name in NAMES:
        target = folder / name
        if old[name] is None:
            target.unlink(missing_ok=True)
        else:
            copy_atomic(run / "previous" / name, target)
    if state(root) != old:
        raise RuntimeError("恢复后校验失败；请保留备份和事务标记。")
    marker.unlink()
    return True


def publish(root, staged_root, run, expected):
    root, staged_root, run = map(lambda p: Path(p).resolve(), (root, staged_root, run))
    check_available(root)
    if state(root) != expected:
        raise RuntimeError("构建期间正式语料被其他程序修改，停止发布。")
    folder, staged = root / "data" / "rag", staged_root / "data" / "rag"
    new = state(staged_root)
    if any(value is None for value in new.values()):
        raise ValueError("待发布文件不完整。")
    if not run.is_relative_to(root / "data" / "library_runs"):
        raise ValueError("运行目录必须位于 data/library_runs。")
    for name in NAMES:
        if expected[name] is not None:
            copy_atomic(folder / name, run / "previous" / name)
    if state(root) != expected:
        raise RuntimeError("备份期间正式语料发生变化，停止发布。")
    marker = folder / JOURNAL
    atomic_write(marker, json.dumps({"run": run.relative_to(root).as_posix(),
                                   "old": expected, "new": new}, indent=2))
    try:
        for name in NAMES:
            copy_atomic(staged / name, folder / name)
        if state(root) != new:
            raise RuntimeError("发布后文件校验失败。")
        marker.unlink()
    except BaseException:
        # If restoration also fails, retain the journal for an explicit recovery.
        recover(root)
        raise
