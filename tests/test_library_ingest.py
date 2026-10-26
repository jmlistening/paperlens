"""State-transition regression tests. No network, model loading or private data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import ingest_library as ingest
import library_transaction as tx
from prepare_corpus import atomic_write


class TransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "data" / "library_runs" / "test"
        self.work = self.run / "work"
        for name in tx.NAMES:
            atomic_write(self.root / "data" / "rag" / name, "old:" + name)
            atomic_write(self.work / "data" / "rag" / name, "new:" + name)
        # Exercise rollback of a new file that did not previously exist.
        (self.root / "data" / "rag" / "library_manifest.json").unlink()
        self.before = tx.state(self.root)

    def test_complete_publication(self):
        tx.publish(self.root, self.work, self.run, self.before)
        self.assertEqual(tx.state(self.root), tx.state(self.work))
        tx.check_available(self.root)
        self.assertEqual(tx.digest(self.run / "previous" / "corpus.json"), self.before["corpus.json"])

    def test_write_failure_restores_complete_previous_snapshot(self):
        original, failed = tx.copy_atomic, False
        def flaky(source, target):
            nonlocal failed
            if target == self.root / "data" / "rag" / "vector_store.npz" and not failed:
                failed = True
                raise OSError("injected write failure")
            return original(source, target)
        with patch.object(tx, "copy_atomic", side_effect=flaky):
            with self.assertRaises(OSError):
                tx.publish(self.root, self.work, self.run, self.before)
        self.assertEqual(tx.state(self.root), self.before)
        tx.check_available(self.root)

    def test_process_crash_blocks_reads_and_can_recover(self):
        code = '''import os,sys
from pathlib import Path
import library_transaction as tx
root,work,run=map(Path,sys.argv[1:])
original=tx.copy_atomic
def crash(source,target):
    original(source,target)
    if target == root/'data'/'rag'/'corpus.json':
        os._exit(17)
tx.copy_atomic=crash
tx.publish(root,work,run,tx.state(root))
'''
        proc = subprocess.run([sys.executable, "-c", code, str(self.root), str(self.work), str(self.run)],
                              cwd=PROJECT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 17, proc.stderr)
        with self.assertRaises(RuntimeError):
            tx.check_available(self.root)
        from search_vector import load_store
        with self.assertRaisesRegex(RuntimeError, "recover"):
            load_store(self.root)
        self.assertTrue(tx.recover(self.root))
        self.assertEqual(tx.state(self.root), self.before)
        self.assertFalse(tx.recover(self.root))

    def test_foreign_change_prevents_publication(self):
        atomic_write(self.root / "data" / "rag" / "corpus.json", "foreign edit")
        changed = tx.state(self.root)
        with self.assertRaises(RuntimeError):
            tx.publish(self.root, self.work, self.run, self.before)
        self.assertEqual(tx.state(self.root), changed)


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pdf = self.root / "data" / "pdfs" / "2610.12345v1.pdf"
        self.pdf.parent.mkdir(parents=True)
        import pymupdf
        with pymupdf.open() as doc:
            doc.new_page().insert_text((72, 100), "Retrieval augmented generation uses source evidence for question answering.")
            doc.save(self.pdf)

    def test_corrupt_pdf_stops_before_creating_run(self):
        self.pdf.write_bytes(b"not a pdf")
        with self.assertRaisesRegex(ValueError, "PDF"):
            ingest.run(self.root)
        self.assertFalse((self.root / "data" / "library_runs").exists())

    def test_missing_original_does_not_silently_shrink_library(self):
        atomic_write(self.root / "data" / "rag" / "corpus.json", json.dumps({
            "papers": [{"arxiv_id": "2610.99999v1", "pdf_sha256": "0" * 64}], "chunks": []}))
        before = tx.state(self.root)
        with self.assertRaisesRegex(ValueError, "原文缺失"):
            ingest.run(self.root)
        self.assertEqual(tx.state(self.root), before)

    def test_embedding_failure_keeps_live_files_and_reports_failure(self):
        import build_vector_index
        atomic_write(self.root / "data" / "rag" / "vector_store.npz", "previous index")
        before = tx.state(self.root)
        with patch.object(build_vector_index, "main", side_effect=RuntimeError("injected model failure")):
            with self.assertRaisesRegex(RuntimeError, "model failure"):
                ingest.run(self.root)
        self.assertEqual(tx.state(self.root), before)
        reports = list((self.root / "data" / "library_runs").glob("*/run.json"))
        self.assertEqual(len(reports), 1)
        self.assertEqual(json.loads(reports[0].read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
