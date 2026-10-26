"""Offline tests for the paired bootstrap comparison helper."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.analysis.paired_compare import (  # noqa: E402
    load_metrics,
    paired_bootstrap,
)

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"


def write_system(folder: Path, values: dict[str, float]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for question_id, value in values.items():
        (folder / f"{question_id}.json").write_text(
            json.dumps(
                {
                    "question_id": question_id,
                    "metrics": {"recall@6": value},
                }
            ),
            encoding="utf-8",
        )


class PairedCompareTests(unittest.TestCase):
    def test_load_and_paired_delta(self):
        TMP_ROOT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as tmp:
            root = Path(tmp)
            write_system(root / "control" / "sys", {"q1": 0.0, "q2": 0.5, "q3": 1.0})
            write_system(root / "treatment" / "sys", {"q1": 1.0, "q2": 0.5, "q3": 1.0})
            control = load_metrics(root / "control", "sys")
            treatment = load_metrics(root / "treatment", "sys")
            report = paired_bootstrap(
                control, treatment, "recall@6", n_bootstrap=200, seed=7
            )
            self.assertEqual(report["n_pairs"], 3)
            self.assertAlmostEqual(report["mean_delta"], 1 / 3, places=12)
            self.assertEqual(report["wins"], 1)
            self.assertEqual(report["losses"], 0)
            self.assertEqual(report["ties"], 2)
            self.assertLessEqual(report["ci95_low"], report["mean_delta"])
            self.assertGreaterEqual(report["ci95_high"], report["mean_delta"])

    def test_missing_system_raises(self):
        TMP_ROOT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as tmp:
            with self.assertRaises(FileNotFoundError):
                load_metrics(Path(tmp), "missing")


if __name__ == "__main__":
    unittest.main()
