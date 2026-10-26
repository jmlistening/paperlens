"""Markdown + JSON report for PaperLens Eval-lite runs (Task E6).

``make_report(results, out_dir)`` takes the per-system summaries produced by
``eval.run_experiment.run_experiment`` (its ``results`` list) and writes:

* ``out_dir/results.json`` -- machine-readable means, bootstrap intervals,
  sample counts, failures and skipped systems.
* ``out_dir/report.md`` -- a human-readable main table.

Statistics policy
-----------------
Every interval is a percentile bootstrap over questions, with a fixed seed so
the report is reproducible. The report performs **no hypothesis tests**. The
specification reserves McNemar/Holm for a 120-question benchmark; Eval-lite
(n=40 in practice) is descriptive only. Failed or invalid system outputs were
already scored 0 by the runner and stay in every denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.analysis.figures import bar_section  # noqa: E402

REPORT_VERSION = "eval_report_v1"
DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 20261025
CONFIDENCE = 0.95
SIGNIFICANCE_MIN_N = 120

METRIC_ORDER = (
    "recall@3",
    "recall@6",
    "recall@10",
    "all_gold@6",
    "mrr@10",
    "ndcg@10",
    "quote_integrity",
    "schema_success",
)
METRIC_LABELS = {
    "recall@3": "Recall@3",
    "recall@6": "Recall@6",
    "recall@10": "Recall@10",
    "all_gold@6": "AllGold@6",
    "mrr@10": "MRR@10",
    "ndcg@10": "nDCG@10",
    "quote_integrity": "Quote integrity",
    "schema_success": "Schema success",
}

SIGNIFICANCE_NOTE = (
    "No significance testing was performed. These are descriptive statistics; "
    "one fixed benchmark run cannot establish a difference between systems."
)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def _seed_for(system: str, metric: str) -> int:
    digest = hashlib.sha256(f"{system}\x1f{metric}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = max(0.0, min(1.0, quantile)) * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    )


def _clean_values(values: Sequence[Any] | None) -> list[float]:
    cleaned: list[float] = []
    for value in values or ():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            cleaned.append(float(value))
    return cleaned


def bootstrap_ci(
    values: Sequence[float] | None,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> dict[str, float | int] | None:
    """Percentile bootstrap CI of the mean; ``None`` when there is no sample.

    The sample mean is reported as ``mean`` (the bootstrap distribution is only
    used for the interval). The result is deterministic for a fixed seed.
    """
    sample = _clean_values(values)
    if not sample:
        return None
    mean = sum(sample) / len(sample)
    if len(sample) == 1:
        return {
            "mean": mean,
            "ci_low": mean,
            "ci_high": mean,
            "n": 1,
            "resamples": 0,
        }
    if resamples <= 0:
        raise ValueError("resamples must be > 0")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    rng = random.Random(seed)
    count = len(sample)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(count):
            total += sample[rng.randrange(count)]
        means.append(total / count)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": mean,
        "ci_low": _percentile(means, alpha),
        "ci_high": _percentile(means, 1.0 - alpha),
        "n": count,
        "resamples": resamples,
    }


# ---------------------------------------------------------------------------
# Building the report model
# ---------------------------------------------------------------------------


def _metric_order(samples: Mapping[str, Any]) -> list[str]:
    ordered = [metric for metric in METRIC_ORDER if metric in samples]
    ordered.extend(sorted(metric for metric in samples if metric not in METRIC_ORDER))
    return ordered


def _build_system(item: Mapping) -> dict:
    name = item.get("system")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"result without a system name: {item!r}")
    name = name.strip()

    raw_samples = item.get("metric_samples")
    samples_map: dict[str, list[float]] = {}
    if isinstance(raw_samples, Mapping):
        for metric, values in raw_samples.items():
            if not isinstance(metric, str):
                continue
            cleaned = _clean_values(values if isinstance(values, Sequence) else None)
            if cleaned:
                samples_map[metric] = cleaned

    metrics: dict[str, dict] = {}
    for metric in _metric_order(samples_map):
        interval = bootstrap_ci(samples_map[metric], seed=_seed_for(name, metric))
        if interval is not None:
            metrics[metric] = interval

    failures = item.get("failures")
    return {
        "system": name,
        "status": str(item.get("status", "unknown")),
        "n": int(item.get("n", 0) or 0),
        "n_failed": int(item.get("n_failed", 0) or 0),
        "n_retrieval": int(item.get("n_retrieval", 0) or 0),
        "tokens": int(item.get("tokens", 0) or 0),
        "failures": [str(value) for value in failures] if isinstance(failures, list) else [],
        "skipped_reason": item.get("skipped_reason"),
        "metrics": metrics,
        "samples": samples_map,
    }


def _format_interval(interval: Mapping | None) -> str:
    if not isinstance(interval, Mapping):
        return "n/a"
    return (
        f"{float(interval['mean']):.3f} "
        f"[{float(interval['ci_low']):.3f}, {float(interval['ci_high']):.3f}]"
    )


def _format_systems_table(systems: Sequence[Mapping], metric_keys: Sequence[str]) -> list[str]:
    headers = ["System", "Status", "n", "Failed", "Tokens"]
    headers.extend(f"{METRIC_LABELS.get(metric, metric)} (95% CI)" for metric in metric_keys)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for system in systems:
        cells = [
            str(system["system"]),
            str(system["status"]),
            str(system["n"]),
            str(system["n_failed"]),
            str(system["tokens"]),
        ]
        for metric in metric_keys:
            cells.append(_format_interval(system["metrics"].get(metric)))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def make_report(results: list[dict], out_dir: Path) -> str:
    """Write ``results.json`` and ``report.md``; return the Markdown text."""
    if isinstance(results, Mapping):  # be forgiving about run.json payloads
        results = results.get("results")  # type: ignore[assignment]
    if not isinstance(results, (list, tuple)):
        raise TypeError("results must be a list of per-system summary dicts")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    systems = [_build_system(item) for item in results]
    metric_keys = sorted(
        {metric for system in systems for metric in system["metrics"]},
        key=lambda metric: (
            METRIC_ORDER.index(metric) if metric in METRIC_ORDER else len(METRIC_ORDER),
            metric,
        ),
    )

    payload = {
        "version": REPORT_VERSION,
        "generated_by": "eval/analysis/make_report.py",
        "confidence": CONFIDENCE,
        "bootstrap_resamples": DEFAULT_RESAMPLES,
        "bootstrap_seed": DEFAULT_SEED,
        "significance_testing": False,
        "significance_note": SIGNIFICANCE_NOTE,
        "significance_min_n": SIGNIFICANCE_MIN_N,
        "n_systems": len(systems),
        "systems": systems,
    }
    (out_path / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines: list[str] = [
        "# PaperLens Eval-lite Report",
        "",
        "Descriptive statistics only. Failed or invalid system outputs are scored 0 "
        "and stay in every denominator; they are never silently dropped.",
        "",
        f"Bootstrap intervals: {DEFAULT_RESAMPLES} percentile resamples of the "
        f"question-level scores, {CONFIDENCE:.0%} interval, seed {DEFAULT_SEED}.",
        "",
        SIGNIFICANCE_NOTE,
        "",
        "## Main table",
        "",
    ]
    if not systems:
        lines.extend(["No systems were run; there are no metrics to report.", ""])
    else:
        lines.extend(_format_systems_table(systems, metric_keys))
        lines.append("")

    skipped = [system for system in systems if system["status"] == "skipped"]
    if skipped:
        lines.extend(["## Skipped systems (no numbers produced)", ""])
        for system in skipped:
            reason = system.get("skipped_reason") or "not_implemented"
            lines.append(f"- `{system['system']}`: skipped ({reason}).")
        lines.append("")

    with_failures = [system for system in systems if system["failures"]]
    if with_failures:
        lines.extend(["## Failed or invalid questions", ""])
        for system in with_failures:
            failed = ", ".join(f"`{name}`" for name in system["failures"])
            lines.append(
                f"- `{system['system']}`: {len(system['failures'])} of "
                f"{system['n']} questions scored 0 ({failed})."
            )
        lines.append("")

    bars = bar_section(systems, "recall@6")
    if bars:
        lines.extend([f"## Recall@6 overview", "", *bars])

    lines.extend(
        [
            "## Limitations",
            "",
            "- Retrieval-first Eval-lite: no answer-quality, claim-level or LLM-judge metric is included.",
            "- A failed, crashed or schema-invalid system output counts as 0; it is never dropped.",
            "- Questions are a fixed benchmark, not a random sample, so bootstrap intervals are descriptive.",
            "- No hypothesis tests were run. Do not present these rows as a confirmed improvement.",
            "- Systems marked `skipped` are declared stubs and contribute no measurement.",
            "- `n` counts attempted questions; metric-level `n` can be smaller when a question has no gold evidence.",
            "",
        ]
    )

    markdown = "\n".join(lines) + "\n"
    (out_path / "report.md").write_text(markdown, encoding="utf-8")
    return markdown


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _latest_run_dir() -> Path | None:
    runs_dir = PROJECT_ROOT / "eval" / "benchmark_v2" / "runs"
    if not runs_dir.is_dir():
        return None
    candidates = [path for path in runs_dir.iterdir() if (path / "run.json").is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a PaperLens Eval-lite report from a run directory."
    )
    parser.add_argument(
        "out_dir",
        nargs="?",
        type=Path,
        default=None,
        help="run directory containing run.json (default: latest under eval/benchmark_v2/runs)",
    )
    parser.add_argument(
        "--run-json",
        type=Path,
        default=None,
        help="explicit path to a run.json",
    )
    args = parser.parse_args(argv)

    if args.run_json is not None:
        run_json_path = args.run_json
        out_dir = args.out_dir or run_json_path.parent
    elif args.out_dir is not None:
        out_dir = args.out_dir
        run_json_path = out_dir / "run.json"
    else:
        latest = _latest_run_dir()
        if latest is None:
            parser.error("no run directory with run.json found under eval/benchmark_v2/runs")
        out_dir = latest
        run_json_path = latest / "run.json"

    if not run_json_path.is_file():
        parser.error(f"run.json not found: {run_json_path}")
    payload = json.loads(run_json_path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if results is None and isinstance(payload.get("systems"), Mapping):
        results = list(payload["systems"].values())
    if not isinstance(results, list):
        parser.error(f"{run_json_path}: expected a 'results' list")

    print(make_report(results, out_dir))
    print(f"\nArtifacts: {Path(out_dir).resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
