"""Paired bootstrap comparison of two Eval-lite system runs."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

DEFAULT_METRICS = ("recall@6", "ndcg@10", "mrr@10")
DEFAULT_BOOTSTRAP = 10000
DEFAULT_SEED = 20261025


def load_metrics(run_dir: Path, system: str) -> dict[str, dict[str, float]]:
    folder = Path(run_dir) / system
    if not folder.is_dir():
        raise FileNotFoundError(f"system results not found: {folder}")
    rows: dict[str, dict[str, float]] = {}
    for path in sorted(folder.glob("*.json")):
        if path.name in {"run.json", "metrics.json", "summary.json"}:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        question_id = data.get("question_id") or path.stem
        metrics = data.get("metrics")
        if isinstance(question_id, str) and isinstance(metrics, dict):
            rows[question_id] = {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
    if not rows:
        raise ValueError(f"no per-question metrics in {folder}")
    return rows


def paired_bootstrap(
    control: dict[str, dict[str, float]],
    treatment: dict[str, dict[str, float]],
    metric: str,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    common = sorted(set(control) & set(treatment))
    if not common:
        raise ValueError("the two runs share no question ids")
    diffs = [
        treatment[qid].get(metric, 0.0) - control[qid].get(metric, 0.0)
        for qid in common
    ]
    rng = random.Random(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        means.append(sum(sample) / len(sample))
    means.sort()
    lower = means[int(0.025 * (len(means) - 1))]
    upper = means[int(0.975 * (len(means) - 1))]
    wins = sum(1 for value in diffs if value > 0)
    losses = sum(1 for value in diffs if value < 0)
    ties = len(diffs) - wins - losses
    return {
        "metric": metric,
        "n_pairs": len(common),
        "control_mean": sum(control[q][metric] for q in common) / len(common),
        "treatment_mean": sum(treatment[q][metric] for q in common) / len(common),
        "mean_delta": sum(diffs) / len(diffs),
        "ci95_low": lower,
        "ci95_high": upper,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def compare(
    control_dir: Path,
    control_system: str,
    treatment_dir: Path,
    treatment_system: str,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    control = load_metrics(control_dir, control_system)
    treatment = load_metrics(treatment_dir, treatment_system)
    return {
        "control": {
            "run_dir": str(control_dir),
            "system": control_system,
            "questions": len(control),
        },
        "treatment": {
            "run_dir": str(treatment_dir),
            "system": treatment_system,
            "questions": len(treatment),
        },
        "comparisons": [
            paired_bootstrap(
                control, treatment, metric,
                n_bootstrap=n_bootstrap, seed=seed,
            )
            for metric in metrics
        ],
        "note": (
            "Descriptive paired bootstrap on a fixed benchmark; no "
            "pre-registration or multiple-comparison correction."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paired bootstrap comparison",
        "",
        f"Control: `{report['control']['system']}` ({report['control']['questions']} questions)",
        f"Treatment: `{report['treatment']['system']}` ({report['treatment']['questions']} questions)",
        "",
        "| Metric | Control | Treatment | Delta | 95% CI | W/L/T |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in report["comparisons"]:
        lines.append(
            f"| {row['metric']} | {row['control_mean']:.4f} | "
            f"{row['treatment_mean']:.4f} | {row['mean_delta']:+.4f} | "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | "
            f"{row['wins']}/{row['losses']}/{row['ties']} |"
        )
    lines += ["", report["note"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-run", required=True, type=Path)
    parser.add_argument("--control-system", required=True)
    parser.add_argument("--treatment-run", required=True, type=Path)
    parser.add_argument("--treatment-system", required=True)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare(
        args.control_run,
        args.control_system,
        args.treatment_run,
        args.treatment_system,
        metrics=tuple(part.strip() for part in args.metrics.split(",") if part.strip()),
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.out.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
