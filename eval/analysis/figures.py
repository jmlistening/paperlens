"""Tiny text-only figures for the PaperLens Eval-lite Markdown report.

No plotting dependency and no image output: every figure is a line of text that
survives copy-paste into Markdown, a terminal or a pull request.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def text_bar(
    value: float | None,
    *,
    width: int = 24,
    maximum: float = 1.0,
) -> str:
    """Render ``value`` as a clamped bar, e.g. ``##########..........``.

    ``None`` and non-finite values render as an empty string so a missing
    metric is visibly absent rather than drawn as zero.
    """
    if value is None or width <= 0:
        return ""
    try:
        number = float(value)
        ceiling = float(maximum)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number) or not math.isfinite(ceiling) or ceiling <= 0:
        return ""
    ratio = max(0.0, min(1.0, number / ceiling))
    filled = int(round(ratio * width))
    return "#" * filled + "." * (width - filled)


def metric_bars(
    summaries: Sequence[Mapping],
    metric: str = "recall@6",
    *,
    width: int = 24,
) -> list[str]:
    """One aligned bar line per system that reports ``metric``."""
    lines: list[str] = []
    name_width = max(
        (len(str(summary.get("system", ""))) for summary in summaries),
        default=0,
    )
    for summary in summaries:
        metrics = summary.get("metrics") or {}
        value = metrics.get(metric)
        if isinstance(value, Mapping):
            value = value.get("mean")
        if value is None:
            continue
        name = str(summary.get("system", ""))
        lines.append(
            f"{name:<{name_width}}  {text_bar(value, width=width)}  {float(value):.3f}"
        )
    return lines


def bar_section(
    summaries: Sequence[Mapping],
    metric: str = "recall@6",
    *,
    width: int = 24,
) -> list[str]:
    """Markdown lines for a text-bar section; empty when nothing is measured."""
    lines = metric_bars(summaries, metric, width=width)
    if not lines:
        return []
    return [f"Mean {metric} (0 to 1):", "", "```text", *lines, "```", ""]
