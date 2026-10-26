"""Links-first digest rendering for the optional arXiv push."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
ALLOWED_HOSTS = {"arxiv.org", "www.arxiv.org"}
FOOTER = (
    "PaperLens 推送默认关闭。你可以在 Streamlit 的「推送设置」页关闭，"
    "或运行 `python push_ctl.py disable`。"
)
SUMMARY_SECTIONS = (
    "研究问题：",
    "核心方法：",
    "主要结果：",
    "阅读全文时需要确认：",
)


def format_summary(text: str) -> str:
    """Put a blank line between the five summary sections.

    The model often returns the five labels separated by a single newline (or
    spaces). Email HTML collapses bare newlines, so normalize the text once and
    let both the HTML and plain-text renderers use the same layout.
    """
    value = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not value:
        return ""
    for label in SUMMARY_SECTIONS:
        value = re.sub(
            r"\s*" + re.escape(label),
            "\n\n" + label,
            value,
            count=1,
        )
    return value.strip()


@dataclass
class Digest:
    subject: str
    html: str
    text: str
    attachments: list[tuple[str, bytes]] = field(default_factory=list)
    paper_keys: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _key(arxiv_id: str) -> str:
    return arxiv_id.split("v")[0]


def _checked_url(value: str) -> str:
    parsed = urlparse(value or "")
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() not in ALLOWED_HOSTS:
        raise ValueError(f"only arXiv links are allowed: {value!r}")
    return value


def build_digest(
    selected: list[dict[str, Any]],
    summaries: Mapping[str, str],
    config: Any,
    pdf_loader: Callable[[dict[str, Any]], tuple[str, bytes] | None] | None = None,
) -> Digest:
    limit = int(getattr(config, "max_per_digest", 3))
    if len(selected) > limit:
        raise ValueError(
            f"selected {len(selected)} papers but max_per_digest is {limit}"
        )

    cards_html: list[str] = []
    cards_text: list[str] = []
    attachments: list[tuple[str, bytes]] = []
    warnings: list[str] = []
    paper_keys: list[str] = []

    for index, paper in enumerate(selected, start=1):
        arxiv_id = paper["arxiv_id"]
        key = _key(arxiv_id)
        paper_keys.append(key)
        title = str(paper.get("title", ""))
        url = _checked_url(paper.get("url") or f"https://arxiv.org/abs/{arxiv_id}")
        assessment = paper.get("assessment") or {}
        score = assessment.get("score", "")
        reason = str(assessment.get("reason", ""))
        limitation = str(assessment.get("limitation", ""))
        summary = format_summary(
            str(summaries.get(key, paper.get("summary", "")))
        )
        categories = ", ".join(str(c) for c in (paper.get("categories") or [])[:3])

        cards_html.append(
            f"""
            <div style="background:#fff;border:1px solid #dfe5ed;border-radius:10px;
                        padding:22px;margin:18px 0;">
              <div style="color:#2563eb;font-size:14px;">
                推荐 {index} · 兴趣匹配分 {html.escape(str(score))}/100
              </div>
              <h2 style="font-size:19px;line-height:1.5;color:#172033;">
                {html.escape(title)}
              </h2>
              <p style="color:#596579;font-size:13px;">{html.escape(categories)}</p>
              <p><strong>推荐理由：</strong>{html.escape(reason)}</p>
              <p style="color:#596579;"><strong>需要注意：</strong>{html.escape(limitation)}</p>
              <div style="border-top:1px solid #e5e7eb;padding-top:16px;line-height:1.9;">
                {html.escape(summary).replace(chr(10), "<br>")}
              </div>
              <p><a href="{html.escape(url, quote=True)}" style="color:#2563eb;">
                打开 arXiv 论文页面</a></p>
            </div>
            """
        )
        cards_text.append(
            f"{index}. {title}\n"
            f"兴趣匹配分：{score}/100\n"
            f"推荐理由：{reason}\n"
            f"需要注意：{limitation}\n"
            f"{summary}\n"
            f"论文链接：{url}"
        )

        if getattr(config, "attach_pdfs", False) and pdf_loader is not None:
            loaded = pdf_loader(paper)
            if loaded is None:
                warnings.append(f"{arxiv_id}: PDF unavailable; link only")
            else:
                filename, content = loaded
                if not content.startswith(b"%PDF-"):
                    warnings.append(f"{arxiv_id}: attachment is not a PDF; link only")
                elif len(content) > MAX_ATTACHMENT_BYTES:
                    warnings.append(f"{arxiv_id}: PDF over 10 MB; link only")
                else:
                    attachments.append((filename, content))

    body = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head><meta charset="utf-8"><title>PaperLens 前沿论文推荐</title></head>
    <body style="margin:0;background:#f3f6fa;">
      <div style="max-width:800px;margin:auto;padding:28px 18px;
                  font-family:Arial,'Microsoft YaHei',sans-serif;
                  color:#263247;line-height:1.7;">
        <h1 style="margin-bottom:4px;color:#172033;">PaperLens</h1>
        <p style="margin-top:0;color:#596579;">
          前沿论文推荐 · 本期 {len(selected)} 篇
        </p>
        {''.join(cards_html)}
        <p style="font-size:12px;color:#667085;border-top:1px solid #e5e7eb;
                  margin-top:24px;padding-top:12px;">
          {html.escape(FOOTER)}
        </p>
      </div>
    </body></html>
    """
    text = (
        "PaperLens｜前沿论文推荐\n\n"
        + "\n\n".join(cards_text)
        + "\n\n"
        + FOOTER
    )
    subject = f"PaperLens｜前沿论文推荐（{len(selected)} 篇）"
    return Digest(
        subject=subject,
        html=body,
        text=text,
        attachments=attachments,
        paper_keys=paper_keys,
        warnings=warnings,
    )


def render_html(digest: Digest) -> str:
    return digest.html


def render_text(digest: Digest) -> str:
    return digest.text
