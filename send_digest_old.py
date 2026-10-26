import argparse
import hashlib
import json
import re
import smtplib
import ssl
from email.message import EmailMessage
from email.policy import SMTP
from email.utils import formatdate, make_msgid
from html import escape
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parent
NOTICE = (
    "中文速览与推荐理由仅依据标题和摘要，未经全文核验。"
    "兴趣匹配分不代表论文质量或准确率。附件为英文原版 PDF。"
)


def html_text(value):
    return escape(str(value)).replace("\n", "<br>")


def checked_url(value):
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https"):
        raise ValueError("论文链接必须使用 HTTP 或 HTTPS。")
    if parts.hostname not in ("arxiv.org", "www.arxiv.org"):
        raise ValueError("论文链接不是预期的 arXiv 地址。")
    return value


def prepare_digest():
    source = ROOT / "data" / "recommendations.json"
    source_bytes = source.read_bytes()
    recommendations = json.loads(source_bytes.decode("utf-8-sig"))

    data = json.loads(
        (ROOT / "data" / "selected_summaries.json").read_text(
            encoding="utf-8-sig"
        )
    )

    current_hash = hashlib.sha256(source_bytes).hexdigest()
    if data.get("recommendations_sha256") != current_hash:
        raise ValueError(
            "推荐清单已变化，请先重新运行 summarize_selected.py。"
        )

    papers = data["papers"]
    if not papers:
        raise ValueError("本轮没有入选论文，无需发送。")

    ids = [p["arxiv_id"] for p in papers]
    if ids != recommendations["selected_ids"]:
        raise ValueError("速览与推荐清单不对应，请重新生成速览。")

    cards = []
    plain_parts = ["PaperLens｜个性化论文推荐", NOTICE]
    attachments = []
    attachment_bytes = 0

    for index, paper in enumerate(papers, start=1):
        assessment = paper["assessment"]
        title = paper["title"]
        summary = paper["summary"]
        url = checked_url(paper["url"])

        filename = re.sub(
            r"[^A-Za-z0-9._-]", "_", paper["arxiv_id"]
        ) + ".pdf"
        pdf = ROOT / "data" / "pdfs" / filename

        if not pdf.is_file():
            raise ValueError(
                f"缺少 {filename}，请先运行 download_selected.py。"
            )

        attachment_bytes += pdf.stat().st_size
        if attachment_bytes > 10 * 1024 * 1024:
            raise ValueError(
                "附件合计超过本阶段设置的 10 MB 上限。"
                "后续可增加拆分邮件功能。"
            )

        pdf_data = pdf.read_bytes()
        if not pdf_data.startswith(b"%PDF-"):
            raise ValueError(f"{filename} 的文件头不是 PDF。")
        attachments.append((filename, pdf_data))

        plain_parts.append(
            f"{index}. {title}\n"
            f"兴趣匹配分：{assessment['score']}/100\n"
            f"推荐理由：{assessment['reason']}\n"
            f"需要注意：{assessment['limitation']}\n\n"
            f"{summary}\n\n"
            f"论文链接：{url}\n"
            f"附件：{filename}"
        )

        cards.append(f"""
        <div style="background:#ffffff;border:1px solid #dfe5ed;
                    border-radius:10px;padding:22px;margin:18px 0;">
          <div style="color:#2563eb;font-size:14px;">
            推荐 {index} · 兴趣匹配分 {assessment['score']}/100
          </div>
          <h2 style="font-size:19px;line-height:1.5;color:#172033;">
            {html_text(title)}
          </h2>
          <p><strong>推荐理由：</strong>
            {html_text(assessment['reason'])}</p>
          <p style="color:#596579;"><strong>需要注意：</strong>
            {html_text(assessment['limitation'])}</p>
          <div style="border-top:1px solid #e5e7eb;
                      padding-top:16px;line-height:1.9;">
            {html_text(summary)}
          </div>
          <p><a href="{escape(url, quote=True)}"
                style="color:#2563eb;">打开 arXiv 论文页面</a></p>
          <p style="font-size:13px;color:#667085;">
            英文 PDF 附件：{html_text(filename)}
          </p>
        </div>
        """)

    html_body = f"""<!DOCTYPE html>
    <html lang="zh-CN">
    <head><meta charset="utf-8"><title>PaperLens 论文推荐</title></head>
    <body style="margin:0;background:#f3f6fa;">
      <div style="max-width:800px;margin:auto;padding:28px 18px;
                  font-family:Arial,'Microsoft YaHei',sans-serif;
                  color:#263247;line-height:1.7;">
        <h1 style="margin-bottom:4px;color:#172033;">PaperLens</h1>
        <p style="margin-top:0;color:#596579;">
          个性化论文推荐 · 本期 {len(papers)} 篇
        </p>
        <p style="font-size:13px;color:#667085;">{html_text(NOTICE)}</p>
        {''.join(cards)}
      </div>
    </body></html>
    """

    preview = ROOT / "data" / "digest_preview.html"
    preview.write_text(html_body, encoding="utf-8")
    print(f"邮件正文预览：{preview}")
    print(f"论文：{len(papers)} 篇，PDF 附件：{len(attachments)} 个")
    print(f"附件原始大小合计：{attachment_bytes / 1024 / 1024:.2f} MB")

    return "\n\n".join(plain_parts), html_body, attachments


def send_digest(plain_body, html_body, attachments):
    config = dotenv_values(ROOT / ".env", encoding="utf-8-sig")
    required = (
        "SMTP_HOST", "SMTP_PORT", "SMTP_USER",
        "SMTP_PASSWORD", "MAIL_TO",
    )
    if any(not (config.get(key) or "").strip() for key in required):
        raise ValueError(".env 缺少邮件配置。")

    sender = config["SMTP_USER"].strip()
    receiver = config["MAIL_TO"].strip()
    pattern = r"[^@\s,;<>]+@[^@\s,;<>]+\.[^@\s,;<>]+"
    for address in (sender, receiver):
        if not address.isascii() or not re.fullmatch(pattern, address):
            raise ValueError("请配置单个、完整的邮箱地址。")

    port = int(config["SMTP_PORT"].strip())
    if port != 465:
        raise ValueError("本程序使用 SSL 连接，请沿用 SMTP_PORT=465。")

    message = EmailMessage()
    message["Subject"] = f"PaperLens｜为你推荐 {len(attachments)} 篇论文"
    message["From"] = sender
    message["To"] = receiver
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid()
    message.set_content(plain_body)
    message.add_alternative(html_body, subtype="html")

    for filename, content in attachments:
        message.add_attachment(
            content,
            maintype="application",
            subtype="pdf",
            filename=filename,
        )

    # 邮件编码会增加体积；这里是项目测试上限，不是邮箱官方限制。
    if len(message.as_bytes(policy=SMTP)) > 15 * 1024 * 1024:
        raise ValueError("编码后的邮件超过本阶段设置的 15 MB 上限。")

    server = None
    try:
        print("正在连接邮箱服务器并发送……", flush=True)
        server = smtplib.SMTP_SSL(
            config["SMTP_HOST"].strip(),
            port,
            context=ssl.create_default_context(),
            timeout=60,
        )
        server.login(sender, config["SMTP_PASSWORD"].strip())
        server.send_message(
            message, from_addr=sender, to_addrs=[receiver]
        )
        print("邮件已被 SMTP 服务器接受，请查看收件箱和垃圾邮件。")
    finally:
        if server is not None:
            try:
                server.quit()
            except (smtplib.SMTPException, OSError):
                server.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--send", action="store_true", help="实际发送邮件"
    )
    args = parser.parse_args()

    plain_body, html_body, attachments = prepare_digest()

    if args.send:
        send_digest(plain_body, html_body, attachments)
    else:
        print("预览已生成。添加 --send 参数可以实际发送。")


if __name__ == "__main__":
    try:
        main()
    except smtplib.SMTPAuthenticationError:
        print("邮箱认证失败，请检查 SMTP 授权码。")
    except smtplib.SMTPRecipientsRefused:
        print("服务器拒绝了收件地址，请检查 MAIL_TO。")
    except smtplib.SMTPResponseException as exc:
        print(f"邮件服务器返回错误，状态码：{exc.smtp_code}")
    except smtplib.SMTPException as exc:
        print(f"邮件通信异常：{type(exc).__name__}")
        print("重试前先检查收件箱，避免重复发送。")
    except OSError:
        print("文件读取或网络连接失败，请检查文件和网络。")
        print("如果已开始发送，请先查看收件箱再决定是否重试。")
    except (ValueError, KeyError, TypeError) as exc:
        print(f"本次处理未完成：{exc}")