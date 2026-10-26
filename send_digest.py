"""PaperLens 邮件推送：默认预览；--send 才发送。

去重键：收件邮箱 + 不含版本号的 arXiv 编号。
accepted 表示 SMTP 接受或人工确认，不能保证进入收件箱。
pending/sending/uncertain 均阻止自动重发；中断后先检查邮箱。
不要删除 data/delivery.sqlite3，否则会丢失去重记录。
"""
import argparse
import hashlib
import json
import re
import smtplib
import sqlite3
import ssl
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
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
BLOCKED = {"accepted", "pending", "sending", "uncertain"}
STATUS_TEXT = {
    "pending": "已预留／未确认完成", "sending": "发送中／未确认完成",
    "accepted": "服务器已接受／人工已确认", "failed": "未成功，可重试",
    "uncertain": "结果不明，需核实",
}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def paper_key(arxiv_id):
    if not isinstance(arxiv_id, str) or not re.fullmatch(
        r"(?:\d{4}\.\d{4,5}|[A-Za-z][A-Za-z0-9.-]*/\d{7})(?:v[1-9]\d*)?",
        arxiv_id,
    ):
        raise ValueError("论文编号不是支持的 arXiv ID 格式。")
    return re.sub(r"v\d+$", "", arxiv_id)


def checked_address(value):
    value = (value or "").strip()
    pattern = r"[^@\s,;<>]+@[^@\s,;<>]+\.[^@\s,;<>]+"
    if not value.isascii() or not re.fullmatch(pattern, value):
        raise ValueError("请在 .env 中配置单个、完整的邮箱地址。")
    local, domain = value.rsplit("@", 1)
    return local + "@" + domain.lower()


def html_text(value):
    return escape(str(value)).replace("\n", "<br>")


def checked_url(value):
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https"):
        raise ValueError("论文链接必须使用 HTTP 或 HTTPS。")
    if parts.hostname not in ("arxiv.org", "www.arxiv.org"):
        raise ValueError("论文链接不是预期的 arXiv 地址。")
    return value


def load_papers():
    source_bytes = (ROOT / "data" / "recommendations.json").read_bytes()
    recommendations = json.loads(source_bytes.decode("utf-8-sig"))
    data = json.loads((ROOT / "data" / "selected_summaries.json").read_text(
        encoding="utf-8-sig"
    ))
    if data.get("recommendations_sha256") != hashlib.sha256(source_bytes).hexdigest():
        raise ValueError("推荐清单已变化，请先重新运行 summarize_selected.py。")
    papers = data["papers"]
    if not isinstance(papers, list):
        raise ValueError("速览 papers 必须是列表。")
    if [p["arxiv_id"] for p in papers] != recommendations["selected_ids"]:
        raise ValueError("速览与推荐清单不对应，请重新生成速览。")
    keys = [paper_key(p["arxiv_id"]) for p in papers]
    if len(keys) != len(set(keys)):
        raise ValueError("入选清单含同一论文的重复记录或多个版本，请先整理清单。")
    return papers


def open_db(root=ROOT):
    folder = Path(root) / "data"
    folder.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(folder / "delivery.sqlite3", timeout=15, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS attempts (
            id TEXT PRIMARY KEY, recipient TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN
              ('pending','sending','accepted','failed','uncertain')),
            kind TEXT NOT NULL, message_id TEXT,
            papers_json TEXT NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, detail TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS deliveries (
            recipient TEXT NOT NULL, paper_key TEXT NOT NULL,
            arxiv_id TEXT NOT NULL, title TEXT NOT NULL,
            attempt_id TEXT NOT NULL REFERENCES attempts(id),
            status TEXT NOT NULL CHECK(status IN
              ('pending','sending','accepted','failed','uncertain')),
            updated_at TEXT NOT NULL,
            PRIMARY KEY(recipient, paper_key)
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_id TEXT NOT NULL REFERENCES attempts(id),
            status TEXT NOT NULL, created_at TEXT NOT NULL, detail TEXT NOT NULL
        );
    """)
    return db


@contextmanager
def transaction(db):
    db.execute("BEGIN IMMEDIATE")
    try:
        yield
        db.execute("COMMIT")
    except BaseException:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise


def eligible_papers(db, recipient, papers, verbose=True):
    eligible = []
    for paper in papers:
        row = db.execute(
            "SELECT status, attempt_id FROM deliveries WHERE recipient=? AND paper_key=?",
            (recipient, paper_key(paper["arxiv_id"])),
        ).fetchone()
        if row is not None and row["status"] in BLOCKED:
            if verbose:
                print(f"跳过 {paper['arxiv_id']}：{STATUS_TEXT[row['status']]}"
                      f"；记录 {row['attempt_id']}")
        else:
            eligible.append(paper)
    return eligible


def reserve(db, recipient, papers, message_id=None, imported=False):
    if not papers:
        raise ValueError("没有可登记的论文。")
    attempt_id = uuid.uuid4().hex
    status = "accepted" if imported else "pending"
    detail = "人工登记此前已收到的邮件；未发送新邮件" if imported else "已预留，尚未调用 SMTP"
    timestamp = now()
    with transaction(db):
        # Re-check under a write lock: two processes cannot claim the same paper.
        if len(eligible_papers(db, recipient, papers, verbose=False)) != len(papers):
            raise ValueError("发送记录已被另一个进程更新。本次未发送，请重新预览。")
        db.execute(
            "INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?)",
            (attempt_id, recipient, status, "manual_import" if imported else "smtp",
             message_id, json.dumps([p["arxiv_id"] for p in papers]), timestamp,
             timestamp, detail),
        )
        for paper in papers:
            db.execute("""
                INSERT INTO deliveries VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(recipient,paper_key) DO UPDATE SET
                  arxiv_id=excluded.arxiv_id, title=excluded.title,
                  attempt_id=excluded.attempt_id, status=excluded.status,
                  updated_at=excluded.updated_at
            """, (recipient, paper_key(paper["arxiv_id"]), paper["arxiv_id"],
                  paper["title"], attempt_id, status, timestamp))
        db.execute("INSERT INTO events(attempt_id,status,created_at,detail) VALUES (?,?,?,?)",
                   (attempt_id, status, timestamp, detail))
    return attempt_id


def set_status(db, attempt_id, status, detail, expected):
    with transaction(db):
        row = db.execute("SELECT status FROM attempts WHERE id=?", (attempt_id,)).fetchone()
        if row is None or row["status"] not in expected:
            raise ValueError("记录状态已改变或记录不存在，请先用 --history 核实。")
        timestamp = now()
        db.execute("UPDATE attempts SET status=?, updated_at=?, detail=? WHERE id=?",
                   (status, timestamp, detail, attempt_id))
        db.execute("UPDATE deliveries SET status=?, updated_at=? WHERE attempt_id=?",
                   (status, timestamp, attempt_id))
        db.execute("INSERT INTO events(attempt_id,status,created_at,detail) VALUES (?,?,?,?)",
                   (attempt_id, status, timestamp, detail))


def show_history(db):
    rows = db.execute("SELECT * FROM attempts ORDER BY created_at DESC, rowid DESC LIMIT 20").fetchall()
    if not rows:
        print("暂无发送记录。旧版邮件不会自动进入本数据库。")
    for row in rows:
        print(f"\n记录：{row['id']}\n时间（UTC）：{row['updated_at']}"
              f"\n收件人：{row['recipient']}\n状态：{STATUS_TEXT[row['status']]}"
              f"\n论文：{', '.join(json.loads(row['papers_json']))}"
              f"\nMessage-ID：{row['message_id'] or '人工登记，无 Message-ID'}"
              f"\n说明：{row['detail']}")
    print(f"\n数据库：{ROOT / 'data' / 'delivery.sqlite3'}")


def resolve_attempt(db, attempt_id, result, note, confirmed_stopped):
    if not confirmed_stopped or result not in {"accepted", "failed"} or not note.strip():
        raise ValueError("人工核实需同时指定 --result accepted/failed、--note 和 --confirm-stopped。"
                         "先确认原发送进程已停止，并检查邮箱。")
    # Do not resolve an old attempt after a newer attempt has taken over its papers.
    with transaction(db):
        row = db.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone()
        if row is None or row["status"] not in {"pending", "sending", "uncertain"}:
            raise ValueError("只能核实 pending、sending 或 uncertain 记录。")
        timestamp = now()
        detail = "人工核实：" + note.strip()
        db.execute("UPDATE attempts SET status=?,updated_at=?,detail=? WHERE id=?",
                   (result, timestamp, detail, attempt_id))
        db.execute("UPDATE deliveries SET status=?,updated_at=? WHERE attempt_id=?",
                   (result, timestamp, attempt_id))
        db.execute("INSERT INTO events(attempt_id,status,created_at,detail) VALUES (?,?,?,?)",
                   (attempt_id, result, timestamp, detail))
    print("人工核实已记录；本命令没有发送邮件。")


def prepare_digest(papers):
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


def build_message(config, recipient, plain_body, html_body, attachments):
    required = ('SMTP_HOST', 'SMTP_PORT', 'SMTP_USER', 'SMTP_PASSWORD')
    if any(not (config.get(key) or '').strip() for key in required):
        raise ValueError('.env 缺少邮件配置。')
    sender = checked_address(config['SMTP_USER'])
    port = int(config['SMTP_PORT'].strip())
    if port != 465:
        raise ValueError('本程序使用 SSL 连接，请沿用 SMTP_PORT=465。')
    message = EmailMessage()
    message['Subject'] = f'PaperLens｜为你推荐 {len(attachments)} 篇论文'
    message['From'] = sender
    message['To'] = recipient
    message['Date'] = formatdate(localtime=True)
    message['Message-ID'] = make_msgid()
    message.set_content(plain_body)
    message.add_alternative(html_body, subtype='html')
    for filename, content in attachments:
        message.add_attachment(content, maintype='application', subtype='pdf', filename=filename)
    # Project limit, not a claimed SMTP provider limit.
    if len(message.as_bytes(policy=SMTP)) > 15 * 1024 * 1024:
        raise ValueError('编码后的邮件超过本阶段设置的 15 MB 上限。')
    return message, sender, port


def error_label(exc):
    # Never persist SMTP response text, credentials, or the complete config.
    code = getattr(exc, 'smtp_code', None)
    return type(exc).__name__ + (f' (SMTP {code})' if code is not None else '')


def deliver(db, config, recipient, papers, message, sender, port):
    attempt_id = reserve(db, recipient, papers, message['Message-ID'])
    print(f'发送记录：{attempt_id}', flush=True)
    server = None
    entered_send = False
    try:
        print('正在连接邮箱服务器……', flush=True)
        server = smtplib.SMTP_SSL(
            config['SMTP_HOST'].strip(), port,
            context=ssl.create_default_context(), timeout=60,
        )
        server.login(sender, config['SMTP_PASSWORD'].strip())
        set_status(db, attempt_id, 'sending', '即将提交邮件', {'pending'})
        entered_send = True
        refused = server.send_message(message, from_addr=sender, to_addrs=[recipient])
        if refused:
            raise smtplib.SMTPRecipientsRefused(refused)
    except BaseException as exc:
        known_rejection = isinstance(exc, (
            smtplib.SMTPRecipientsRefused, smtplib.SMTPSenderRefused,
            smtplib.SMTPDataError, smtplib.SMTPHeloError,
            smtplib.SMTPNotSupportedError,
        ))
        status = 'failed' if not entered_send or known_rejection else 'uncertain'
        try:
            set_status(db, attempt_id, status, error_label(exc), {'pending', 'sending'})
        except (sqlite3.Error, ValueError):
            print('发送状态未能写回。预留记录会阻止自动重发，请用 --history 核实。')
        if status == 'uncertain':
            print('邮件可能已经发出，结果不明；已阻止自动重发。请检查收件箱和垃圾邮件。')
        else:
            print('本次未成功提交邮件，可排除故障后重试。')
        raise
    else:
        # SMTP acceptance and a local SQLite commit cannot be one atomic transaction.
        # If the commit fails or the process dies here, 'sending' remains blocked.
        try:
            set_status(db, attempt_id, 'accepted', 'SMTP 已接受，不等于已进入收件箱', {'sending'})
        except (sqlite3.Error, ValueError):
            print('SMTP 已接受邮件，但本地确认写入失败。不要直接重发，请先核实记录。')
            raise
        print('邮件已被 SMTP 服务器接受，发送记录已保存。请查看收件箱和垃圾邮件。')
    finally:
        if server is not None:
            try:
                server.quit()
            except Exception:
                try:
                    server.close()
                except Exception:
                    pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--send', action='store_true', help='实际发送尚未发送的论文')
    actions.add_argument('--history', action='store_true', help='查看最近 20 次发送或人工登记')
    actions.add_argument('--mark-existing', action='store_true', help='人工登记当前清单中此前已收到的论文，不发送邮件')
    actions.add_argument('--resolve', metavar='ATTEMPT_ID', help='人工核实一条不确定或中断的记录，不发送邮件')
    parser.add_argument('--result', choices=['accepted', 'failed'])
    parser.add_argument('--note', default='')
    parser.add_argument('--confirm-stopped', action='store_true', help='确认原发送进程已经停止')
    args = parser.parse_args(argv)
    if not args.resolve and (args.result or args.note or args.confirm_stopped):
        parser.error('--result、--note、--confirm-stopped 只能与 --resolve 一起使用。')
    db = open_db()
    try:
        if args.history:
            show_history(db)
            return
        if args.resolve:
            resolve_attempt(db, args.resolve, args.result, args.note, args.confirm_stopped)
            return
        config = dotenv_values(ROOT / '.env', encoding='utf-8-sig')
        recipient = checked_address(config.get('MAIL_TO'))
        papers = load_papers()
        selected = eligible_papers(db, recipient, papers)
        print(f'本轮候选 {len(papers)} 篇；可处理 {len(selected)} 篇。')
        if not selected:
            # Avoid displaying a stale preview from an earlier batch.
            (ROOT / 'data' / 'digest_preview.html').write_text(
                '<!doctype html><meta charset="utf-8"><title>PaperLens</title>'
                '<h1>本轮没有可发送的新论文</h1>'
                '<p>请通过 --history 查看已发送或待核实的记录。</p>', encoding='utf-8')
            print('本轮没有可发送的新论文，未连接邮件服务器。')
            return
        if args.mark_existing:
            attempt_id = reserve(db, recipient, selected, imported=True)
            print(f'已人工登记 {len(selected)} 篇此前收到的论文；未发送邮件。记录：{attempt_id}')
            return
        bodies = prepare_digest(selected)
        if not args.send:
            print('仅生成预览，未连接邮件服务器，也没有登记为已发送。')
            print('确认后可运行：python send_digest.py --send')
            return
        message, sender, port = build_message(config, recipient, *bodies)
        deliver(db, config, recipient, selected, message, sender, port)
    finally:
        db.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n操作已中断。若已开始发送，请用 --history 查看记录并核实邮箱。')
        sys.exit(130)
    except smtplib.SMTPAuthenticationError:
        print('邮箱认证失败，请检查 SMTP 授权码。')
        sys.exit(1)
    except smtplib.SMTPException as exc:
        print(f'邮件通信未正常完成：{error_label(exc)}；请用 --history 查看状态。')
        sys.exit(1)
    except sqlite3.Error:
        print('发送记录数据库操作失败；请保留数据库，先检查是否有其他进程或磁盘问题。')
        sys.exit(1)
    except OSError:
        print('文件读取或网络连接失败。若已开始发送，请先检查 --history 和邮箱。')
        sys.exit(1)
    except (ValueError, KeyError, TypeError) as exc:
        print(f'本次处理未完成：{exc}')
        sys.exit(1)
