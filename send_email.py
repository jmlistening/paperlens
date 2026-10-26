import json
import re
import smtplib
import ssl
from pathlib import Path
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

from dotenv import dotenv_values


def main():
    project_dir = Path(__file__).resolve().parent
    config = dotenv_values(project_dir / ".env", encoding="utf-8-sig")

    required = [
        "SMTP_HOST", "SMTP_PORT", "SMTP_USER",
        "SMTP_PASSWORD", "MAIL_TO",
    ]
    if any(not (config.get(key) or "").strip() for key in required):
        print("邮箱配置不完整，请检查 .env。")
        return

    host = config["SMTP_HOST"].strip()
    sender = config["SMTP_USER"].strip()
    password = config["SMTP_PASSWORD"].strip()
    receiver = config["MAIL_TO"].strip()

    try:
        port = int(config["SMTP_PORT"])
    except ValueError:
        print("SMTP_PORT 必须是数字，本次使用 465。")
        return

    # 本次仅支持一个收件地址
    address_pattern = r"[^@\s,;<>]+@[^@\s,;<>]+\.[^@\s,;<>]+"
    if not all(
        address.isascii() and re.fullmatch(address_pattern, address)
        for address in [sender, receiver]
    ):
        print("请检查发件和收件地址，填写完整邮箱，不要填写姓名或多个地址。")
        return

    # 读取第一篇论文和此前生成的中文速览
    papers_file = project_dir / "data" / "papers.json"
    summary_file = project_dir / "data" / "first_summary.md"

    if not papers_file.exists() or not summary_file.exists():
        print("缺少论文数据或解读文件，请先完成抓取和解读。")
        return

    papers = json.loads(papers_file.read_text(encoding="utf-8"))
    if not papers:
        print("论文列表为空。")
        return

    paper = papers[0]
    summary = summary_file.read_text(encoding="utf-8")

    # 防止重新抓取后，把旧解读和新论文附件混在一起
    if f"论文链接：{paper['url']}" not in summary:
        print("中文解读与当前第一篇论文不匹配，请重新运行 summarize_paper.py。")
        return

    filename = re.sub(r"[^A-Za-z0-9._-]", "_", paper["arxiv_id"]) + ".pdf"
    pdf_file = project_dir / "data" / "pdfs" / filename

    if not pdf_file.exists():
        print("缺少对应 PDF，请先运行 download_pdfs.py。")
        return

    # 这是本次测试设定的附件上限，不代表邮箱的官方上限
    if pdf_file.stat().st_size > 10 * 1024 * 1024:
        print("本篇 PDF 超过本次测试的 10MB 上限，暂不发送。")
        return

    pdf_bytes = pdf_file.read_bytes()
    if not pdf_bytes.startswith(b"%PDF-"):
        print("附件不是可识别的 PDF，请检查下载结果。")
        return

    # 组装邮件：正文加附件
    message = EmailMessage()
    message["Subject"] = "PaperLens 测试日报｜中文速览与论文 PDF"
    message["From"] = sender
    message["To"] = receiver
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid()

    message.set_content(
        "这是 PaperLens 的第一封论文测试邮件。\n"
        "正文仅依据论文标题和摘要生成，未经全文核验。\n"
        "附件为对应的英文原版 PDF。\n\n"
        + summary
    )
    message.add_attachment(
        pdf_bytes,
        maintype="application",
        subtype="pdf",
        filename=filename,
    )

    print(f"准备发送论文：{paper['title']}", flush=True)
    print(f"附件：{filename}", flush=True)
    print("正在连接邮箱服务器……", flush=True)

    server = None
    try:
        server = smtplib.SMTP_SSL(
            host,
            port,
            context=ssl.create_default_context(),
            timeout=60,
        )
        server.login(sender, password)
        server.send_message(
            message,
            from_addr=sender,
            to_addrs=[receiver],
        )
    except smtplib.SMTPAuthenticationError as exc:
        print(f"邮箱登录失败，状态码：{exc.smtp_code}")
        print("请检查邮箱地址、客户端授权码，以及 SMTP 服务是否开启。")
    except smtplib.SMTPRecipientsRefused:
        print("服务器拒绝了收件地址，请检查 MAIL_TO。")
    except smtplib.SMTPResponseException as exc:
        print(f"邮箱服务器返回错误，状态码：{exc.smtp_code}")
    except (smtplib.SMTPException, OSError) as exc:
        print(f"连接或发送异常，类型：{type(exc).__name__}")
        print("请先查看邮箱是否收到，再决定是否重试，避免重复发送。")
    else:
        print("邮件已被 SMTP 服务器接受！")
        print("请到收件箱查看；如未出现，也检查垃圾邮件文件夹。")
    finally:
        if server is not None:
            # 退出连接报错，不应把已被接受的邮件误判为发送失败
            try:
                server.quit()
            except (smtplib.SMTPException, OSError):
                server.close()


if __name__ == "__main__":
    main()