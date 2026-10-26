import json
import re
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
MAX_BYTES = 25 * 1024 * 1024  # 本阶段设置的单篇下载上限


def is_pdf(path):
    if not path.is_file():
        return False
    with path.open("rb") as file:
        return file.read(5) == b"%PDF-"


def main():
    source = ROOT / "data" / "recommendations.json"
    if not source.exists():
        print("请先运行 rank_papers.py，生成推荐清单。")
        return

    data = json.loads(source.read_text(encoding="utf-8-sig"))
    paper_map = {
        paper["arxiv_id"]: paper
        for paper in data["ranked_papers"]
    }
    selected_ids = data["selected_ids"]

    if not selected_ids:
        print("本轮没有入选论文，无需下载。")
        return

    if any(paper_id not in paper_map for paper_id in selected_ids):
        print("推荐清单不完整，请重新运行 rank_papers.py。")
        return

    output_dir = ROOT / "data" / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)

    ready = 0
    attempted_request = False

    with requests.Session() as session:
        for index, paper_id in enumerate(selected_ids, start=1):
            paper = paper_map[paper_id]
            filename = re.sub(
                r"[^A-Za-z0-9._-]", "_", paper_id
            ) + ".pdf"
            target = output_dir / filename
            temporary = target.with_suffix(".pdf.part")

            print(f"\n[{index}/{len(selected_ids)}] {paper['title']}")

            if is_pdf(target):
                print(f"已有 PDF，跳过下载：{filename}")
                ready += 1
                continue

            # 两次实际下载请求之间留出间隔。
            if attempted_request:
                time.sleep(3)
            attempted_request = True

            url = paper["pdf_url"]
            if url.startswith("http://"):
                url = "https://" + url[len("http://"):]

            try:
                print("正在下载……", flush=True)
                with session.get(
                    url,
                    stream=True,
                    timeout=(15, 60),
                ) as response:
                    response.raise_for_status()
                    total = 0

                    with temporary.open("wb") as file:
                        for chunk in response.iter_content(
                            chunk_size=65536
                        ):
                            if not chunk:
                                continue
                            total += len(chunk)
                            if total > MAX_BYTES:
                                raise ValueError(
                                    "超过本阶段设置的 25 MB 下载上限。"
                                )
                            file.write(chunk)

                if not is_pdf(temporary):
                    raise ValueError("下载内容不是有效的 PDF 文件头。")

                temporary.replace(target)
                ready += 1
                print(f"下载完成：{filename}（{total / 1024 / 1024:.2f} MB）")

            except requests.RequestException as exc:
                print(f"网络下载失败：{type(exc).__name__}")
            except (OSError, ValueError) as exc:
                print(f"文件处理失败：{exc}")
            finally:
                if temporary.exists():
                    temporary.unlink()

    print(f"\n入选论文 PDF 已就绪：{ready}/{len(selected_ids)}")
    print(f"保存目录：{output_dir}")

    if ready != len(selected_ids):
        print("还有文件未下载成功，可以重新运行本程序补下载。")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"本次处理未完成：{exc}")