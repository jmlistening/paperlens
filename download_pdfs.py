import json
import re
import time
from pathlib import Path

import requests


def main():
    project_dir = Path(__file__).resolve().parent
    input_file = project_dir / "data" / "papers.json"

    if not input_file.exists():
        print("请先运行 fetch_papers.py 获取论文。")
        return

    papers = json.loads(input_file.read_text(encoding="utf-8"))

    output_dir = project_dir / "data" / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 先下载候选列表的前3篇；后续改为下载筛选后的论文
    selected = papers[:3]
    success_count = 0

    with requests.Session() as session:
        for index, paper in enumerate(selected, start=1):
            # 清理文件名，避免特殊字符造成保存失败
            paper_id = paper["arxiv_id"]
            filename = re.sub(r"[^A-Za-z0-9._-]", "_", paper_id) + ".pdf"
            target = output_dir / filename
            temporary = target.with_suffix(".pdf.part")

            print(f"\n[{index}/{len(selected)}] {paper['title']}", flush=True)

            if target.exists():
                print(f"本地已有文件，跳过：{target.name}")
                success_count += 1
                continue

            if index > 1:
                time.sleep(3)

            try:
                url = paper["pdf_url"]
                if not url:
                    print("没有 PDF 地址，跳过。")
                    continue

                # 将 arXiv 返回的 HTTP 地址改为 HTTPS
                if url.startswith("http://"):
                    url = "https://" + url[len("http://"):]

                with session.get(
                    url,
                    stream=True,
                    timeout=(15, 60),
                ) as response:
                    response.raise_for_status()

                    size = 0
                    with temporary.open("wb") as file:
                        for chunk in response.iter_content(chunk_size=65536):
                            if not chunk:
                                continue
                            size += len(chunk)

                            # 当前测试限制每个文件最多25MB
                            if size > 25 * 1024 * 1024:
                                raise ValueError("文件超过25MB，保留链接供手动下载。")

                            file.write(chunk)

                # 避免把网页或错误提示保存成 PDF
                with temporary.open("rb") as file:
                    if file.read(5) != b"%PDF-":
                        raise ValueError("返回内容不是可识别的 PDF。")

                temporary.replace(target)
                success_count += 1
                print(f"下载成功：{target.name}（{size / 1024 / 1024:.2f} MB）")

            except (requests.RequestException, OSError, ValueError) as exc:
                print(f"本篇下载失败：{exc}")
                print("继续处理下一篇。")
            finally:
                if temporary.exists():
                    temporary.unlink()

    print(f"\n完成：本地已准备 {success_count}/{len(selected)} 个 PDF。")
    print(f"保存位置：{output_dir}")


if __name__ == "__main__":
    main()