import os
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SERVER = Path(r"D:\zotero_translate\server_new")

TRANSLATOR = (
    SERVER / "zotero-pdf2zh-next-venv"
    / "Scripts" / "pdf2zh_next.exe"
)
CONFIG = SERVER / "config" / "config.toml"
SOURCE = ROOT / "data" / "pdfs" / "2610.19398v1.pdf"


def main():
    for path in (TRANSLATOR, CONFIG, SOURCE):
        if not path.is_file():
            raise FileNotFoundError(f"找不到文件：{path}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (
        ROOT / "data" / "translations"
        / SOURCE.stem / f"same_backend_{run_id}"
    )
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "translation.log"

    # 避免当前终端的 PDF2ZH 环境变量覆盖 Zotero 配置。
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("PDF2ZH_")
    }
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["PDF2ZH_GUI"] = "false"
    env["PDF2ZH_DEBUG"] = "false"

    # 模型、密钥、思考设置等从 Zotero 配置读取。
    # 其余参数对应你之前发来的 Zotero 设置截图。
    command = [
        str(TRANSLATOR),
        str(SOURCE),
        "--config-file", str(CONFIG),
        "--deepseek",
        "--lang-in", "en",
        "--lang-out", "zh-CN",
        "--qps", "10",
        "--auto-enable-ocr-workaround",
        "--watermark-output-mode", "no_watermark",
        "--output", str(output),
        "--pages", "1-2",
        "--only-include-translated-page",
        "--report-interval", "5",
    ]

    print("使用 Zotero 服务目录中的翻译器。", flush=True)
    print("模型等设置读取自 Zotero 的 config.toml。", flush=True)
    print("本次只翻译前两页，用于检查效果。", flush=True)
    print(f"输出目录：{output}", flush=True)
    print(f"完整日志：{log_path}", flush=True)

    # 同时显示日志并保存，不用再开一个窗口查看。
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command,
            cwd=SERVER,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"翻译器退出码：{return_code}，请检查日志：{log_path}"
        )

    mono = list(output.glob("*.mono.pdf"))
    dual = list(output.glob("*.dual.pdf"))
    if not mono or not dual:
        raise RuntimeError(
            f"未同时找到中文版和双语版，请检查：{log_path}"
        )

    print("\n中文版和双语版已生成，请检查译文与排版。")
    os.startfile(str(output))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        print(f"\n运行未完成：{exc}")
        raise SystemExit(1)