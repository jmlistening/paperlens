import os
import subprocess
from datetime import datetime
from pathlib import Path

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parent
TRANSLATOR_ENV = "zotero-pdf2zh-next-venv"


def main():
    # 本次先测试 RCER 的前两页。
    source_pdf = ROOT / "data" / "pdfs" / "2610.19398v1.pdf"
    if not source_pdf.is_file():
        raise ValueError(f"找不到原版 PDF：{source_pdf}")

    config = dotenv_values(ROOT / ".env", encoding="utf-8-sig")
    api_key = (config.get("DEEPSEEK_API_KEY") or "").strip()
    model = (config.get("DEEPSEEK_MODEL") or "").strip()
    base_url = (
        config.get("DEEPSEEK_BASE_URL") or ""
    ).strip().rstrip("/")

    if not api_key or not model:
        raise ValueError(".env 缺少 DeepSeek API Key 或模型名称。")

    # 当前使用工具内置的 DeepSeek 服务，连接官方接口。
    if base_url not in (
        "https://api.deepseek.com",
        "https://api.deepseek.com/v1",
    ):
        raise ValueError(
            "当前脚本适用于 DeepSeek 官方接口。"
            "你的 BASE_URL 不同，请告诉我服务商名称，不要发送密钥。"
        )

    conda_exe = os.environ.get("CONDA_EXE", "")
    if not conda_exe or not Path(conda_exe).is_file():
        raise ValueError(
            "没有找到 Conda，请在激活 paperlens 的 Anaconda Prompt 中运行。"
        )

    # 每次测试使用新目录，避免把旧文件误认为本次结果。
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (
        ROOT / "data" / "translations"
        / "2610.19398v1" / f"preview_{run_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    child_env = os.environ.copy()

    # 密钥通过子进程环境变量传递，不放在命令行参数中。
    child_env["PDF2ZH_DEEPSEEK_API_KEY"] = api_key

    # 确保本次输出中文版和双语版，并使用命令行运行。
    child_env["PDF2ZH_NO_MONO"] = "false"
    child_env["PDF2ZH_NO_DUAL"] = "false"
    child_env["PDF2ZH_GUI"] = "false"
    child_env["PDF2ZH_DEBUG"] = "false"

    command = [
        conda_exe,
        "run",
        "--no-capture-output",
        "-n",
        TRANSLATOR_ENV,
        "pdf2zh_next",
        str(source_pdf),
        "--deepseek",
        "--deepseek-model",
        model,
        "--deepseek-thinking-mode",
        "disabled",
        "--lang-in",
        "en",
        "--lang-out",
        "zh-CN",
        "--pages",
        "1-2",
        "--only-include-translated-page",
        "--output",
        str(output_dir),
        "--qps",
        "2",
        "--pool-max-workers",
        "2",
        "--no-auto-extract-glossary",
        "--watermark-output-mode",
        "watermarked",
        "--report-interval",
        "5",
    ]

    print("开始翻译 RCER 的前两页。", flush=True)
    print(f"使用模型：{model}", flush=True)
    print("目标：中文 PDF + 双语对照 PDF", flush=True)
    print(f"输出目录：{output_dir}", flush=True)
    print("下面显示翻译工具的运行日志……\n", flush=True)

    result = subprocess.run(
        command,
        cwd=ROOT,
        env=child_env,
        check=False,
    )

    if result.returncode != 0:
        print(f"\n翻译工具异常退出，退出码：{result.returncode}")
        print("请把日志末尾的报错截图发给我。")
        return

    pdf_files = sorted(output_dir.rglob("*.pdf"))
    mono_files = [p for p in pdf_files if "mono" in p.name.lower()]
    dual_files = [p for p in pdf_files if "dual" in p.name.lower()]

    print("\n本次生成的 PDF：")
    for path in pdf_files:
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"  {path.name}（{size_mb:.2f} MB）")

    if not mono_files or not dual_files:
        print("尚未找到完整的 mono / dual 两类文件，请检查日志。")
        return

    print("\n已找到中文版和双语版文件，请打开检查翻译与排版。")
    print("注意：这次只是前两页测试，不是全文翻译。")
    os.startfile(str(output_dir))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f"运行未完成：{exc}")
    except KeyboardInterrupt:
        print("\n已中断本次运行，请确认翻译进程已经停止。")