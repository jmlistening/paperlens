import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from prepare_corpus import atomic_write, sha256
from run_pipeline import pipeline_lock

ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    atomic_write(
        path,
        json.dumps(value, ensure_ascii=False, indent=2),
    )


def run_stage(script, arguments):
    """使用当前虚拟环境的 Python 执行一个步骤。"""
    print(f"\n正在运行：{script}", flush=True)

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    subprocess.run(
        [sys.executable, str(ROOT / script), *arguments],
        cwd=ROOT,
        env=env,
        check=True,
    )


def fresh_result(script, arguments, filename, run_dir, query):
    """归档旧结果，再运行脚本，防止误读旧文件。"""
    path = ROOT / "data" / "rag" / filename

    if path.exists():
        path.replace(run_dir / ("previous_" + filename))

    run_stage(script, arguments)

    raw = path.read_bytes()
    result = json.loads(raw)

    if (
        result.get("query") != query
        or not isinstance(result.get("hits"), list)
    ):
        raise ValueError(
            f"{filename} 与本次问题不一致或格式异常。"
        )

    atomic_write(
        run_dir / filename,
        raw.decode("utf-8"),
    )

    return result, sha256(raw)


def execute(query, paper, run_dir, report):
    # 第一步：检索最多 20 条候选。
    arguments = [query, "--top-k", "20"]

    if paper:
        arguments += ["--paper", paper]

    candidates, input_hash = fresh_result(
        "search_evidence.py",
        arguments,
        "evidence_results.json",
        run_dir,
        query,
    )

    report["candidate_count"] = len(candidates["hits"])
    report["candidate_sha256"] = input_hash
    filename = "evidence_results.json"

    # 第二步：候选超过 3 条时进行重排序。
    if len(candidates["hits"]) > 3:
        selected, selected_hash = fresh_result(
            "rerank_evidence.py",
            ["--top-k", "3"],
            "reranked_results.json",
            run_dir,
            query,
        )

        if selected["rerank"]["input_sha256"] != input_hash:
            raise ValueError(
                "重排序使用的候选不是本次检索结果。"
            )

        filename = "reranked_results.json"

    else:
        selected = candidates
        selected_hash = input_hash
        print("候选不超过 3 条，跳过重排序。", flush=True)

    report["evidence_file"] = filename
    report["evidence_sha256"] = selected_hash

    if not selected["hits"]:
        report["status"] = "insufficient_evidence"
        print(
            "没有可用证据，本次停止，不调用回答接口。",
            flush=True,
        )
        return

    # 第三步：生成答案，并确认它属于本次运行。
    answers = ROOT / "data" / "rag" / "answers"
    before = set(answers.glob("*/answer_result.json"))

    run_stage(
        "answer_question.py",
        ["--evidence", filename],
    )

    new_files = (
        set(answers.glob("*/answer_result.json")) - before
    )

    if len(new_files) != 1:
        raise ValueError(
            "无法唯一确定本次新答案，请检查 answers 目录。"
        )

    answer_path = new_files.pop()
    answer = json.loads(
        answer_path.read_text(encoding="utf-8")
    )

    if (
        answer.get("query") != query
        or answer.get("input_sha256") != selected_hash
    ):
        raise ValueError(
            "答案与本次问题或证据不一致。"
        )

    shutil.copy2(
        answer_path,
        run_dir / "answer_result.json",
    )

    preview = answer_path.with_name("answer_preview.txt")
    if preview.exists():
        shutil.copy2(
            preview,
            run_dir / "answer_preview.txt",
        )

    report["status"] = "completed"
    report["answer_status"] = answer["answer"]["status"]
    report["answer_source"] = str(
        answer_path.relative_to(ROOT)
    )


def main():
    parser = argparse.ArgumentParser(
        description="PaperLens 一条命令完成证据问答"
    )
    parser.add_argument(
        "query",
        help="用英文双引号包住问题",
    )
    parser.add_argument(
        "--paper",
        help="可选：限定完整 arXiv 编号",
    )
    args = parser.parse_args()

    query = args.query.strip()
    if not query or len(query) > 2000:
        parser.error("问题须为 1～2000 字符。")

    for script in (
        "search_evidence.py",
        "rerank_evidence.py",
        "answer_question.py",
    ):
        if not (ROOT / script).is_file():
            raise FileNotFoundError(f"缺少 {script}")

    (ROOT / "data").mkdir(exist_ok=True)

    # 复用已有流程锁。
    # 导入 run_pipeline 不会执行邮件推送。
    with pipeline_lock():
        stamp = datetime.now(timezone.utc).strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        run_dir = (
            ROOT / "data" / "rag" / "qa_runs" / stamp
        )
        run_dir.mkdir(parents=True)

        report = {
            "query": query,
            "paper_filter": args.paper,
            "started_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "status": "running",
        }

        save_json(run_dir / "run.json", report)
        print(
            f"本次完整记录：{run_dir}",
            flush=True,
        )

        try:
            execute(query, args.paper, run_dir, report)

        except (Exception, KeyboardInterrupt) as error:
            report["status"] = (
                "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed"
            )
            report["error_type"] = type(error).__name__

            print(
                "\n流程已停止，后续步骤不会执行。",
                flush=True,
            )
            raise

        finally:
            report["finished_at"] = datetime.now(
                timezone.utc
            ).isoformat()

            save_json(run_dir / "run.json", report)
            print(
                f"\n本次记录目录：{run_dir}",
                flush=True,
            )


if __name__ == "__main__":
    main()