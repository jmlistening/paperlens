"""Frozen pilot evaluation. --check is free; only --run invokes existing CLIs."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUITE = Path("eval/pilot_v1")
GROUPS = ("qa_runs", "agent_runs", "rerank_runs", "answers")
MODES = ("fixed", "agent_one", "agent_multi")
CALL_LIMITS = {"fixed": 2, "agent_one": 2, "agent_multi": 4}


def select_modes(value):
    modes = value.split(",")
    if len(set(modes)) != len(modes) or set(modes) - set(MODES):
        raise ValueError("模式须为 fixed、agent_one、agent_multi，不能重复。")
    return tuple(m for m in MODES if m in modes)


def verify_continuation(root, path, cases, modes):
    prior = read(path)
    statuses = {(t["case_id"], t["mode"]): t["status"] for t in prior["tasks"]}
    for case in cases:
        for mode in modes:
            if statuses.get((case["id"], mode)) != "pending":
                raise ValueError("续跑只允许原批次 pending 的任务；禁止重跑已尝试的模式。")
    requested = {(c["id"], m) for c in cases for m in modes}
    parent_hash = digest(path)
    for previous_path in (root / "eval/results").glob("*/manifest.json"):
        previous = read(previous_path)
        if previous.get("continuation", {}).get("sha256") == parent_hash:
            attempted = {(t["case_id"], t["mode"]) for t in previous.get("tasks", [])
                         if t["status"] != "pending"}
            if requested & attempted:
                raise ValueError("该续跑已有尝试记录；为防重复付费，请先检查最新结果包。")
    # Only bookkeeping code changed; original business code, questions and data must match.
    frozen = {n: sha for n, sha in prior["input_sha256"].items()
              if n not in ("run_eval.py", "test_eval.py")}
    assert_frozen(root, frozen)
    return prior


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def select_cases(suite, value):
    cases = suite["cases"]
    if not value:
        return cases
    ids = value.split(",")
    if len(set(ids)) != len(ids) or set(ids) - {c["id"] for c in cases}:
        raise ValueError("题号不存在或重复；示例：--cases Q01 或 --cases Q02,Q03")
    return [c for c in cases if c["id"] in ids]


def check(root, cases):
    # No .env access, embedding loading, downloads, or model requests.
    from agent_tools import LocalTools
    checked = set()
    for case in cases:
        pdf = root / case["pdf_path"]
        if not pdf.is_file() or digest(pdf) != case["pdf_sha256"]:
            raise ValueError(f"{case['id']} 的 PDF 缺失或版本变化；不能沿用旧评测依据。")
        if case["paper"] not in checked:
            tool = LocalTools(root, case["paper"])
            if not tool.eligible:
                raise ValueError(f"{case['paper']} 没有可检索片段。")
            print(f"{case['paper']}：{len(tool.eligible)} 条可检索向量", flush=True)
            checked.add(case["paper"])
    print(f"检查通过：{len(cases)} 道题。未加载嵌入模型、未调用模型 API。", flush=True)


def inputs(root, cases):
    paths = list(root.glob("*.py"))
    paths += [root / "data/rag" / n for n in
              ("corpus.json", "corpus_clean.json", "vector_store.npz")]
    paths += list((root / SUITE).glob("*"))
    paths += [root / c["pdf_path"] for c in cases]
    return {p.relative_to(root).as_posix(): digest(p)
            for p in sorted(set(paths)) if p.is_file()}


def assert_frozen(root, expected):
    for name, sha in expected.items():
        path = root / name
        if not path.is_file() or digest(path) != sha:
            raise ValueError(f"冻结文件发生变化：{name}；停止后续运行。")


def inventory(root):
    return {g: {p for p in (root / "data/rag" / g).glob("*") if p.is_dir()}
            for g in GROUPS}


def command(root, case, mode):
    script = "run_qa.py" if mode == "fixed" else "run_agent.py"
    cmd = [sys.executable, "-u", str(root / script), case["query"], "--paper", case["paper"]]
    if mode != "fixed":
        cmd += ["--max-rounds", "1" if mode == "agent_one" else "3",
                "--max-calls", "2" if mode == "agent_one" else "4"]
    return cmd


def usage_total(entries):
    values = [e.get("usage") for e in entries]
    known = [u["total_tokens"] for u in values if isinstance(u, dict)
             and type(u.get("total_tokens")) is int and u["total_tokens"] >= 0]
    return {"known_total_tokens": sum(known),
            "unknown_usage_calls": len(values) - len(known),
            "complete": len(values) == len(known)}


def collect(root, before, folder, case, mode, exit_code, elapsed):
    after = inventory(root)
    fresh = {g: sorted(after[g] - before[g]) for g in GROUPS}
    # Never select a pre-existing or simply 'latest' answer.
    primary_group = "qa_runs" if mode == "fixed" else "agent_runs"
    primary = fresh[primary_group]
    allowed = {"qa_runs", "rerank_runs", "answers"} if mode == "fixed" else {"agent_runs", "answers"}
    if len(primary) != 1 or any(len(v) > 1 or (v and g not in allowed) for g, v in fresh.items()):
        raise ValueError("无法唯一关联本次记录；可能存在并行问答。后续任务停止。")
    report = read(primary[0] / "run.json")
    if report.get("query") != case["query"] or report.get("paper_filter") != case["paper"]:
        raise ValueError("运行记录的问题或论文范围不匹配。")
    for group, dirs in fresh.items():
        for source in dirs:
            shutil.copytree(source, folder / "records" / group / source.name,
                            ignore=shutil.ignore_patterns("previous_*"))
    if mode == "fixed":
        responses = [read(p / "raw_response.json") for g in ("rerank_runs", "answers")
                     for p in fresh[g] if (p / "raw_response.json").is_file()]
        usage = usage_total(responses)
        expected = int(report.get("candidate_count", 0) > 3) + int(report.get("status") == "completed")
        successful = exit_code == 0 and report.get("status") in ("completed", "insufficient_evidence")
        # Both distinct legacy responses prove both possible calls returned,
        # even when final schema/citation validation subsequently failed.
        all_calls_returned = (len(fresh["rerank_runs"]) == 1 and len(fresh["answers"]) == 1
                              and len(responses) == 2)
        count_known = all_calls_returned or (successful and len(responses) == expected)
        if not count_known:
            # A timed-out legacy API call may have no response or attempt record.
            usage.update(complete=False, unknown_usage_calls=None)
        attempts = len(responses) if count_known else None
        models = sorted({r["model"] for r in responses if r.get("model")})
        rounds = 1 if "candidate_count" in report else None
    else:
        entries = report.get("model_calls", [])
        usage = usage_total(entries)
        attempts = report.get("api_attempts")
        models = sorted({r["model"] for r in entries if r.get("model")})
        rounds = sum(e.get("kind") == "retrieval" for e in report.get("trace", []))
        successful = exit_code == 0 and report.get("status") == "completed"
    answer_file = primary[0] / "answer_result.json"
    answer_status = read(answer_file).get("answer", {}).get("status") if answer_file.is_file() else None
    if answer_file.is_file() and read(answer_file).get("query") != case["query"]:
        raise ValueError("答案与本次问题不一致。")
    return {"case_id": case["id"], "mode": mode, "process_exit_code": exit_code,
            "pipeline_status": report.get("status"), "workflow_completed": successful,
            "answer_self_status": answer_status, "wall_seconds": round(elapsed, 3),
            "retrieval_rounds": rounds, "api_attempts": attempts, "models": models,
            "usage": usage, "human_quality": None,
            "source_run": primary[0].relative_to(root).as_posix()}


def run_task(root, session, case, mode):
    folder = session / "tasks" / f"{case['id']}_{mode}"
    folder.mkdir(parents=True, exist_ok=False)
    cmd = command(root, case, mode)
    save(folder / "invocation.json", {"case_id": case["id"], "command": cmd})
    before = inventory(root)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    started = time.perf_counter()
    # Run the unchanged original CLI. Gold criteria are never supplied to it.
    with (folder / "console.txt").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, cwd=root, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            process.stdout.close()
    elapsed = time.perf_counter() - started
    result = collect(root, before, folder, case, mode, code, elapsed)
    save(folder / "metrics.json", result)
    return result


def export(session):
    target = session.with_suffix(".zip")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(session.rglob("*")):
            if p.is_file():
                z.write(p, Path(session.name) / p.relative_to(session))
    print(f"\n评测记录包：{target}", flush=True)


def write_summary(session, manifest):
    lines = ["# PaperLens 试评测记录", "", f"批次状态：{manifest['status']}", "",
             "任务状态：" + "；".join(f"{s}={sum(t['status'] == s for t in manifest['tasks'])}"
                                     for s in ("completed", "failed", "pending")), "",
             "质量尚未评分；模型自评状态不代表正确率。", "",
             "| 题号 | 模式 | 流程状态 | 模型自评 | 端到端秒数 | 请求次数 | 已知 Token | 用量完整 |",
             "|---|---|---|---|---:|---:|---:|---|"]
    for r in manifest["results"]:
        u = r["usage"]
        lines.append(f"| {r['case_id']} | {r['mode']} | {r['pipeline_status']} | "
                     f"{r['answer_self_status'] or '无'} | {r['wall_seconds']} | "
                     f"{r['api_attempts'] if r['api_attempts'] is not None else '未知'} | "
                     f"{u['known_total_tokens']} | {'是' if u['complete'] else '否，不能视为总量'} |")
    lines += ["", "耗时包含原 CLI 进程启动、嵌入模型加载、检索和 API；不包含预检查和记录打包。",
              "这是单次流程计时，受缓存与网络影响，不是纯模型/检索速度基准。",
              "", "人工核对依据见 frozen/eval/pilot_v1/REFERENCES.md；结果在 tasks 下。"]
    (session / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="PaperLens 三模式小规模试评测；默认仅检查")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true")
    action.add_argument("--run", action="store_true", help="调用本机已配置模型，会产生 API 用量")
    parser.add_argument("--cases", help="题号用逗号隔开；--run 必须显式指定")
    parser.add_argument("--modes", default=",".join(MODES), help="只运行选定模式，逗号分隔")
    parser.add_argument("--continue-from", type=Path, help="原批次目录：仅运行其中尚未尝试的任务")
    args = parser.parse_args()
    if args.run and not args.cases:
        parser.error("先用 --run --cases Q01 试跑，不能默认启动全部付费题目。")
    suite = read(ROOT / SUITE / "questions.json")
    cases = select_cases(suite, args.cases)
    selected_modes = select_modes(args.modes)
    prior_path = None
    if args.continue_from:
        prior_path = (ROOT / args.continue_from / "manifest.json").resolve()
        verify_continuation(ROOT, prior_path, cases, selected_modes)
    check(ROOT, cases)
    if not args.run:
        return 0
    # Separate lock prevents two evaluation batches; existing CLIs retain their own pipeline lock.
    from run_pipeline import pipeline_lock
    with pipeline_lock(ROOT / "eval/runtime"):
        prior = verify_continuation(ROOT, prior_path, cases, selected_modes) if prior_path else None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        session = ROOT / "eval/results" / stamp
        session.mkdir(parents=True)
        frozen = inputs(ROOT, cases)
        for source in list(ROOT.glob("*.py")) + list((ROOT / SUITE).glob("*")):
            if source.is_file():
                dest = session / "frozen" / source.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
        packages = {}
        for package in ("numpy", "faiss-cpu", "sentence-transformers", "torch", "openai", "rank-bm25"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        manifest = {"suite": suite["version"], "cases": [c["id"] for c in cases],
                    "input_sha256": frozen, "python": sys.version, "platform": platform.platform(),
                    "packages": packages, "status": "running", "results": [],
                    "harness_version": "pilot_v1_1_bookkeeping_only",
                    "selected_modes": list(selected_modes),
                    "tasks": [{"case_id": c["id"], "mode": m, "status": "pending"}
                              for c in cases for m in selected_modes],
                    "maximum_model_calls": sum(CALL_LIMITS[m] for m in selected_modes) * len(cases),
                    "timing": "CLI wall time including process/model loading; excludes check, copy and export",
                    "quality": "manual review required; model self-status is not a score"}
        if prior_path:
            manifest["continuation"] = {"source": str(prior_path), "sha256": digest(prior_path),
                                        "unchanged_business_inputs_verified": True}
            save(session / "prior_manifest.json", prior)
        save(session / "manifest.json", manifest)
        print(f"开始 {len(cases)} 题 × {len(selected_modes)} 模式，最多 {manifest['maximum_model_calls']} 次模型请求。失败即停，无自动重试。", flush=True)
        try:
            for case in cases:
                offset = (int(case["id"][1:]) - 1) % 3
                modes = [m for m in MODES[offset:] + MODES[:offset] if m in selected_modes]
                for mode in modes:
                    assert_frozen(ROOT, frozen)
                    task = next(t for t in manifest["tasks"] if t["case_id"] == case["id"] and t["mode"] == mode)
                    task["status"] = "running"
                    save(session / "manifest.json", manifest)
                    print(f"\n=== {case['id']} / {mode} ===", flush=True)
                    result = run_task(ROOT, session, case, mode)
                    task["status"] = "completed" if result["workflow_completed"] else "failed"
                    manifest["results"].append(result)
                    save(session / "manifest.json", manifest)
                    assert_frozen(ROOT, frozen)
                    if not result["workflow_completed"]:
                        raise RuntimeError("本任务未完成，后续模式已停止。不要直接重复整批，以免重复付费。")
                    models = {m for r in manifest["results"] for m in r["models"]}
                    if len(models) > 1:
                        raise RuntimeError("不同调用返回了不同模型名；停止，不能作为同模型对照。")
            manifest["status"] = "completed_pending_human_review"
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            for task in manifest["tasks"]:
                if task["status"] == "running":
                    task["status"] = "failed"
            manifest.update(status="stopped", error_type=type(exc).__name__, error=str(exc))
            print(f"评测停止：{exc}", flush=True)
            return 1
        finally:
            save(session / "manifest.json", manifest)
            write_summary(session, manifest)
            export(session)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"检查未通过：{error}", flush=True)
        raise SystemExit(1)
