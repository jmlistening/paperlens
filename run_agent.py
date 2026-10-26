"""PaperLens bounded retrieval agent. Existing fixed run_qa.py stays unchanged."""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from agent_policy import ANSWER_EXTRA, DECISION_PROMPT, normalized_query, validate_decision
from answer_question import PROMPT as ANSWER_PROMPT, render_answer, validate_answer
from answer_quality import review_agent_answer, render_quality
from prepare_corpus import atomic_write, sha256
from run_pipeline import pipeline_lock

ROOT = Path(__file__).resolve().parent
MAX_PAYLOAD_CHARS = 70000


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2))


class LiveBackend:
    simulated = False

    def __init__(self, root):
        self.root, self.client, self.model = root, None, None

    def __call__(self, prompt, payload, max_tokens):
        if self.client is None:
            from dotenv import load_dotenv
            from openai import OpenAI
            load_dotenv(self.root / ".env")
            key, self.model = os.getenv("DEEPSEEK_API_KEY", "").strip(), os.getenv("DEEPSEEK_MODEL", "").strip()
            if not key or not self.model:
                raise ValueError("本地 .env 缺少 DEEPSEEK_API_KEY 或 DEEPSEEK_MODEL。")
            self.client = OpenAI(api_key=key,
                                 base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                                 timeout=120.0, max_retries=0)
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "system", "content": prompt},
                                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}}, max_tokens=max_tokens)
        except Exception as exc:
            # Avoid writing provider exceptions that might contain credentials or request headers.
            raise RuntimeError(f"模型请求未完成（{type(exc).__name__}）；没有自动重试。") from None
        choice = response.choices[0]
        return {"model": response.model, "finish_reason": choice.finish_reason,
                "content": choice.message.content,
                "usage": response.usage.model_dump() if response.usage else None}


class AgentRun:
    def __init__(self, root, query, paper, tools, backend, max_rounds=3, max_calls=4,
                 output_dir=None):
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 2000:
            raise ValueError("问题须为 1～2000 字符。")
        if type(max_rounds) is not int or not 1 <= max_rounds <= 3:
            raise ValueError("检索轮数须为 1～3。")
        if type(max_calls) is not int or not 2 <= max_calls <= 4:
            raise ValueError("模型调用上限须为 2～4。")
        self.root, self.query, self.paper = Path(root), query.strip(), paper
        self.tools, self.backend = tools, backend
        self.max_rounds, self.max_calls = max_rounds, max_calls
        self.stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + "_agent"
        if output_dir is None:
            # Production default: unchanged location under the live library.
            self.folder = self.root / "data" / "rag" / "agent_runs" / self.stamp
            self.history_folder = self.root / "data" / "rag" / "answers" / self.stamp
        else:
            # Evaluation mode: every artifact, including live history, stays
            # under the caller's output directory and never touches data/rag.
            self.folder = Path(output_dir).resolve() / self.stamp
            self.history_folder = self.folder / "history"
        self.registry, self.pool, self.chunk_ids, self.history = {}, [], {}, set()
        simulated = getattr(backend, "simulated", True)
        self.report = {"schema_version": 1, "mode": "offline_simulation" if simulated else "live",
                       "query": self.query, "paper_filter": paper, "started_at": now(), "status": "running",
                       "corpus_sha256": tools.metadata["corpus_sha256"],
                       "limits": {"retrieval_rounds": max_rounds, "model_calls": max_calls,
                                  "decision_output_tokens": 1600, "answer_output_tokens": 4096,
                                  "payload_chars": MAX_PAYLOAD_CHARS, "visible_candidates": 32,
                                  "answer_evidence": 6, "api_timeout_seconds": 120, "automatic_retries": 0},
                       "api_attempts": 0, "model_calls": [], "trace": [],
                       "code_hashes": {n: sha256((ROOT / n).read_bytes()) for n in
                                       ("run_agent.py", "agent_tools.py", "agent_policy.py", "answer_question.py", "answer_quality.py", "answer_schema.py")},
                       "prompt_hashes": {"decision": sha256(DECISION_PROMPT.encode()),
                                         "answer": sha256((ANSWER_PROMPT + ANSWER_EXTRA).encode())}}

    def checkpoint(self):
        save(self.folder / "run.json", self.report)

    def event(self, kind, **values):
        self.report["trace"].append({"kind": kind, **values})
        self.checkpoint()

    def call(self, role, prompt, payload, max_tokens):
        if self.report["api_attempts"] >= self.max_calls:
            raise RuntimeError("模型调用预算已用尽。")
        if len(prompt) + len(json.dumps(payload, ensure_ascii=False)) > MAX_PAYLOAD_CHARS:
            raise ValueError("本轮证据上下文超过限制，未调用模型。")
        self.tools.check_current()
        number = self.report["api_attempts"] + 1
        prefix = self.folder / "calls" / f"{number:02d}_{role}"
        save(prefix.with_suffix(".request.json"), {"system": prompt, "payload": payload, "max_tokens": max_tokens})
        self.report["api_attempts"] = number
        entry = {"number": number, "role": role, "status": "started", "usage": None}
        self.report["model_calls"].append(entry)
        self.checkpoint()
        start = time.perf_counter()
        try:
            raw = self.backend(prompt, payload, max_tokens)
            save(prefix.with_suffix(".response.json"), raw)
            entry.update(model=raw.get("model"), usage=raw.get("usage"), finish_reason=raw.get("finish_reason"))
            if raw.get("finish_reason") != "stop" or not isinstance(raw.get("content"), str):
                raise ValueError("模型输出未完整结束，停止且不自动重试。")
            value = json.loads(raw["content"])
            entry["status"] = "received"
            self.tools.check_current()
            return value
        except BaseException as exc:
            entry.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            entry["elapsed_seconds"] = round(time.perf_counter() - start, 3)
            self.checkpoint()

    def retrieve(self, tool, query, round_number, keep=()):
        print(f"阶段：第 {round_number} 轮本地检索 · {tool}", flush=True)
        hits = self.tools.search(tool, query, 20 if round_number == 1 else 12)
        keys, new = [], 0
        for hit in hits:
            cid = hit.get("chunk_id")
            canonical = self.tools.canonical.get(cid)
            if canonical is None or any(hit.get(k) != v for k, v in canonical.items()):
                raise ValueError("工具结果与当前索引不一致。")
            if self.paper and canonical["arxiv_id"] != self.paper:
                raise ValueError("工具结果越过指定论文范围。")
            if cid not in self.chunk_ids:
                eid = f"C{len(self.registry) + 1}"
                self.chunk_ids[cid] = eid
                self.registry[eid] = canonical
                new += 1
            keys.append(self.chunk_ids[cid])
        self.pool = list(dict.fromkeys([*keep, *keys, *self.pool]))[:32]
        self.history.add((tool, normalized_query(query)))
        save(self.folder / f"retrieval_{round_number:02d}.json",
             {"tool": tool, "query": query, "paper_filter": self.paper,
              "corpus_sha256": self.report["corpus_sha256"], "hits": hits, "candidate_ids": keys})
        save(self.folder / "candidates.json", self.registry)
        self.event("retrieval", round=round_number, tool=tool, query=query,
                   candidate_count=len(keys), new_count=new,
                   sources=[{"id": k, "paper": self.registry[k]["arxiv_id"],
                             "page": self.registry[k]["page_number"]} for k in keys])
        return new

    def usage_summary(self):
        calls = self.report["model_calls"]
        known = [c["usage"] for c in calls if isinstance(c.get("usage"), dict)
                 and type(c["usage"].get("total_tokens")) is int and c["usage"]["total_tokens"] >= 0]
        return {"known_total_tokens": sum(u.get("total_tokens", 0) or 0 for u in known),
                "unknown_usage_calls": len(calls) - len(known),
                "complete": len(calls) == len(known)}

    def execute(self):
        self.retrieve("search_dense", self.query, 1)
        decision = {"assessment": "insufficient", "missing": ["未检索到可用的本地原文证据。"],
                    "selected_ids": [], "action": "finish", "query": "", "reason": "没有可用证据"}
        stop = "empty_retrieval"
        for step in range(1, self.max_rounds + 1):
            if not self.pool:
                break
            # Reserve one model call for the final grounded answer.
            if self.report["api_attempts"] >= self.max_calls - 1:
                stop = "call_budget"
                break
            can_search = step < self.max_rounds and self.report["api_attempts"] + 2 < self.max_calls
            print(f"阶段：第 {step} 轮评估证据覆盖", flush=True)
            payload = {"question": self.query, "paper_filter": self.paper,
                       "supplementary_search_allowed": can_search,
                       "history": [e for e in self.report["trace"] if e["kind"] in ("retrieval", "decision")],
                       "candidates": [{"id": k, "title": self.registry[k]["title"],
                                       "arxiv_id": self.registry[k]["arxiv_id"],
                                       "page": self.registry[k]["page_number"],
                                       "text": self.registry[k]["text"]} for k in self.pool]}
            decision = validate_decision(self.call("decision", DECISION_PROMPT, payload, 1600), set(self.pool))
            self.event("decision", round=step, **decision)
            if decision["action"] == "finish":
                stop = "sufficient" if decision["assessment"] == "sufficient" else "model_stopped_with_gaps"
                break
            if not can_search:
                stop = "round_budget" if step >= self.max_rounds else "call_budget"
                break
            key = (decision["action"], normalized_query(decision["query"]))
            if key in self.history:
                stop = "repeated_query"
                break
            new = self.retrieve(decision["action"], decision["query"], step + 1, decision["selected_ids"])
            if not new:
                stop = "no_new_evidence"
                break
        self.event("stop", reason=stop)
        self.report["stop_reason"] = stop
        evidence = {f"E{i}": self.registry[k] for i, k in enumerate(decision["selected_ids"], 1)}
        evidence_record = {"query": self.query, "paper_filter": self.paper,
                           "corpus_sha256": self.report["corpus_sha256"], "evidence": evidence,
                           "candidate_ids": decision["selected_ids"]}
        save(self.folder / "selected_evidence.json", evidence_record)
        input_hash = sha256((self.folder / "selected_evidence.json").read_bytes())
        if evidence:
            print("阶段：根据已核验出处生成回答", flush=True)
            payload = {"question": self.query,
                       "coverage": {"assessment": decision["assessment"], "missing": decision["missing"]},
                       "evidence": [{"id": k, "title": u["title"], "text": u["text"]} for k, u in evidence.items()]}
            answer = self.call("answer", ANSWER_PROMPT + ANSWER_EXTRA, payload, 4096)
            answer, quality = review_agent_answer(answer, evidence)
            # Ending the loop cannot erase a gap already identified on these same sources.
            if decision["assessment"] != "sufficient":
                if answer["status"] == "answered":
                    answer["status"] = "partial"
                answer["limitations"] = list(dict.fromkeys(answer["limitations"] + decision["missing"]))
            validate_answer(answer, evidence)
        else:
            answer = {"status": "insufficient", "points": [], "limitations": decision["missing"]}
            quality = {"version": "answer_quality_v1", "model_answer_status": None,
                       "quote_extensions": [], "warnings": [], "coverage": [],
                       "coverage_validation": "not_applicable_no_evidence",
                       "semantic_verification": "not_performed", "needs_review": False}
        self.tools.check_current()
        usage = self.usage_summary()
        try:
            run_ref = self.folder.relative_to(self.root).as_posix()
        except ValueError:
            run_ref = self.folder.as_posix()
        summary = {"mode": self.report["mode"], "stop_reason": stop,
                   "model_calls": self.report["api_attempts"],
                   "retrieval_rounds": sum(e["kind"] == "retrieval" for e in self.report["trace"]),
                   "limits": self.report["limits"], "usage": usage,
                   "run": run_ref}
        result = {"created_at": now(), "query": self.query,
                  "model": self.report["model_calls"][-1].get("model") if self.report["model_calls"] else "未调用模型",
                  "corpus_sha256": self.report["corpus_sha256"], "input_sha256": input_hash,
                  "prompt_version": "bounded_retrieval_agent_v1_1", "usage": usage,
                  "answer": answer, "evidence": evidence, "agent": summary,
                  "answer_quality": quality,
                  "agent_trace": self.report["trace"]}
        preview = render_answer(self.query, answer, evidence) + "\n" + render_quality(quality)
        save(self.folder / "answer_result.json", result)
        atomic_write(self.folder / "answer_preview.txt", preview)
        # Offline simulations are never inserted into the user's real answer history.
        if self.report["mode"] == "live" and self.history_folder is not None:
            atomic_write(self.history_folder / "answer_preview.txt", preview)
            save(self.history_folder / "answer_result.json", result)
        self.report.update(status="completed", answer_status=answer["status"], usage=usage)
        print(f"阶段：完成 · {answer['status']} · 模型调用 {summary['model_calls']} 次", flush=True)
        return result

    def run(self):
        start = time.perf_counter()
        self.checkpoint()
        print(f"本次完整记录：{self.folder}", flush=True)
        try:
            return self.execute()
        except BaseException as exc:
            self.report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                               error_type=type(exc).__name__, usage=self.usage_summary())
            raise
        finally:
            self.report.update(finished_at=now(), elapsed_seconds=round(time.perf_counter() - start, 3))
            self.checkpoint()


def main():
    parser = argparse.ArgumentParser(description="PaperLens 本地证据补检索 Agent")
    parser.add_argument("query")
    parser.add_argument("--paper")
    parser.add_argument("--max-rounds", type=int, default=3, choices=range(1, 4))
    parser.add_argument("--max-calls", type=int, default=4, choices=range(2, 5))
    parser.add_argument("--dry-run", action="store_true", help="仅验证索引与参数，不加载嵌入模型或调用 API")
    args = parser.parse_args()
    if not 1 <= len(args.query.strip()) <= 2000:
        parser.error("问题须为 1～2000 字符。")
    from agent_tools import LocalTools
    try:
        with pipeline_lock(ROOT):
            tools = LocalTools(ROOT, args.paper)
            if args.dry_run:
                print(f"检查通过：{len(tools.eligible)} 条可检索向量；最多 {args.max_rounds} 轮检索、{args.max_calls} 次模型调用。")
                print("仅检查配置与本地文件；没有加载嵌入模型、调用模型 API 或新建问答记录。")
                return 0
            AgentRun(ROOT, args.query, args.paper, tools, LiveBackend(ROOT), args.max_rounds, args.max_calls).run()
        return 0
    except Exception as exc:
        print(f"Agent 未完成：{exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
