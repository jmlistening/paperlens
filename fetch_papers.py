"""按当前兴趣 profile 中的检索配置获取候选；不调用大模型或发送邮件。"""
import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VERSION = "discovery_v1_rate_limit_fix"
COOLDOWN_SECONDS = 15 * 60


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.stem + "_", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def check_integer(obj, key, low, high):
    value = obj.get(key)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"search.{key} 必须是 {low}—{high} 的整数。")
    return value


def check_terms(value, required=True):
    if not isinstance(value, list) or len(value) > 10 or (required and not value):
        raise ValueError("每组 any_terms 需要 1—10 个词组；require_any_terms 可为空。")
    checked = []
    for term in value:
        # 只接收普通英文关键词，不接收用户拼接的查询表达式。
        if not isinstance(term, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 -]{0,79}", term):
            raise ValueError("检索词仅支持英文、数字、空格和连字符，长度不超过 80。")
        term = " ".join(term.split())
        if term.casefold() not in {x.casefold() for x in checked}:
            checked.append(term)
    return checked


def parse_profile(profile):
    """校验已解码的 interests 配置并返回检索配置。

    从 read_profile 中抽出，供增量推送复用同一套字段校验，
    不读取文件、不联网。传入 dict 缺失或字段非法时抛 ValueError。
    """
    if not isinstance(profile, dict) or not isinstance(profile.get("search"), dict):
        raise ValueError("兴趣配置缺少 search 配置，请使用本次更新版配置。")
    config = profile["search"]
    days = check_integer(config, "lookback_days", 1, 365)
    maximum = check_integer(config, "max_candidates", 1, 20)
    per_query = check_integer(config, "per_query_limit", 1, 30)
    if per_query < maximum:
        raise ValueError("per_query_limit 请设置为不小于 max_candidates。")
    groups = config.get("groups")
    if not isinstance(groups, list) or not 1 <= len(groups) <= 5:
        raise ValueError("search.groups 需要 1—5 组检索设置。")
    checked, names = [], set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("每个检索组必须是对象。")
        name = group.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 60:
            raise ValueError("检索组需要 1—60 字的名称。")
        name = name.strip()
        if name in names:
            raise ValueError("检索组名称不能重复。")
        names.add(name)
        checked.append({
            "name": name,
            "any_terms": check_terms(group.get("any_terms")),
            "require_any_terms": check_terms(group.get("require_any_terms", []), False),
        })
    return {
        "lookback_days": days, "max_candidates": maximum,
        "per_query_limit": per_query, "groups": checked,
    }


def read_profile(path):
    raw = path.read_bytes()
    profile = json.loads(raw.decode("utf-8-sig"))
    return profile, parse_profile(profile), hashlib.sha256(raw).hexdigest()


def term_clause(terms):
    return "(" + " OR ".join(
        f'ti:"{term}" OR abs:"{term}"' for term in terms
    ) + ")"


def build_queries(config, now):
    end = now.astimezone(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=config["lookback_days"])
    # arXiv submittedDate 使用 UTC，格式为 YYYYMMDDHHMM。
    date_clause = f"submittedDate:[{start:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}]"
    queries = []
    for group in config["groups"]:
        clauses = [term_clause(group["any_terms"])]
        if group["require_any_terms"]:
            clauses.append(term_clause(group["require_any_terms"]))
        clauses.append(date_clause)
        queries.append({"name": group["name"], "query": " AND ".join(clauses)})
    return queries, start, end


def identity(pid):
    match = re.fullmatch(r"(\d{4}\.\d{4,5}|[A-Za-z.-]+/\d{7})(?:v(\d+))?", pid)
    if not match:
        raise ValueError("arXiv 返回了无效的论文编号。")
    return match.group(1), int(match.group(2) or 0)


def normalize_result(result):
    pid = result.get_short_id()
    identity(pid)
    title = " ".join(result.title.split())
    abstract = " ".join(result.summary.split())
    if not title or not abstract:
        raise ValueError("arXiv 返回的论文缺少标题或摘要。")
    return {
        "arxiv_id": pid, "title": title,
        "authors": [author.name for author in result.authors],
        "published": result.published.astimezone(timezone.utc).isoformat(),
        "updated": result.updated.astimezone(timezone.utc).isoformat(),
        "abstract": abstract, "categories": list(result.categories),
        "url": "https://arxiv.org/abs/" + pid,
        "pdf_url": "https://arxiv.org/pdf/" + pid,
    }


def merge_groups(buckets, maximum, run_id, retrieved_at):
    """按组轮流取候选，按论文基础编号去重，保留抓到的最高版本。"""
    pool, origins, queues = {}, {}, []
    for name, papers in buckets:
        queue = []
        for rank, paper in enumerate(papers, start=1):
            key, version = identity(paper["arxiv_id"])
            if key not in pool or version > identity(pool[key]["arxiv_id"])[1]:
                pool[key] = paper
            origins.setdefault(key, {})[name] = min(rank, origins.get(key, {}).get(name, rank))
            if key not in queue:
                queue.append(key)
        queues.append(iter(queue))

    picked, seen = [], set()
    while len(picked) < maximum:
        advanced = False
        for queue in queues:
            for key in queue:
                if key in seen:
                    continue
                seen.add(key)
                picked.append(key)
                advanced = True
                break
            if len(picked) >= maximum:
                break
        if not advanced:
            break
    result = []
    for key in picked:
        paper = dict(pool[key])
        paper["discovery"] = {
            "run_id": run_id, "retrieved_at": retrieved_at,
            "matched_queries": [
                {"name": name, "rank": rank} for name, rank in origins[key].items()
            ],
        }
        result.append(paper)
    return result, len(pool)



class CooldownActive(RuntimeError):
    pass


def check_cooldown(data, now):
    path = data / "arxiv_cooldown.json"
    if not path.exists():
        return
    state = json.loads(path.read_text(encoding="utf-8"))
    until = datetime.fromisoformat(state["retry_not_before_utc"])
    if until.tzinfo is None:
        raise ValueError("冷却记录缺少时区，请检查 data/arxiv_cooldown.json。")
    if now < until:
        minutes = max(1, int((until - now).total_seconds() + 59) // 60)
        local_time = until.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        raise CooldownActive(
            f"arXiv 冷却中，本次未联网。至少再等 {minutes} 分钟；"
            f"本机时间 {local_time} 后可手动尝试。该时间不保证服务恢复。"
        )


def is_rate_limit(error):
    # 常见 arxiv.py 版本提供 status；保留与已观察到报错格式的兼容。
    status = getattr(error, "status", None)
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    return status == 429 or response_status == 429 or bool(
        re.search(r"\bHTTP\s+429\b", str(error), re.IGNORECASE)
    )


def record_cooldown(data, now):
    until = now + timedelta(seconds=COOLDOWN_SECONDS)
    atomic_json(data / "arxiv_cooldown.json", {
        "reason": "HTTP 429", "recorded_at_utc": now.isoformat(),
        "retry_not_before_utc": until.isoformat(),
        "policy": "本地保守冷却 15 分钟；不是服务端承诺的恢复时间",
    })
    return until


def main(dry_run=False):
    # Lazy import: profile_config imports parse_profile from this module.
    from profile_config import profile_path

    _, config, profile_hash = read_profile(profile_path(ROOT))
    now = datetime.now(timezone.utc)
    queries, start, end = build_queries(config, now)
    print(f"首次提交时间范围（UTC）：{start.isoformat()} 至 {end.isoformat()}")
    print(f"检索组：{len(queries)}；每组最多 {config['per_query_limit']}；最终候选最多 {config['max_candidates']}")
    for query in queries:
        print(f"\n[{query['name']}] {query['query']}")
    if dry_run:
        print("\n配置检查通过。仅预览查询，未联网、未修改数据。")
        return

    import arxiv

    data = ROOT / "data"
    check_cooldown(data, now)
    run_id = now.strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = data / "discovery_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "version": VERSION, "run_id": run_id, "status": "running",
        "started_at": now.isoformat(), "profile_sha256": profile_hash,
        "config": config, "queries": queries, "groups": [],
        "window_start_utc": start.isoformat(), "window_end_utc": end.isoformat(),
    }
    atomic_json(run_dir / "run.json", report)
    try:
        # 共用客户端串行请求，拉长间隔；关闭库的自动重试，避免遇到 429 连续请求。
        client = arxiv.Client(page_size=config["per_query_limit"], delay_seconds=10.0, num_retries=0)
        buckets = []
        for query in queries:
            print(f"\n正在检索：{query['name']}……", flush=True)
            search = arxiv.Search(
                query=query["query"], max_results=config["per_query_limit"],
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )
            papers, outside, received = [], 0, 0
            for item in client.results(search):
                received += 1
                paper = normalize_result(item)
                published = datetime.fromisoformat(paper["published"])
                if not start <= published <= end:
                    outside += 1
                    continue
                papers.append(paper)
            buckets.append((query["name"], papers))
            report["groups"].append({
                "name": query["name"], "received": received,
                "within_window": len(papers), "outside_window": outside,
            })
            print(f"该组返回 {received} 篇，时间范围内保留 {len(papers)} 篇。", flush=True)
            atomic_json(run_dir / "run.json", report)

        papers, unique_count = merge_groups(buckets, config["max_candidates"], run_id, now.isoformat())
        # 保存本轮快照，便于追溯查询与去重行为。
        atomic_json(run_dir / "query_results.json", [
            {"name": name, "papers": values} for name, values in buckets
        ])
        atomic_json(run_dir / "papers.json", papers)
        atomic_json(data / "papers.json", papers)
        report.update({
            "status": "completed", "unique_count_before_limit": unique_count,
            "candidate_count": len(papers),
            "papers_sha256": hashlib.sha256((run_dir / "papers.json").read_bytes()).hexdigest(),
        })
        for index, paper in enumerate(papers, start=1):
            names = ", ".join(item["name"] for item in paper["discovery"]["matched_queries"])
            print(f"{index:2d}. [{names}] {paper['title']}")
        print(f"\n合并去重后 {unique_count} 篇，本轮保留 {len(papers)} 篇候选。")
        print(f"候选已保存到：{data / 'papers.json'}")
        if not papers:
            print("本轮候选为空，已写入空列表；请检查检索范围，不要继续评分。")
        print("本命令未调用大模型、未下载 PDF、未更新问答索引、未发送邮件。")
    except BaseException as error:
        if isinstance(error, Exception) and is_rate_limit(error):
            until = record_cooldown(data, datetime.now(timezone.utc))
            report.update({
                "status": "rate_limited", "error_type": type(error).__name__,
                "http_status": 429, "retry_not_before_utc": until.isoformat(),
            })
            local_time = until.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            raise CooldownActive(
                f"arXiv 返回 HTTP 429，已停止，未自动重试。"
                f"本机时间 {local_time} 后再手动尝试；冷却期间重复运行不会联网。"
                "15 分钟是本地等待策略，不保证届时恢复。"
            ) from error
        report.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error_type": type(error).__name__})
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(run_dir / "run.json", report)
        print(f"本次抓取记录：{run_dir / 'run.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="仅检查配置并显示查询，不联网或写数据")
    args = parser.parse_args()
    try:
        main(dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("抓取已中断，请勿继续评分，先检查本轮记录。")
        raise SystemExit(130)
    except Exception as error:
        print(f"抓取未完成（{type(error).__name__}）：{error}")
        print("请勿继续评分或发送；data/papers.json 可能仍是旧候选。")
        raise SystemExit(1)
