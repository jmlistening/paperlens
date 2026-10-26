"""Conservative, offline checks; neither punctuation nor self-review proves truth."""
from copy import deepcopy
import re

from answer_question import validate_answer
from answer_schema import VERSION as ANSWER_SCHEMA_VERSION, normalize_answer_schema


def normalized(text):
    return " ".join(text.split())


def has_terminal_boundary(text):
    value = normalized(text).rstrip('"”’')
    return bool(re.search(r"[.!?。！？][\')\]}]*$|\.\s*\(\d{1,3}[a-z]?\)$", value))


def extend_formula_quote(quote, source):
    """Append only an immediately following, numbered equation from the same source.

    Ambiguous occurrences, ordinary prose continuations and overlong equations
    are left for review. No punctuation or equation is generated.
    """
    q, s = normalized(quote), normalized(source)
    if has_terminal_boundary(q) or s.count(q) != 1:
        return None
    start = s.index(q)
    tail = s[start + len(q):start + 800]
    # A short formula prefix must immediately precede its first math operator.
    # Do not jump across prose or a sentence to reach a later equation.
    match = re.match(r"^\s+[A-Za-zΑ-ω][\w{}^+−′'(),\[\] ]{0,60}\s*[=∈≤≥].*?(?:\.\s*\(\d{1,3}[a-z]?\)|\(\d{1,3}[a-z]?\)\.)(?=\s|$)", tail)
    if not match:
        return None
    expanded = s[start:start + len(q) + match.end()]
    if len(expanded) > 800:
        return None
    return expanded


def check_coverage(value, point_count):
    """Validate model-supplied mappings, not semantic coverage of the question."""
    if not isinstance(value, list) or not 1 <= len(value) <= 6:
        raise ValueError("覆盖自查缺失，或不是 1～6 项列表。")
    seen = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != {"aspect", "status", "point_numbers", "limitation"}:
            raise ValueError("覆盖自查字段不完整。")
        aspect, status, numbers, limitation = (row[k] for k in ("aspect", "status", "point_numbers", "limitation"))
        if not isinstance(aspect, str) or not 1 <= len(aspect.strip()) <= 300 or aspect.strip() in seen:
            raise ValueError("覆盖自查的问题要点为空、重复或过长。")
        seen.add(aspect.strip())
        if status not in ("covered", "partial", "missing"):
            raise ValueError("覆盖自查状态无效。")
        if (not isinstance(numbers, list) or any(type(n) is not int or not 1 <= n <= point_count for n in numbers)
                or len(set(numbers)) != len(numbers)):
            raise ValueError("覆盖自查引用了不存在或重复的回答序号。")
        if not isinstance(limitation, str) or len(limitation) > 500:
            raise ValueError("覆盖自查的缺口说明无效。")
        if status == "covered" and (not numbers or limitation.strip()):
            raise ValueError("已覆盖的自查项必须关联回答，且不能同时声明缺口。")
        if status != "covered" and not limitation.strip():
            raise ValueError("未完全覆盖的自查项必须说明缺口。")
        if status == "missing" and numbers:
            raise ValueError("标为未覆盖，却关联了回答序号；需核对该项状态。")
        if status == "partial" and not numbers:
            raise ValueError("标为部分覆盖，但对应回答序号为空；无法定位已回答的部分。")
    return deepcopy(value)


def coverage_display_rows(value, point_count):
    """Read-only per-item display; never correct model claims or relax validation."""
    if not isinstance(value, list) or not 1 <= len(value) <= 6:
        return []
    aspects = [r.get("aspect", "").strip() for r in value
               if isinstance(r, dict) and isinstance(r.get("aspect"), str)]
    labels = {"covered": "自评已覆盖", "partial": "自评部分覆盖", "missing": "自评未覆盖"}
    rows = []
    for number, item in enumerate(value, 1):
        issue = ""
        try:
            check_coverage([item], point_count)
            if aspects.count(item["aspect"].strip()) > 1:
                raise ValueError("问题要点重复，需核对。")
        except ValueError as exc:
            issue = str(exc)
        raw = item if isinstance(item, dict) else {}
        aspect, status, numbers, limitation = (raw.get(k) for k in
                                               ("aspect", "status", "point_numbers", "limitation"))
        # Do not stringify malformed nested structures as if they were valid mappings.
        number_text = ("、".join(map(str, numbers)) or "无") if (
            isinstance(numbers, list) and len(numbers) <= 6
            and all(type(n) is int for n in numbers)) else "格式无效"
        rows.append({"自查项": number,
                     "问题要点": aspect[:300] if isinstance(aspect, str) else "格式无效",
                     "模型原始状态": labels.get(status, "格式无效") if isinstance(status, str) else "格式无效",
                     "对应回答": number_text,
                     "模型声明的缺口": limitation[:500] if isinstance(limitation, str) else "格式无效",
                     "结构检查": "待核对" if issue else "通过（仅结构）",
                     "核对说明": issue})
    return rows


def review_agent_answer(answer, evidence):
    reviewed, schema_changes = normalize_answer_schema(answer)
    # Normalizing an absent optional field never bypasses citation validation.
    validate_answer(reviewed, evidence)
    report = {"version": "answer_quality_v1_2", "model_answer_status": answer["status"],
              "answer_schema_version": ANSWER_SCHEMA_VERSION,
              "schema_normalizations": schema_changes,
              "quote_extensions": [], "warnings": [], "coverage": [],
              "coverage_validation": "unavailable",
              "semantic_verification": "not_performed"}
    report["warnings"].extend({"code": c["code"], "message": c["message"]} for c in schema_changes)
    for point_number, point in enumerate(reviewed["points"], 1):
        for citation_number, citation in enumerate(point["citations"], 1):
            quote = citation["quote"]
            expanded = extend_formula_quote(quote, evidence[citation["id"]]["text"])
            location = {"point": point_number, "citation": citation_number, "evidence_id": citation["id"]}
            if expanded:
                citation["quote"] = expanded
                report["quote_extensions"].append({**location, "kind": "adjacent_numbered_formula",
                                                   "original_quote": quote, "expanded_quote": expanded})
            elif not has_terminal_boundary(quote):
                report["warnings"].append({**location, "code": "quote_boundary_review",
                    "message": f"第 {point_number} 点第 {citation_number} 条引文末尾未检测到完整句界，请结合原文核对；程序未补写。"})
    try:
        report["coverage"] = check_coverage(reviewed.get("coverage"), len(reviewed["points"]))
        report["coverage_validation"] = "valid_structure_only"
    except ValueError as exc:
        # A missing/invalid self-review must not discard an otherwise grounded answer
        # or be silently interpreted as fully covered. Raw model output is retained.
        report["warnings"].append({"code": "coverage_review_unavailable", "message": str(exc)})
    else:
        gaps = [f"{r['aspect']}：{r['limitation']}" for r in report["coverage"] if r["status"] != "covered"]
        if gaps:
            if reviewed["status"] == "answered":
                reviewed["status"] = "partial"
            reviewed["limitations"] = list(dict.fromkeys(reviewed["limitations"] + gaps))
    report["needs_review"] = bool(report["warnings"])
    validate_answer(reviewed, evidence)
    return reviewed, report


def render_quality(report):
    lines = ["\n回答检查（不代表语义正确性已验证）："]
    if report["quote_extensions"]:
        lines.append(f"已从同一证据连续补入 {len(report['quote_extensions'])} 处编号公式；原始摘录保留在 JSON 中。")
    for warning in report["warnings"]:
        lines.append("待核对：" + warning["message"])
    labels = {"covered": "模型自评已覆盖", "partial": "模型自评部分覆盖", "missing": "模型自评未覆盖"}
    for row in report["coverage"]:
        numbers = "、".join(map(str, row["point_numbers"])) or "无"
        lines.append(f"- {row['aspect']}：{labels[row['status']]}；回答序号 {numbers}。{row['limitation']}")
    return "\n".join(lines)
