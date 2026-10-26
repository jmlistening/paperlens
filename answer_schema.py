"""Narrow, auditable schema normalization; never invent answer content or gaps."""
from copy import deepcopy

VERSION = "answer_schema_v1_missing_optional_limitations"


def normalize_answer_schema(answer):
    """Only an absent limitations field on an answered response defaults to [].

    Explicit null/wrong types, partial/insufficient omissions, invented quotes,
    and missing coverage are not repaired here. Callers must still validate.
    """
    normalized = deepcopy(answer)
    changes = []
    if isinstance(normalized, dict) and normalized.get("status") == "answered" and "limitations" not in normalized:
        normalized["limitations"] = []
        changes.append({"code": "missing_limitations_on_answered", "field": "limitations",
                        "original_presence": False, "replacement": [],
                        "message": "模型自评 answered 但漏写 limitations；程序按空列表处理，仅修复结构，不代表不存在局限或答案正确。"})
    return normalized, changes
