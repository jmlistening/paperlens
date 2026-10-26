# Answer Repair Report (one-shot, opt-in)

- Generated at: `2026-10-25T05:09:25+00:00`
- Eval directory: `eval\benchmark_v2\runs\answer_eval120`
- Benchmark: `eval\benchmark_v2\eval120`
- Answer model: `deepseek-flash`
- Dry run: False
- Selected: 8; repaired: 4; skipped: 2; still invalid: 2

This pass only repairs checkpoints that were `answer_status == "invalid"` or `status == "failed"`. It is citation/schema-level repair, not a correctness or faithfulness judgement.

## Repaired

| Question | Answer status | Previous error | Attempt | Tokens (before -> after) |
|---|---|---|---|---|
| q018 | answered | 引用原句长度无效。 | 2 | 3247 -> 7208 |
| q101 | answered | invalid JSON: Expecting property name enclosed in double quotes | 2 | 2905 -> 6640 |
| q114 | answered | E2 的引文不在原文中，回答未通过校验。 | 2 | 3409 -> 7509 |
| q127 | partial | 引用原句长度无效。 | 2 | 2529 -> 5544 |

## Skipped (not repaired)

| Question | Reason |
|---|---|
| q005 | no usable evidence: no evidence recorded in checkpoint |
| q007 | no usable evidence: no evidence recorded in checkpoint |

## Still invalid

| Question | Original error | Repair error |
|---|---|---|
| q009 | APIConnectionError: Connection error. | E2 的引文不在原文中，回答未通过校验。 |
| q108 | E2 的引文不在原文中，回答未通过校验。 | E2 的引文不在原文中，回答未通过校验。 |

## Already valid (no-op)

| Question | Answer status |
|---|---|
| q001 | answered |
| q002 | partial |
| q004 | answered |
| q010 | answered |
| q011 | answered |
| q012 | answered |
| q013 | answered |
| q014 | answered |
| q015 | partial |
| q016 | partial |
| q017 | answered |
| q019 | answered |
| q020 | answered |
| q021 | answered |
| q023 | answered |
| q025 | answered |
| q027 | answered |
| q028 | answered |
| q029 | answered |
| q030 | answered |
| q032 | partial |
| q033 | partial |
| q035 | answered |
| q036 | answered |
| q037 | answered |
| q038 | answered |
| q040 | answered |
| q102 | answered |
| q103 | answered |
| q104 | answered |
| q105 | answered |
| q106 | answered |
| q107 | answered |
| q109 | answered |
| q110 | partial |
| q111 | answered |
| q112 | answered |
| q113 | answered |
| q115 | answered |
| q116 | answered |
| q117 | answered |
| q118 | answered |
| q119 | answered |
| q121 | answered |
| q122 | answered |
| q123 | answered |
| q124 | answered |
| q125 | partial |
| q126 | answered |
| q128 | answered |
| q129 | answered |
| q130 | partial |

## Notes

- A checkpoint with no usable evidence is skipped and never repaired.
- A failed repair keeps the original error/raw response and records `repaired_attempt` in the checkpoint.
- A successful repair overwrites only that checkpoint; token and latency totals are the sum of the original and repair calls.
- `data/`, benchmark questions/qrels and the retrieval run are never written.
