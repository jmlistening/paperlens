# PaperLens Benchmark v2 标注指南（Task E5b）

本目录的题目用于 held-out Eval-lite。目标是 40 题（可扩到 60），覆盖事实、
数值、机理、多跳、部分可答、不可答六类。题目、gold quote、页码和 qrels 一经
冻结（`manifest_v1.json`）不再修改；任何改动都必须新建版本并重跑全部系统。

## 1. 文件与流程

| 文件 | 作用 |
|---|---|
| `splits.draft.json` | 40 篇 test 论文（family 级隔离；含 `prepared_path`） |
| `generate_questions.py` | 调模型生成候选；`--limit 3` 做 pilot |
| `questions.draft.jsonl` | 候选题目，一行一个 JSON 对象 |
| `validate_questions.py` | 程序校验候选与整套题 |
| `build_qrels.py` | 把 gold quote 映射到 `index/corpus_clean.json` 的 parent chunk |
| `qrels.jsonl` | 每题的 `gold_parent_ids` / `answerability` / `missing_aspects` |
| `freeze.py` | 记录 questions/qrels/splits/index 的 SHA-256 与计数 |
| `manifest_v1.json` | 冻结清单；`freeze_id` 由 artifact 哈希决定 |
| `generation_usage.json` | 真实生成时的 model / prompt hash / token 用量 |

标准流程：

1. `python eval/benchmark_v2/generate_questions.py --limit 3`（pilot）；
2. `python eval/benchmark_v2/validate_questions.py --questions ... --splits ...`；
3. 作者逐题人工复核、编辑或拒绝（更新 `review`）；
4. 若存在第二标注者，对至少 25% 的题做独立复核；没有第二人时在 README 写明
   single-annotator limitation，**不要**伪造 κ；
5. `build_qrels.py` 生成 qrels；
6. 用户确认题目后才运行 `freeze.py`；冻结后只跑一次正式评测。

## 2. 题目 schema（JSONL，一行一题）

```json
{
  "question_id": "q001",
  "paper_ids": ["2607.05956v1"],
  "question_zh": "...",
  "query_en": "...",
  "type": "factual|numeric|mechanism|multi_hop|partial|unanswerable",
  "answerability": "answerable|partial|unanswerable",
  "gold_points": [
    {"requirement": "...", "quote": "...", "pdf_page": 3, "section": "..."}
  ],
  "missing_aspects": [],
  "source_title": "...",
  "generator": {"model": "...", "prompt_sha256": "...", "created_at": "..."},
  "review": {"status": "pending", "reviewer": null, "timestamp": null}
}
```

- `paper_ids`：单篇题一个 ID；跨论文多跳题至少两个 family。
- `query_en`：检索系统实际使用的英文查询，避免中文混入 BM25 条件。
- `gold_points`：2–4 个为宜；每个 point 对应一个可独立核验的要求。
- `review.status`：`pending` / `approved` / `needs_edit` / `rejected`。
  人工修改后必须写 `reviewer` 与 ISO-8601 `timestamp`。

## 3. 程序校验规则（硬性）

`validate_candidate` / `validate_question_set` 会拒绝：

1. `quote` 不是所引页面的空白规范化子串；或 `pdf_page` 不存在。
2. `quote` 只出现在摘要中（摘要里出现、正文其他位置也出现的 quote 可以保留）。
3. 整套可答题中，非摘要 gold quote 占比低于 **70%**。
4. `numeric` 题的 quote 缺少数字或缺少单位（如 `40%`、`8 tokens`）。
5. 空白规范化后重复的 `question_zh`。
6. `answerable` / `partial` 题没有 gold point；`type` 与 `answerability` 冲突。

`quote` 要求逐字，不可翻译、改写、拼接或跨页。只允许空白（换行/多空格）差异。
数字题的单位必须与数字相邻；只有表号（如 “Table 2”）不算单位。

## 4. 题型与判定

| type | answerability | 说明 |
|---|---|---|
| `factual` | `answerable` | 单点事实，1–2 个 gold point |
| `numeric` | `answerable` | 必须引用带单位的数值 |
| `mechanism` | `answerable` | 解释方法/流程/因果 |
| `multi_hop` | `answerable` | 需组合多段或两篇论文 |
| `partial` | `partial` | 论文只覆盖问题的一部分；`missing_aspects` 必填 |
| `unanswerable` | `unanswerable` | 论文未提供；gold_points 为空，`missing_aspects` 说明缺失内容 |

不可答题应覆盖四类：内容缺失、假前提、超出论文范围、需要外部知识。不要把
“检索不到”误标成“论文没有”。

## 5. qrels 规则

`build_qrels.py` 对每个 gold quote 做空白规范化，然后在
`eval/benchmark_v2/index/corpus_clean.json` 的 parent chunk 中查找包含它的
**全部** chunk，写入 `gold_parent_ids`（去重、排序）。规则：

- 一个 quote 命中多个 parent chunk 时全部保留，Recall 以 parent 为单位。
- `answerability == "unanswerable"` 的题一律 `gold_parent_ids: []`。
- 没有命中任何 chunk 的 quote 写入 summary 的 `unmatched_quotes`，需人工处理；
  不要通过放宽查询或改写 qrels 掩盖。
- qrels 与 questions 必须一一对应。

## 6. 冻结与隐私

- `freeze.py` 记录 questions、qrels、splits、index manifest 的 SHA-256、题目
  数量、类型分布、答案性分布和 review 状态。
- 相同输入重复冻结返回同一 `freeze_id` 与 `already_frozen: true`；输入变化时
  必须显式 `--overwrite`。
- 公开仓库只放 arXiv ID、标题、题目、页码、短 quote / 哈希；不放 PDF 或全文。
- 真实生成用量写入 `generation_usage.json`（model、prompt_sha256、tokens），
  不得伪造 token 数；测试一律注入 fake `chat_fn`，禁止调用真实 API。
