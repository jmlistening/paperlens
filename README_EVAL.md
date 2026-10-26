# PaperLens Eval-Lite：实验运行与报告

本文件描述 Track E 的 **Task E6**：批量实验 runner、断点续跑、token 预算，以及
JSON + Markdown 报告。E6 只提供**可运行的评测基础设施**；本任务没有运行真实
benchmark，也没有产生任何可写进简历的指标。所有实测数字必须在 Task E7 之后写入
`eval/CLAIMS.md`，并以 run manifest hash + n 为证据。

## 1. 新增文件

| 文件 | 作用 |
|---|---|
| `eval/run_experiment.py` | 实验 runner：按系统跑 benchmark、逐题 checkpoint、resume、budget；`ExperimentDeps.run_system` 是测试注入点 |
| `eval/experiments/matrix.yaml` | 系统矩阵：BM25/dense/RRF 已实现；rerank 与 answer 系统是显式 stub |
| `eval/analysis/make_report.py` | `make_report(results, out_dir)`：写 `results.json` + `report.md`，含 n 与 95% bootstrap CI |
| `eval/analysis/figures.py` | 纯文本条形图，无绘图依赖 |
| `tests/test_run_experiment.py` | 离线测试：合成 questions/qrels + fake system，覆盖失败、resume、budget、报告 |

## 2. Benchmark 输入

`benchmark_dir`（默认 `eval/benchmark_v2/`）必须包含：

- `questions.jsonl` 或 `questions.json`
- `qrels.jsonl` 或 `qrels.json`

JSON 文件可以是列表，或 `{"questions": [...]}` / `{"qrels": [...]}` 包装。每题至少：

```json
{"question_id": "q001", "question_zh": "...", "query_en": "...", "type": "factual", "answerability": "answerable"}
```

qrels 记录：

```json
{"question_id": "q001", "gold_parent_ids": ["p_..."], "answerability": "answerable"}
```

约束：

- `question_id` 必须唯一；不同 id 不能映射到同一个 checkpoint 文件名（例如 `a/b` 与 `a_b` 会冲突并报错）。
- gold 字段（`gold_parent_ids`、`gold_points` 等）**不会**传给 system：runner 调用前会剥离，避免答案泄漏。
- 没有 qrels 记录的题目按“无 gold”处理：计入 n 与失败率，但不进入检索指标分母。

## 3. 系统矩阵

`eval/experiments/matrix.yaml` 声明 6 个系统：

| 系统 | 类型 | 状态 | 说明 |
|---|---|---|---|
| `bm25_top6` | retrieval | 已实现 | BM25Okapi k1=1.5, b=0.75, top-6 |
| `dense_top6` | retrieval | 已实现 | E5-small + FAISS，query_zh，top-6 |
| `hybrid_rrf_top6` | retrieval | 已实现 | dense top-50 + BM25 top-50，RRF k=60，top-6 |
| `hybrid_rrf_rerank_top6` | retrieval | **stub** | 需要 LLM rerank；`implemented: false`，跳过而不是伪造结果 |
| `fixed_rag_matched` | answer | **stub** | 需要 LLM；E6 不调用模型 |
| `agent_single_round` | answer | **stub** | 需要 LLM；E6 不调用模型 |

检索系统的唯一输出是：

```json
{"retrieved_parent_ids": ["...", "..."], "tokens": 0}
```

`implemented: false` 的系统在报告中标记为 `skipped (not_implemented)`，不产生 n 或指标。
以后接入 answer 系统时：

1. 实现一个新的 `ExperimentDeps` 适配器（`run_system(system, question) -> dict`）；
2. 在 matrix 中把该系统改为 `implemented: true`；
3. 保持与 matched 系统相同的 evidence 上限与 answer prompt，否则不可做因果比较。

本任务明确**不调用 LLM、不访问 arXiv、不联网、不读 `data/`**。只有 `RetrievalDeps`
在真实运行时读取 `benchmark_dir/index/`。

## 4. 运行命令

Task E7 的手动 gate 示例（本次未执行）：

```powershell
Set-Location <repo>

& python eval\run_experiment.py `
  --matrix eval\experiments\matrix.yaml `
  --systems bm25_top6,dense_top6,hybrid_rrf_top6 `
  --budget 200000 `
  --out eval\benchmark_v2\runs\20261025_retrieval_v1

& python eval\analysis\make_report.py `
  eval\benchmark_v2\runs\20261025_retrieval_v1
```

不指定 `--out` 时写入 `eval/benchmark_v2/runs/<UTC timestamp>/`。`--no-report` 只打印
摘要。`make_report.py` 不带参数时读取 `eval/benchmark_v2/runs/` 下最新的 run。

## 5. 输出布局

```text
<out_dir>/
  <system>/
    <question_id>.json    # 每题一个 checkpoint（成功或失败都在）
    run.json              # 该系统的 status / n / metrics / failures / tokens
  run.json                # 聚合：matrix hash、budget 状态、stop_reason、results
  results.json            # make_report 生成：均值、95% CI、n、failures
  report.md               # make_report 生成的主表与限制
```

checkpoint 关键字段：`status`（`ok` / `failed` / `invalid`）、`error`、
`retrieved_parent_ids`、`tokens`、`metrics`、`attempt`、`finished_at`。

## 6. Resume、失败与预算语义

**Resume**

- 每个题目完成后立即原子写 checkpoint。
- 再次运行会跳过 `status` 为 `ok` / `failed` / `invalid` 且 `question_id` 匹配的 checkpoint。
- 损坏或写了一半的 checkpoint 会自动重跑。
- 默认连失败题也跳过（失败已按 0 计分）。要重试失败题：
  `--retry-failed`，或删除对应 checkpoint 文件。`attempt` 会递增。
- `--limit N` 是调试用上限（每个系统只跑前 N 题）；结束后 `status=incomplete`、
  `stop_reason=limit`，不带 limit 重跑会继续。

**Budget**

- `--budget` 或 matrix 的 `budget.max_tokens` 限制记录到的 token 总数。
- 在启动下一题**之前**检查预算；已在运行的一题允许完成，因此最终可能超出上限一题。
- 预算耗尽时：`run.json` 写 `status=budget_exhausted`、`budget.stopped=true`、
  `budget.tokens_used`、`budget.remaining_question_ids`；退出码仍为 0（这是干净的停止，不是错误）。
- 去掉 budget 重跑即可从 checkpoint 继续。

**失败策略**

- system 抛异常 → `status=failed`，error 记录类型与消息（截断 500 字符）。
- 返回非 dict、`retrieved_parent_ids` 不是合法 id 列表、未知 status → `status=invalid`。
- 显式 `failed` / `error` / `timeout` / `schema_invalid` / `not_implemented` → `failed`。
- 失败或 invalid 的题：检索指标记 0（仅当该题有 gold），`schema_success=0`；
  永远留在分母里，**不会静默丢弃**。

## 7. 指标与统计

每题计算的确定性指标：

- `recall@3`、`recall@6`、`recall@10`（gold parent chunks 命中比例）
- `all_gold@6`（全部 gold 都在 top-6）
- `mrr@10`、`ndcg@10`（rel=2 gold；可选 qrels `rel` 映射）
- `quote_integrity`（仅当输出含 citations 时；失败为 0）
- `schema_success`（合法输出为 1，失败为 0）

无 gold 的题（如 unanswerable）不进入检索指标分母，但仍计入 `n` 与失败率。
系统均值只在有样本的题上取平均；失败题的 0 始终包含在内。

`make_report` 的统计约定：

- 95% percentile bootstrap，2000 次重采样，固定 seed `20261025`（每个
  (system, metric) 由 SHA-256 派生独立 seed），报告可复现。
- 主表每格为 `mean [ci_low, ci_high]`，并列出 `n`、失败数、tokens。
- **不做显著性检验。** 设计文档要求 120 题才考虑 McNemar/Holm；40 题 Eval-lite
  只报告描述性统计与区间。报告与 `results.json` 都显式写
  `significance_testing: false`，不要把这些行解读为“显著提升”。
- 跳过（stub）的系统单独列出，不给数字。

## 8. 离线测试

```powershell
& python -I -B -X utf8 -m unittest discover -s tests -p "test_run_experiment.py" -v
```

测试使用合成 questions/qrels 与 fake system（3 题、1 个失败、resume、budget stop、
n=40 报告），只写 `tests/.tmp/`，不读 `data/`、不加载模型、不联网。E6 的测试不运行
真实 benchmark；Task E7 的第一次真实运行是人工 gate。

## 9. 限制（必须随结果一起说明）

- 本任务没有真实测量结果；`eval/CLAIMS.md` 中 Recall@k 等仍为 `TODO`。
- 检索指标只说明“gold parent 是否被召回”，不代表答案正确、faithfulness 或引用被支持。
- 40 题不足以支撑系统间差异的显著性结论；不得只报有利子集。
- BM25 与 dense 使用同一 `query_en` 做 apples-to-apples；dense 可另报 `query_zh` 条件。
- 失败计 0 的规则会压低系统分数；这是有意的诚实口径，不是待修复的缺陷。
- 报告中的 bootstrap 区间反映题目间变异，不代表对总体的推断。
