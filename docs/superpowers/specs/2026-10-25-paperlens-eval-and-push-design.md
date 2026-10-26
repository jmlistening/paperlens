# PaperLens Eval-Lite + 163 Push Design

日期：2026-10-25  
状态：已由用户确认方向，进入实施计划  
范围：M0 冻结与仓库卫生；Track P（163 优先的可选 arXiv 推送 MVP）；Track E（40–60 题 Eval-lite 与确定性检索/引用指标）。M3（claim 验证/校准拒答）、M4（GitHub/面经收尾）在本设计中有边界，但不在首个实施计划内。

## 1. 意图

PaperLens 已经有一套真实跑过的数据/索引/问答/邮件链路，但缺少可写进简历的正式评测：目前只有 6 道 dev 题，只有 Q01 在 fixed/agent_one/agent_multi 各跑过 1 次，人工质量分为空。用户要求：

1. 真实 baseline vs 改进的实测数字，不编造；
2. 自动推送前沿 arXiv 论文，可关闭，163 邮箱优先；
3. 产品与评测同时推进；
4. 后续上传 GitHub，并用于面试练习。

本设计把“评测”和“推送”拆成两条互不阻塞的轨道，共享 M0 的冻结与仓库卫生。两条轨道都只写新目录/新入口，不改旧失败记录与旧索引。

## 2. 当前事实（以代码与落盘状态为准）

- 问答：8 篇 dev 论文、95 页、2,505 原始片段、453 清洗片段、460 条 384 维 E5 向量；fixed RAG = dense top-20 → DeepSeek 0–3 重排 → top-3 → JSON 答案，引文必须是证据原文的空白规范化子串。
- Agent：有界控制器，最多 3 轮/4 次调用；正式 Q01 的 agent_one/agent_multi 都只有 1 轮检索/2 次调用。
- 数据：2,125 篇 prepared 全文（dataset_v1 999 + dataset_history_v1 1,126），42,682 页、151,533,086 字符；2,119 条模型初筛建议（core 796/adjacent 911/out_of_scope 411/uncertain 1）；human_reviewed=0；candidate_qa=verified_qa=frozen_test_qa=0；2,133 条记录 license_status=not_verified。
- 评测：6 题 dev pilot；Q01 三模式各 1 次，25,729 tokens；12/12 引文逐字校验通过；human_quality=null；无 accuracy/faithfulness/refusal/Recall@k 代码。
- 推送：fetch → rank（LLM 0–100，top_k=3，min_score=60）→ summarize → send_digest；delivery.sqlite3 有状态机；3 次 discovery 全部 failed/429；无 scheduler、无推送开关/退订；当前 selected_summaries 哈希与 recommendations 不匹配，直接发送会被拒。
- 工程：显式命令下 103 项离线测试通过；默认 unittest discover 只收 16 项；不是 git 仓库；无 README.md/requirements/LICENSE/CI；docs 约 101MB 且含隐私文本。

## 3. 目标、非目标、反目标

### 目标

- G1：构建 40–60 题 held-out benchmark，family 级隔离，parent-chunk qrels，题型覆盖事实/数值/机理/多跳/部分可答/不可答。
- G2：在 matched evidence budget 下比较 BM25、dense、RRF hybrid、rerank on/off、fixed 与 Agent；产出 Recall@3/6/10、MRR@10、nDCG@10、quote integrity、schema success、p50/p95 延迟、tokens/query，带 n 与 95% CI。
- G3：实现 163 优先的可选每日 arXiv 推送：默认关闭、启停/暂停/静默、预览、首次发送人工批准、watermark + seen ledger、arXiv family 去重、163/QQ SMTP preset、Windows Task Scheduler、离线测试。
- G4：仓库可复现：git init、README、requirements、.env.example、CI、脱敏 demo、claims ledger。
- G5：面经包：3–5 条中英文 bullet、20–30 道问答、2 次 mock（M4）。

### 非目标

- 不在 M1 做 120 题双标注 + 11 系统 + 统计检验的完整 benchmark。
- 不在 M1 做 claim-level judge/faithfulness；M3 再做，且必须有独立 judge 或人工审计。
- 不把 2,125 篇全量建索引；M1 只建 40 篇 test index。
- 不发布 PDF/全文；不把 Q01 单题结果当准确率。
- 不把推送可靠性指标与问答质量指标混在一张表。

### 反目标

- 任何“为了好看”而预填百分比、把 dev 题当 test、把模型自评当人工分、把 SMTP accepted 说成收件成功，都视为失败。
- 任何未经用户明确批准的真实邮件发送、任何把 .env/授权码提交到仓库的行为，都视为失败。

## 4. 约束

- 平台：Windows 11，Python 3.11.17，解释器 `python`；Streamlit 1.66、FAISS、sentence-transformers、rank-bm25、PyMuPDF 已安装。
- 旧数据/旧索引/旧失败记录保持不动；新产物写新目录。
- 磁盘：D: 约 15GB 可用；不得生成无上限的 raw trace/zip。
- arXiv 限流：串行、冷却、bounded retry；429 不得静默跳天。
- 邮件：163/QQ 用授权码；SMTP accepted ≠ inbox；默认 links-first，不默认附 PDF。
- 安全：秘密只进 `.env`；日志/状态掩码；GitHub 只发代码、配置模板、arXiv ID/元数据、自建标注，不发 PDF/全文/个人 DB。

## 5. 总体架构

### M0 共享冻结（先做）

- `git init`（私有、本地；不 push）。
- `.gitignore`：排除 `data/`、`.env`、`__pycache__/`、`.idea/`、`.vscode/`、模型缓存、PDF、旧运行产物、docs 大证据目录。
- `requirements.txt`：按当前环境 pin；另加 CPU/GitHub 说明。
- `.env.example`：只写变量名/占位符。
- `tests/__init__.py`：让默认 discover 收集 tests/ 下测试。
- `eval/CLAIMS.md`：每条简历声明 → 结果文件/图/表 → manifest hash → n。
- `eval/baseline_snapshot_20261025.json`：当前索引/数据/测试数量/哈希的只读快照。
- `README.md` 骨架：定位、现状、quickstart（待填）、已实现 vs 已实测、局限。

### Track P：推送

新增文件（复用旧脚本为库，不重写）：

- `push_config.py` / `push_config.json`：唯一配置源；默认 `enabled=false`；provider=163/qq/custom；不存密码。
- `push_state.py` + `data/frontier.sqlite3`：papers/candidates/runs/watermarks/impressions；paper_key 去版本号；slot 幂等；seen ledger。
- `push_fetch.py`：watermark-overlap-catchup 窗口；复用 `fetch_papers` 的 profile/query/normalize/merge；429 冷却；候选上限与溢出队列。
- `push_rank.py`：抽出 `rank_papers.rank_batch`；冷启动 = llm_only；可选 favorite 相似度；min_score 只作用于 llm 分；类别多样性。
- `mail_transport.py`：163/QQ/custom preset；465 SSL / 587 STARTTLS；login/noop/test/send；错误分类；不记录密码。
- `push_digest.py`：links-first HTML/plain；无追踪像素；页脚关闭说明；附件默认关。
- `push_run.py`：门禁（enabled/paused/quiet）→ slot → fetch → dedup → rank → summarize → digest → preview/send；失败退出码；连续失败自动暂停。
- `push_ctl.py` / `push_status.py`：init/doctor/enable/disable/pause/resume/preview/send/test-email/approve-first-send/schedule/history/status。
- `pages/3_推送设置.py`：总开关、频率/时间、provider、附件、暂停/静默、测试邮件、预览、首次发送批准、最近 runs；不显示密码。
- `scripts/register_push_task.ps1`、`unregister_push_task.ps1`、`run_push.bat`；`docs/PUSH.md`。
- `tests/test_push_*.py`：fake arXiv/LLM/SMTP/clock；默认 discover 可收集。

### Track E：Eval-lite

新增文件：

- `eval/benchmark_v2/select_corpus.py` + `splits.json`：40 篇 test（24 篇 2026 + 16 篇 2020–2025），排除 8 个 dev family、out_of_scope/uncertain、92 篇 quality flagged、近重复。
- `eval/benchmark_v2/build_test_index.py`：用与 dev 相同的 `prepare_corpus` + `clean_corpus` 重新解析 40 篇 PDF（不是直接用 prepared pages），切块/嵌入/FAISS；写入 `eval/benchmark_v2/index`；默认 `data/rag` 不变。
- `eval/benchmark_v2/leakage_audit.py`：family/title/abstract/full-text 近重复审计；dev/test 重叠必须 0。
- `eval/benchmark_v2/generate_questions.py` + `prompts/`：半自动出题；每题带 question_zh/query_en/type/difficulty/answerability/gold points（quote/pdf_page/section）。
- `eval/benchmark_v2/validate_questions.py`：quote 必须是 page text 子串；page/chunk 存在；数字题含数字+单位；跨论文题 ≥2 family；去重；≥70% gold spans 不在 abstract。
- `eval/benchmark_v2/review_app.py` + `ANNOTATION.md`：人工复核/编辑/拒绝；记录 reviewer/timestamp；25% 第二标注（若有人）。
- `eval/benchmark_v2/build_qrels.py`：parent-chunk 级 qrels；child 映射 parent；answerability/missing_aspects。
- `eval/benchmark_v2/freeze.py` + `manifest_v1.json`：论文/PDF/pages/chunker/model/index/questions/gold/split 哈希。
- `retrieval_core.py`：dense/BM25/RRF 可配置；`search_evidence.py`/`search_bm25.py`/`agent_tools.py` 复用或薄适配；默认行为不变。
- `eval/metrics_v2/metrics.py` + `score_run.py`：Recall@3/6/10、MRR@10、nDCG@10、quote integrity、schema success、latency、tokens。
- `eval/run_experiment.py` + `eval/experiments/matrix.yaml`：按系统跑 benchmark；per-task checkpoint/resume；token budget；失败计 fail。
- `eval/analysis/make_report.py` + `figures.py`：JSON + Markdown 主表；n/CI；placeholder for M3 judge。
- `tests/test_benchmark_*.py`、`tests/test_metrics_v2.py`、`tests/test_retrieval_core.py`。

### 共享

- `requirements.txt`、`.env.example`、`README.md`、`.github/workflows/ci.yml`、`eval/CLAIMS.md`。
- 所有新测试默认 discover 可收集；纯 Python 测试与重依赖测试分 job（M4）。

## 6. Track E 详细设计

### 6.1 语料与 split

- 测试池：core/adjacent、未 flagged、非 8 个 dev family；40 篇；24 篇 2026 recent + 16 篇 2020–2025 historical；core:adjacent 约 3:2。
- Family = 去版本号 arXiv base ID；同一 family 只能在一个 split；test 论文不进 `data/rag`。
- 近重复：title Jaccard ≥0.8、abstract Jaccard ≥0.75、全文 embedding cosine ≥0.9 不跨 dev/test；报告 0 重叠。
- 公开版只发 arXiv ID、标题、题目、页码、短 quote/哈希；pdf/全文不入库。

### 6.2 题集（40 题 MVP，可扩到 60）

40 题分布建议：factual 10、numeric 5、mechanism 7、multi-hop 6（3 within + 3 cross）、partial 4、unanswerable 8（4 absent + 2 false-premise + 2 out-of-scope）。60 题按比例扩到 15/8/10/9/6/12。每题 2–4 个 gold points，约 100–160 个 gold spans。

出题流程：LLM 生成候选 → 程序校验 → 作者 100% 复核 → 25% 第二标注（若有人）→ 10 题 dev smoke → 冻结。没有第二标注者时，README 写 single-annotator limitation，不报 κ。

### 6.3 检索与系统矩阵

必须先测确定性检索，不依赖 judge：

- `bm25_top6`：query_en，BM25Okapi k1=1.5,b=0.75，同 filter/parent dedup。
- `dense_top6`：E5-small，query_zh，同 filter/parent dedup。
- `hybrid_rrf_top6`：dense top-50 + BM25 top-50，RRF k=60，top-6。
- `hybrid_rrf_rerank_top6`：hybrid top-20 → 现有 LLM rerank → grade≥2 → top-6。
- `fixed_rag_matched`：hybrid+rerank top-6，统一 answer prompt。
- `agent_single_round`：dense top-20 → 1 次 decision 选 ≤6 → 同一 answer prompt。
- `agent_multi_round`：最多 3 轮，可补 dense/BM25 top-12；记录 rounds/stop_reason；只有 >1 轮的题才计入 multi-round 增益。
- `oracle_evidence`：gold parent chunks 作为 evidence，同 cap/prompt；只作生成上界。

M1 不跑 claim verification/refusal judge；M3 再加。

### 6.4 指标与统计

- Recall@3/6/10：gold parent chunks 在 top-k 的比例；AllGold@6；MRR@10；nDCG@10（rel=2 gold, 1 同页/邻接, 0 其他）。
- quote integrity：引文空白规范化后是 cited evidence 子串的比例，硬门 1.00。
- schema success / normalization rate。
- p50/p95 latency（in-process 分段 + CLI wall，冷启动单列）、tokens/query（prompt/completion/cached/total）。
- 40 题只报描述性统计 + paired bootstrap 95% CI；不宣称 10–15pp 显著。120 题才做 McNemar/Holm。
- Test-once：冻结后只跑一次；任何改动新建 v2，全部系统重跑。

### 6.5 公平性

- fixed 与 agent 的证据上限、answer prompt、后处理必须统一；production fixed 单列，不与 matched 系统做因果比较。
- closed-book 只评 answer correctness，不放进要求 citation 的 strict 表。
- BM25/dense 使用同一 query source（query_en）做 apples-to-apples；dense 另报 query_zh 条件。
- 主表只放预注册的 primary comparison；其他进 exploratory。

## 7. Track P 详细设计

### 7.1 配置与开关

`push_config.json` 默认：

```json
{
  "enabled": false,
  "paused": false,
  "quiet_mode": false,
  "frequency": "daily",
  "time_local": "08:30",
  "provider": "163",
  "attach_pdfs": false,
  "window_hours": 72,
  "overlap_hours": 24,
  "catchup_days": 7,
  "fetch_pool_limit": 40,
  "rank_max_candidates": 20,
  "max_per_digest": 3,
  "min_new_papers": 1,
  "auto_pause_after_failures": 3
}
```

- 默认关闭；`enabled=false` 时 `push_run` 在任何网络/模型/SMTP 调用前退出。
- `paused` 完全跳过；`quiet_mode` 只生成预览不发信。
- 首次真实摘要邮件必须 `approve-first-send --confirm`；计划任务不能越过。
- 预览永不连 SMTP、不写 delivery 账本。

### 7.2 163 / QQ

- 163：`smtp.163.com`，465 SSL（默认），587 STARTTLS 可选；`SMTP_USER` 为完整地址；`SMTP_PASSWORD` 为客户端授权码；From 与账号一致。
- QQ：`smtp.qq.com`，465 SSL；16 位授权码；From 一致。
- 通用：`SMTP_HOST`/`SMTP_PORT`/`SMTP_SECURITY`；465→SMTP_SSL，587→STARTTLS，25 仅 localhost；默认证书校验开启。
- 授权码只进 `.env`；日志只记录异常类型/SMTP code；状态掩码 `a***@163.com`。

### 7.3 增量与去重

- `paper_key` = 去版本号 arXiv base ID；版本号单独记录。
- `delivery.sqlite3` 仍是邮件唯一权威：accepted/pending/sending/uncertain 阻止重发；failed 可重试；uncertain 需人工 `--resolve`。
- `frontier.sqlite3` 记录 seen/evaluated/selected/delivered；低分论文不每天重评。
- watermark 仅在抓取成功后推进；`start=max(watermark-overlap, now-catchup)`；首次 `initial_lookback_days=7`。
- 候选上限：fetch_pool_limit=40、rank_max_candidates=20、max_per_digest=3；溢出写 queued，FIFO，TTL 14 天。
- 已 ignored/已读/已投递/已评估/8 篇 dev 论文跳过。

### 7.4 调度与可靠性

- Windows Task Scheduler：`run_push.bat` → `push_run.py --scheduled`；`StartWhenAvailable`、`MultipleInstances=IgnoreNew`、`RestartCount=2`、`RestartInterval=10m`、`ExecutionTimeLimit=2h`。
- 默认 Interactive 登录类型；关机/未登录会错过 on-time，报告必须区分 on-time 与 catch-up。
- 连续 3 次失败或 SMTP uncertain 自动暂停；`push_status.py` 提供 stale/next_expected/queue/tokens。
- 429：bounded retry + jitter，冷却期不重试；一次 429 不得静默跳天。

### 7.5 邮件与安全

- links-first：标题、arXiv ID/category、中文速览、兴趣分、reason/limitation、arXiv 链接；附件默认关，开启也按 10MB raw/15MB encoded 降级为链接。
- HTML 全转义；无远程图片/追踪像素；链接只允许 arxiv.org/www.arxiv.org；页脚给 UI/CLI 关闭路径，不伪造一键退订。
- 状态/日志不出现完整邮箱/授权码/笔记/阅读历史。
- Streamlit 绑定 127.0.0.1，headless，关闭 usage stats。

## 8. M0 冻结与 GitHub 卫生

- 不删除旧数据/旧失败/旧交付目录；先 gitignore，再决定归档。
- `.gitignore` 至少排除：`.env`、`data/`、`backups/`、`__pycache__/`、`.idea/`、`.vscode/`、`data/models/`、`*.pdf`、`*.zip`、`docs/handoff/evidence_*/`、`docs/stage1*/**/deliveries/`。
- `requirements.txt` 记录当前环境关键包；GitHub 另给 CPU-only 安装说明。
- `eval/CLAIMS.md` 初始只列“已实现/已跑过/已实测”三类，所有数字留空或写 TODO。
- 首次提交只包含代码/文档/配置模板/测试，不包含 data/docs 大目录。

## 9. 简历声明政策

- 每条 bullet 必须映射到 `eval/CLAIMS.md` 的一行：claim → 结果 JSON/表/图 → manifest hash → n。
- 未测量写 `TODO`，不写 `[X%]` 的具体值；目标值写在 `target` 字段，不与 `measured` 混。
- 负结果/不显著必须如实写；不得只报有利子组。
- 推送指标与 QA 指标分表；SMTP accepted 不写成 delivered。

## 10. 风险与缓解

| 风险 | 缓解 |
|---|---|
| test index 用 prepared pages 导致与 dev 不可比 | 用同一 prepare_corpus/clean_corpus 重解析 40 篇 PDF；gold quote 对 test index 实际 chunk 校验 |
| 40 题 power 不足 | 只报描述性 + bootstrap CI；不宣称显著；60 题作为可选扩展；M3 不依赖小样本显著 |
| judge 自偏好（只有 DeepSeek） | M1 不做 judge；M3 需独立 provider 或本地 NLI + 人工审计；否则只报确定性指标 |
| 第二标注者缺失 | 写 single-annotator limitation；不报 κ；不把 Claude 当第二人类 |
| arXiv 429 | bound retry + jitter；冷却；如实报缺天；预览模式可先跑 |
| 163 授权码/风控 | 只让用户本人在 `.env` 填；先 test-email 登录，再 test-email --send --confirm；首封可能进垃圾箱 |
| SMTP accepted ≠ inbox | 只报 accepted/uncertain；不写 delivered |
| 磁盘不足 | 新产物写新目录，raw trace 上限；90 天 retention；先清理重复 PaperLens_* 目录需用户确认 |
| PDF/全文许可 | GitHub 不发 PDF/全文；只发 ID/题目/短 quote/哈希；逐篇 license 另核 |
| 旧文档漂移 | README/简历以 manifest/state 为准，不引用旧 handoff 数字 |

## 11. 已定决策与留给实现者的选择

已定：

- 163 优先，QQ 作为 preset；默认关闭；首次发送人工批准。
- M1 做 40 题 Eval-lite + 确定性指标；不做 claim judge。
- test index 用 40 篇 PDF 重新解析，不用 prepared pages。
- 不把 2,125 篇全量索引；不发布 PDF/全文。
- 旧数据/旧失败/旧索引不动；新目录/新入口。
- 4 条主线共享 M0；推送与评测可并行。

留给实现者：

- frontier.sqlite3 的精确表结构（在本文约束内）。
- 邮件 HTML/plain 模板的具体排版。
- Task Scheduler 的精确 XML/PS1 参数（在本文约束内）。
- test fixture 的具体论文/题目选择（由 select_corpus 产出后人工复核）。
- requirements 的精确 pin 列表（按当前环境导出后人工确认）。
