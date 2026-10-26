# PaperLens Claims Ledger

每条准备写进简历/README/面试材料的声明，都必须能追到结果文件、manifest hash 和样本量。未测量的写 `TODO`，不写具体百分比；目标值与实测值分列。

| Claim | Evidence path | Status | n | Measured value | Notes |
|---|---|---|---|---|---|
| 产品索引 48 篇 / 16,648 片段 / 2,997 向量 | `data/rag/library_manifest.json`; `data/rag/corpus_clean.json`; `data/rag/vector_store.npz` | measured | 48 papers | 2,997 vectors, 384d | 8 dev + 40 benchmark evaluated; old 8-paper index backed up |
| 历史开发索引 8 篇 / 453 父块 / 460 向量 | `data/rag_dev8_backup_20261025` | measured | 8 papers | 460 vectors, 384d | historical backup, not the current product index |
| Q01 三模式各 1 次 | `eval/results/20261021_021954_983665/` | measured | 1 question | 25,729 tokens | 流程证据，不是准确率 |
| 引文逐字校验 12/12 | Q01 answer records | measured | 12 citations | 12/12 | 只证明 quote 存在，不证明 claim 被支持 |
| 2,125 篇 prepared 全文 | `data/dataset_v1/state.json`; `data/dataset_history_v1/state.json` | measured | 2,125 prepared | 42,682 pages | 不是 indexed/evaluated |
| 2,119 条模型初筛建议 | screening state/reports | measured | 2,119 valid suggestions | core 796 / adjacent 911 / out_of_scope 411 / uncertain 1 | human_reviewed=0 |
| 769 项离线测试 | explicit unittest commands | measured | 769 tests | 767 passed / 2 skipped | `tests/` 753（2 skipped）+ `test_eval.py` 16；default discover 收集 753 |
| 产品中文 query rewrite | `query_rewrite.py`; `search_evidence.py` | implemented, not measured | — | page-blind translation + cache + fallback | 尚未在产品索引上做正式对照 |
| Push selected impressions | `push_run.py`; `data/push/frontier.sqlite3` impressions | implemented, not measured | — | run_id/position/final_score/channel recorded | 尚无个性化 ranking 指标 |
| 40 篇 held-out test index | `eval/benchmark_v2/index_manifest.json` | measured | 40 papers | 2,537 vectors, 384d | 用同一 prepare/clean 管线重解析 PDF |
| 泄漏审计 | `eval/benchmark_v2/leakage_report.json` | measured | 40 test / 80 reserve | dev-test overlap 0; near-dup 0 | 索引 namespace 干净 |
| 163 推送端到端 | local `data/push` ledger (not published) | measured | one real send | SMTP accepted; inbox delivery not claimed | 单次本地验证，不是批量送达率；个人日志不进入公共仓库 |
| 163 首封 digest | `data/push/runs/7d81b1de6dea4a6083de1c028eb8359d/digest_preview.html` | measured | 2 papers | preview_ready | 预览与发送内容对应 |
| 31 题 held-out benchmark v1（historical, model-reviewed） | `eval/benchmark_v2/benchmark_v1_manifest.json`; `questions.curated.jsonl`; `qrels.curated.jsonl` | measured | 31 questions / 42 gold parents | frozen 2026-10-25 | human_reviewed=false; 68 gold points, 42 matched parents |
| 60 题 held-out benchmark v2（model-reviewed, current） | `eval/benchmark_v2/benchmark_v2_manifest.json`; `questions.v2.jsonl`; `qrels.v2.jsonl` | measured | 60 questions / 83 gold parents | frozen 2026-10-25 | human_reviewed=false; variant+base merged |
| v2 qrels coverage / sensitivity | `eval/benchmark_v2/qrels_coverage_v2.json` | measured | 145 gold points | 132 matched (91.0%); 13 unmatched; 40/60 single parent | unmatched-as-miss: BM25 R@6 0.768, rerank 0.843 |
| v2 query source audit | `eval/benchmark_v2/query_audit_v2.json` | measured | 60 questions | page-blind translation; 40.8% mean token overlap with gold; 5/60 4-gram | question-level leakage risk remains |
| v2.120 distractor benchmark | `eval/benchmark_v2/benchmark_v2_120_manifest.json`; `index120_manifest.json` | measured | 120 papers / 60 questions | 40 evaluated + 80 distractor papers | questions unchanged; qrels rebuilt |
| v2.120 distractor baseline | `eval/benchmark_v2/reports/benchmark_v2_120_retrieval/report.md` | measured | 60 questions | BM25 R@6 0.822; dense_en 0.678; hybrid 0.808; dense_zh 0.142 | paper hit@6: 0.983/0.917/0.983; still within-paper |
| v2.120 distractor + rerank | `eval/benchmark_v2/reports/benchmark_v2_120_retrieval/rerank_report.md` | measured | 60 questions | R@6 0.872; nDCG 0.830; MRR 0.838; 2 failed; 399,725 tokens | Recall gain shrinks and CI crosses 0 |
| v2.120 rerank vs BM25（paired） | `.../benchmark_v2_120_retrieval/paired_rerank_vs_bm25.md` | measured | 60 pairs | R@6 +0.050 [-0.044,0.139]; nDCG +0.170 [0.073,0.263]; MRR +0.193 [0.080,0.304] | Recall not conclusive; nDCG/MRR robust |
| v2.120 rerank vs RRF（paired） | `.../benchmark_v2_120_retrieval/paired_rerank_vs_hybrid.md` | measured | 60 pairs | R@6 +0.064 [-0.033,0.164]; nDCG +0.174 [0.075,0.273]; MRR +0.200 [0.083,0.314] | Recall not conclusive |
| v2 BM25/dense/RRF baseline | `eval/benchmark_v2/reports/benchmark_v2_retrieval/report.md` | measured | 60 questions | BM25 R@6 0.831 [0.747,0.906]; dense_en 0.719; hybrid 0.808; dense_zh 0.175 | descriptive only; no significance test |
| v2 Hybrid RRF + DeepSeek rerank | `eval/benchmark_v2/reports/benchmark_v2_retrieval/rerank_report.md` | measured | 60 questions | R@6 0.908 (conservative 0.843); nDCG 0.847; MRR 0.839; 1 failed; 408,643 tokens | descriptive; 1 failure counted 0 |
| v2 Rerank vs BM25（paired） | `eval/benchmark_v2/reports/benchmark_v2_retrieval/paired_rerank_vs_bm25.md` | measured | 60 pairs | R@6 +0.078 [0.011,0.147]; nDCG +0.173 [0.091,0.253]; MRR +0.174 [0.069,0.275] | no pre-registration / multiple-comparison correction |
| v2 Rerank vs plain RRF（paired） | `eval/benchmark_v2/reports/benchmark_v2_retrieval/paired_rerank_vs_hybrid.md` | measured | 60 pairs | R@6 +0.100 [0.017,0.183]; nDCG +0.186 [0.097,0.275]; MRR +0.199 [0.094,0.303] | descriptive |
| v1 BM25/dense/RRF Recall@k | `eval/benchmark_v2/reports/benchmark_v1_retrieval/report.md` | measured | 31 questions | BM25 R@6 0.839 [0.710,0.935]; dense_en 0.726; hybrid 0.758; dense_zh 0.177 | historical; descriptive |
| v1 plain RRF hybrid 超过 BM25 | `eval/benchmark_v2/reports/benchmark_v1_retrieval/report.md` | not supported | 31 questions | hybrid R@6 0.758 < BM25 0.839 | plain RRF 为负；v2 同样为负 |
| v1 Hybrid RRF + DeepSeek rerank | `eval/benchmark_v2/reports/benchmark_v1_retrieval/rerank_report.md` | measured | 31 questions | R@6 0.903; nDCG 0.847; MRR 0.860; 0 failed; 215,142 tokens | historical; 描述性；无预注册 |
| v1 Rerank vs BM25（paired） | `eval/benchmark_v2/reports/benchmark_v1_retrieval/paired_rerank_vs_bm25.md` | measured | 31 pairs | nDCG +0.190 [0.078,0.306]; MRR +0.241 [0.100,0.383]; Recall +0.065 [-0.016,0.161] | historical; Recall CI 跨 0 |
| v1 Rerank vs plain RRF（paired） | `eval/benchmark_v2/reports/benchmark_v1_retrieval/paired_rerank_vs_hybrid.md` | measured | 31 pairs | Recall +0.145 [0.032,0.274]; nDCG +0.242 [0.122,0.372]; MRR +0.293 [0.151,0.436] | historical; 描述性 |
| Answer-level citation eval（not accuracy, post-repair） | `eval/benchmark_v2/reports/answer_eval_v2_120/report.md`; `aggregate.json` | measured | 60 questions | 47 answered/9 partial/4 invalid; schema+quote 0.933; cited gold parent 0.822 [0.722,0.917]; gold-point citation 0.774 [0.676,0.864]; 169,306 tokens; p95 5.38s | citation-level only; no accuracy/faithfulness/judge/human review |
| Answer repair pass | `eval/benchmark_v2/reports/answer_eval_v2_120/repair_report.md` | measured | 8 selected | 4 repaired; 2 skipped (no retrieval evidence); 2 still invalid | pre-repair: 44/8/8, schema 0.867, cited 0.764, gold-point 0.725 |
| Gold-citation precision（not general citation correctness） | `eval/benchmark_v2/reports/answer_eval_v2_120/citation_precision.json` | measured | 56 answered/partial | cited gold parent precision 0.629; citation occurrence gold-quote precision 0.681 | deterministic; cited chunk may be relevant but outside the narrow gold set |
| Out-of-scope refusal check | `eval/benchmark_v2/reports/answer_eval_refusal/refusal_summary.json`; `report.md` | measured | 12 questions | 12/12 insufficient; 0 answered/partial/invalid; schema+quote 1.0; 22,267 tokens; p95 1.28s | focused template set, not full absent-info/false-premise benchmark |
| Fixed-RAG vs bounded-Agent A/B (5-question pilot) | `eval/benchmark_v2/reports/agent_eval_pilot/summary.json`; `PILOT_NOTES.md`; `eval/agent_eval.py`; `tests/test_agent_eval.py` | measured (pilot) | 5 questions | fixed schema/quote 0.800, cited gold 0.800, gold-point 0.600, 12,967 tokens; single schema/quote 1.000 (n=3), cited gold/gold-point 0.200, 29,297; multi schema/quote 1.000, cited gold 0.600, gold-point 0.500, 58,418 | n=5 convenience sample; no CI/significance; citation-level only; fixed q009 invalid; single 2 harness abstentions; full 60 not run; pilot fingerprint `cb60fa50…`, current actual-file scheme `3e1d11df…` |
| claim-level citation precision | `eval/` judge results | TODO | 0 | TODO | 不在 Phase 1；需独立 judge |
| 个性化排序 NDCG/precision@k | reading/impressions | TODO | 0 | TODO | 反馈数据不足，只能 pilot |

## 写作规则

- `measured` 必须同时有结果文件 + manifest hash + n；缺一项就是 `TODO`。
- `target` 写在设计文档，不写进本表当作结果。
- 负结果/不显著照实写；不得只报有利子组。
- 推送的“SMTP accepted”不得写成“delivered/inbox”。
