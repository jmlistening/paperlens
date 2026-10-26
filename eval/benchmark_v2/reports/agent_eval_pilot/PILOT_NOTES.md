# Agent A/B 5-question pilot (2026-10-25)

This is a **5-question convenience pilot**, not the full 60-question A/B and not
an answer-accuracy evaluation. It was run to validate the harness end to end on
the frozen index with real DeepSeek calls.

## Provenance

- Command: `eval/agent_eval.py --benchmark-dir eval/benchmark_v2/eval120 --retrieval-run eval/benchmark_v2/runs/eval120_rerank --retrieval-system hybrid_rrf_rerank_top6 --systems fixed_rag,agent_single,agent_multi --question-ids q001,q002,q004,q009,q010 --out .../agent_eval_pilot`
- Questions: `q001,q002,q004,q009,q010` (all five have `status=ok` fixed retrieval records; `q005`/`q007` were excluded because their fixed retrieval records failed).
- Answer/decision model: `deepseek-flash` (from `.env`).
- Benchmark fingerprint in the generated summary: `cb60fa504e7fd6cc39314c2b4fc3b0746c78c9686d18c0a149b24a1097aaa497`
  (produced before actual corpus/vector hashes were added to the fingerprint).
- The current harness reports `3e1d11dfe080e7bfbc73065ac887c4a77e09f8bc362efc68d3cb09936d17c890`
  for the same files. Underlying hashes: questions
  `b37ea872ef48cc05ba980f8c6517efd6c285624eb78dd765a1d8976ce5cb1d08`, qrels
  `c78358326cbc91c2e45ac4c99c4665cc725eca7dd6d94a5965536c9341c3131d`, manifest
  `8a4d7cf12fc642faff5d18d933b2fe897d3c38c626f9e720d446f62fd01da48b`, corpus_clean
  `d861fa31cb5457b9191e63cc54cdd88f66dc1aa4e6b1eb5e713db0a72ca9a3d1`, vector_store
  `b73b83aef5537f99dded63d2e126e22f880247f0e6b261980ce1d6d880e161fe`.
- The pilot omitted `--rag-dir`, so fixed and agent both used
  `eval/benchmark_v2/eval120/index` (symlink to `index120`); the agent embedding
  model was `intfloat/multilingual-e5-small`, query prefix `query: `. The
  generated `summary.json` predates the `agent_index` provenance block; the
  current harness records that block and would reject a mismatched `--rag-dir`
  corpus.
- The guardrail changes (actual-file fingerprint, retrieval configuration,
  scoring versions and agent-index identity) changed the run signature.
  Re-running the same command into this old output directory will not reuse
  these checkpoints; use a new empty `--out` if you need to rerun.
- Evidence budget: 6 units for every system; answer prompt: `answer_question.PROMPT`.

## Results

| System | Answer status | Schema / quote | Cited gold parent | Gold-point citation | Tools / rounds (mean) | Tokens | Latency mean / p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|
| `fixed_rag` | 4 answered / 1 invalid | 0.800 (n=5) | 0.800 (n=5) | 0.600 (n=5) | 0.0 / 0.0 | 12,967 | 1,752 / 1,631 / 2,351 |
| `agent_single` | 2 answered / 3 insufficient | 1.000 (n=3) | 0.200 (n=5) | 0.200 (n=5) | 1.0 / 1.0 | 29,297 | 6,235 / 2,739 / 15,934 |
| `agent_multi` | 4 answered / 1 partial | 1.000 (n=5) | 0.600 (n=5) | 0.500 (n=5) | 1.6 / 1.6 | 58,418 | 4,238 / 4,480 / 5,457 |

`schema`/`quote` means exclude harness-generated abstentions; that is why
`agent_single` has n=3 while the other systems have n=5.

## Observations

- **fixed q009 is invalid**: the answer cites `E2` for several quotes, but the
  validator reports `E2 的引文不在原文中`. This is a model citation-ID/quote
  error, counted as schema/quote 0 and excluded from no denominator. The fixed
  control is therefore 4/5 schema-valid on this pilot.
- **agent_single q001/q004 abstained at `round_budget`**: the decision model
  requested a supplemental search even though the payload said
  `supplementary_search_allowed=false`; with `max_rounds=1` the harness stopped
  without selected evidence. These are marked `answer_source=harness_abstention`
  and excluded from schema/quote means, but remain in status/citation counts.
  `q009` was a model-generated `insufficient`; `q002`/`q010` were answered.
- **agent_multi recovered q001/q004/q009 with a second round** (2 rounds on
  those three, 1 round on q002/q010). It beat `agent_single` on cited gold
  (0.6 vs 0.2) but **did not beat `fixed_rag`** on cited gold or gold-point
  citation in this sample (0.6 vs 0.8; 0.5 vs 0.6).
- **Gold-point recall is capped by qrels coverage**: for these five questions
  only 9/12 gold points appear in the frozen corpus (q001 1/2, q002 3/3,
  q004 2/2, q009 2/3, q010 1/2), so the mean cannot reach 1.0.
- **Cost/latency are not end-to-end comparisons**: fixed tokens are answer-only
  and exclude the precomputed rerank cost; agent tokens include decisions and
  answers. `agent_single`'s mean latency includes the 17s first-search embedding
  model load (p50 2.7s); `agent_multi` ran warm after it. The generated report's
  `agent_single` answer-latency mean (916.7 ms) includes two 0 ms abstentions;
  the three measured answer calls average ~1,528 ms. The current harness
  excludes no-call records and prints the measured n.
- The pilot omitted `--rag-dir`, so fixed and agent both used
  `eval/benchmark_v2/eval120/index`; the agent and fixed corpora match.
- **n=5, convenience sample, no CI or significance test.** These numbers are
  descriptive and must not be used as answer accuracy, faithfulness, or a
  general multi-round gain. The full 60-question A/B remains unrun.

## Files

- `summary.json`, `report.md`: combined result and caveats.
- `fixed_rag/`, `agent_single/`, `agent_multi/`: per-system `aggregate.json`,
  `report.md`, and per-question `q*.json` checkpoints.
- Raw answers are in the checkpoints (`raw_response`); the evidence text can be
  reconstructed from the frozen index by the recorded parent/chunk IDs.
