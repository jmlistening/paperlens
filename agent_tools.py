"""Read-only local retrieval tools. No model API, downloads or shell execution."""
import json
import re
from pathlib import Path

import numpy as np

from library_transaction import check_available, digest
from search_bm25 import tokenize
from search_evidence import exclusion_reason, section_events, select_hits
from search_vector import load_store


def _load_eval_store(folder):
    """Load a frozen benchmark index directory without the library journal.

    ``retrieval_core.load_index`` performs the same structural checks as
    ``search_vector.load_store`` but accepts an arbitrary directory. It is
    imported lazily so production imports stay unchanged.
    """
    from retrieval_core import load_index

    index, metadata, _parents = load_index(folder)
    return index, metadata


class LocalTools:
    def __init__(self, root, paper=None, rag_dir=None, verify_sources=True):
        """Build read-only retrieval tools.

        Production defaults (``rag_dir=None``, ``verify_sources=True``) keep the
        original behaviour exactly: load ``<root>/data/rag`` through
        ``load_store``, enforce the library journal, and verify every source PDF
        under ``<root>/data/pdfs``.

        Evaluation mode passes ``rag_dir`` (for a frozen benchmark index) and
        ``verify_sources=False``. It then loads the index from ``rag_dir``,
        skips the pipeline journal check and skips PDF source checks. Corpus and
        index files remain hash-checked so a mid-run change is still detected.

        A ``rag_dir`` that resolves to ``<root>/data/rag`` is the live library,
        not a frozen index: it always uses ``load_store`` and the journal.
        Frozen-index mode requires ``verify_sources=False`` and rejects the
        unsafe combination explicitly.
        """
        self.root, self.paper = Path(root).resolve(), paper
        self.verify_sources = bool(verify_sources)
        live_rag_dir = (self.root / "data" / "rag").resolve()
        self.rag_dir = (
            Path(rag_dir).resolve() if rag_dir is not None else live_rag_dir
        )
        self.production_layout = self.rag_dir == live_rag_dir
        if self.production_layout:
            self.index, self.metadata = load_store(self.root)
        else:
            if self.verify_sources:
                raise ValueError(
                    "冻结索引模式需要 verify_sources=False；"
                    "verify_sources=True 只适用于 data/rag 生产库。"
                )
            self.index, self.metadata = _load_eval_store(self.rag_dir)
        folder = self.rag_dir
        self.files = {folder / n: digest(folder / n) for n in
                      ("corpus.json", "corpus_clean.json", "vector_store.npz")}
        chunks = json.loads((folder / "corpus_clean.json").read_text(encoding="utf-8"))["chunks"]
        self.events = section_events(chunks)
        self.canonical = {u["chunk_id"]: u for u in self.metadata["units"]}
        scoped = [u for u in self.canonical.values() if not paper or u["arxiv_id"] == paper]
        if not scoped:
            raise ValueError("索引中没有这个完整论文编号，请先累计入库。")
        if self.verify_sources:
            originals = {u["pdf_path"]: u["pdf_sha256"] for u in scoped}
            for name, expected in originals.items():
                path = (self.root / name).resolve()
                if not path.is_relative_to(self.root / "data" / "pdfs") or digest(path) != expected:
                    raise ValueError("原文缺失或已变化，请先检查论文库并重新入库。")
                self.files[path] = expected
        self.eligible = [u for u in scoped if not exclusion_reason(u, self.events)]
        self.model = None
        self.check_current()

    def check_current(self):
        # The journal guards the mutable production library regardless of the
        # PDF-verification flag. A frozen benchmark index is read-only, so
        # evaluation mode skips it.
        if self.production_layout:
            check_available(self.root)
        if any(digest(path) != expected for path, expected in self.files.items()):
            raise ValueError("运行期间语料、索引或原文发生变化，本次停止。")

    def search(self, tool, query, top_k=12):
        if tool not in ("search_dense", "search_bm25"):
            raise ValueError("工具不在允许列表中。")
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 2000:
            raise ValueError("检索词须为 1～2000 字符。")
        self.check_current()
        if tool == "search_dense":
            hits = self._dense(query.strip(), top_k)
        else:
            hits = self._bm25(query.strip(), top_k)
        self.check_current()
        return hits

    def _dense(self, query, top_k):
        if self.model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"阶段：加载本地检索模型（{device}）", flush=True)
            self.model = SentenceTransformer(
                self.metadata["model_name"], device=device,
                revision=self.metadata.get("model_revision"),
                cache_folder=str(self.root / "data" / "models"), local_files_only=True)
        text = self.metadata["query_prefix"] + query
        limit = min(self.metadata["max_tokens"], int(self.model.max_seq_length))
        if len(self.model.tokenizer(text, truncation=False)["input_ids"]) > limit:
            raise ValueError("检索词超过本地模型长度上限，请缩短问题。")
        vector = self.model.encode([text], prompt="", normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
        vector = np.ascontiguousarray(vector, dtype="float32")
        if (vector.shape != (1, self.index.d) or not np.isfinite(vector).all()
                or not np.allclose(np.linalg.norm(vector), 1.0, atol=1e-4)):
            raise ValueError("查询向量无效。")
        hits, _ = select_hits(self.index, self.metadata["units"], vector,
                              self.events, top_k, self.paper)
        return hits

    def _bm25(self, query, top_k):
        from rank_bm25 import BM25Okapi
        if re.search(r"[\u3400-\u9fff]", query):
            raise ValueError("关键词检索工具要求英文关键词，中文问题可使用语义检索。")
        words = list(dict.fromkeys(tokenize(query)))
        if not words:
            raise ValueError("关键词检索没有有效英文词或数字。")
        pairs = [(u, tokenize(u["text"])) for u in self.eligible]
        pairs = [(u, terms) for u, terms in pairs if terms]
        if not pairs:
            return []
        scores = BM25Okapi([terms for _, terms in pairs], k1=1.5, b=0.75).get_scores(words)
        matched = [i for i, (_, terms) in enumerate(pairs) if set(words).intersection(terms)]
        matched.sort(key=lambda i: (-float(scores[i]), pairs[i][0]["chunk_id"]))
        hits, seen = [], set()
        for i in matched:
            unit = pairs[i][0]
            if unit["parent_chunk_id"] in seen:
                continue
            seen.add(unit["parent_chunk_id"])
            hits.append({**unit, "rank": len(hits) + 1, "score": float(scores[i])})
            if len(hits) == top_k:
                break
        return hits
