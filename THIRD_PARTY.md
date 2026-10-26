# Third-party components

The MIT license applies to this repository's source code. Dependencies keep
their own licenses. `requirements.txt` is the authoritative list; verify the
exact version and license before redistribution.

| Component | License | Note |
|---|---|---|
| PyMuPDF (`fitz`) | **AGPL-3.0 or Artifex commercial** | Important: network/closed-source distribution may trigger AGPL obligations. Replace it or obtain a commercial license if needed. |
| Streamlit | Apache-2.0 | |
| FAISS (`faiss-cpu`) | MIT | |
| sentence-transformers | Apache-2.0 | |
| PyTorch | BSD-3-Clause | |
| Transformers | Apache-2.0 | |
| OpenAI Python client | Apache-2.0 | Used with a DeepSeek-compatible endpoint. |
| `arxiv` | MIT | |
| `rank-bm25` | Apache-2.0 | |
| `python-dotenv` | BSD-3-Clause | |
| Jinja2 | BSD-3-Clause | |
| NumPy | BSD-3-Clause | |
| pandas | BSD-3-Clause | |
| scikit-learn | BSD-3-Clause | |
| `requests` | Apache-2.0 | |
| `httpx` | BSD-3-Clause | |
| pydantic | MIT | |
| PyYAML | MIT | |
| tqdm | MPL-2.0 AND MIT | |

Also note:

- arXiv API terms and rate limits apply when fetching metadata or PDFs.
- 163/QQ SMTP provider terms apply. Do not use this project for spam or
  unsolicited bulk email.
- Model provider terms apply to API use and generated outputs.
- Paper content remains with its authors/publishers. This repository does not
  ship PDFs or full texts.
- The 2,125-paper prepared corpus is not included; paper licenses have not been
  verified individually.

This file is informational, not legal advice. If you plan commercial or
closed-source distribution, review the dependencies (especially PyMuPDF) with
a qualified lawyer.
