# Privacy

PaperLens is **local-first**. It has no telemetry, no analytics, and no hosted backend. Your research interests, questions, answers, push history, and recipient address stay on the machine that runs the app.

## What stays local

| Data | Default location | In Git? |
|---|---|---|
| Active research profile | `data/push/profile.json` | No (gitignored) |
| Recipient + verification state | `data/push/ui_state.json` | No (gitignored) |
| Delivery frontier / watermark | `data/push/frontier.sqlite3` | No (gitignored) |
| Sent digest history | `data/push/runs/` | No (gitignored) |
| Q&A records and evidence | `data/rag/answers/`, `data/rag/qa_runs/` | No (gitignored) |
| SMTP / model credentials | `.env` | No (gitignored) |
| Public profile example | `interests.example.json` | Yes — generic only |

The UI, status JSON, and logs show the recipient as a mask (for example `d***@example.com`). The raw address is only in `data/push/ui_state.json`.

## If you expose the app

The personal instance is **single-user and has no authentication**. The advanced research pages read the local `data/` directory: they can show your research profile, previous Q&A, sent push history, and recently delivered papers. Anyone who can open the URL can also trigger previews or manual sends.

- Keep the personal instance on `127.0.0.1`.
- For remote access, use an SSH tunnel or Tailscale; do not open port 8501 to the public internet.
- For a public demo, use synthetic data, disable real SMTP/model calls, and hide or disable the personal-data pages. Never mount your real `data/` or `.env` in a demo.

## Before publishing a fork

Run the repository scanner:

```powershell
python scripts/check_public_repo.py
python scripts/check_public_repo.py --history
```

The default scan checks tracked files for real-looking email addresses, `sk-` API keys, and non-placeholder `SMTP_PASSWORD=` values. `--history` performs a read-only audit and reports paths/commits; it never rewrites history.

If you want to publish a clean repository, prefer a **fresh public repository** from a clean tree over publishing a history that once contained a personal profile or handoff notes. `gitignore` and deleting a file do not remove it from old commits.

## Public demo checklist

- [ ] Synthetic profile, synthetic Q&A, synthetic push history.
- [ ] No real `.env`, SMTP credentials, or API keys.
- [ ] Real sends disabled; preview uses fake fixtures if needed.
- [ ] Personal-data pages hidden or clearly marked as demo data.
- [ ] Banner: “Demo — no real emails.”
- [ ] Optional auth in front of the demo to prevent API-cost abuse.
