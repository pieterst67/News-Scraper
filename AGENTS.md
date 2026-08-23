# AGENTS.md

Daily Dutch news digest. Single entrypoint: `news_scraper.py` (RSS → embeddings → clustering → gpt-5.2 Dutch briefings → HTML email).

## Running

- Run `venv/bin/python news_scraper.py` with the existing `venv/`. `requirements.txt` is missing trafilatura, tiktoken, requests, lxml, numpy — never rebuild the venv from it.
- Import-time crash if the `nl_NL.UTF-8` locale is missing (news_scraper.py:52).
- A full run spends OpenAI + Cloudflare API quota and sends a real email. For experiments, copy the DB and set `DB_PATH` to the copy (helper: `copy_database_for_experiment()`).

## Environment

- Config lives in untracked `.env`. README's var list is stale: code also requires `CF_ACCOUNT_ID`, `CF_API_TOKEN`, `EMAIL_CC`, `EMAIL_BCC` and reads `DB_PATH`, `BROWSER_DOMAINS`. `TOPICS`/`PUZZLE_KEYWORDS` are unused legacy.
- `.env.lc`/`.env.nyt` hold site credentials (untracked) — never commit or print their values.

## Behavior edits must not break

- Collect: skip URLs already in DB, articles > 48 h old or < 500 chars; throttle on robots.txt `crawl-delay` (baseline 2.5 s, ±40% jitter); backoff retries with 30–60 s cooloff after 3 consecutive failures.
- Fetch order: GET first, Cloudflare Browser Rendering as fallback; hosts in `BROWSER_DOMAINS` go straight to Cloudflare.
- Clustering: needs ≥ 20 unprocessed articles; drops clusters with < 3; a briefing from the last 3 days at cosine similarity ≥ 0.80 makes a cluster a "continuing story".
- Output: Dutch only (prompt contract), `gpt-5.2` with strict JSON schema, digest capped by hardcoded `READ_LIMIT_WORDS = 2000`.

## Dead / special files

- `paywall_lc_scraper.py` — standalone Playwright lc.nl scraper (creds in `.env.lc`); not imported by the pipeline.
- `news_scraper_topics.py_` — disabled legacy pipeline (trailing underscore); do not edit.
- No tests, lint, or CI: verify changes by running the pipeline against a DB copy.
