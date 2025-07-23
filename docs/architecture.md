# Architecture Overview

This project is a small job search and ranking tool built with **FastAPI**. Jobs are scraped from multiple boards via [JobSpy](https://pypi.org/project/python-jobspy/) and stored in a local SQLite database. Users can swipe through jobs in a simple web UI and record feedback that is later used for training a matching model. Optionally, job descriptions are processed with an Ollama language model to generate summaries and text embeddings.

## Components

- **Web API / UI** – `app/main.py`
  - Defines FastAPI routes and Jinja2 templates under `app/templates`.
  - Fetch jobs in a background task and stream progress messages.
  - Serve a swipe interface where each job can be liked or rejected.
  - Provide statistics and basic management endpoints.
- **Job Scraping** – `jobspy.scrape_jobs`
  - Called from the background task to collect jobs from sites like LinkedIn, Indeed, ZipRecruiter, Glassdoor and Google.
- **Database** – SQLite stored at the path in the `DATABASE` env var.
  - Tables: `jobs`, `feedback`, `embeddings`, `summaries` and `job_tags`.
  - `jobs` holds raw postings. `feedback` records user ratings and reasons.
  - `job_tags` stores skills extracted from each description.
  - `embeddings` and `summaries` are optionally populated using Ollama.
- **Ollama Integration**
  - When `OLLAMA_BASE_URL` is set, descriptions are summarized and embedded via Ollama’s API. The model name defaults to `llama3` and can be changed with `OLLAMA_MODEL`.
  - Summaries and embeddings are generated in `process_all_jobs`.
  - The embedding text now includes the job title and company so the model
    captures more context when ranking roles.
- **Client UI**
  - HTML templates render pages for searching, swiping, viewing stats and managing stored jobs. Styling lives in `app/static/style.css`.
  - The stats page shows performance for several algorithms (logistic regression, random forest, SVM and a clustering approach). The app trains all of them and keeps the one with the highest recall. Metrics for each candidate are displayed in a table so you can compare their accuracy, precision, recall and F1 score. Results are calculated on a 10% holdout set when possible.
- **Tests** – under `tests/` using `pytest` to cover database operations and job fetching logic.
- **Docker** – `Dockerfile` and `build.sh` build a container image. GitHub Actions automatically build and push on merges to `main`.

## Development Quick Start

1. **Install dependencies** (Python 3.12):
   ```bash
   pip install -r requirements.txt
   ```
2. **Run tests** to ensure the environment is working:
   ```bash
   pytest -q
   ```
3. **Start the app** locally:
   ```bash
   uvicorn app.main:app --reload
   ```
   Visit `http://localhost:8000` to fetch jobs and swipe.

Alternatively, build the Docker image with `./build.sh` and run it via `docker run -p 8000:8000 job-ranker`.

## Codex Environment Setup

The Codex container already includes Python 3.12. To reproduce the CI checks you only need to install dependencies and run the test suite:

```bash
pip install -r requirements.txt
pytest -q
```

No additional services are required. Network restrictions may prevent some job boards from being scraped, but the tests use mocked data so they should pass offline.
