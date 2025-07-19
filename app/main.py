import os
import sqlite3
import time
import json
from typing import List, Dict, Optional

import logging
from collections import deque
from threading import Thread

import pandas as pd
import requests
from fastapi import FastAPI, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jobspy import scrape_jobs

DATABASE = os.environ.get("DATABASE", "jobs.db")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")

# Allow separate models for embeddings and rephrasing while keeping backwards
# compatibility with the original `OLLAMA_MODEL` variable.
_default_model = os.environ.get("OLLAMA_MODEL", "llama3")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", _default_model)
OLLAMA_REPHRASE_MODEL = os.environ.get("OLLAMA_REPHRASE_MODEL", _default_model)
OLLAMA_ENABLED = bool(OLLAMA_BASE_URL)

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
BUILD_NUMBER = os.environ.get("GITHUB_RUN_NUMBER") or os.environ.get("BUILD_NUMBER", "dev")
templates.env.globals["build_number"] = BUILD_NUMBER

# Store progress messages for the fetch process
logger = logging.getLogger("job_fetch")
logging.basicConfig(level=logging.INFO)
progress_logs = deque(maxlen=100)

REWRITE_PROMPT_TEMPLATE = '''
Rewrite the following job description into concise Markdown without preamble.
Omit any sections that cannot be filled from the text. Use this structure:

---
**Company:** short overview

**Technical Requirements**
- item one
- item two

**Soft Skills**
- item one
- item two

**Description**
A short paragraph summarizing the role.
---

Job posting:
"""{description}"""
'''


def ensure_model_downloaded() -> None:
    if not OLLAMA_ENABLED:
        return
    for model in {OLLAMA_EMBED_MODEL, OLLAMA_REPHRASE_MODEL}:
        try:
            requests.post(f"{OLLAMA_BASE_URL}/api/pull", json={"name": model}, timeout=120)
        except Exception as exc:
            logger.info(f"Failed to pull model {model}: {exc}")


def embed_text(text: str) -> List[float]:
    if not OLLAMA_ENABLED:
        return []
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={
                "model": OLLAMA_EMBED_MODEL,
                "prompt": text,
                "options": {"num_ctx": 4096},
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("embedding", [])
    except Exception as exc:
        logger.info(f"Embedding failed: {exc}")
        return []


def generate_summary(text: str) -> str:
    if not OLLAMA_ENABLED:
        return ""
    prompt = REWRITE_PROMPT_TEMPLATE.format(description=text)
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_REPHRASE_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_ctx": 4096},
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as exc:
        logger.info(f"Summary generation failed: {exc}")
        return ""


def process_all_jobs() -> None:
    if not OLLAMA_ENABLED:
        return
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT id, description FROM jobs")
    rows = cur.fetchall()
    for job_id, desc in rows:
        if not desc:
            continue
        cur.execute("SELECT 1 FROM summaries WHERE job_id=?", (job_id,))
        have_sum = cur.fetchone()
        cur.execute("SELECT 1 FROM embeddings WHERE job_id=?", (job_id,))
        have_emb = cur.fetchone()
        if have_sum and have_emb:
            continue
        summary = generate_summary(desc) if not have_sum else None
        embedding = embed_text(desc) if not have_emb else None
        if summary is not None:
            cur.execute(
                "INSERT OR IGNORE INTO summaries(job_id, summary) VALUES(?, ?)",
                (job_id, summary),
            )
        if embedding is not None:
            cur.execute(
                "INSERT OR IGNORE INTO embeddings(job_id, embedding) VALUES(?, ?)",
                (job_id, json.dumps(embedding)),
            )
        conn.commit()
    conn.close()


def format_salary(min_amount: float, max_amount: float, currency: str) -> str:
    """Return a salary range rounded to the nearest thousand with 'k' suffix."""
    def to_k(val: float) -> str:
        return f"{round(val / 1000):.0f}k"

    return f"{to_k(min_amount)} - {to_k(max_amount)} {currency}"

templates.env.globals["format_salary"] = format_salary


def time_since_posted(date_str: str) -> str:
    """Return relative age like '2d' or '1w 3d' for the given date string."""
    if not date_str:
        return ""

    date_str = str(date_str).lower().strip()
    now = pd.Timestamp.utcnow()

    # Handle common relative phrases
    import re

    m = re.search(r"(\d+)\s*day", date_str)
    if m:
        days = int(m.group(1))
    else:
        m = re.search(r"(\d+)\s*week", date_str)
        if m:
            days = int(m.group(1)) * 7
        else:
            m = re.search(r"(\d+)\s*month", date_str)
            if m:
                days = int(m.group(1)) * 30
            else:
                if "today" in date_str or "just" in date_str:
                    days = 0
                elif "yesterday" in date_str:
                    days = 1
                else:
                    try:
                        dt = pd.to_datetime(date_str, utc=True, errors="coerce")
                        if pd.isna(dt):
                            return ""
                        days = (now - dt).days
                    except Exception:
                        return ""

    if days < 7:
        return f"{days}d"
    weeks = days // 7
    days = days % 7
    if weeks < 4:
        return f"{weeks}w {days}d" if days else f"{weeks}w"
    months = weeks // 4
    weeks = weeks % 4
    return f"{months}m {weeks}w" if weeks else f"{months}m"


templates.env.globals["time_since_posted"] = time_since_posted

def log_progress(message: str) -> None:
    """Log a progress message to the logger and internal buffer."""
    logger.info(message)
    progress_logs.append(message)



def init_db() -> None:
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site TEXT,
            title TEXT,
            company TEXT,
            location TEXT,
            date_posted TEXT,
            description TEXT,
            interval TEXT,
            min_amount REAL,
            max_amount REAL,
            currency TEXT,
            job_url TEXT UNIQUE,
            rating_count INTEGER DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            liked INTEGER,
            reason TEXT,
            rated_at INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings(
            job_id INTEGER PRIMARY KEY,
            embedding TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS summaries(
            job_id INTEGER PRIMARY KEY,
            summary TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_jobs(df: pd.DataFrame) -> None:
    if df.empty:
        return
    cols = {
        "site": None,
        "title": None,
        "company": None,
        "location": None,
        "date_posted": None,
        "description": None,
        "interval": None,
        "min_amount": None,
        "max_amount": None,
        "currency": None,
        "job_url": None,
    }
    # Keep only the columns we know about
    df = df.loc[:, df.columns.intersection(cols.keys())]
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    for _, row in df.iterrows():
        values = tuple(row.get(c) for c in cols)
        cur.execute(
            """
            INSERT OR IGNORE INTO jobs(site,title,company,location,date_posted,description,interval,min_amount,max_amount,currency,job_url)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            values,
        )
    conn.commit()
    conn.close()
    if OLLAMA_ENABLED:
        process_all_jobs()


def get_random_job() -> Optional[Dict]:
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT j.*, s.summary FROM jobs j
        LEFT JOIN summaries s ON j.id = s.job_id
        ORDER BY rating_count ASC, RANDOM() LIMIT 1
        """
    )
    row = cur.fetchone()
    columns = [c[0] for c in cur.description]
    conn.close()
    if row:
        return dict(zip(columns, row))
    return None


def get_job(job_id: int) -> Optional[Dict]:
    """Return a job by id including any summary."""
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT j.*, s.summary FROM jobs j
        LEFT JOIN summaries s ON j.id = s.job_id
        WHERE j.id=?
        """,
        (job_id,),
    )
    row = cur.fetchone()
    columns = [c[0] for c in cur.description]
    conn.close()
    if row:
        return dict(zip(columns, row))
    return None


def list_jobs_by_feedback() -> List[Dict]:
    """Return jobs ordered by number of positive ratings."""
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT j.*,
               COALESCE(SUM(CASE WHEN f.liked=1 THEN 1 ELSE 0 END), 0) AS likes,
               COALESCE(SUM(CASE WHEN f.liked=0 THEN 1 ELSE 0 END), 0) AS dislikes,
               CASE WHEN s.job_id IS NOT NULL THEN 1 ELSE 0 END AS has_summary,
               CASE WHEN e.job_id IS NOT NULL THEN 1 ELSE 0 END AS has_embedding
        FROM jobs j
        LEFT JOIN feedback f ON j.id = f.job_id
        LEFT JOIN summaries s ON j.id = s.job_id
        LEFT JOIN embeddings e ON j.id = e.job_id
        GROUP BY j.id
        ORDER BY likes DESC
        """
    )
    rows = cur.fetchall()
    columns = [c[0] for c in cur.description]
    conn.close()
    return [dict(zip(columns, r)) for r in rows]


def aggregate_job_stats() -> Dict:
    """Return aggregate statistics about the stored jobs."""
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM jobs")
    total_jobs = cur.fetchone()[0]

    cur.execute("SELECT site, COUNT(*) FROM jobs GROUP BY site")
    by_site = {site: count for site, count in cur.fetchall()}

    cur.execute("SELECT date_posted, COUNT(*) FROM jobs GROUP BY date_posted")
    by_date = {date: count for date, count in cur.fetchall()}

    cur.execute(
        "SELECT AVG(min_amount), AVG(max_amount) FROM jobs WHERE min_amount IS NOT NULL AND max_amount IS NOT NULL"
    )
    avg_min, avg_max = cur.fetchone()

    cur.execute("SELECT COUNT(*) FROM summaries")
    summaries_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM embeddings")
    embeddings_count = cur.fetchone()[0]

    conn.close()
    return {
        "total_jobs": total_jobs,
        "by_site": by_site,
        "by_date": by_date,
        "avg_min_pay": avg_min or 0,
        "avg_max_pay": avg_max or 0,
        "jobs_with_summaries": summaries_count,
        "jobs_with_embeddings": embeddings_count,
    }


def increment_rating_count(job_id: int) -> None:
    """Increment the rating count for a job."""
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT rating_count FROM jobs WHERE id=?", (job_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return
    count = row[0] + 1
    cur.execute("UPDATE jobs SET rating_count=? WHERE id=?", (count, job_id))
    conn.commit()
    conn.close()


def record_feedback(job_id: int, liked: bool, reason: Optional[str]) -> None:
    """Store feedback and increment rating count."""
    increment_rating_count(job_id)
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO feedback(job_id, liked, reason, rated_at) VALUES(?,?,?,?)",
        (job_id, int(liked), reason, int(time.time())),
    )
    conn.commit()
    conn.close()


def fetch_jobs_task(search_term: str, location: str, sites: List[str]) -> None:
    """Background task to fetch job postings and store them."""
    google_term = f"{search_term} jobs in {location}"
    jobs: List[pd.DataFrame] = []
    for site in sites:
        log_progress(f"Fetching from {site}")
        try:
            df = scrape_jobs(
                site_name=site,
                search_term=search_term,
                google_search_term=google_term,
                location=location,
                country_indeed="USA",
                results_wanted=50,
                hours_old=168,
                description_format="html",
                linkedin_fetch_description=site == "linkedin",
                verbose=1,
            )
            jobs.append(df)
            log_progress(f"{site}: {len(df)} jobs")
        except Exception as exc:
            log_progress(f"Skipping {site}: {exc}")
    if jobs:
        combined = pd.concat(jobs, ignore_index=True)
        save_jobs(combined)
        log_progress(f"Saved {len(combined)} jobs")
        if OLLAMA_ENABLED:
            process_all_jobs()
    log_progress("Done")


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    if OLLAMA_ENABLED:
        def worker():
            ensure_model_downloaded()
            process_all_jobs()

        Thread(target=worker, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def search_form(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})


@app.post("/fetch")
def fetch_jobs(
    request: Request,
    background_tasks: BackgroundTasks,
    search_term: str = Form(...),
    location: str = Form(...),
    sites: List[str] = Form(...),
):
    """Start job fetching in the background and show progress."""
    progress_logs.clear()
    background_tasks.add_task(fetch_jobs_task, search_term, location, sites)
    return templates.TemplateResponse("progress.html", {"request": request})


@app.get("/progress")
def progress() -> JSONResponse:
    """Return current fetch progress logs."""
    done = len(progress_logs) > 0 and progress_logs[-1] == "Done"
    return JSONResponse({"logs": list(progress_logs), "done": done})


@app.get("/swipe", response_class=HTMLResponse)
def swipe(request: Request, job_id: Optional[int] = None):
    job = get_job(job_id) if job_id else get_random_job()
    if not job:
        return templates.TemplateResponse("no_jobs.html", {"request": request})
    return templates.TemplateResponse("swipe.html", {"request": request, "job": job})


@app.get("/reject/{job_id}", response_class=HTMLResponse)
def reject_form(request: Request, job_id: int):
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs WHERE id=?", (job_id,))
    row = cur.fetchone()
    columns = [c[0] for c in cur.description]
    conn.close()
    if not row:
        return RedirectResponse("/swipe", status_code=303)
    job = dict(zip(columns, row))
    return templates.TemplateResponse("reject.html", {"request": request, "job": job})


@app.post("/feedback")
def feedback(job_id: int = Form(...), liked: int = Form(...), reason: str = Form("", max_length=200)):
    record_feedback(job_id, bool(liked), reason or None)
    return RedirectResponse("/swipe", status_code=303)


@app.get("/stats", response_class=HTMLResponse)
def stats(request: Request):
    jobs = list_jobs_by_feedback()
    agg = aggregate_job_stats()
    return templates.TemplateResponse(
        "stats.html", {"request": request, "jobs": jobs, "stats": agg}
    )


def cleanup_jobs() -> int:
    """Delete jobs without a description and remove duplicate title/company entries."""
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()

    cur.execute("DELETE FROM jobs WHERE description IS NULL OR description = ''")
    deleted_missing = cur.rowcount

    cur.execute(
        """
        DELETE FROM jobs
        WHERE id NOT IN (
            SELECT MIN(id) FROM jobs GROUP BY title, company
        )
        """
    )
    deleted_dupes = cur.rowcount
    conn.commit()
    conn.close()
    return deleted_missing + deleted_dupes


@app.get("/manage", response_class=HTMLResponse)
def manage(request: Request):
    return templates.TemplateResponse("manage.html", {"request": request, "deleted": None})


@app.post("/cleanup", response_class=HTMLResponse)
def cleanup(request: Request):
    deleted = cleanup_jobs()
    return templates.TemplateResponse(
        "manage.html", {"request": request, "deleted": deleted}
    )


def reprocess_jobs_task() -> None:
    """Delete existing summaries/embeddings and regenerate them."""
    if not OLLAMA_ENABLED:
        return
    log_progress("Clearing summaries and embeddings")
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute("DELETE FROM summaries")
    cur.execute("DELETE FROM embeddings")
    conn.commit()
    conn.close()
    log_progress("Regenerating data")
    process_all_jobs()
    log_progress("Done")


@app.post("/reprocess", response_class=HTMLResponse)
def reprocess(request: Request, background_tasks: BackgroundTasks):
    progress_logs.clear()
    background_tasks.add_task(reprocess_jobs_task)
    return templates.TemplateResponse("progress.html", {"request": request})
