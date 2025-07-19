import os
import sqlite3
import time
from typing import List, Dict, Optional

import logging
from collections import deque

import pandas as pd
from fastapi import FastAPI, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jobspy import scrape_jobs

DATABASE = os.environ.get("DATABASE", "jobs.db")

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
BUILD_NUMBER = os.environ.get("GITHUB_RUN_NUMBER") or os.environ.get("BUILD_NUMBER", "dev")
templates.env.globals["build_number"] = BUILD_NUMBER

# Store progress messages for the fetch process
logger = logging.getLogger("job_fetch")
logging.basicConfig(level=logging.INFO)
progress_logs = deque(maxlen=100)

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
            elo REAL DEFAULT 1000,
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


def get_random_job() -> Optional[Dict]:
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs ORDER BY rating_count ASC, RANDOM() LIMIT 1")
    row = cur.fetchone()
    columns = [c[0] for c in cur.description]
    conn.close()
    if row:
        return dict(zip(columns, row))
    return None


def list_jobs_by_elo() -> List[Dict]:
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs ORDER BY elo DESC")
    rows = cur.fetchall()
    columns = [c[0] for c in cur.description]
    conn.close()
    return [dict(zip(columns, r)) for r in rows]


def update_elo_single(job_id: int, liked: bool, k: int = 32) -> None:
    """Update ELO score treating the rating as a match against a baseline."""
    baseline = 1000
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT elo, rating_count FROM jobs WHERE id=?", (job_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return
    elo, count = row
    expected = 1 / (1 + 10 ** ((baseline - elo) / 400))
    actual = 1 if liked else 0
    elo = elo + k * (actual - expected)
    count += 1
    cur.execute("UPDATE jobs SET elo=?, rating_count=? WHERE id=?", (elo, count, job_id))
    conn.commit()
    conn.close()


def record_feedback(job_id: int, liked: bool, reason: Optional[str]) -> None:
    """Store feedback and update ELO."""
    update_elo_single(job_id, liked)
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
    log_progress("Done")


@app.on_event("startup")
def on_startup() -> None:
    init_db()


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
def swipe(request: Request):
    job = get_random_job()
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
    jobs = list_jobs_by_elo()
    return templates.TemplateResponse("stats.html", {"request": request, "jobs": jobs})
