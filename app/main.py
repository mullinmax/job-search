import os
import sqlite3
import time
from typing import List, Dict, Optional

import logging
from collections import deque
from threading import Thread

import io
import pandas as pd
from fastapi import FastAPI, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jobspy import scrape_jobs

from .config import (
    DATABASE,
    OLLAMA_ENABLED,
    BUILD_NUMBER,
)
from .db import (
    init_db,
    save_jobs,
    get_random_job,
    get_job,
    list_jobs_by_feedback,
    list_liked_jobs,
    aggregate_job_stats,
    increment_rating_count,
    record_feedback,
    cleanup_jobs,
)
from .ai import (
    ensure_model_downloaded,
    embed_text,
    generate_summary,
    render_markdown,
    process_all_jobs,
)
from .model import train_model, predict_unrated, evaluate_model

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
    train_model()
    if OLLAMA_ENABLED:
        def worker():
            ensure_model_downloaded()
            process_all_jobs()
            train_model()

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
    predictions = predict_unrated()
    model_stats = evaluate_model()
    return templates.TemplateResponse(
        "stats.html",
        {
            "request": request,
            "jobs": jobs,
            "stats": agg,
            "predictions": predictions,
            "model_stats": model_stats,
        },
    )




@app.get("/manage", response_class=HTMLResponse)
def manage(request: Request):
    return templates.TemplateResponse("manage.html", {"request": request, "deleted": None})


@app.post("/cleanup", response_class=HTMLResponse)
def cleanup(request: Request):
    deleted = cleanup_jobs()
    return templates.TemplateResponse(
        "manage.html", {"request": request, "deleted": deleted}
    )


def clear_ai_data_task() -> None:
    """Delete all summaries and embeddings."""
    if not OLLAMA_ENABLED:
        return
    log_progress("Clearing summaries and embeddings")
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute("DELETE FROM summaries")
    cur.execute("DELETE FROM embeddings")
    conn.commit()
    conn.close()
    log_progress("Done")


def reprocess_jobs_task() -> None:
    """Generate AI data for roles missing it."""
    if not OLLAMA_ENABLED:
        return
    log_progress("Generating data for missing jobs")
    process_all_jobs()
    log_progress("Done")


@app.post("/reprocess", response_class=HTMLResponse)
def reprocess(request: Request, background_tasks: BackgroundTasks):
    progress_logs.clear()
    background_tasks.add_task(reprocess_jobs_task)
    return templates.TemplateResponse("progress.html", {"request": request})


@app.post("/delete_ai", response_class=HTMLResponse)
def delete_ai(request: Request, background_tasks: BackgroundTasks):
    progress_logs.clear()
    background_tasks.add_task(clear_ai_data_task)
    return templates.TemplateResponse("progress.html", {"request": request})


@app.get("/export_likes")
def export_likes():
    """Download positively rated jobs as an Excel file."""
    df = list_liked_jobs()
    if not df.empty:
        df["Date Rated"] = pd.to_datetime(df["rated_at"], unit="s").dt.date
        df["Pay Range"] = df.apply(
            lambda r: format_salary(r["min_amount"], r["max_amount"], r["currency"])
            if pd.notnull(r["min_amount"]) and pd.notnull(r["max_amount"])
            else "",
            axis=1,
        )
        df = df.rename(
            columns={
                "company": "Company",
                "title": "Job Title",
                "location": "Location",
                "date_posted": "Date Posted",
                "job_url": "Link",
            }
        )
        df["Notes"] = ""
        df = df[
            [
                "Company",
                "Job Title",
                "Location",
                "Date Posted",
                "Date Rated",
                "Pay Range",
                "Notes",
                "Link",
            ]
        ]
    else:
        df = pd.DataFrame(
            columns=[
                "Company",
                "Job Title",
                "Location",
                "Date Posted",
                "Date Rated",
                "Pay Range",
                "Notes",
                "Link",
            ]
        )
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    headers = {
        "Content-Disposition": "attachment; filename=liked_jobs.xlsx"
    }
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )
