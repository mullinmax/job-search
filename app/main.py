import os
import time
from typing import List, Dict, Optional, Tuple

import html
import random
from difflib import SequenceMatcher

import logging
from collections import deque
from threading import Thread, Lock

import io
import pandas as pd
from fastapi import FastAPI, Form, Request, BackgroundTasks, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .scrapers import (
    scrape_with_jobspy,
    scrape_with_linkedin,
    scrape_with_jobfunnel,
    RATE_LIMIT,
)

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
    delete_job,
    mark_not_duplicates,
    find_duplicate_jobs,
    _transfer_feedback,
)
from .database import connect_db
from .ai import (
    ensure_model_downloaded,
    embed_text,
    generate_summary,
    render_markdown,
    process_all_jobs,
    regenerate_job_ai,
)
from .utils import sanitize_html
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
Start with two bullet lists: first list the required technologies and then list any bonus or nice-to-have skills. Group alternatives with '/'.
Skip duplicate items and omit sections that cannot be filled.

---
**Required Skills**
- item one
- item two

**Bonus Skills**
- item one
- item two

**Soft Skills**
- item one
- item two
---

Job posting:
"""{description}"""
'''

def format_salary(min_amount: float, max_amount: float, currency: str) -> str:
    """Return a salary range rounded to the nearest thousand with 'k' suffix."""
    def to_k(val: float) -> str:
        return f"{round(val / 1000):.0f}k"
    cur = "$" if not currency or currency.upper() == "USD" else currency
    return f"{cur}{to_k(min_amount)} - {cur}{to_k(max_amount)}"

templates.env.globals["format_salary"] = format_salary


def clean_location(loc: str) -> str:
    """Strip country information leaving 'City, ST' or just state."""
    if not loc:
        return ""
    parts = [p.strip() for p in str(loc).split(",")]
    while parts and parts[-1].lower() in {"us", "usa", "united states", "u.s."}:
        parts.pop()
    return ", ".join(parts)


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


def highlight_diffs(a: Dict, b: Dict) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return HTML strings for fields of two jobs with differences marked.

    The description field is rendered as Markdown/HTML so bullet lists display
    correctly. Segments that differ are wrapped in ``<mark>`` tags. This may
    break complex HTML slightly but keeps the original formatting."""

    def markup(x: str, y: str, render: bool = False) -> Tuple[str, str]:
        if render:
            x = render_markdown(x)
            y = render_markdown(y)
        sm = SequenceMatcher(None, x or "", y or "")
        a_parts: List[str] = []
        b_parts: List[str] = []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            sub_a = x[i1:i2]
            sub_b = y[j1:j2]
            if op == "equal":
                a_parts.append(sub_a)
                b_parts.append(sub_b)
            else:
                if sub_a:
                    a_parts.append(f"<mark>{html.escape(sub_a)}</mark>")
                if sub_b:
                    b_parts.append(f"<mark>{html.escape(sub_b)}</mark>")
        a_html = "".join(a_parts)
        b_html = "".join(b_parts)
        return sanitize_html(a_html), sanitize_html(b_html)

    fields = [
        ("title", False),
        ("company", False),
        ("location", False),
        ("description", True),
    ]
    res_a: Dict[str, str] = {}
    res_b: Dict[str, str] = {}
    for f, render in fields:
        res_a[f + "_html"], res_b[f + "_html"] = markup(str(a.get(f) or ""), str(b.get(f) or ""), render)
    res_a["summary"] = a.get("summary")
    res_b["summary"] = b.get("summary")
    res_a["id"] = a.get("id")
    res_b["id"] = b.get("id")
    res_a["site"] = a.get("site")
    res_b["site"] = b.get("site")
    return res_a, res_b

def log_progress(message: str) -> None:
    """Log a progress message to the logger and internal buffer."""
    logger.info(message)
    progress_logs.append(message)





fetch_lock = Lock()


def fetch_jobs_task(search_term: str, location: str, sites: List[str]) -> None:
    """Background task to fetch job postings and store them."""
    if not fetch_lock.acquire(blocking=False):
        log_progress("Another fetch task is running")
        return
    try:
        google_term = f"{search_term} jobs in {location}"
        jobs: List[pd.DataFrame] = []
        for site in sites:
            log_progress(f"Fetching from {site}")
            try:
                df = scrape_with_jobspy(site, search_term, google_term, location, 50)
                jobs.append(df)
                log_progress(f"{site}: {len(df)} jobs")
            except Exception as exc:
                log_progress(f"Skipping {site}: {exc}")
            time.sleep(RATE_LIMIT)

        # Additional libraries
        log_progress("Fetching from linkedin-jobs-scraper")
        try:
            df = scrape_with_linkedin(search_term, location, 50)
            if not df.empty:
                jobs.append(df)
                log_progress(f"linkedin-extra: {len(df)} jobs")
        except Exception as exc:
            log_progress(f"LinkedIn scraper failed: {exc}")
        time.sleep(RATE_LIMIT)

        log_progress("Fetching from jobfunnel")
        try:
            df = scrape_with_jobfunnel(search_term, location, 50)
            if not df.empty:
                jobs.append(df)
                log_progress(f"jobfunnel: {len(df)} jobs")
        except Exception as exc:
            log_progress(f"JobFunnel failed: {exc}")

        if jobs:
            combined = pd.concat(jobs, ignore_index=True)
            save_jobs(combined)
            log_progress(f"Saved {len(combined)} jobs")
            if OLLAMA_ENABLED:
                process_all_jobs()
        log_progress("Done")
    finally:
        fetch_lock.release()


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


@app.get("/rate/{job_id}", response_class=HTMLResponse)
def rate_form(request: Request, job_id: int, liked: int):
    job = get_job(job_id)
    if not job:
        return RedirectResponse("/swipe", status_code=303)
    return templates.TemplateResponse(
        "rate.html",
        {"request": request, "job": job, "liked": liked, "tags": job.get("tags", [])},
    )


@app.post("/feedback")
def feedback(job_id: int = Form(...), liked: int = Form(...), tags: List[str] = Form(None)):
    record_feedback(job_id, bool(liked), tags or [])
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
            "container_class": "container-fluid",
        },
    )


@app.post("/train", response_class=HTMLResponse)
def train(request: Request):
    """Retrain the logistic regression model."""
    train_model()
    return RedirectResponse("/stats", status_code=303)




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
    conn = connect_db()
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


def regen_job_task(job_id: int) -> None:
    """Regenerate AI data for a single role."""
    if not OLLAMA_ENABLED:
        return
    log_progress(f"Regenerating job {job_id}")
    regenerate_job_ai(job_id)
    log_progress("Done")


def regen_jobs_task(job_ids: List[int]) -> None:
    """Regenerate AI data for multiple roles."""
    if not OLLAMA_ENABLED:
        return
    for jid in job_ids:
        log_progress(f"Regenerating job {jid}")
        regenerate_job_ai(jid)
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


@app.post("/regen_job/{job_id}", response_class=HTMLResponse)
def regen_job(request: Request, job_id: int, background_tasks: BackgroundTasks):
    progress_logs.clear()
    background_tasks.add_task(regen_job_task, job_id)
    return templates.TemplateResponse("progress.html", {"request": request})


@app.post("/regen_jobs", response_class=HTMLResponse)
def regen_jobs(request: Request, background_tasks: BackgroundTasks, job_ids: List[int] = Form(...)):
    progress_logs.clear()
    background_tasks.add_task(regen_jobs_task, job_ids)
    return templates.TemplateResponse("progress.html", {"request": request})


@app.get("/export_likes")
def export_likes():
    """Download positively rated jobs as an Excel file."""
    df = list_liked_jobs()
    if not df.empty:
        rated = pd.to_datetime(df["rated_at"], unit="s", errors="coerce")
        df["Farmed Date"] = rated.dt.strftime("%-m/%-d/%Y")

        df["min_amount"] = pd.to_numeric(df["min_amount"], errors="coerce")
        df["max_amount"] = pd.to_numeric(df["max_amount"], errors="coerce")
        df["Pay Range"] = df.apply(
            lambda r: format_salary(r["min_amount"], r["max_amount"], r["currency"])
            if (
                pd.notnull(r["min_amount"])
                and pd.notnull(r["max_amount"])
                and r["min_amount"] > 0
                and r["max_amount"] > 0
            )
            else "",
            axis=1,
        )
        df = df.rename(
            columns={
                "site": "Source",
                "company": "Company",
                "title": "Job Title",
                "location": "Location",
                "date_posted": "Date Posted",
                "job_url": "Link",
            }
        )
        df["Location"] = df["Location"].apply(clean_location)
        dates = pd.to_datetime(df["Date Posted"], errors="coerce")
        formatted = dates.dt.strftime("%-m/%-d/%Y")
        df["Date Posted"] = formatted.where(dates.notna(), df["Date Posted"])
        df["Notes"] = ""
        df = df[
            [
                "Source",
                "Company",
                "Job Title",
                "Location",
                "Date Posted",
                "Farmed Date",
                "Pay Range",
                "Notes",
                "Link",
            ]
        ]
    else:
        df = pd.DataFrame(
            columns=[
                "Source",
                "Company",
                "Job Title",
                "Location",
                "Date Posted",
                "Farmed Date",
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

@app.get("/dedup", response_class=HTMLResponse)
def dedup(request: Request):
    pairs = find_duplicate_jobs()
    if not pairs:
        return templates.TemplateResponse(
            "dedup.html", {"request": request, "pair": None, "remaining": 0}
        )
    a, b, _ = pairs[0]
    a_html, b_html = highlight_diffs(a, b)
    return templates.TemplateResponse(
        "dedup.html",
        {
            "request": request,
            "pair": (a_html, b_html),
            "remaining": len(pairs),
        },
    )


@app.post("/dedup_action", response_class=HTMLResponse)
def dedup_action(pair_ids: str = Form(...), dup: int = Form(...)):
    id1, id2 = [int(x) for x in pair_ids.split(",")]
    if dup:
        j1 = get_job(id1)
        j2 = get_job(id2)

        def parse_date(d: str) -> pd.Timestamp:
            """Return parsed UTC timestamp or UTC Timestamp.min if invalid."""
            if not d:
                return pd.Timestamp.min.tz_localize("UTC")
            try:
                dt = pd.to_datetime(d, utc=True, errors="coerce")
            except Exception:
                return pd.Timestamp.min.tz_localize("UTC")
            return dt if not pd.isna(dt) else pd.Timestamp.min.tz_localize("UTC")

        if j1 and j1.get("site") == "upload":
            keep, remove = id1, id2
        elif j2 and j2.get("site") == "upload":
            keep, remove = id2, id1
        else:
            t1 = parse_date(j1.get("date_posted")) if j1 else pd.Timestamp.min.tz_localize("UTC")
            t2 = parse_date(j2.get("date_posted")) if j2 else pd.Timestamp.min.tz_localize("UTC")
            if t1 >= t2:
                keep, remove = id1, id2
            else:
                keep, remove = id2, id1

        conn = connect_db()
        cur = conn.cursor()
        _transfer_feedback(cur, remove, keep)
        conn.commit()
        conn.close()
        delete_job(remove)
    else:
        mark_not_duplicates(id1, id2)
    return RedirectResponse("/dedup", status_code=303)


@app.post("/delete_job/{job_id}", response_class=HTMLResponse)
def delete_job_endpoint(request: Request, job_id: int):
    delete_job(job_id)
    return RedirectResponse("/dedup", status_code=303)


def import_custom_csv(data: bytes) -> int:
    """Import user-supplied CSV data and mark all roles as matches."""
    df = pd.read_csv(io.BytesIO(data))
    if df.empty:
        return 0
    jobs = pd.DataFrame({
        "site": "upload",
        "title": df.get("Job Title"),
        "company": df.get("Company"),
        "location": df.get("City"),
        "date_posted": df.get("Posted Date"),
        "description": "",
        "interval": "",
        "min_amount": None,
        "max_amount": None,
        "currency": None,
        "job_url": df.get("Hyperlink"),
    })
    ids = save_jobs(jobs)
    for row, job_id in zip(df.itertuples(), ids):
        rated = pd.to_datetime(getattr(row, "Farmed Date", None), errors="coerce")
        ts = int(rated.timestamp()) if not pd.isna(rated) else None
        record_feedback(job_id, True, [], rated_at=ts)
    return len(ids)


@app.post("/upload_csv", response_class=HTMLResponse)
async def upload_csv(request: Request, file: UploadFile):
    data = await file.read()
    import_custom_csv(data)
    return RedirectResponse("/manage", status_code=303)
