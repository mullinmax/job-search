import json
import sqlite3
import time
from typing import Dict, List, Optional

import pandas as pd

from . import main as app_main
from .ai import OLLAMA_ENABLED, process_all_jobs, render_markdown
from .model import train_model, _model


def init_db() -> None:
    conn = sqlite3.connect(app_main.DATABASE)
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
    df = df.loc[:, df.columns.intersection(cols.keys())]
    conn = sqlite3.connect(app_main.DATABASE)
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
    conn = sqlite3.connect(app_main.DATABASE)
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
        job = dict(zip(columns, row))
        if job.get("summary"):
            job["summary"] = render_markdown(job["summary"])
        return job
    return None


def get_job(job_id: int) -> Optional[Dict]:
    conn = sqlite3.connect(app_main.DATABASE)
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
        job = dict(zip(columns, row))
        if job.get("summary"):
            job["summary"] = render_markdown(job["summary"])
        return job
    return None


def list_jobs_by_feedback() -> List[Dict]:
    conn = sqlite3.connect(app_main.DATABASE)
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
    conn = sqlite3.connect(app_main.DATABASE)
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
    conn = sqlite3.connect(app_main.DATABASE)
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
    increment_rating_count(job_id)
    conn = sqlite3.connect(app_main.DATABASE)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO feedback(job_id, liked, reason, rated_at) VALUES(?,?,?,?)",
        (job_id, int(liked), reason, int(time.time())),
    )
    conn.commit()
    conn.close()

    retrain = False
    if _model is not None:
        conn = sqlite3.connect(app_main.DATABASE)
        cur = conn.cursor()
        cur.execute("SELECT embedding FROM embeddings WHERE job_id=?", (job_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            emb = json.loads(row[0])
            pred = int(_model.predict([emb])[0])
            retrain = pred != int(liked)
    else:
        retrain = True

    if retrain:
        train_model()


def cleanup_jobs() -> int:
    conn = sqlite3.connect(app_main.DATABASE)
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


def list_liked_jobs() -> pd.DataFrame:
    """Return a DataFrame of positively rated jobs with rating timestamps."""
    conn = sqlite3.connect(app_main.DATABASE)
    query = """
        SELECT j.company, j.title, j.location, j.date_posted,
               f.rated_at, j.min_amount, j.max_amount, j.currency, j.job_url
        FROM jobs j
        JOIN feedback f ON j.id = f.job_id
        WHERE f.liked = 1
        ORDER BY f.rated_at DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
