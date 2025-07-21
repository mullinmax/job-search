import json
import sqlite3
import time
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import main as app_main
from .ai import OLLAMA_ENABLED, process_all_jobs, render_markdown
from .utils import sanitize_html
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
            job_url TEXT,
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
            tags TEXT,
            rated_at INTEGER
        )
        """
    )
    # Add new columns to existing feedback tables without losing data
    cur.execute("PRAGMA table_info(feedback)")
    existing_cols = {r[1] for r in cur.fetchall()}
    if "tags" not in existing_cols:
        cur.execute("ALTER TABLE feedback ADD COLUMN tags TEXT")
    if "rated_at" not in existing_cols:
        cur.execute("ALTER TABLE feedback ADD COLUMN rated_at INTEGER")
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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS clean_jobs(
            job_id INTEGER PRIMARY KEY,
            title TEXT,
            company TEXT,
            min_amount REAL,
            max_amount REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS job_tags(
            job_id INTEGER,
            tag TEXT,
            PRIMARY KEY(job_id, tag)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS not_duplicates(
            job_id1 INTEGER,
            job_id2 INTEGER,
            PRIMARY KEY(job_id1, job_id2)
        )
        """
    )
    conn.commit()
    conn.close()


def save_jobs(df: pd.DataFrame) -> List[int]:
    """Insert jobs and return their resulting IDs after deduplication."""
    if df.empty:
        return []
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
    inserted_urls = []
    for _, row in df.iterrows():
        values = tuple(row.get(c) for c in cols)
        try:
            cur.execute(
                """
                INSERT INTO jobs(site,title,company,location,date_posted,description,interval,min_amount,max_amount,currency,job_url)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(job_url) DO UPDATE SET
                    site=excluded.site,
                    title=excluded.title,
                    company=excluded.company,
                    location=excluded.location,
                    date_posted=excluded.date_posted,
                    description=excluded.description,
                    interval=excluded.interval,
                    min_amount=excluded.min_amount,
                    max_amount=excluded.max_amount,
                    currency=excluded.currency
                """,
                values,
            )
        except sqlite3.OperationalError as e:
            if "ON CONFLICT" in str(e):
                cur.execute(
                    """
                    INSERT INTO jobs(site,title,company,location,date_posted,description,interval,min_amount,max_amount,currency,job_url)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    values,
                )
            else:
                raise
        inserted_urls.append(row.get("job_url"))

    # Merge duplicates by job_url and description
    cur.execute(
        "SELECT job_url FROM jobs WHERE job_url IS NOT NULL GROUP BY job_url HAVING COUNT(*)>1"
    )
    urls = [r[0] for r in cur.fetchall()]
    for url in urls:
        cur.execute(
            "SELECT id, date_posted FROM jobs WHERE job_url=? ORDER BY CASE WHEN date_posted IS NULL THEN 1 ELSE 0 END, date_posted DESC, id DESC",
            (url,),
        )
        rows = cur.fetchall()
        keep = rows[0][0]
        for rid, _ in rows[1:]:
            _transfer_feedback(cur, rid, keep)
            cur.execute("DELETE FROM summaries WHERE job_id=?", (rid,))
            cur.execute("DELETE FROM embeddings WHERE job_id=?", (rid,))
            cur.execute("DELETE FROM jobs WHERE id=?", (rid,))

    cur.execute(
        "SELECT description FROM jobs WHERE description IS NOT NULL AND description != '' GROUP BY description HAVING COUNT(*)>1"
    )
    descs = [r[0] for r in cur.fetchall()]
    for desc in descs:
        cur.execute(
            "SELECT id, date_posted FROM jobs WHERE description=? ORDER BY CASE WHEN date_posted IS NULL THEN 1 ELSE 0 END, date_posted DESC, id DESC",
            (desc,),
        )
        rows = cur.fetchall()
        keep = rows[0][0]
        for rid, _ in rows[1:]:
            _transfer_feedback(cur, rid, keep)
            cur.execute("DELETE FROM summaries WHERE job_id=?", (rid,))
            cur.execute("DELETE FROM embeddings WHERE job_id=?", (rid,))
            cur.execute("DELETE FROM jobs WHERE id=?", (rid,))

    # Identify resulting IDs for inserted URLs
    ids: List[int] = []
    for url in inserted_urls:
        cur.execute(
            "SELECT id FROM jobs WHERE job_url=? ORDER BY CASE WHEN date_posted IS NULL THEN 1 ELSE 0 END, date_posted DESC, id DESC LIMIT 1",
            (url,),
        )
        row = cur.fetchone()
        ids.append(row[0] if row else None)

    conn.commit()
    conn.close()
    if OLLAMA_ENABLED:
        process_all_jobs()
    return ids


def get_random_job() -> Optional[Dict]:
    conn = sqlite3.connect(app_main.DATABASE)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT j.id, j.site,
               COALESCE(c.title, j.title) AS title,
               COALESCE(c.company, j.company) AS company,
               j.location, j.date_posted, j.description, j.interval,
               COALESCE(c.min_amount, j.min_amount) AS min_amount,
               COALESCE(c.max_amount, j.max_amount) AS max_amount,
               j.currency, j.job_url, j.rating_count,
               s.summary, e.embedding,
               GROUP_CONCAT(t.tag) AS tags
        FROM jobs j
        LEFT JOIN summaries s ON j.id = s.job_id
        LEFT JOIN clean_jobs c ON j.id = c.job_id
        LEFT JOIN embeddings e ON j.id = e.job_id
        LEFT JOIN job_tags t ON j.id = t.job_id
        GROUP BY j.id
        ORDER BY rating_count ASC, RANDOM() LIMIT 20
        """
    )
    rows = cur.fetchall()
    columns = [c[0] for c in cur.description]
    conn.close()
    if not rows:
        return None

    jobs = []
    for row in rows:
        job = dict(zip(columns, row))
        if job.get("summary"):
            job["summary"] = sanitize_html(render_markdown(job["summary"]))
        if job.get("description"):
            job["description"] = sanitize_html(job["description"])
        if job.get("tags"):
            job["tags"] = [t for t in str(job["tags"]).split(',') if t]
        else:
            job["tags"] = []
        jobs.append(job)

    if _model is not None:
        weights = []
        expected_dim = getattr(_model, "n_features_in_", None)
        for job in jobs:
            try:
                vec = json.loads(job.get("embedding") or "null")
            except Exception:
                vec = None
            prob = 0.5
            if vec and (expected_dim is None or len(vec) == expected_dim):
                try:
                    prob = float(_model.predict_proba([vec])[0, 1])
                except Exception:
                    prob = 0.5
            job["predicted_confidence"] = prob
            job["predicted_match"] = prob >= 0.5
            weights.append(1.0 + prob)
        chosen = random.choices(jobs, weights=weights, k=1)[0]
    else:
        chosen = random.choice(jobs)
        chosen["predicted_confidence"] = None
        chosen["predicted_match"] = None

    chosen.pop("embedding", None)
    return chosen


def get_job(job_id: int) -> Optional[Dict]:
    conn = sqlite3.connect(app_main.DATABASE)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT j.id, j.site,
               COALESCE(c.title, j.title) AS title,
               COALESCE(c.company, j.company) AS company,
               j.location, j.date_posted, j.description, j.interval,
               COALESCE(c.min_amount, j.min_amount) AS min_amount,
               COALESCE(c.max_amount, j.max_amount) AS max_amount,
               j.currency, j.job_url, j.rating_count,
               s.summary, e.embedding,
               GROUP_CONCAT(t.tag) AS tags
        FROM jobs j
        LEFT JOIN summaries s ON j.id = s.job_id
        LEFT JOIN clean_jobs c ON j.id = c.job_id
        LEFT JOIN embeddings e ON j.id = e.job_id
        LEFT JOIN job_tags t ON j.id = t.job_id
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
            job["summary"] = sanitize_html(render_markdown(job["summary"]))
        if job.get("description"):
            job["description"] = sanitize_html(job["description"])
        if job.get("tags"):
            job["tags"] = [t for t in str(job["tags"]).split(',') if t]
        else:
            job["tags"] = []
        if _model is not None and job.get("embedding"):
            expected_dim = getattr(_model, "n_features_in_", None)
            try:
                vec = json.loads(job.get("embedding") or "null")
            except Exception:
                vec = None
            if vec and (expected_dim is None or len(vec) == expected_dim):
                try:
                    prob = float(_model.predict_proba([vec])[0, 1])
                    job["predicted_confidence"] = prob
                    job["predicted_match"] = prob >= 0.5
                except Exception:
                    pass
        if "predicted_confidence" not in job:
            job["predicted_confidence"] = None
            job["predicted_match"] = None
        job.pop("embedding", None)
        return job
    return None


def list_jobs_by_feedback() -> List[Dict]:
    conn = sqlite3.connect(app_main.DATABASE)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT j.id, j.site,
               COALESCE(c.title, j.title) AS title,
               COALESCE(c.company, j.company) AS company,
               j.location, j.date_posted, j.description, j.interval,
               COALESCE(c.min_amount, j.min_amount) AS min_amount,
               COALESCE(c.max_amount, j.max_amount) AS max_amount,
               j.currency, j.job_url, j.rating_count,
               COALESCE(SUM(CASE WHEN f.liked=1 THEN 1 ELSE 0 END), 0) AS likes,
               COALESCE(SUM(CASE WHEN f.liked=0 THEN 1 ELSE 0 END), 0) AS dislikes,
               CASE WHEN s.job_id IS NOT NULL THEN 1 ELSE 0 END AS has_summary,
               CASE WHEN e.job_id IS NOT NULL THEN 1 ELSE 0 END AS has_embedding
        FROM jobs j
        LEFT JOIN feedback f ON j.id = f.job_id
        LEFT JOIN summaries s ON j.id = s.job_id
        LEFT JOIN embeddings e ON j.id = e.job_id
        LEFT JOIN clean_jobs c ON j.id = c.job_id
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
        """
        SELECT AVG(COALESCE(c.min_amount, j.min_amount)),
               AVG(COALESCE(c.max_amount, j.max_amount))
        FROM jobs j
        LEFT JOIN clean_jobs c ON j.id = c.job_id
        WHERE COALESCE(c.min_amount, j.min_amount) IS NOT NULL
          AND COALESCE(c.max_amount, j.max_amount) IS NOT NULL
        """
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


def record_feedback(
    job_id: int,
    liked: bool,
    tags: List[str] | None,
    rated_at: Optional[int] = None,
) -> None:
    """Insert a feedback entry and retrain the model if needed."""
    increment_rating_count(job_id)
    conn = sqlite3.connect(app_main.DATABASE)
    cur = conn.cursor()
    if rated_at is None:
        rated_at = int(time.time())
    cur.execute(
        "INSERT INTO feedback(job_id, liked, tags, rated_at) VALUES(?,?,?,?)",
        (job_id, int(liked), ",".join(tags or []), rated_at),
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
        SELECT j.site,
               COALESCE(c.company, j.company) AS company,
               COALESCE(c.title, j.title) AS title,
               j.location, j.date_posted,
               f.rated_at,
               COALESCE(c.min_amount, j.min_amount) AS min_amount,
               COALESCE(c.max_amount, j.max_amount) AS max_amount,
               j.currency, j.job_url
        FROM jobs j
        JOIN feedback f ON j.id = f.job_id
        LEFT JOIN clean_jobs c ON j.id = c.job_id
        WHERE f.liked = 1
        ORDER BY f.rated_at DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def delete_job(job_id: int) -> None:
    """Remove a job and associated AI data."""
    conn = sqlite3.connect(app_main.DATABASE)
    cur = conn.cursor()
    cur.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    cur.execute("DELETE FROM summaries WHERE job_id=?", (job_id,))
    cur.execute("DELETE FROM embeddings WHERE job_id=?", (job_id,))
    cur.execute("DELETE FROM feedback WHERE job_id=?", (job_id,))
    cur.execute("DELETE FROM job_tags WHERE job_id=?", (job_id,))
    conn.commit()
    conn.close()


def _transfer_feedback(cur: sqlite3.Cursor, src: int, dest: int) -> None:
    """Move feedback rows from src to dest if dest has none."""
    cur.execute("SELECT COUNT(*) FROM feedback WHERE job_id=?", (dest,))
    if cur.fetchone()[0] == 0:
        cur.execute(
            "SELECT liked, tags, rated_at FROM feedback WHERE job_id=?",
            (src,),
        )
        rows = cur.fetchall()
        if rows:
            cur.executemany(
                "INSERT INTO feedback(job_id, liked, tags, rated_at) VALUES(?,?,?,?)",
                [(dest, r[0], r[1], r[2]) for r in rows],
            )
            cur.execute("SELECT rating_count FROM jobs WHERE id=?", (dest,))
            cnt = cur.fetchone()[0] or 0
            cnt += len(rows)
            cur.execute(
                "UPDATE jobs SET rating_count=? WHERE id=?",
                (cnt, dest),
            )
        cur.execute("DELETE FROM feedback WHERE job_id=?", (src,))
    cur.execute("SELECT tag FROM job_tags WHERE job_id=?", (src,))
    tags = [r[0] for r in cur.fetchall()]
    if tags:
        cur.executemany(
            "INSERT OR IGNORE INTO job_tags(job_id, tag) VALUES(?, ?)",
            [(dest, t) for t in tags],
        )
    cur.execute("DELETE FROM job_tags WHERE job_id=?", (src,))


def mark_not_duplicates(id1: int, id2: int) -> None:
    """Record that two jobs are not duplicates."""
    if id1 > id2:
        id1, id2 = id2, id1
    conn = sqlite3.connect(app_main.DATABASE)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO not_duplicates(job_id1, job_id2) VALUES(?, ?)",
        (id1, id2),
    )
    conn.commit()
    conn.close()


def find_duplicate_jobs(threshold: float = 0.85) -> List[Tuple[Dict, Dict, float]]:
    """Return pairs of potentially duplicate jobs based on text similarity."""
    conn = sqlite3.connect(app_main.DATABASE)
    df = pd.read_sql_query(
        "SELECT id, site, title, company, location, description FROM jobs", conn
    )
    cur = conn.cursor()
    cur.execute("SELECT job_id1, job_id2 FROM not_duplicates")
    ignored = {tuple(sorted(row)) for row in cur.fetchall()}
    conn.close()
    if df.empty:
        return []
    texts = (
        df["title"].fillna("") + " " + df["company"].fillna("") + " " + df["description"].fillna("")
    ).str.lower()
    vec = TfidfVectorizer().fit_transform(texts)
    sim = cosine_similarity(vec)
    pairs = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if sim[i, j] >= threshold:
                key = tuple(sorted((df.iloc[i]["id"], df.iloc[j]["id"])))
                if key not in ignored:
                    pairs.append(
                        (df.iloc[i].to_dict(), df.iloc[j].to_dict(), float(sim[i, j]))
                    )
    return pairs
