import os
import sqlite3
from collections import deque
from pathlib import Path
import sys

import pandas as pd
import pytest

import importlib

# Ensure the app package can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the module fresh for each test to respect DATABASE patching

@pytest.fixture
def main(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE", str(db_path))
    module = importlib.import_module("app.main")
    importlib.reload(module)
    monkeypatch.setattr(module, "DATABASE", str(db_path))
    logs = deque(maxlen=100)
    monkeypatch.setattr(module, "progress_logs", logs)
    return module


def test_init_db_creates_tables(main):
    main.init_db()
    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    conn.close()
    assert "jobs" in tables
    assert "feedback" in tables


def test_save_and_get_random_job(main):
    main.init_db()
    df = pd.DataFrame([
        {
            "site": "test",
            "title": "Data Engineer",
            "company": "ACME",
            "location": "Remote",
            "date_posted": "today",
            "description": "desc",
            "interval": "year",
            "min_amount": 100,
            "max_amount": 200,
            "currency": "USD",
            "job_url": "http://example.com/1",
        }
    ])
    main.save_jobs(df)
    job = main.get_random_job()
    assert job["title"] == "Data Engineer"
    assert job["company"] == "ACME"


def test_update_elo_single(main):
    main.init_db()
    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO jobs(site,title,company,location,date_posted,description,interval,min_amount,max_amount,currency,job_url,elo,rating_count)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,1000,0)
        """,
        (
            "test",
            "job",
            "comp",
            "loc",
            "date",
            "desc",
            "year",
            100,
            200,
            "USD",
            "http://example.com/2",
        ),
    )
    job_id = cur.lastrowid
    conn.commit()
    conn.close()

    main.update_elo_single(job_id, True)
    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT elo, rating_count FROM jobs WHERE id=?", (job_id,))
    elo, count = cur.fetchone()
    conn.close()
    assert round(elo, 2) == 1016.0
    assert count == 1


def test_record_feedback(main):
    main.init_db()
    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO jobs(site,title,company,location,date_posted,description,interval,min_amount,max_amount,currency,job_url)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "test",
            "job",
            "comp",
            "loc",
            "date",
            "desc",
            "year",
            100,
            200,
            "USD",
            "http://example.com/3",
        ),
    )
    job_id = cur.lastrowid
    conn.commit()
    conn.close()

    main.record_feedback(job_id, True, "good")
    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM feedback")
    feedback_count = cur.fetchone()[0]
    cur.execute("SELECT rating_count FROM jobs WHERE id=?", (job_id,))
    rating_count = cur.fetchone()[0]
    conn.close()

    assert feedback_count == 1
    assert rating_count == 1


def test_fetch_jobs_task(main, monkeypatch):
    main.init_db()

    def fake_scrape_jobs(site_name, **kwargs):
        return pd.DataFrame([
            {
                "site": site_name,
                "title": "Title",
                "company": "Comp",
                "location": "Loc",
                "date_posted": "d",
                "description": "desc",
                "interval": "year",
                "min_amount": 1,
                "max_amount": 2,
                "currency": "USD",
                "job_url": f"http://example.com/{site_name}",
            }
        ])

    monkeypatch.setattr(main, "scrape_jobs", fake_scrape_jobs)

    main.fetch_jobs_task("dev", "here", ["indeed", "linkedin"])

    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM jobs")
    count = cur.fetchone()[0]
    conn.close()

    assert count == 2
    assert main.progress_logs[-1] == "Done"

