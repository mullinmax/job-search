import os
import sqlite3
import json
from collections import deque
from pathlib import Path
import sys

import pandas as pd
import pytest
import time

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
    assert "job_tags" in tables


def test_init_db_adds_tags_column(main):
    # Simulate an older schema without the tags column
    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE feedback(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            liked INTEGER,
            rated_at INTEGER
        )
        """
    )
    cur.execute(
        "INSERT INTO feedback(job_id, liked, rated_at) VALUES(1, 1, 0)"
    )
    conn.commit()
    conn.close()

    # Running init_db should add the missing column without touching data
    main.init_db()
    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(feedback)")
    cols = {r[1] for r in cur.fetchall()}
    cur.execute("SELECT liked FROM feedback WHERE job_id=1")
    row = cur.fetchone()
    conn.close()
    assert "tags" in cols
    assert row[0] == 1


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


def test_save_jobs_dedup_by_url(main):
    main.init_db()
    df = pd.DataFrame([
        {
            "site": "t",
            "title": "Engineer",
            "company": "A",
            "location": "L",
            "date_posted": "2024-01-01",
            "description": "old",
            "interval": "year",
            "min_amount": 1,
            "max_amount": 2,
            "currency": "USD",
            "job_url": "http://dup.com/1",
        },
        {
            "site": "t",
            "title": "Engineer",
            "company": "A",
            "location": "L",
            "date_posted": "2024-02-01",
            "description": "new",
            "interval": "year",
            "min_amount": 1,
            "max_amount": 2,
            "currency": "USD",
            "job_url": "http://dup.com/1",
        },
    ])
    main.save_jobs(df)
    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT description FROM jobs")
    row = cur.fetchone()
    conn.close()
    assert row[0] == "new"


def test_save_jobs_dedup_by_description(main):
    main.init_db()
    df = pd.DataFrame([
        {
            "site": "t",
            "title": "Engineer",
            "company": "A",
            "location": "L",
            "date_posted": "2024-01-01",
            "description": "same desc",
            "interval": "year",
            "min_amount": 1,
            "max_amount": 2,
            "currency": "USD",
            "job_url": "http://desc.com/1",
        },
        {
            "site": "t",
            "title": "Engineer",
            "company": "A",
            "location": "L",
            "date_posted": "2024-02-01",
            "description": "same desc",
            "interval": "year",
            "min_amount": 1,
            "max_amount": 2,
            "currency": "USD",
            "job_url": "http://desc.com/2",
        },
    ])
    main.save_jobs(df)
    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT job_url FROM jobs")
    url = cur.fetchone()[0]
    conn.close()
    assert url == "http://desc.com/2"


def test_get_random_job_sanitizes_html(main):
    main.init_db()
    df = pd.DataFrame([
        {
            "site": "t",
            "title": "Eng",
            "company": "C",
            "location": "L",
            "date_posted": "d",
            "description": "<script>alert(1)</script><b>good</b>",
            "interval": "year",
            "min_amount": 1,
            "max_amount": 2,
            "currency": "USD",
            "job_url": "http://safe.com/1",
        }
    ])
    main.save_jobs(df)
    job = main.get_random_job()
    assert "<script" not in job["description"]


def test_highlight_diffs_sanitizes(main):
    a = {
        "title": "<b>x</b>",
        "company": "A",
        "location": "L",
        "description": "<img src=x onerror=alert(1)>",
        "site": "s",
    }
    b = {
        "title": "<b>x</b>",
        "company": "A",
        "location": "L",
        "description": "ok",
        "site": "s",
    }
    res_a, res_b = main.highlight_diffs(a, b)
    assert "onerror" not in res_a["description_html"].lower()
    assert "<script" not in res_a["description_html"].lower()


def test_increment_rating_count(main):
    main.init_db()
    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO jobs(site,title,company,location,date_posted,description,interval,min_amount,max_amount,currency,job_url,rating_count)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,0)
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

    main.increment_rating_count(job_id)
    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT rating_count FROM jobs WHERE id=?", (job_id,))
    count = cur.fetchone()[0]
    conn.close()
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

    main.record_feedback(job_id, True, ["good"])
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

    monkeypatch.setattr(main, "scrape_with_jobspy", lambda *a, **k: fake_scrape_jobs(a[0]))
    monkeypatch.setattr(main, "scrape_with_linkedin", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(main, "scrape_with_jobfunnel", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(time, "sleep", lambda x: None)

    main.fetch_jobs_task("dev", "here", ["indeed", "linkedin"])

    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM jobs")
    count = cur.fetchone()[0]
    conn.close()

    assert count == 1
    assert main.progress_logs[-1] == "Done"


def test_cleanup_jobs(main):
    main.init_db()
    df = pd.DataFrame([
        {
            "site": "t",
            "title": "Dev",
            "company": "Comp",
            "location": "L",
            "date_posted": "d",
            "description": "",
            "interval": "year",
            "min_amount": 1,
            "max_amount": 2,
            "currency": "USD",
            "job_url": "http://example.com/1",
        },
        {
            "site": "t",
            "title": "Dev",
            "company": "Comp",
            "location": "L",
            "date_posted": "d",
            "description": "desc",
            "interval": "year",
            "min_amount": 1,
            "max_amount": 2,
            "currency": "USD",
            "job_url": "http://example.com/2",
        },
        {
            "site": "t",
            "title": "Dev",
            "company": "Comp",
            "location": "L",
            "date_posted": "d",
            "description": "desc2",
            "interval": "year",
            "min_amount": 1,
            "max_amount": 2,
            "currency": "USD",
            "job_url": "http://example.com/3",
        },
    ])
    main.save_jobs(df)
    deleted = main.cleanup_jobs()
    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM jobs")
    count = cur.fetchone()[0]
    conn.close()
    assert deleted == 2
    assert count == 1


def test_aggregate_job_stats(main):
    main.init_db()
    df = pd.DataFrame([
        {
            "site": "a",
            "title": "T1",
            "company": "C1",
            "location": "L",
            "date_posted": "2024-01-01",
            "description": "d",
            "interval": "year",
            "min_amount": 100,
            "max_amount": 200,
            "currency": "USD",
            "job_url": "http://x.com/1",
        },
        {
            "site": "b",
            "title": "T2",
            "company": "C2",
            "location": "L",
            "date_posted": "2024-01-02",
            "description": "d2",
            "interval": "year",
            "min_amount": 200,
            "max_amount": 300,
            "currency": "USD",
            "job_url": "http://x.com/2",
        },
        {
            "site": "a",
            "title": "T3",
            "company": "C3",
            "location": "L",
            "date_posted": "2024-01-01",
            "description": "d3",
            "interval": "year",
            "min_amount": None,
            "max_amount": None,
            "currency": None,
            "job_url": "http://x.com/3",
        },
    ])
    main.save_jobs(df)
    stats = main.aggregate_job_stats()
    assert stats["total_jobs"] == 3
    assert stats["by_site"] == {"a": 2, "b": 1}
    assert stats["by_date"] == {"2024-01-01": 2, "2024-01-02": 1}
    assert stats["avg_min_pay"] == pytest.approx(150)
    assert stats["avg_max_pay"] == pytest.approx(250)


def test_render_markdown(main):
    html = main.render_markdown("---\n**Test**\n---")
    assert "<strong>Test</strong>" in html
    assert "---" not in html


def test_render_markdown_lists(main):
    text = "**Technical Requirements**\n- a\n- b"
    html = main.render_markdown(text)
    assert html.count("<li>") == 2


def test_render_markdown_dedup(main):
    text = "**Skills**\n- Python\n- Python\n- Java"
    html = main.render_markdown(text)
    assert html.count("<li>") == 2
    assert "Python" in html and "Java" in html


def test_render_markdown_prefix(main):
    text = "markdown Technical Requirements\n- A\n- B"
    html = main.render_markdown(text)
    assert "markdown" not in html.lower()


def test_clear_ai_data_and_reprocess_tasks(main, monkeypatch):
    main.init_db()
    df = pd.DataFrame([
        {
            "site": "t",
            "title": "T1",
            "company": "C1",
            "location": "L",
            "date_posted": "d",
            "description": "d1",
            "interval": "year",
            "min_amount": 1,
            "max_amount": 2,
            "currency": "USD",
            "job_url": "http://e.com/1",
        }
    ])
    main.save_jobs(df)
    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    cur.execute("INSERT INTO summaries(job_id, summary) VALUES(1, 's')")
    cur.execute("INSERT INTO embeddings(job_id, embedding) VALUES(1, 'e')")
    conn.commit()
    conn.close()

    # Test clearing
    monkeypatch.setattr(main, "OLLAMA_ENABLED", True)
    main.clear_ai_data_task()
    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM summaries")
    assert cur.fetchone()[0] == 0
    cur.execute("SELECT COUNT(*) FROM embeddings")
    assert cur.fetchone()[0] == 0
    conn.close()
    assert main.progress_logs[-1] == "Done"

    # Test generating missing data
    called = {}

    def fake_process():
        called["x"] = True

    monkeypatch.setattr(main, "process_all_jobs", fake_process)
    main.reprocess_jobs_task()
    assert called.get("x")
    assert main.progress_logs[-1] == "Done"


def test_evaluate_model(main):
    main.init_db()
    import importlib
    import app.config as cfg
    import app.model as model_mod
    importlib.reload(cfg)
    importlib.reload(model_mod)
    main.train_model = model_mod.train_model
    main.evaluate_model = model_mod.evaluate_model
    # Ensure the model module uses the database for this test

    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    jobs = [
        (
            "t",
            "J1",
            "C1",
            "L",
            "d",
            "desc",
            "year",
            1,
            2,
            "USD",
            "http://e.com/1",
            json.dumps([0.0, 0.0]),
            0,
        ),
        (
            "t",
            "J2",
            "C2",
            "L",
            "d",
            "desc",
            "year",
            1,
            2,
            "USD",
            "http://e.com/2",
            json.dumps([1.0, 1.0]),
            1,
        ),
    ]
    for job in jobs:
        cur.execute(
            """
            INSERT INTO jobs(site,title,company,location,date_posted,description,interval,min_amount,max_amount,currency,job_url)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            job[:11],
        )
        job_id = cur.lastrowid
        cur.execute(
            "INSERT INTO embeddings(job_id, embedding) VALUES(?, ?)",
            (job_id, job[11]),
        )
        cur.execute(
            "INSERT INTO feedback(job_id, liked, tags, rated_at) VALUES(?,?,?,0)",
            (job_id, job[12], ""),
        )
    conn.commit()
    conn.close()
    main.train_model()
    stats = main.evaluate_model()
    assert stats["total"] == 2
    assert stats["tp"] + stats["tn"] + stats["fp"] + stats["fn"] == 2
    assert 0.0 <= stats["accuracy"] <= 1.0


def test_list_liked_jobs(main):
    main.init_db()
    df = pd.DataFrame([
        {
            "site": "t",
            "title": "Dev",
            "company": "C",
            "location": "L",
            "date_posted": "d",
            "description": "desc",
            "interval": "year",
            "min_amount": 1,
            "max_amount": 2,
            "currency": "USD",
            "job_url": "http://e.com/liked",
        }
    ])
    main.save_jobs(df)
    ts = int(pd.Timestamp("2025-07-21").timestamp())
    main.record_feedback(1, True, "", rated_at=ts)
    liked = main.list_liked_jobs()
    assert len(liked) == 1
    assert liked["company"].iloc[0] == "C"


def test_find_duplicate_jobs(main):
    main.init_db()
    df = pd.DataFrame([
        {"site": "a", "title": "Engineer", "company": "X", "location": "L", "date_posted": "d", "description": "work on things", "interval": "year", "min_amount": 1, "max_amount": 2, "currency": "USD", "job_url": "http://a.com/1"},
        {"site": "b", "title": "Engineer", "company": "X", "location": "L", "date_posted": "d", "description": "working on things", "interval": "year", "min_amount": 1, "max_amount": 2, "currency": "USD", "job_url": "http://a.com/2"},
    ])
    main.save_jobs(df)
    pairs = main.find_duplicate_jobs(0.5)
    assert pairs
    a, b, sim = pairs[0]
    assert a["title"] == b["title"]


def test_mark_not_duplicates(main):
    main.init_db()
    df = pd.DataFrame([
        {"site": "a", "title": "Engineer", "company": "X", "location": "L", "date_posted": "d", "description": "desc", "interval": "year", "min_amount": 1, "max_amount": 2, "currency": "USD", "job_url": "http://a.com/1"},
        {"site": "b", "title": "Engineer", "company": "X", "location": "L", "date_posted": "d", "description": "desc slightly different", "interval": "year", "min_amount": 1, "max_amount": 2, "currency": "USD", "job_url": "http://a.com/2"},
    ])
    main.save_jobs(df)
    pairs = main.find_duplicate_jobs(0.5)
    assert pairs
    id1 = pairs[0][0]["id"]
    id2 = pairs[0][1]["id"]
    main.mark_not_duplicates(id1, id2)
    pairs = main.find_duplicate_jobs(0.5)
    assert pairs == []


def test_train_model_single_class(main):
    """Model training should handle only one feedback class gracefully."""
    main.init_db()
    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO jobs(site,title,company,location,date_posted,description,interval,min_amount,max_amount,currency,job_url)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "t",
            "J1",
            "C1",
            "L",
            "d",
            "desc",
            "year",
            1,
            2,
            "USD",
            "http://e.com/1",
        ),
    )
    job_id = cur.lastrowid
    cur.execute(
        "INSERT INTO embeddings(job_id, embedding) VALUES(?, ?)",
        (job_id, json.dumps([0.0, 0.0])),
    )
    conn.commit()
    conn.close()

    import app.model as model_mod
    import app.db as db_mod
    model_mod._model = None
    db_mod._model = None
    model_mod.DATABASE = main.DATABASE

    # Should not raise even though only one class is present
    main.record_feedback(job_id, True, "")

    assert model_mod._model is None


def test_predict_unrated_skips_invalid_embeddings(main):
    main.init_db()
    import importlib
    import app.model as model_mod
    importlib.reload(model_mod)
    main.train_model = model_mod.train_model
    main.predict_unrated = model_mod.predict_unrated
    model_mod.DATABASE = main.DATABASE
    model_mod._model = None
    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    jobs = [
        ("t", "J1", "C1", "L", "d", "desc", "year", 1, 2, "USD", "http://e.com/1", json.dumps([0.0, 0.0])),
        ("t", "J2", "C2", "L", "d", "desc", "year", 1, 2, "USD", "http://e.com/2", json.dumps([1.0, 1.0])),
        ("t", "J3", "C3", "L", "d", "desc", "year", 1, 2, "USD", "http://e.com/3", json.dumps([0.5, 0.5])),
        ("t", "J4", "C4", "L", "d", "desc", "year", 1, 2, "USD", "http://e.com/4", json.dumps([])),
    ]
    for job in jobs:
        cur.execute(
            """
            INSERT INTO jobs(site,title,company,location,date_posted,description,interval,min_amount,max_amount,currency,job_url)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            job[:11],
        )
        job_id = cur.lastrowid
        cur.execute(
            "INSERT INTO embeddings(job_id, embedding) VALUES(?, ?)",
            (job_id, job[11]),
        )
        if job[4] == "d" and job[1] in {"J1", "J2"}:
            cur.execute(
                "INSERT INTO feedback(job_id, liked, tags, rated_at) VALUES(?,?,?,0)",
                (job_id, 1 if job[1] == "J2" else 0, ""),
            )
    conn.commit()
    conn.close()

    main.train_model()
    preds = main.predict_unrated()
    ids = {p["id"] for p in preds}
    assert 3 in ids
    assert 4 not in ids


def test_clean_location(main):
    assert main.clean_location("Cincinnati, OH, US") == "Cincinnati, OH"
    assert main.clean_location("Ohio, United States") == "Ohio"


def test_export_likes_formats_fields(main, monkeypatch):
    main.init_db()
    df = pd.DataFrame([
        {
            "site": "t",
            "title": "Dev",
            "company": "ACME",
            "location": "Cincinnati, OH, US",
            "date_posted": "2025-07-20",
            "description": "d",
            "interval": "year",
            "min_amount": 0,
            "max_amount": 0,
            "currency": "USD",
            "job_url": "http://e.com/1",
        }
    ])
    main.save_jobs(df)
    ts = int(pd.Timestamp("2025-07-21").timestamp())
    main.record_feedback(1, True, "", rated_at=ts)

    captured = {}

    def fake_to_excel(self, buf, index=False, engine=None):
        captured["df"] = self.copy()

    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel)
    main.export_likes()

    out = captured["df"]
    assert list(out.columns)[0] == "Source"
    assert out.loc[0, "Location"] == "Cincinnati, OH"
    assert out.loc[0, "Pay Range"] == ""
    assert out.loc[0, "Date Posted"] == "7/20/2025"
    assert out.loc[0, "Date Rated"] == "7/21/2025"


def test_import_custom_csv_marks_match(main):
    main.init_db()
    csv_data = (
        "Company,Job Title,City,Posted Date,Farmed Date,Pay Range,Notes (Mostly missing skills),Hyperlink\n"
        "ACME,Engineer,,6/6/2025,6/10/2025,, ,http://e.com/1\n"
    )
    count = main.import_custom_csv(csv_data.encode())
    assert count == 1
    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT id, site FROM jobs")
    jid, site = cur.fetchone()
    cur.execute("SELECT liked FROM feedback WHERE job_id=?", (jid,))
    fb = cur.fetchone()
    conn.close()
    assert site == "upload"
    assert fb[0] == 1


def test_save_jobs_preserves_feedback_on_dup(main):
    main.init_db()
    df1 = pd.DataFrame([
        {
            "site": "t",
            "title": "Dev",
            "company": "ACME",
            "location": "L",
            "date_posted": "2024-06-01",
            "description": "old",
            "interval": "year",
            "min_amount": 1,
            "max_amount": 2,
            "currency": "USD",
            "job_url": "http://dup.com/1",
        }
    ])
    ids = main.save_jobs(df1)
    main.record_feedback(ids[0], True, "", rated_at=1000)
    df2 = pd.DataFrame([
        {
            "site": "t",
            "title": "Dev",
            "company": "ACME",
            "location": "L",
            "date_posted": "2024-06-05",
            "description": "new",
            "interval": "year",
            "min_amount": 1,
            "max_amount": 2,
            "currency": "USD",
            "job_url": "http://dup.com/1",
        }
    ])
    main.save_jobs(df2)
    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT description,id FROM jobs")
    desc, jid = cur.fetchone()
    cur.execute("SELECT liked, rated_at FROM feedback WHERE job_id=?", (jid,))
    fb = cur.fetchone()
    conn.close()
    assert desc == "new"
    assert fb[0] == 1 and fb[1] == 1000


def test_dedup_action_keeps_uploaded(main):
    main.init_db()
    id1 = main.save_jobs(pd.DataFrame([
        {"site": "upload", "title": "Dev", "company": "A", "job_url": "u1", "description": "x1", "date_posted": "1"}
    ]))[0]
    id2 = main.save_jobs(pd.DataFrame([
        {"site": "t", "title": "Dev", "company": "A", "job_url": "u2", "description": "x2", "date_posted": "0"}
    ]))[0]
    main.dedup_action(f"{id1},{id2}", 1)
    conn = sqlite3.connect(main.DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT site FROM jobs")
    site = cur.fetchone()[0]
    conn.close()
    assert site == "upload"

