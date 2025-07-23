import sqlite3
import importlib
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import types

# Provide a minimal stub for app.main to avoid circular imports
stub_main = types.ModuleType("app.main")
sys.modules["app.main"] = stub_main

import app.db as db

@pytest.fixture
def tag_db(tmp_path, monkeypatch):
    db_path = tmp_path / "tags.db"
    monkeypatch.setenv("DATABASE", str(db_path))
    importlib.reload(db)
    db.init_db()
    yield db


def test_tag_significance_uses_feedback_tags(tag_db):
    conn = db.connect_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO jobs(site,title,company,location,date_posted,description,interval,min_amount,max_amount,currency,job_url) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        ("s", "t", "c", "l", "d", "desc", "year", 1, 2, "USD", "url"),
    )
    job_id = cur.lastrowid
    cur.execute(
        "INSERT INTO feedback(job_id, liked, tags, rated_at) VALUES(?,?,?,0)",
        (job_id, 1, "Python",),
    )
    conn.commit()
    conn.close()

    stats = tag_db.tag_significance()
    tags = {s["tag"] for s in stats}
    assert "Python" in tags


