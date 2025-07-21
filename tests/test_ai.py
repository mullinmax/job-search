import sys
from pathlib import Path
import types
import importlib
import sys
import sqlite3
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_response(text):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": text}

    return Resp()


@pytest.fixture
def ai_module(monkeypatch):
    # Provide a minimal stub for app.main to avoid circular imports
    stub_main = types.ModuleType("app.main")
    monkeypatch.setitem(sys.modules, "app.main", stub_main)
    ai = importlib.import_module("app.ai")
    importlib.reload(ai)
    yield ai
    # Remove imported modules so later tests can reload with correct env
    for mod in ["app.ai", "app.config"]:
        sys.modules.pop(mod, None)


def test_clean_company_rejects_long(ai_module, monkeypatch):
    ai = ai_module
    monkeypatch.setattr(ai, "OLLAMA_ENABLED", True)
    monkeypatch.setattr(
        ai.requests,
        "post",
        lambda *a, **k: make_response(
            "Hired by Matrix Explanation: 'Inc.' is a corporate suffix"
        ),
    )
    assert ai.clean_company("Hired by Matrix Inc.") == "Hired by Matrix Inc."


def test_clean_company_accepts_clean(ai_module, monkeypatch):
    ai = ai_module
    monkeypatch.setattr(ai, "OLLAMA_ENABLED", True)
    monkeypatch.setattr(
        ai.requests,
        "post",
        lambda *a, **k: make_response("JPMorgan Chase"),
    )
    assert ai.clean_company("JPMorgan Chase & Co.") == "JPMorgan Chase"


def test_regenerate_job_ai_updates_clean_data(tmp_path, monkeypatch):
    db_path = tmp_path / "t.db"
    stub_main = types.ModuleType("app.main")
    stub_main.DATABASE = str(db_path)
    monkeypatch.setitem(sys.modules, "app.main", stub_main)

    import importlib
    ai = importlib.import_module("app.ai")
    importlib.reload(ai)
    db = importlib.import_module("app.db")
    importlib.reload(db)

    monkeypatch.setattr(ai, "OLLAMA_ENABLED", True)
    monkeypatch.setattr(ai, "generate_summary", lambda t: "sum")
    monkeypatch.setattr(ai, "embed_text", lambda t: [0.0])
    monkeypatch.setattr(ai, "clean_title", lambda t: t)
    monkeypatch.setattr(ai, "clean_company", lambda t: "Clean")
    monkeypatch.setattr(ai, "infer_salary", lambda t: (1.0, 2.0))

    db.init_db()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO jobs(site,title,company,location,date_posted,description,interval,currency,job_url)
        VALUES('s','t','C LLC','L','d','desc','year','USD','u')
        """
    )
    job_id = cur.lastrowid
    conn.commit()
    conn.close()

    ai.regenerate_job_ai(job_id)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT company, min_amount, max_amount FROM clean_jobs WHERE job_id=?", (job_id,))
    row = cur.fetchone()
    conn.close()

    assert row == ("Clean", 1.0, 2.0)

    for mod in ["app.ai", "app.db", "app.main", "app.model", "app.config"]:
        sys.modules.pop(mod, None)
