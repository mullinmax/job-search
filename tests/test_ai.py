import sys
from pathlib import Path
import types
import importlib
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


def test_embedding_includes_title_company(ai_module, tmp_path, monkeypatch):
    ai = ai_module
    stub_main = sys.modules["app.main"]
    db_path = tmp_path / "db.db"
    stub_main.DATABASE = str(db_path)

    import sqlite3

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE jobs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            company TEXT,
            description TEXT,
            min_amount REAL,
            max_amount REAL
        )
        """
    )
    cur.execute("CREATE TABLE summaries(job_id INTEGER PRIMARY KEY, summary TEXT)")
    cur.execute(
        "CREATE TABLE embeddings(job_id INTEGER PRIMARY KEY, embedding TEXT)"
    )
    cur.execute(
        "CREATE TABLE clean_jobs(job_id INTEGER PRIMARY KEY, title TEXT, company TEXT, min_amount REAL, max_amount REAL)"
    )
    cur.execute(
        "CREATE TABLE job_tags(job_id INTEGER, tag TEXT, PRIMARY KEY(job_id, tag))"
    )
    cur.execute(
        "INSERT INTO jobs(id,title,company,description,min_amount,max_amount) VALUES(1,'Data Scientist','ACME','Analyze data',1,2)"
    )
    conn.commit()
    conn.close()

    captured = {}

    def fake_embed(text: str):
        captured["text"] = text
        return []

    monkeypatch.setattr(ai, "OLLAMA_ENABLED", True)
    monkeypatch.setattr(ai, "embed_text", fake_embed)
    monkeypatch.setattr(ai, "generate_summary", lambda x: "")
    monkeypatch.setattr(ai, "generate_tags", lambda x: [])
    monkeypatch.setattr(ai, "clean_title", lambda x: x)
    monkeypatch.setattr(ai, "clean_company", lambda x: x)
    monkeypatch.setattr(ai, "infer_salary", lambda x: None)

    ai.process_all_jobs()

    text = captured.get("text", "")
    assert "Data Scientist" in text
    assert "ACME" in text


def test_process_all_jobs_generates_tags(ai_module, tmp_path, monkeypatch):
    ai = ai_module
    stub_main = sys.modules["app.main"]
    db_path = tmp_path / "tags.db"
    stub_main.DATABASE = str(db_path)

    import sqlite3

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE jobs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            company TEXT,
            description TEXT,
            min_amount REAL,
            max_amount REAL
        )
        """
    )
    cur.execute("CREATE TABLE summaries(job_id INTEGER PRIMARY KEY, summary TEXT)")
    cur.execute(
        "CREATE TABLE embeddings(job_id INTEGER PRIMARY KEY, embedding TEXT)"
    )
    cur.execute(
        "CREATE TABLE clean_jobs(job_id INTEGER PRIMARY KEY, title TEXT, company TEXT, min_amount REAL, max_amount REAL)"
    )
    cur.execute(
        "CREATE TABLE job_tags(job_id INTEGER, tag TEXT, PRIMARY KEY(job_id, tag))"
    )
    cur.execute(
        "INSERT INTO jobs(id,title,company,description,min_amount,max_amount) VALUES(1,'Dev','C','do stuff',1,2)"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(ai, "OLLAMA_ENABLED", True)
    monkeypatch.setattr(ai, "embed_text", lambda x: [])
    monkeypatch.setattr(ai, "generate_summary", lambda x: "")
    monkeypatch.setattr(ai, "generate_tags", lambda x: ["Python", "SQL"])
    monkeypatch.setattr(ai, "clean_title", lambda x: x)
    monkeypatch.setattr(ai, "clean_company", lambda x: x)
    monkeypatch.setattr(ai, "infer_salary", lambda x: None)

    ai.process_all_jobs()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT tag FROM job_tags ORDER BY tag")
    tags = [r[0] for r in cur.fetchall()]
    conn.close()
    assert tags == ["Python", "SQL"]
