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
