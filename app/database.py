import os
import sqlite3
from .config import DATABASE


def _get_db_path() -> str:
    """Return the current database path."""
    try:
        from . import main as app_main
        return getattr(app_main, "DATABASE", DATABASE)
    except Exception:
        return os.environ.get("DATABASE", DATABASE)

def connect_db() -> sqlite3.Connection:
    """Return a SQLite connection with a generous timeout."""
    return sqlite3.connect(_get_db_path(), timeout=30)
