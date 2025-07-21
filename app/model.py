import json
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import DATABASE

_model: LogisticRegression | None = None


def train_model() -> None:
    """Train a logistic regression model from feedback and embeddings."""
    global _model
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT f.liked, e.embedding
        FROM feedback f
        JOIN embeddings e ON f.job_id = e.job_id
        """
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        _model = None
        return
    # Filter out invalid or empty embeddings
    X_list = []
    y_list = []
    for liked, emb in rows:
        vec = json.loads(emb)
        if not vec:
            continue
        if X_list and len(vec) != len(X_list[0]):
            continue
        X_list.append(vec)
        y_list.append(liked)
    if not X_list:
        _model = None
        return
    X = np.array(X_list)
    y = np.array(y_list)
    # Skip training until we have at least two classes to avoid sklearn errors
    if len(set(y)) < 2:
        _model = None
        return
    _model = LogisticRegression(max_iter=1000)
    _model.fit(X, y)


def predict_unrated() -> List[Dict]:
    """Return predictions for unrated jobs sorted by match then confidence."""
    if _model is None:
        return []
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT j.id, j.title, j.company, e.embedding
        FROM jobs j
        JOIN embeddings e ON j.id = e.job_id
        WHERE j.id NOT IN (SELECT job_id FROM feedback)
        """
    )
    rows = cur.fetchall()
    conn.close()
    results = []
    expected_dim = getattr(_model, "n_features_in_", None)
    for job_id, title, company, emb in rows:
        vec = json.loads(emb)
        if not vec:
            continue
        if expected_dim is not None and len(vec) != expected_dim:
            continue
        try:
            prob = float(_model.predict_proba([vec])[0, 1])
        except Exception:
            continue
        match = prob >= 0.5
        results.append(
            {
                "id": job_id,
                "title": title,
                "company": company,
                "confidence": prob,
                "match": match,
            }
        )
    results.sort(key=lambda x: (x["match"], x["confidence"]), reverse=True)
    return results


def predict_job(job_id: int) -> Optional[Tuple[bool, float]]:
    """Return match prediction and probability for a single job."""
    if _model is None:
        return None
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT embedding FROM embeddings WHERE job_id=?", (job_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    try:
        vec = json.loads(row[0])
    except Exception:
        return None
    expected_dim = getattr(_model, "n_features_in_", None)
    if not vec or (expected_dim is not None and len(vec) != expected_dim):
        return None
    try:
        prob = float(_model.predict_proba([vec])[0, 1])
    except Exception:
        return None
    return prob >= 0.5, prob


def evaluate_model() -> Dict[str, float | int]:
    """Return basic accuracy stats comparing predictions to feedback."""
    if _model is None:
        return {
            "total": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT f.liked, e.embedding
        FROM feedback f
        JOIN embeddings e ON f.job_id = e.job_id
        """
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return {
            "total": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    expected_dim = getattr(_model, "n_features_in_", None)
    X_list = []
    y_list = []
    for liked, emb in rows:
        vec = json.loads(emb)
        if not vec:
            continue
        if expected_dim is not None and len(vec) != expected_dim:
            continue
        X_list.append(vec)
        y_list.append(liked)
    if not X_list:
        return {
            "total": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    X = np.array(X_list)
    y = np.array(y_list)
    preds = _model.predict(X)

    tp = int(np.sum((preds == 1) & (y == 1)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    total = len(y)

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
    }
