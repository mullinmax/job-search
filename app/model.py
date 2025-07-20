import json
import sqlite3
from typing import Dict, List

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
    X = np.array([json.loads(r[1]) for r in rows])
    y = np.array([r[0] for r in rows])
    _model = LogisticRegression(max_iter=1000)
    _model.fit(X, y)


def predict_unrated() -> List[Dict]:
    """Return predictions for unrated jobs sorted by confidence."""
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
    for job_id, title, company, emb in rows:
        vec = json.loads(emb)
        prob = float(_model.predict_proba([vec])[0, 1])
        results.append({"id": job_id, "title": title, "company": company, "confidence": prob})
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


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

    X = np.array([json.loads(r[1]) for r in rows])
    y = np.array([r[0] for r in rows])
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
