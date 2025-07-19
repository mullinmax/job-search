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
