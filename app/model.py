import json
from .database import connect_db
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer

from .config import DATABASE

_model: LogisticRegression | None = None
_tag_binarizer: MultiLabelBinarizer | None = None
_eval_set: Tuple[np.ndarray, np.ndarray] | None = None


def train_model() -> None:
    """Train a logistic regression model from feedback, embeddings and tags."""
    global _model, _tag_binarizer
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT f.liked, e.embedding, f.tags, GROUP_CONCAT(t.tag)
        FROM feedback f
        JOIN embeddings e ON f.job_id = e.job_id
        LEFT JOIN job_tags t ON f.job_id = t.job_id
        GROUP BY f.id
        """
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        _model = None
        _tag_binarizer = None
        _eval_set = None
        return
    X_emb: List[List[float]] = []
    tag_sets: List[List[str]] = []
    y_list: List[int] = []
    for liked, emb, fb_tags, job_tags in rows:
        vec = json.loads(emb)
        if not vec:
            continue
        if X_emb and len(vec) != len(X_emb[0]):
            continue
        tags = []
        if fb_tags:
            tags.extend(t.strip() for t in str(fb_tags).split(',') if t.strip())
        if job_tags:
            tags.extend(t.strip() for t in str(job_tags).split(',') if t.strip())
        seen = set()
        uniq = []
        for t in tags:
            tl = t.lower()
            if tl not in seen:
                seen.add(tl)
                uniq.append(t)
        X_emb.append(vec)
        tag_sets.append(uniq)
        y_list.append(liked)
    if not X_emb:
        _model = None
        _tag_binarizer = None
        _eval_set = None
        return
    _tag_binarizer = MultiLabelBinarizer()
    tag_matrix = _tag_binarizer.fit_transform(tag_sets)
    X = np.hstack([np.array(X_emb), tag_matrix])
    y = np.array(y_list)
    if len(set(y)) < 2:
        _model = None
        _tag_binarizer = None
        _eval_set = None
        return
    rng = np.random.default_rng(0)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    test_size = max(1, int(len(y) * 0.1)) if len(y) > 1 else 0
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    if len(train_idx) < 2 or len(set(y[train_idx])) < 2:
        train_idx = idx
        test_idx = []
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    _model = LogisticRegression(max_iter=1000)
    _model.fit(X_train, y_train)
    _eval_set = (X_test, y_test)


def predict_unrated() -> List[Dict]:
    """Return predictions for unrated jobs sorted by match then confidence."""
    if _model is None:
        return []
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT j.id, j.title, j.company, e.embedding,
               GROUP_CONCAT(t.tag)
        FROM jobs j
        JOIN embeddings e ON j.id = e.job_id
        LEFT JOIN job_tags t ON j.id = t.job_id
        WHERE j.id NOT IN (SELECT job_id FROM feedback)
        GROUP BY j.id
        """
    )
    rows = cur.fetchall()
    conn.close()
    results = []
    expected_dim = getattr(_model, "n_features_in_", None)
    for job_id, title, company, emb, tags in rows:
        vec = json.loads(emb)
        if not vec:
            continue
        tag_list = [t.strip() for t in str(tags).split(',') if t.strip()] if tags else []
        if _tag_binarizer is not None:
            tag_vec = _tag_binarizer.transform([tag_list])[0]
            feat = np.hstack([vec, tag_vec])
        else:
            feat = vec
        if expected_dim is not None and len(feat) != expected_dim:
            continue
        try:
            prob = float(_model.predict_proba([feat])[0, 1])
            match = bool(_model.predict([feat])[0])
        except Exception:
            continue
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
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT e.embedding, GROUP_CONCAT(t.tag)
        FROM embeddings e
        LEFT JOIN job_tags t ON e.job_id = t.job_id
        WHERE e.job_id=?
        GROUP BY e.job_id
        """,
        (job_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    try:
        vec = json.loads(row[0])
    except Exception:
        return None
    tags = row[1] if row and len(row) > 1 else None
    tag_list = [t.strip() for t in str(tags).split(',') if t.strip()] if tags else []
    if _tag_binarizer is not None:
        tag_vec = _tag_binarizer.transform([tag_list])[0]
        feat = np.hstack([vec, tag_vec])
    else:
        feat = vec
    expected_dim = getattr(_model, "n_features_in_", None)
    if not vec or (expected_dim is not None and len(feat) != expected_dim):
        return None
    try:
        prob = float(_model.predict_proba([feat])[0, 1])
        match = bool(_model.predict([feat])[0])
    except Exception:
        return None
    return match, prob


def evaluate_model() -> Dict[str, float | int]:
    """Return accuracy metrics using the holdout set from training."""
    if _model is None or _eval_set is None:
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

    X_test, y_test = _eval_set
    if len(y_test) == 0:
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

    preds = _model.predict(X_test)
    tp = int(np.sum((preds == 1) & (y_test == 1)))
    tn = int(np.sum((preds == 0) & (y_test == 0)))
    fp = int(np.sum((preds == 1) & (y_test == 0)))
    fn = int(np.sum((preds == 0) & (y_test == 1)))
    total = len(y_test)

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
