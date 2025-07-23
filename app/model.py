import json
from typing import Dict, List, Optional, Tuple, Set, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from .database import connect_db


class _ModelWrapper:
    """Wrapper that standardizes prediction API across model types."""

    def __init__(
        self, name: str, model: Any, description: str, *, use_tags: bool
    ) -> None:
        self.name = name
        self.model = model
        self.description = description
        self.use_tags = use_tags
        self._cluster_labels: Dict[int, int] | None = None
        self.expected_dim: int | None = None

    def _combine(self, X_emb: np.ndarray, tags: np.ndarray) -> np.ndarray:
        if self.use_tags and tags.size:
            return np.hstack([X_emb, tags])
        return X_emb

    def fit(self, X_emb: np.ndarray, tags: np.ndarray, y: np.ndarray) -> None:
        X = self._combine(X_emb, tags)
        self.expected_dim = X.shape[1]
        if isinstance(self.model, KMeans):
            self.model.fit(X)
            clusters = self.model.predict(X)
            labels: Dict[int, int] = {}
            for c in np.unique(clusters):
                mask = clusters == c
                pos = int(np.sum(y[mask] == 1))
                neg = int(np.sum(y[mask] == 0))
                labels[c] = 1 if pos >= neg else 0
            self._cluster_labels = labels
        else:
            self.model.fit(X, y)

    def predict(self, X_emb: np.ndarray, tags: np.ndarray) -> np.ndarray:
        X = self._combine(X_emb, tags)
        if isinstance(self.model, KMeans):
            clusters = self.model.predict(X)
            return np.array([self._cluster_labels.get(int(c), 0) for c in clusters])
        return self.model.predict(X)

    def predict_proba(self, X_emb: np.ndarray, tags: np.ndarray) -> np.ndarray:
        X = self._combine(X_emb, tags)
        if isinstance(self.model, KMeans):
            preds = self.predict(X_emb, tags)
            return np.vstack([1 - preds, preds]).T.astype(float)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            probs = 1 / (1 + np.exp(-scores))
            return np.vstack([1 - probs, probs]).T
        preds = self.predict(X_emb, tags)
        return np.vstack([1 - preds, preds]).T.astype(float)


_model: _ModelWrapper | None = None
_models: List[_ModelWrapper] = []
_model_metrics: Dict[str, Dict[str, Any]] = {}
_tag_binarizer: MultiLabelBinarizer | None = None
_eval_set: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
_train_set: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
_known_tags: Set[str] = set()


def train_model() -> None:
    """Train multiple models and keep the one with best recall."""
    global _model, _models, _model_metrics, _tag_binarizer, _eval_set, _train_set, _known_tags
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
        _train_set = None
        _known_tags = set()
        _models = []
        _model_metrics = {}
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
            tags.extend(t.strip() for t in str(fb_tags).split(",") if t.strip())
        if job_tags:
            tags.extend(t.strip() for t in str(job_tags).split(",") if t.strip())
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
        _train_set = None
        _known_tags = set()
        _models = []
        _model_metrics = {}
        return
    _tag_binarizer = MultiLabelBinarizer()
    tag_matrix = _tag_binarizer.fit_transform(tag_sets)
    _known_tags = set(_tag_binarizer.classes_)

    X_emb_np = np.array(X_emb)
    y = np.array(y_list)
    _train_set = (X_emb_np, tag_matrix, y)
    if len(set(y)) < 2:
        _model = None
        _tag_binarizer = None
        _eval_set = None
        _train_set = None
        _known_tags = set()
        _models = []
        _model_metrics = {}
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
    X_train_emb = X_emb_np[train_idx]
    X_train_tags = tag_matrix[train_idx]
    y_train = y[train_idx]
    X_test_emb = X_emb_np[test_idx]
    X_test_tags = tag_matrix[test_idx]
    y_test = y[test_idx]

    base_models = [
        ("logreg", LogisticRegression(max_iter=1000), "Logistic Regression"),
        ("forest", RandomForestClassifier(n_estimators=100), "Random Forest"),
        ("svm", SVC(kernel="linear", probability=True), "Support Vector Machine"),
        (
            "mlp",
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500),
            "Neural Network",
        ),
        (
            "kmeans",
            KMeans(n_clusters=2, n_init="auto", random_state=0),
            "KMeans Clustering",
        ),
    ]
    candidates: List[_ModelWrapper] = []
    for name, model, desc in base_models:
        params = model.get_params() if hasattr(model, "get_params") else {}
        candidates.append(
            _ModelWrapper(
                f"{name}_tags",
                model.__class__(**params),
                f"{desc} + tags",
                use_tags=True,
            )
        )
        candidates.append(
            _ModelWrapper(
                f"{name}_base",
                model.__class__(**params),
                f"{desc} (no tags)",
                use_tags=False,
            )
        )

    metrics: Dict[str, Dict[str, Any]] = {}
    for cand in candidates:
        cand.fit(X_train_emb, X_train_tags, y_train)
        eval_emb = X_test_emb if len(X_test_emb) else X_train_emb
        eval_tags = X_test_tags if len(X_test_tags) else X_train_tags
        eval_y = y_test if len(X_test_emb) else y_train
        preds = cand.predict(eval_emb, eval_tags)
        tp = int(np.sum((preds == 1) & (eval_y == 1)))
        tn = int(np.sum((preds == 0) & (eval_y == 0)))
        fp = int(np.sum((preds == 1) & (eval_y == 0)))
        fn = int(np.sum((preds == 0) & (eval_y == 1)))
        total = len(eval_y)
        acc = (tp + tn) / total if total else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        metrics[cand.name] = {
            "name": cand.name,
            "description": cand.description,
            "total": total,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }

    _models = candidates
    _model_metrics = metrics
    best_key = max(metrics.values(), key=lambda m: (m["recall"], m["f1"]))["name"]
    _model = next(c for c in candidates if c.name == best_key)

    _eval_set = (X_test_emb, X_test_tags, y_test)


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
    expected_dim = _model.expected_dim
    for job_id, title, company, emb, tags in rows:
        vec = json.loads(emb)
        if not vec:
            continue
        tag_list = (
            [t.strip() for t in str(tags).split(",") if t.strip()] if tags else []
        )
        if _tag_binarizer is not None:
            filtered = [t for t in tag_list if t in _known_tags]
            tag_vec = _tag_binarizer.transform([filtered])[0]
        else:
            tag_vec = np.array([])
        feat = np.hstack([vec, tag_vec]) if _model.use_tags and tag_vec.size else vec
        if expected_dim is not None and len(feat) != expected_dim:
            continue
        try:
            prob = float(
                _model.predict_proba(np.array([vec]), np.array([tag_vec]))[0, 1]
            )
            match = bool(_model.predict(np.array([vec]), np.array([tag_vec]))[0])
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
    tag_list = [t.strip() for t in str(tags).split(",") if t.strip()] if tags else []
    if _tag_binarizer is not None:
        filtered = [t for t in tag_list if t in _known_tags]
        tag_vec = _tag_binarizer.transform([filtered])[0]
    else:
        tag_vec = np.array([])
    feat = np.hstack([vec, tag_vec]) if _model.use_tags and tag_vec.size else vec
    expected_dim = _model.expected_dim
    if not vec or (expected_dim is not None and len(feat) != expected_dim):
        return None
    try:
        prob = float(_model.predict_proba(np.array([vec]), np.array([tag_vec]))[0, 1])
        match = bool(_model.predict(np.array([vec]), np.array([tag_vec]))[0])
    except Exception:
        return None
    return match, prob


def evaluate_model() -> Dict[str, float | int]:
    """Return accuracy metrics using available data."""
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

    if _eval_set is not None and len(_eval_set[2]) > 0:
        X_emb_eval, tag_eval, y_eval = _eval_set
    elif _train_set is not None:
        X_emb_eval, tag_eval, y_eval = _train_set
    else:
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

    metrics = _model_metrics.get(_model.name)
    if metrics is None:
        preds = _model.predict(X_emb_eval, tag_eval)
        tp = int(np.sum((preds == 1) & (y_eval == 1)))
        tn = int(np.sum((preds == 0) & (y_eval == 0)))
        fp = int(np.sum((preds == 1) & (y_eval == 0)))
        fn = int(np.sum((preds == 0) & (y_eval == 1)))
        total = len(y_eval)
        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        metrics = {
            "name": _model.name,
            "description": _model.description,
            "total": total,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    result = dict(metrics)
    result["models"] = list(_model_metrics.values())
    result["model_name"] = _model.description
    return result


def find_outliers(threshold: float = 0.75) -> List[Dict]:
    """Return rated jobs the models disagree with."""
    if not _models:
        return []
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT j.id, j.title, j.company, e.embedding,
               GROUP_CONCAT(t.tag), f.liked, MIN(f.rated_at)
        FROM jobs j
        JOIN feedback f ON j.id = f.job_id
        JOIN embeddings e ON j.id = e.job_id
        LEFT JOIN job_tags t ON j.id = t.job_id
        GROUP BY j.id
        """
    )
    rows = cur.fetchall()
    conn.close()

    results = []
    for job_id, title, company, emb, tags, liked, rated_at in rows:
        try:
            vec = json.loads(emb)
        except Exception:
            continue
        tag_list = (
            [t.strip() for t in str(tags).split(",") if t.strip()] if tags else []
        )
        if _tag_binarizer is not None:
            filtered = [t for t in tag_list if t in _known_tags]
            tag_vec = _tag_binarizer.transform([filtered])[0]
        else:
            tag_vec = np.array([])
        counts = {1: 0, 0: 0}
        for mdl in _models:
            feat = mdl._combine(np.array([vec]), np.array([tag_vec]))
            if mdl.expected_dim is not None and feat.shape[1] != mdl.expected_dim:
                continue
            try:
                pred = int(mdl.predict(np.array([vec]), np.array([tag_vec]))[0])
            except Exception:
                continue
            counts[pred] += 1
        total = counts[1] + counts[0]
        if total == 0:
            continue
        majority = counts[1] / total
        predicted = majority >= 0.5
        if predicted != bool(liked) and (
            majority >= threshold or 1 - majority >= threshold
        ):
            results.append(
                {
                    "id": job_id,
                    "title": title,
                    "company": company,
                    "liked": bool(liked),
                    "predicted": predicted,
                    "confidence": majority if predicted else 1 - majority,
                    "rated_at": rated_at,
                }
            )
    results.sort(key=lambda r: r["confidence"], reverse=True)
    return results
