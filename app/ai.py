import json
import logging
from typing import List, Optional, Tuple

import requests

from .config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_REPHRASE_MODEL,
    OLLAMA_ENABLED,
)
from . import main as app_main

logger = logging.getLogger("job_fetch")

REWRITE_PROMPT_TEMPLATE = '''
Rewrite the following job description with an eye toward removing fluff. Strip
company mission statements, marketing language, generic disclaimers and physical
demands. Boil lengthy sentences down to short phrases so only the true
requirements and responsibilities remain.

Return concise Markdown organized into bullet lists. Group alternatives with '/' and skip duplicates or empty sections:

---
**Required Skills**
- item one
- item two

**Bonus Skills**
- item one
- item two

**Soft Skills**
- item one
- item two

**Responsibilities**
- item one
- item two
---

Job posting:
"""{description}"""
'''

CLEAN_COMPANY_PROMPT = (
    "Clean up the company name: '{name}'. Remove corporate suffixes like Inc., LLC or Ltd. "
    "Correct the capitalization and return ONLY the cleaned name without any extra text."
)
CLEAN_TITLE_PROMPT = "Simplify the job title by removing words like full-time or hybrid but keep codes: '{title}'. Return the cleaned title." 
SALARY_PROMPT = (
    "Extract the annual salary range from this text. "
    "If an hourly rate is given, multiply by 40 and 52 to convert. "
    "Respond with two numbers like '50000,70000' or leave blank if unknown:\n{text}"
)

TAG_PROMPT = (
    "List concise, canonical tags for important skills or requirements in this job description. "
    "Each bullet point from the generated summary should become a tag. "
    "Use common abbreviations where appropriate to keep tags short. "
    "Return the tags as a comma separated list only.\n{text}"
)


def ensure_model_downloaded() -> None:
    if not OLLAMA_ENABLED:
        return
    for model in {OLLAMA_EMBED_MODEL, OLLAMA_REPHRASE_MODEL}:
        try:
            requests.post(f"{OLLAMA_BASE_URL}/api/pull", json={"name": model}, timeout=120)
        except Exception as exc:
            logger.info(f"Failed to pull model {model}: {exc}")


def embed_text(text: str) -> List[float]:
    if not OLLAMA_ENABLED:
        return []
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text, "options": {"num_ctx": 4096}},
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("embedding", [])
    except Exception as exc:
        logger.info(f"Embedding failed: {exc}")
        return []


def generate_summary(text: str) -> str:
    if not OLLAMA_ENABLED:
        return ""
    prompt = REWRITE_PROMPT_TEMPLATE.format(description=text)
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_REPHRASE_MODEL, "prompt": prompt, "stream": False, "options": {"num_ctx": 8192}},
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as exc:
        logger.info(f"Summary generation failed: {exc}")
        return ""


def clean_company(name: str) -> str:
    if not OLLAMA_ENABLED or not name:
        return name
    prompt = CLEAN_COMPANY_PROMPT.format(name=name)
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_REPHRASE_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        resp = r.json().get("response", name).strip()
        if (
            len(resp) > len(name) + 2
            or ":" in resp
            or "\n" in resp
        ):
            return name
        return resp
    except Exception as exc:
        logger.info(f"Company cleanup failed: {exc}")
        return name


def clean_title(title: str) -> str:
    if not OLLAMA_ENABLED or not title:
        return title
    prompt = CLEAN_TITLE_PROMPT.format(title=title)
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_REPHRASE_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", title).strip()
    except Exception as exc:
        logger.info(f"Title cleanup failed: {exc}")
        return title


from typing import Optional, Tuple
import re
from difflib import SequenceMatcher


def infer_salary(text: str) -> Optional[Tuple[float, float]]:
    if not OLLAMA_ENABLED or not text:
        return None
    prompt = SALARY_PROMPT.format(text=text)
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_REPHRASE_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        resp = r.json().get("response", "")
    except Exception as exc:
        logger.info(f"Salary extraction failed: {exc}")
        return None
    nums = re.findall(r"\d+(?:\.\d+)?", resp)
    if not nums:
        return None
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    val = float(nums[0])
    return val, val


def generate_tags(text: str) -> List[str]:
    if not OLLAMA_ENABLED or not text:
        return []
    prompt = TAG_PROMPT.format(text=text)
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_REPHRASE_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        resp = r.json().get("response", "")
    except Exception as exc:
        logger.info(f"Tag generation failed: {exc}")
        return []
    tags = [t.strip() for t in resp.split(',') if t.strip()]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in tags:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            unique.append(t)
    return unique


from markdown import markdown
from .utils import sanitize_html


def render_markdown(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r'^`?`?\s*markdown\s*', '', text, flags=re.I)
    lines = [ln for ln in text.splitlines() if ln.strip() != "---"]
    cleaned = "\n".join(lines)
    fixed = []
    seen = set()
    for ln in cleaned.splitlines():
        if ln.lstrip().startswith("-"):
            normalized = ln.strip().lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            if fixed and fixed[-1].strip():
                fixed.append("")
            fixed.append(ln)
        else:
            fixed.append(ln)
    cleaned = "\n".join(fixed)
    html = markdown(cleaned)
    return sanitize_html(html)


from .database import connect_db


def process_all_jobs() -> None:
    if not OLLAMA_ENABLED:
        return
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT id, title, company, description, min_amount, max_amount FROM jobs")
    rows = cur.fetchall()
    for job_id, title, company, desc, min_amt, max_amt in rows:
        if not desc:
            continue
        cur.execute("SELECT 1 FROM summaries WHERE job_id=?", (job_id,))
        have_sum = cur.fetchone()
        cur.execute("SELECT 1 FROM embeddings WHERE job_id=?", (job_id,))
        have_emb = cur.fetchone()
        cur.execute("SELECT 1 FROM clean_jobs WHERE job_id=?", (job_id,))
        have_clean = cur.fetchone()
        cur.execute("SELECT 1 FROM job_tags WHERE job_id=?", (job_id,))
        have_tags = cur.fetchone()
        summary = generate_summary(desc) if not have_sum else None
        text_for_embedding = f"{title}\n{company}\n{desc}"
        embedding = embed_text(text_for_embedding) if not have_emb else None
        clean_data = None
        tags = generate_tags(desc) if not have_tags else None
        if not have_clean:
            salary = infer_salary(desc) or (min_amt, max_amt)
            clean_data = (
                clean_title(title),
                clean_company(company),
                salary[0],
                salary[1],
            )
        if summary is not None:
            cur.execute(
                "INSERT OR IGNORE INTO summaries(job_id, summary) VALUES(?, ?)",
                (job_id, summary),
            )
        if embedding is not None:
            cur.execute(
                "INSERT OR IGNORE INTO embeddings(job_id, embedding) VALUES(?, ?)",
                (job_id, json.dumps(embedding)),
            )
        if clean_data is not None:
            cur.execute(
                "INSERT OR IGNORE INTO clean_jobs(job_id, title, company, min_amount, max_amount) VALUES(?, ?, ?, ?, ?)",
                (job_id, *clean_data),
            )
        if tags is not None and tags:
            cur.executemany(
                "INSERT OR IGNORE INTO job_tags(job_id, tag) VALUES(?, ?)",
                [(job_id, t) for t in tags],
            )
        conn.commit()
    conn.close()


def regenerate_job_ai(job_id: int) -> None:
    if not OLLAMA_ENABLED:
        return
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT title, company, description, min_amount, max_amount FROM jobs WHERE id=?",
        (job_id,),
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return
    title, company, desc, min_amt, max_amt = row
    summary = generate_summary(desc) if desc else ""
    text_for_embedding = f"{title}\n{company}\n{desc}" if desc else ""
    embedding = embed_text(text_for_embedding) if desc else []
    tags = generate_tags(desc) if desc else []
    salary = infer_salary(desc) or (min_amt, max_amt)
    clean_data = (
        clean_title(title),
        clean_company(company),
        salary[0],
        salary[1],
    )
    cur.execute(
        "INSERT OR REPLACE INTO summaries(job_id, summary) VALUES(?, ?)",
        (job_id, summary),
    )
    cur.execute(
        "INSERT OR REPLACE INTO embeddings(job_id, embedding) VALUES(?, ?)",
        (job_id, json.dumps(embedding)),
    )
    cur.execute(
        "INSERT OR REPLACE INTO clean_jobs(job_id, title, company, min_amount, max_amount) VALUES(?, ?, ?, ?, ?)",
        (job_id, *clean_data),
    )
    cur.execute("DELETE FROM job_tags WHERE job_id=?", (job_id,))
    if tags:
        cur.executemany(
            "INSERT INTO job_tags(job_id, tag) VALUES(?, ?)",
            [(job_id, t) for t in tags],
        )
    conn.commit()
    conn.close()

def _likely_related(a: str, b: str) -> bool:
    """Return True if the tags share significant text overlap."""
    a_norm = re.sub(r"[^a-z0-9 ]+", " ", a.lower()).strip()
    b_norm = re.sub(r"[^a-z0-9 ]+", " ", b.lower()).strip()
    if not a_norm or not b_norm:
        return False
    if a_norm in b_norm or b_norm in a_norm:
        return True
    tokens_a = set(a_norm.split())
    tokens_b = set(b_norm.split())
    if tokens_a <= tokens_b or tokens_b <= tokens_a:
        return True
    sm = SequenceMatcher(None, a_norm, b_norm)
    return sm.ratio() >= 0.85


def are_tags_equivalent(tag1: str, tag2: str) -> bool:
    """Use Ollama to decide if two tags mean the same thing."""
    if not OLLAMA_ENABLED:
        return False
    if tag1.lower() == tag2.lower():
        return True
    prompt = (
        "Do the following job skill tags refer to the same concept? "
        f"1. {tag1}\n2. {tag2}\nAnswer yes or no."
    )
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_REPHRASE_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        r.raise_for_status()
        resp = r.json().get("response", "").strip().lower()
        return resp.startswith("yes")
    except Exception as exc:
        logger.info(f"Tag comparison failed: {exc}")
        return False


def consolidate_similar_tags() -> dict[str, str]:
    """Merge similar tags and update the database."""
    if not OLLAMA_ENABLED:
        return {}
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT tag FROM job_tags")
    tags = [r[0] for r in cur.fetchall()]
    replacements: dict[str, str] = {}
    for i, t1 in enumerate(tags):
        for t2 in tags[i + 1 :]:
            if t2 in replacements or t1 in replacements:
                continue
            if not _likely_related(t1, t2):
                continue
            if are_tags_equivalent(t1, t2):
                keep = t1 if len(t1) <= len(t2) else t2
                drop = t2 if keep == t1 else t1
                replacements[drop] = keep
                app_main.log_progress(f"Consolidating '{drop}' -> '{keep}'")
    for old, new in replacements.items():
        cur.execute("UPDATE job_tags SET tag=? WHERE tag=?", (new, old))
        cur.execute("SELECT id, tags FROM feedback WHERE tags LIKE ?", (f"%{old}%",))
        rows = cur.fetchall()
        for fid, tag_str in rows:
            tag_list = [
                new if t.strip() == old else t.strip() for t in str(tag_str).split(",") if t.strip()
            ]
            cur.execute("UPDATE feedback SET tags=? WHERE id=?", (",".join(tag_list), fid))
    conn.commit()
    conn.close()
    return replacements

