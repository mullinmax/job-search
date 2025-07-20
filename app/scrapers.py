import os
import tempfile
import time
from typing import List
import pandas as pd
from jobspy import scrape_jobs

RATE_LIMIT = 15  # seconds between scrapers


def scrape_with_jobspy(site: str, search_term: str, google_term: str, location: str, results: int) -> pd.DataFrame:
    """Fetch jobs using the JobSpy library."""
    df = scrape_jobs(
        site_name=site,
        search_term=search_term,
        google_search_term=google_term,
        location=location,
        country_indeed="USA",
        results_wanted=results,
        hours_old=168,
        description_format="html",
        linkedin_fetch_description=site == "linkedin",
        verbose=1,
    )
    return df


def scrape_with_linkedin(search_term: str, location: str, limit: int = 50) -> pd.DataFrame:
    """Fetch jobs using linkedin-jobs-scraper if available."""
    try:
        from linkedin_jobs_scraper import LinkedinScraper
        from linkedin_jobs_scraper.query import Query, QueryOptions
        from linkedin_jobs_scraper.events import Events
    except Exception:
        return pd.DataFrame()

    jobs: List[dict] = []

    def on_data(data):
        jobs.append({
            "site": "linkedin-extra",
            "title": data.title,
            "company": data.company,
            "location": data.place,
            "date_posted": data.date,
            "description": data.description,
            "interval": "",
            "min_amount": None,
            "max_amount": None,
            "currency": None,
            "job_url": data.link,
        })

    scraper = LinkedinScraper(headless=True, max_workers=1, slow_mo=2)
    scraper.on(Events.DATA, on_data)

    queries = [Query(search_term, options=QueryOptions(locations=[location], limit=limit))]
    try:
        scraper.run(queries)
    except Exception:
        return pd.DataFrame()

    time.sleep(1)
    return pd.DataFrame(jobs)


def scrape_with_jobfunnel(search_term: str, location: str, limit: int = 50) -> pd.DataFrame:
    """Fetch jobs using JobFunnel CLI."""
    try:
        from jobfunnel.config.cli import get_config_manager
        from jobfunnel.backend.jobfunnel import JobFunnel
    except Exception:
        return pd.DataFrame()

    temp_dir = tempfile.mkdtemp()
    cfg = {
        "master_csv_file": os.path.join(temp_dir, "master.csv"),
        "block_list_file": os.path.join(temp_dir, "block.json"),
        "duplicates_list_file": os.path.join(temp_dir, "dupes.json"),
        "cache_folder": os.path.join(temp_dir, "cache"),
        "log_file": os.path.join(temp_dir, "funnel.log"),
        "log_level": 20,
        "no_scrape": False,
        "search": {
            "keywords": search_term,
            "province_or_state": "",
            "city": location,
            "radius": 25,
            "similar_results": False,
            "max_listing_days": 7,
            "company_block_list": [],
            "locale": "CANADA_ENGLISH",
            "providers": ["INDEED"],
            "remoteness": "IN_PERSON",
        },
        "delay": {
            "max_duration": 20,
            "min_duration": 15,
            "algorithm": "CONSTANT",
            "random": False,
            "converging": False,
        },
    }

    cfg_mgr = get_config_manager(cfg)
    job_funnel = JobFunnel(cfg_mgr)
    try:
        job_funnel.run()
    except Exception:
        return pd.DataFrame()

    csv_path = cfg_mgr.master_csv_file
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df["site"] = "jobfunnel"
    return df
