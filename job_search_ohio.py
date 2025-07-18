#!/usr/bin/env python3
"""
job_search_ohio.py

Searches for Senior Data Engineer positions in Ohio across multiple job boards
using the JobSpy library and exports results to a CSV file.
"""

import csv
from jobspy import scrape_jobs


def main():
    # Define search parameters
    site_names = [
        "indeed",
        "linkedin",
        "zip_recruiter",
        "glassdoor",
        "google"
    ]
    search_term = "senior data engineer"
    location = "Ohio, USA"
    # Google Jobs requires its own search string
    google_search_term = "senior data engineer jobs in Ohio"

    # Collect jobs from each site individually. If one site fails, continue
    jobs = []
    for site in site_names:
        try:
            data = scrape_jobs(
                site_name=site,
                search_term=search_term,
                google_search_term=google_search_term,
                location=location,
                country_indeed="USA",        # required for Indeed & Glassdoor
                results_wanted=50,           # number of results per site
                hours_old=168,               # only listings posted in the last 7 days
                description_format="markdown",
                verbose=1
            )
            jobs.append(data)
        except Exception as exc:
            print(f"Skipping {site} due to error: {exc}")

    if not jobs:
        print("No job data collected.")
        return

    # Combine results
    import pandas as pd
    jobs = pd.concat(jobs, ignore_index=True)

    # Output a quick summary
    print(f"Found {len(jobs)} total job postings.")
    print(jobs.head())

    # Save to CSV
    output_file = "senior_data_engineer_jobs_ohio.csv"
    jobs.to_csv(
        output_file,
        quoting=csv.QUOTE_NONNUMERIC,
        escapechar="\\",
        index=False
    )
    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    main()
