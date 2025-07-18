#!/usr/bin/env python3
"""
job_search_ohio.py

Searches for Senior Data Engineer positions in Ohio across multiple job boards
using the JobSpy library and exports results to a CSV file.
"""

import csv
import os

import pandas as pd
from jobspy import scrape_jobs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
    jobs = pd.concat(jobs, ignore_index=True)

    # Output a quick summary
    print(f"Found {len(jobs)} total job postings.")
    print(jobs.head())

    # Save to CSV with similarity information
    output_file = "senior_data_engineer_jobs_ohio.csv"

    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)

        def build_text(df: pd.DataFrame) -> pd.Series:
            return (
                df.get("title", "").fillna("")
                + " "
                + df.get("company", "").fillna("")
                + " "
                + df.get("description", "").fillna("")
            )

        existing_text = build_text(existing)
        new_text = build_text(jobs)
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(existing_text.tolist() + new_text.tolist())
        existing_vecs = vectors[: len(existing_text)]
        new_vecs = vectors[len(existing_text) :]
        sim_matrix = cosine_similarity(new_vecs, existing_vecs)
        best_indices = sim_matrix.argmax(axis=1)
        best_scores = sim_matrix.max(axis=1)
        jobs["most_similar_post"] = existing.iloc[best_indices]["title"].values
        jobs["similarity_score"] = best_scores
        combined = pd.concat([existing, jobs], ignore_index=True)
    else:
        jobs["most_similar_post"] = ""
        jobs["similarity_score"] = None
        combined = jobs

    combined.to_csv(
        output_file,
        quoting=csv.QUOTE_NONNUMERIC,
        escapechar="\\",
        index=False,
    )
    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    main()
