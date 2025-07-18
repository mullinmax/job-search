# Job Search Ohio

This repository contains a simple Python script to search for **Senior Data Engineer** positions in Ohio using the [JobSpy](https://github.com/pavinjohnson/python-jobspy) library.

The script queries multiple job boards and saves the results to a CSV file for further analysis.

## Requirements

- Python 3.10+
- `python-jobspy` package

Install dependencies via pip:

```bash
pip install -U python-jobspy
```

## Usage

Run the script to fetch the latest listings:

```bash
python job_search_ohio.py
```

This will search Indeed, LinkedIn, ZipRecruiter, Glassdoor, and Google Jobs for "Senior Data Engineer" roles located in Ohio. The results will be written to `senior_data_engineer_jobs_ohio.csv` in the repository directory.

Feel free to modify the script or search parameters to suit your needs.
