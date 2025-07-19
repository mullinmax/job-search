# Job Classifier

This project lets you fetch job postings from multiple sites and quickly swipe through them, storing feedback about whether each role is a good match.

## Usage

1. Build the Docker image:

```bash
docker build -t job-classifier .
```

2. Run the container:

```bash
docker run -p 8000:8000 job-classifier
```

3. Visit `http://localhost:8000` to enter search terms and fetch jobs.
   After results are saved you can swipe through postings one at a time
   using the green ✓ button to mark a good match or the red ✖ button to
   reject it. When rejecting a job you will be asked to provide a reason.
   Feedback is stored in a SQLite database inside the container so you can review liked and rejected jobs later.

The application uses [JobSpy](https://pypi.org/project/python-jobspy/) to scrape
job boards and FastAPI for the web interface.
