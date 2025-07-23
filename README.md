# Job Ranker

This project lets you fetch job postings from multiple sites and quickly swipe through them, storing feedback about which roles are a good fit.

## Usage

1. Build the Docker image using the provided script which automatically
   increments the build number:

```bash
./build.sh
```

2. Run the container:

```bash
docker run -p 8000:8000 job-ranker
```

3. Alternatively start it with Docker Compose which persists the database under `/config` and lets you configure environment variables like an Ollama API endpoint:

```yaml
version: '3'
services:
  app:
    image: job-ranker
    ports:
      - "8000:8000"
    environment:
      - DATABASE=/config/jobs.db
      - OLLAMA_BASE_URL=http://ollama:11434
      # Optional: use different models for embeddings and rephrasing
      - OLLAMA_EMBED_MODEL=llama3
      - OLLAMA_REPHRASE_MODEL=llama3
    volumes:
      - ./data:/config
```

`OLLAMA_EMBED_MODEL` and `OLLAMA_REPHRASE_MODEL` default to the value of
`OLLAMA_MODEL` (or `llama3` if unset) so existing setups continue to work.

Then run:

```bash
docker compose up
```
4. Visit `http://localhost:8000` to enter search terms and fetch jobs.
   After results are saved you can swipe through postings one at a time
   using the green ✓ button to mark a good match or the red ✖ button to
   reject it. After choosing either option you will be shown a list of
  tags extracted from the posting and can select which ones influenced
  your decision. These tags are stored along with your feedback and
  incorporated when training the matching model. Each algorithm is now
  trained twice: once using only the embeddings and again with the tag
  vectors appended. The statistics page lists accuracy metrics for all
  variants so you can compare their performance.
   If any job boards fail to fetch (for example due to network restrictions) the
  progress screen now shows detailed logs so you can see which sources were skipped.
  A statistics page summarizes stored jobs and reports model accuracy, precision and recall using a 10% holdout set so you can track its performance. It also lists how many roles have been rated positively, negatively or not at all.

The application uses [JobSpy](https://pypi.org/project/python-jobspy/) to scrape
job boards and FastAPI for the web interface. If LinkedIn descriptions are
missing, the scraper now passes `linkedin_fetch_description=True` which visits
each LinkedIn job page to pull the full text. When Ollama is enabled the
AI-generated summary strips company mission statements or disclaimers and
separates **Required Skills** from **Bonus Skills** so it's clear which
technologies are mandatory. Embeddings include the title and company name along
with the full description for better ranking accuracy.

## Documentation

See [docs/architecture.md](docs/architecture.md) for a walkthrough of how the application is structured and how to set up a development environment.
