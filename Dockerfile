FROM python:3.12-slim
WORKDIR /app
ENV DATABASE=/config/jobs.db
VOLUME ["/config"]

ARG BUILD_NUMBER=dev
ENV BUILD_NUMBER=${BUILD_NUMBER}

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
