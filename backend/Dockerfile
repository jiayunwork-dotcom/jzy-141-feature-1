FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY tests ./tests
COPY pytest.ini ./pytest.ini

EXPOSE 8000

# Default command runs the API; run the suite in-container with:
#   docker compose run --rm backend pytest
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
