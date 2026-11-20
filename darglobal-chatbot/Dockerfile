FROM python:3.11-slim

# Avoid interactive prompts and .pyc files; flush logs immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application.
COPY app ./app
COPY static ./static
COPY scraper ./scraper

# Render/most PaaS set $PORT; default to 8000 for local runs.
ENV PORT=8000
EXPOSE 8000

# Start the server. Shell form so $PORT is expanded at runtime.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
