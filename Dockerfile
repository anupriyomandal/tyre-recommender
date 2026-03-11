FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn

# Copy the rest of the project
COPY . .

# Railway injects $PORT at runtime
CMD uvicorn src.api.server:app --host 0.0.0.0 --port ${PORT:-8000}
