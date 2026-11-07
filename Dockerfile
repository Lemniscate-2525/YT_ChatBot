# Production multi-stage Dockerfile for YouTube RAG Assistant
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install runtime system dependencies (curl for container health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Layer cache dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create dedicated non-root user for principle of least privilege
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

# Copy application source code
COPY . .

# Ensure storage directories exist and have proper permissions for non-root user
RUN mkdir -p /app/chroma_db /app/data && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

