# Stage 1: Builder
FROM python:3.10-slim-buster as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH="/app:$PYTHONPATH"

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    pkg-config \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create virtual environment and install packages
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -U pip setuptools wheel

COPY requirements.txt /app/
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt && \
    find /opt/venv -type d -name "__pycache__" -exec rm -r {} + && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    find /opt/venv -type f -name "*.pyo" -delete && \
    rm -rf ~/.cache/pip/*

# Stage 2: Runtime (minimal)
FROM python:3.10-slim-buster as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH" \
    TF_ENABLE_ONEDNN_OPTS=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONOPTIMIZE=0

WORKDIR /app

# Install only essential runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsndfile1 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    mkdir -p /app/pretrained_models /app/.cache && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /app/pretrained_models /app/.cache

# Copy only necessary files from builder
COPY --from=builder /opt/venv /opt/venv

# Copy all application files from the 'app' directory, including subdirectories
COPY --chown=appuser:appuser /app/ /app/

# Remove unnecessary files from virtual environment
RUN find /opt/venv -type d -name "tests" -exec rm -rf {} + && \
    find /opt/venv -type d -name "examples" -exec rm -rf {} + && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    find /opt/venv -type f -name "*.pyo" -delete && \
    find /opt/venv -type d -name "__pycache__" -exec rm -r {} +

USER appuser
EXPOSE 5000

CMD ["gunicorn", \
    "--bind", "0.0.0.0:5000", \
    "--workers", "2", \
    "--threads", "4", \
    "--worker-class", "sync", \
    "--worker-tmp-dir", "/dev/shm", \
    "--access-logfile", "-", \
    "--log-level", "info", \
    "--timeout", "120", \
    "--preload", \
    "app:app"]
