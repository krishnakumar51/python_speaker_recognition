# Stage 1: Builder
FROM python:3.10-slim-buster as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    # Clean up pip cache
    rm -rf ~/.cache/pip/*

# Stage 2: Runtime
FROM python:3.10-slim-buster as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # TensorFlow optimizations
    TF_ENABLE_ONEDNN_OPTS=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    # Python optimizations
    PYTHONOPTIMIZE=2

WORKDIR /app

# Install only required runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* && \
    # Create cache directories
    mkdir -p /app/pretrained_models /app/.cache && \
    # Add non-root user
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /app/pretrained_models /app/.cache

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv

# Copy only necessary files
COPY --chown=appuser:appuser models/ /app/models/
COPY --chown=appuser:appuser *.py /app/

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5000

# Run with optimized settings
CMD ["gunicorn", \
    "--bind", "0.0.0.0:5000", \
    "--workers", "2", \
    "--threads", "4", \
    "--worker-class", "gthread", \
    "--worker-tmp-dir", "/dev/shm", \
    "--access-logfile", "-", \
    "app:app"]