# Stage 1: Builder
FROM python:3.10-slim-buster AS builder

# Set environment variables to prevent .pyc files and enable faster builds
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    pkg-config && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create virtual environment and install packages in the builder stage
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -U pip setuptools wheel

# Copy requirements and install dependencies
COPY requirements.txt . 
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt && \
    find /opt/venv -type d -name "__pycache__" -exec rm -r {} + && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    find /opt/venv -type f -name "*.pyo" -delete

# Stage 2: Runtime (minimal)
FROM python:3.10-slim-buster AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    TF_ENABLE_ONEDNN_OPTS=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONOPTIMIZE=0

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsndfile1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy only the virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv

# Copy application files without specific ownership
COPY /app/ /app/

# Create necessary directories (root will automatically have permission)
RUN mkdir -p /app/pretrained_models /app/.cache

# Remove unnecessary files from virtual environment
RUN find /opt/venv -type d -name "tests" -exec rm -rf {} + && \
    find /opt/venv -type d -name "examples" -exec rm -rf {} + && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    find /opt/venv -type f -name "*.pyo" -delete && \
    find /opt/venv -type d -name "__pycache__" -exec rm -r {} +

EXPOSE 8000

# Use Uvicorn with optimized settings for FastAPI
CMD ["uvicorn", "fast:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
