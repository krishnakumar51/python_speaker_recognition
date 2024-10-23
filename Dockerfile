# Stage 1: Build and Dependency Installation
FROM python:3.10-slim-buster AS build

# Set environment variables to prevent .pyc files and enable TensorFlow optimizations
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_ENABLE_ONEDNN_OPTS=0

# Install essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libsndfile1 \
    build-essential \
    libffi-dev \
    python3-dev \
    linux-headers-amd64 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy only the requirements file (to leverage Docker layer caching)
COPY requirements.txt /app/

# Install Python dependencies without cache
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# Clean up build dependencies to reduce the image size
RUN apt-get purge -y gcc g++ build-essential python3-dev linux-headers-amd64 && apt-get autoremove -y

# Copy the rest of the application code
COPY . /app/

# Stage 2: Production Image
FROM python:3.10-slim-buster AS final

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_ENABLE_ONEDNN_OPTS=0

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy installed dependencies and the application from the build stage
COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=build /app /app

# Expose the Flask port
EXPOSE 5000

# Use Gunicorn to run the Flask app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
