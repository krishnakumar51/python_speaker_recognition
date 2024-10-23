# Stage 1: Build and Dependency Installation
FROM python:3.10-slim-buster AS build

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_ENABLE_ONEDNN_OPTS=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libsndfile1 \
    build-essential \
    libffi-dev \
    python3-dev \
    linux-headers-amd64 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy only the requirements file (for better caching)
COPY requirements.txt /app/

# Install pip and project dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# Copy the rest of the application code
COPY . /app/

# Clean up build dependencies to reduce size (optional, if you don't need them in final)
RUN apt-get remove -y gcc build-essential libffi-dev python3-dev linux-headers-amd64 && apt-get autoremove -y

# Stage 2: Production Image
FROM python:3.10-slim-buster AS final

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_ENABLE_ONEDNN_OPTS=0

# Install only the runtime dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the installed dependencies and application code from the build stage
COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=build /app /app

# Expose the Flask port
EXPOSE 5000

# Command to run the Flask app using Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
