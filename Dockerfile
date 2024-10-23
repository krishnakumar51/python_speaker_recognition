# Stage 1: Build Stage - Python 3.10.14 slim with dev tools for cleanup
FROM python:3.10.14-slim AS build

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_ENABLE_ONEDNN_OPTS=0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install necessary tools for code cleanup (only for the build stage)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir autoflake isort pylint flake8 --verbose

# Install project dependencies (still in build stage)
RUN pip install --no-cache-dir -r requirements.txt --verbose

# Copy the rest of the application code
COPY . /app/

# Clean up unused imports and organize the imports
RUN autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive /app && \
    isort /app

# Run linting to ensure the code is clean (optional step)
RUN pylint /app --disable=all --enable=unused-import --ignore-patterns="__init__.py"

# Stage 2: Final Production Stage
FROM python:3.10.14-slim AS final

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_ENABLE_ONEDNN_OPTS=0

# Install system dependencies required for runtime only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*

# Set the working directory
WORKDIR /app

# Copy only the essential runtime files from the build stage
COPY --from=build /app /app

# Install only the runtime dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt --verbose && \
    rm -rf /root/.cache/pip

# Expose the port that Flask will use
EXPOSE 5000

# Command to run the Flask app using Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
