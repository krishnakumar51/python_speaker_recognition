# Base image with Python 3.10.14 slim
FROM python:3.10.14-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_ENABLE_ONEDNN_OPTS=0

# Install system dependencies and clean up unnecessary files
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install project dependencies without caching and remove pip cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --verbose

# Copy the rest of the application code
COPY . /app/

# Create an empty __init__.py file if needed
RUN touch /app/__init__.py

# Run autoflake and isort to clean up imports and organize them in a single step
RUN pip install --no-cache-dir autoflake isort --verbose && \
    autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive /app && \
    isort /app && \
    pip uninstall -y autoflake isort && \
    rm -rf /root/.cache/pip

# Expose the port that Flask will use
EXPOSE 5000

# Command to run the Flask app using Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
