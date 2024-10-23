# Base image with Python 3.10.14 slim
FROM python:3.10.14-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_ENABLE_ONEDNN_OPTS=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install necessary tools for code cleanup
RUN pip install --upgrade pip && \
    pip install autoflake isort pylint flake8 --verbose && \
    rm -rf /root/.cache/pip

# Install project dependencies
RUN pip install -r requirements.txt --verbose && \
    rm -rf /root/.cache/pip

# Copy the rest of the application code
COPY . /app/

# Create an empty __init__.py file if needed
RUN touch /app/__init__.py

# Clean up unused imports and organize the imports
RUN autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive /app && \
    isort /app

# Optionally run linting to ensure the code is clean (optional step)
RUN pylint /app --disable=all --enable=unused-import --ignore-patterns="__init__.py"

# Expose the port that Flask will use
EXPOSE 5000

# Command to run the Flask app using Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
