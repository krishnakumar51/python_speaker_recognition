# Base image with Python 3.10.14
FROM python:3.10.14-alpine

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_ENABLE_ONEDNN_OPTS=0  

# Install system dependencies
RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --verbose && \
    rm -rf /root/.cache/pip  # Remove pip cache after installation

# Copy the rest of the application code
COPY . /app/

# Expose the port that Flask will use
EXPOSE 5000

# Command to run the Flask app using Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
