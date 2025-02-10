# filepath: c:\Users\sonic\Desktop\bp\Dockerfile
FROM python:3.10-slim

# Prevent Python from writing pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Command to run your app (gunicorn picks up bp:app, adjust port via the $PORT environment variable)
CMD ["sh", "-c", "gunicorn bp:app --bind 0.0.0.0:${PORT}"]