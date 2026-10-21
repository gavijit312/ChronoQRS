# Use a lightweight, official Python 3.10 slim image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for scientific libraries and building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies (ensuring PyTorch installs the lightweight CPU version)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code and model checkpoints into the container
COPY . .

# Expose the default port (Render uses dynamic ports, but 10000 is standard fallback)
EXPOSE 10000

# Run FastAPI using Uvicorn, binding to 0.0.0.0 and using Render's dynamic $PORT variable
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}