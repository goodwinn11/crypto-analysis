# Dockerfile for Crypto GPT-5 Application
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create config directory
RUN mkdir -p /app/config && chmod 700 /app/config

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p /app/frontend /app/backend

# Ensure crypto_module exists
RUN if [ ! -f crypto_module.py ] && [ -f crypto_module_stub.py ]; then \
    cp crypto_module_stub.py crypto_module.py; \
    fi

# Set permissions
RUN if [ -d "/app/config" ]; then chmod 700 /app/config; fi

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "backend/app.py"]