# Base image
FROM python:3.12-slim

# Working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8502

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health || exit 1

# Entrypoint
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]
