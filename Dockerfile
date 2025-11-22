FROM python:3.11-slim

# Prevent Python from buffering output (better logs on Railway)
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies required by sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY app/ ./app

# Expose Railway port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
