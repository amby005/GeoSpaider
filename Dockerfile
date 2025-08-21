# ---- base: small, fast, CPU-only ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Pillow & build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg62-turbo-dev \
    zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY src ./src

# Default env so the app knows where the Chroma DB will be mounted in the container
ENV DB_DIR=/data/vector_store \
    COLLECTION_NAME=satellite_images

# Default command (no GUI in containers, so don't try to open images)
CMD ["python", "-m", "src.demo", "--mode", "pure", "--no-describe"]
