# Istanbul AI - Production Dockerfile
# Multi-stage build optimized for production scalability
# Supports: CPU and GPU (NVIDIA T4) deployments
# Designed for: 10,000 monthly users (~50 concurrent peak)

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim as builder

# Build arguments
ARG ENABLE_GPU=false

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements
COPY production_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r production_requirements.txt

# Install PyTorch based on GPU availability
RUN if [ "$ENABLE_GPU" = "true" ]; then \
        pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118; \
    else \
        pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi

# ============================================
# Stage 2: Production
# ============================================
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash istanbul

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=istanbul:istanbul . .

# Create necessary directories
RUN mkdir -p /app/logs /app/models /data/cache /data/models && \
    chown -R istanbul:istanbul /app /data

# Switch to non-root user
USER istanbul

# Expose port (Render uses PORT env var)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Start production server
CMD uvicorn production_server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2 --log-level info

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
