# Cloud Run Dockerfile - AI Istanbul Backend
# Optimized for Cloud Run deployment with AWS RDS

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first for layer caching
COPY backend/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# NOTE: spaCy removed - LLM handles language detection and intent classification
# This saves ~30s cold start time and reduces image size

# Copy the backend application (this should include ml/deep_learning/models/)
COPY backend/ ./

# Verify ONNX model files were copied
RUN echo "=== Verifying ONNX model files copied ===" && \
    ls -lah /app/ml/deep_learning/models/ 2>&1 && \
    echo "--- Checking for ONNX files specifically ---" && \
    ls -lh /app/ml/deep_learning/models/ncf_model.* 2>&1 && \
    echo "========================================="

# Copy unified_system for UnifiedLLMService (required for LLM operations)
COPY unified_system/ ./unified_system/

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8080

# Health check - longer start period for Cloud Run
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Run the application with timeout settings
CMD exec uvicorn main_modular:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1 \
    --timeout-keep-alive 65 \
    --timeout-graceful-shutdown 10
