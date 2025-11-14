FROM python:3.11-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app

# Environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOME=/home/app
ENV STREAMLIT_HOME=/home/app/.streamlit
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y transformers tokenizers || true && \
    pip install --no-cache-dir transformers==4.46.1 tokenizers==0.20.0 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy application code
COPY . .

# Fix permissions
RUN chown -R app:app /app /home/app
USER app

# Create data and static directories
RUN mkdir -p data static .streamlit

# Expose only the main port for Hugging Face Spaces
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run uvicorn server via startup.py (single foreground process)
CMD ["uvicorn", "startup:enhanced_app", "--host", "0.0.0.0", "--port", "7860", \
     "--log-level", "info", "--access-log", "--workers", "1", "--reload", "false"]
