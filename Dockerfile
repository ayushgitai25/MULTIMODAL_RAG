FROM python:3.11-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /app/data /app/static /home/app/.streamlit && \
    chmod 777 /app/data /app/static

WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOME=/home/app
ENV STREAMLIT_HOME=/home/app/.streamlit
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV NGINX_PORT=7860

# Install system dependencies (Python + Nginx)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    libsndfile1 \
    git \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y transformers tokenizers || true && \
    pip install --no-cache-dir transformers==4.46.1 tokenizers==0.20.0 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy application code
COPY . .

# Copy Nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Fix permissions
RUN chown -R app:app /app /home/app /app/data /var/log/nginx && \
    chmod -R 755 /app/data /var/log/nginx

# Switch to non-root user
USER app

# Expose HF Spaces port
EXPOSE 7860

# Health check via Nginx
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Use supervisor to manage multiple processes
CMD ["bash", "-c", \
    "uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info & " \
    "streamlit run frontend_app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false & " \
    "nginx -g 'daemon off;'"]
