FROM python:3.11-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOME=/home/app
ENV STREAMLIT_HOME=/home/app/.streamlit
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install system dependencies (ffmpeg + build tools for audio/video tasks)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Fix permissions
RUN chown -R app:app /app /home/app
USER app

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Run both backend (FastAPI) and frontend (Streamlit)
CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info & streamlit run frontend_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --browser.gatherUsageStats=false"]
