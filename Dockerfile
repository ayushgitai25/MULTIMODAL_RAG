FROM python:3.10-slim

# Install bash and essentials (for HF Spaces compatibility)
RUN apt-get update && apt-get install -y bash curl procps && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF default UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Copy requirements and install dependencies as user
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application files with ownership
COPY --chown=user:user . .

# Create data directory with permissions
RUN mkdir -p data && chmod 755 data

EXPOSE 7860 8000

# Run as JSON array (exec form) - no shell parsing issues
CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info & streamlit run frontend_app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false --server.enableXsrfProtection=false"]
