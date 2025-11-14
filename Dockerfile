FROM python:3.10-slim

# Install system dependencies as root
RUN apt-get update && apt-get install -y \
    bash \
    curl \
    procps \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && chown -R 1000:1000 /usr/bin/ffmpeg /usr/bin/ffprobe

# Install Python dependencies as root (executables go to /usr/local/bin)
COPY --chown=root:root requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user (HF default UID 1000)
RUN useradd -m -u 1000 user

# Set working directory and copy files with user ownership
WORKDIR /home/user/app
COPY --chown=user:user . .
RUN mkdir -p data && chown -R user:user data && chmod 755 data

# Switch to non-root user for runtime
USER user

EXPOSE 7860 8000

# Run as JSON array (exec form) - binaries now in PATH
CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info & streamlit run frontend_app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false --server.enableXsrfProtection=false"]
