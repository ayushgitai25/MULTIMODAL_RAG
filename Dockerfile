FROM python:3.10-slim

# Create non-root user (HF default UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Copy requirements and install dependencies as user
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application files with ownership
COPY --chown=user . .

# Create data directory with permissions
RUN mkdir -p data && chmod 755 data

EXPOSE 7860 8000

# Run backend in background, frontend with XSRF disabled
CMD ["bash", "-c", \
     "uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info & " \
     "streamlit run frontend_app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false --server.enableXsrfProtection=false"]
