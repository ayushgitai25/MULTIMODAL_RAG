FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (assume utils, config are in place)
COPY . .

# Create data directory
RUN mkdir -p data

# Expose ports: 7860 for Streamlit (HF default), 8000 for internal FastAPI
EXPOSE 7860 8000

# Run backend with uvicorn in background, then frontend
CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info & streamlit run frontend_app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
