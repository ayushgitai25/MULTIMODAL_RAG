FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including ffmpeg for audio processing
RUN apt-get update && apt-get install -y \
    build-essential curl ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code files
COPY . ./

# Streamlit environment setup
ENV STREAMLIT_HOME=/tmp/.streamlit
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose ports for uvicorn and streamlit
EXPOSE 8000 8501

# Run uvicorn and streamlit concurrently:
# uvicorn runs FastAPI backend on port 8000,
# streamlit runs frontend on port 8501
ENTRYPOINT /bin/sh -c "uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run frontend_app.py --server.port 8501 --server.address 0.0.0.0 & wait"


