FROM python:3.11-slim

RUN useradd --create-home --shell /bin/bash app
WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOME=/home/app
ENV STREAMLIT_HOME=/home/app/.streamlit
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Hard-pin transformers first, then the rest
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y transformers tokenizers || true && \
    pip install --no-cache-dir transformers==4.46.1 tokenizers==0.20.0 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R app:app /app /home/app
USER app

EXPOSE 8000 8501

CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info & streamlit run frontend_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --browser.gatherUsageStats=false"]
