---
title: 🤖 Multimodal RAG API
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Multimodal RAG with Gemini

A production-ready FastAPI + Streamlit application for Retrieval-Augmented Generation (RAG) across **PDF**, **Image**, and **Audio** modalities.

## 🚀 Features

- **📄 PDF Processing**: Extract text and generate embeddings for document search
- **🖼️ Image Analysis**: CLIP-based visual embeddings for image understanding  
- **🎵 Audio Transcription**: Wav2Vec2 speech-to-text with audio embeddings
- **🧠 Gemini Integration**: Google Gemini 1.5 Flash for intelligent responses
- **📚 Vector Search**: FAISS index for fast similarity search across modalities
- **🌐 Web Interface**: Streamlit UI with file upload and query capabilities

## 🛠️ Architecture

- **Backend**: FastAPI serving on port 7860 (main process)
- **Frontend**: Streamlit running as internal subprocess
- **Embeddings**: CLIP (multimodal), Sentence Transformers (text), Wav2Vec2 (audio)
- **Storage**: FAISS vector stores per modality with automatic cleanup
- **Deployment**: Docker container optimized for Hugging Face Spaces

## 🔧 API Endpoints

All API calls are prefixed with `/api/`:

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/api/upload_pdf` | POST | Upload and index PDF | `file` (PDF) |
| `/api/upload_image` | POST | Upload and index image | `file` (JPG/PNG) |
| `/api/upload_audio` | POST | Upload and transcribe audio | `file` (WAV/MP3) |
| `/api/query` | POST | Query across modalities | `{"query": str, "mode": "pdf/image/audio"}` |
| `/health` | GET | Health check | - |

## 📦 Deployment

### Hugging Face Spaces

1. **Push to Repository**: Commit all files and push to your Hugging Face Space repo
2. **Factory Rebuild**: In Space settings, trigger "Factory Rebuild" to clear cache
3. **Monitor Logs**: Watch for "FastAPI application ready on port 7860" message
4. **Access UI**: Visit your Space URL - should load Streamlit interface

### Local Development

