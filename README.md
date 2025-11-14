---
title: Multimodal RAG App
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Multimodal RAG with Gemini

A FastAPI + Streamlit app for PDF, image, and audio RAG using CLIP, Wav2Vec2, and Gemini.

## Usage
1. Upload a file (PDF/Image/Audio).
2. Enter a query in the chosen mode.
3. Get AI-generated responses with retrieved context.

Built with LangChain, FAISS, and Google Gemini. Deploys via Docker on Hugging Face Spaces.
