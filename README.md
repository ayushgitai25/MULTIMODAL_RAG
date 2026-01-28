---
title: Multimodal RAG App
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🤖 Multimodal RAG Intelligence Engine

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Open%20in%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ayushhgface25/Multimodal_RAG)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![Powered by Gemini](https://img.shields.io/badge/AI-Gemini%202.5-orange)](https://deepmind.google/technologies/gemini/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](https://www.docker.com/)

**A scalable, context-aware AI agent capable of "seeing" images, "reading" documents, and "listening" to audio files to answer natural language queries.**

This project implements a **Unified Multimodal Search Space** by orchestrating distinct vector embedding models (CLIP & Wav2Vec2) into a partitioned FAISS backend. It allows users to chat with unstructured data with high groundedness and low latency.

🔗 **[Click Here to Try the Live App](https://huggingface.co/spaces/ayushhgface25/Multimodal_RAG)**

---

## 📸 Live Application Preview

| **Document Analysis (PDF)** | **Visual Q&A (Image)** |
|:---:|:---:|
| ![PDF Demo](https://github.com/ayushgitai25/MULTIMODAL_RAG/blob/main/pdf.png) | ![Image Demo](https://github.com/ayushgitai25/MULTIMODAL_RAG/blob/main/imgg.png) |
| *Extracts text & embedded images for grounded Q&A* | *Understands visual scenes using CLIP embeddings* |

---

## 🏗️ System Architecture 

The system utilizes a **Microservices-style architecture** with an Async FastAPI backend and a Streamlit frontend. It bridges the dimension gap between Text/Image (512D) and Audio (768D) using a dual-indexing strategy.

*Have utilized Gemini to create a sophisticated diagram architecture.*

![System Architecture Diagram](https://github.com/ayushgitai25/MULTIMODAL_RAG/blob/main/architecture.png)

---

## 🚀 Key Features

* **📄 Document Intelligence (PDF):**
    * Extracts text using **PyMuPDF**.
    * Identify and embed images *inside* PDFs using **CLIP**.
    * Recursive chunking (500 tokens) for optimal context retention.
* **🖼️ Visual Understanding (Image):**
    * Uses **OpenAI CLIP (ViT-B/32)** to map images into a 512-dimensional vector space.
    * Enables semantic search (e.g., searching for "a financial chart" retrieves the relevant image).
* **🎵 Audio Analysis (Audio):**
    * Processes raw audio (WAV/MP3/M4A) using **Facebook Wav2Vec2**.
    * Dual-path indexing: **Acoustic embeddings** (768D) for feature analysis + **ASR Transcription** (text-to-CLIP) for semantic query compatibility.
* **⚡ High-Performance Architecture:**
    * **Async FastAPI Backend:** Non-blocking file uploads and query processing.
    * **Streamlit Frontend:** Clean, responsive UI for interacting with the agent.
    * **Dockerized:** Fully containerized deployment for consistency across environments.

---

## 🛠️ Technical Architecture

This system solves the "Dimension Mismatch" problem in multimodal AI by maintaining specialized vector stores that feed into a unified generation layer.

| Component | Technology | Role |
| :--- | :--- | :--- |
| **LLM** | Google Gemini 2.5 Flash | Final answer generation using grounded context. |
| **Embeddings** | OpenAI CLIP | Text & Image alignment (512 Dimensions). |
| **Audio** | Facebook Wav2Vec2 | Speech-to-Text & Acoustic Features (768 Dimensions). |
| **Vector DB** | FAISS | Partitioned indices for fast similarity search. |
| **Backend** | FastAPI (Async) | Microservice handling inference and state. |
| **Frontend** | Streamlit | User interface and visualization. |

---

## 💻 Local Installation

To run this application locally, you will need a Google Gemini API Key.

1.  **Clone the repository**
    ```bash
    git clone [https://huggingface.co/spaces/ayushhgface25/Multimodal_RAG](https://huggingface.co/spaces/ayushhgface25/Multimodal_RAG)
    cd Multimodal_RAG
    ```

2.  **Set up Environment Variables**
    Create a `.env` file or export your key:
    ```bash
    export GOOGLE_API_KEY="your_actual_api_key_here"
    ```

3.  **Access the App**
    Open your browser and navigate to: `http://localhost:7860`

---

## 📖 Usage Guide

1.  **Select Mode:** Choose the tab for your data type (PDF, Image, or Audio).
2.  **Upload File:** Drag and drop your file.
    * *System automatically performs ETL (Extract, Transform, Load) and indexing.*
3.  **Ask Questions:**
    * *PDF:* "Summarize the key findings on page 3."
    * *Image:* "Describe the architectural diagram in this picture."
    * *Audio:* "What did the speaker say about the budget?"
4.  **View Context:** Expand the "Retrieved Context" section to see exactly which text chunks or image patches the AI used to generate the answer.

---

## 📂 Project Structure

```text
.
├── app.py                # FastAPI backend endpoints
├── streamlit_app.py      # Frontend UI logic
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
└── utils/
    ├── processor.py      # CLIP/Wav2Vec2 embedding logic
    ├── llm_handler.py    # Gemini API integration
    └── config.py         # Configuration settings






