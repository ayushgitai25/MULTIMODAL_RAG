"""
Multimodal FastAPI Application for RAG with PDF, Image, and Audio Support.

- Handles file uploads (PDF, image, audio with M4A support)
- Indexes into separate FAISS vector stores per modality
- Returns media information along with query answers for frontend display
- Tracks uploaded files globally for media retrieval
"""

import io, base64, os
from PIL import Image
import logging
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain.embeddings.base import Embeddings

from pydub import AudioSegment
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for tracking uploaded files and media
uploaded_files = {
    "pdf": None,
    "image": None, 
    "audio": None
}

# Cache for base64 encoded media
media_cache = {}

os.makedirs("data", exist_ok=True)

transcriber = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

from utils.processor import process_pdf, embed_text, create_multimodal_message, embed_image, embed_audio
from utils.llm_handler import get_llm, query_llm
from config import GOOGLE_API_KEY

llm = get_llm()
image_data_store = {}

class TextEmbeddings(Embeddings):
    """Wraps text embedding functions for compatibility with LangChain FAISS."""
    def embed_documents(self, texts):
        return [embed_text(t) for t in texts]
    def embed_query(self, text):
        return embed_text(text)

class AudioEmbeddings(Embeddings):
    """Wraps audio embedding functions for compatibility with LangChain FAISS."""
    def embed_documents(self, audio_paths):
        return [embed_audio(f) for f in audio_paths]
    def embed_query(self, audio_path):
        return embed_audio(audio_path)

text_embeddings_obj = TextEmbeddings()
audio_embeddings_obj = AudioEmbeddings()

# Vector Store Initialization
pdf_vector_store = FAISS.from_embeddings(
    text_embeddings=[("", np.zeros(512))],
    embedding=text_embeddings_obj,
    metadatas=[{"type": "dummy"}]
)
image_vector_store = FAISS.from_embeddings(
    text_embeddings=[("", np.zeros(512))],
    embedding=text_embeddings_obj,
    metadatas=[{"type": "dummy"}]
)
audio_index_vector_store = FAISS.from_embeddings(
    text_embeddings=[("", np.zeros(768))],
    embedding=audio_embeddings_obj,
    metadatas=[{"type": "dummy"}]
)
audio_query_vector_store = FAISS.from_embeddings(
    text_embeddings=[("", np.zeros(512))],
    embedding=text_embeddings_obj,
    metadatas=[{"type": "dummy"}]
)

def convert_m4a_to_wav(m4a_path, wav_path):
    """Convert M4A audio to WAV format using pydub."""
    audio = AudioSegment.from_file(m4a_path, format="m4a")
    audio.export(wav_path, format="wav")

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF and track for media display."""
    global pdf_vector_store, uploaded_files
    pdf_vector_store = FAISS.from_embeddings(
        text_embeddings=[("", np.zeros(512))],
        embedding=text_embeddings_obj,
        metadatas=[{"type": "dummy"}]
    )
    
    logger.info(f"Received PDF upload: {file.filename}")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_path = os.path.join("data", file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Store PDF for media reference
    uploaded_files["pdf"] = {
        "filename": file.filename,
        "path": file_path,
        "mode": "pdf",
        "base64": None  # We'll generate base64 if needed
    }

    try:
        docs, embeddings, _ = process_pdf(file_path)
        # Store metadata in documents for later retrieval
        for i, doc in enumerate(docs):
            doc.metadata["uploaded_file"] = file.filename
            doc.metadata["mode"] = "pdf"
        
        pdf_vector_store.add_embeddings(
            [(doc.page_content, emb) for doc, emb in zip(docs, embeddings)],
            metadatas=[doc.metadata for doc in docs]
        )
        logger.info(f"PDF '{file.filename}' processed and indexed.")
        return {
            "status": "success", 
            "message": f"PDF '{file.filename}' processed and indexed.",
            "media": {
                "type": "pdf",
                "filename": file.filename,
                "base64": None,  # We'll return base64 or URL
                "size": len(await file.read())
            }
        }
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """Upload image and store base64 for immediate display."""
    global image_vector_store, image_data_store
    image_vector_store = FAISS.from_embeddings(
        text_embeddings=[("", np.zeros(512))],
        embedding=text_embeddings_obj,
        metadatas=[{"type": "dummy"}]
    )
    image_data_store.clear()

    logger.info(f"Received image upload: {file.filename}")
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")

    file_path = os.path.join("data", file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Store image information
    uploaded_files["image"] = {
        "filename": file.filename,
        "path": file_path,
        "mode": "image"
    }

    try:
        pil_image = Image.open(file_path).convert("RGB")
        image_id = f"image_{len(image_data_store)}"
        
        # Store base64 for immediate use
        buf = io.BytesIO()
        pil_image.save(buf, format='PNG')
        image_data_store[image_id] = base64.b64encode(buf.getvalue()).decode()
        
        emb = embed_image(pil_image)
        doc = Document(
            page_content=f"[Image: {image_id}]",
            metadata={
                "type": "image", 
                "image_id": image_id, 
                "source": "image", 
                "filename": file.filename,
                "mode": "image"
            }
        )
        image_vector_store.add_embeddings([(doc.page_content, emb)], metadatas=[doc.metadata])
        logger.info(f"Image '{file.filename}' embedded and indexed.")
        
        return {
            "status": "success", 
            "message": f"Image '{file.filename}' processed and indexed.",
            "media": {
                "type": "image",
                "filename": file.filename,
                "base64": image_data_store[image_id],
                "size": len(buf.getvalue())
            }
        }
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload audio and store for playback."""
    global audio_index_vector_store, audio_query_vector_store
    audio_index_vector_store = FAISS.from_embeddings(
        text_embeddings=[("", np.zeros(768))],
        embedding=audio_embeddings_obj,
        metadatas=[{"type": "dummy"}]
    )
    audio_query_vector_store = FAISS.from_embeddings(
        text_embeddings=[("", np.zeros(512))],
        embedding=text_embeddings_obj,
        metadatas=[{"type": "dummy"}]
    )

    logger.info(f"Received audio upload: {file.filename}")

    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a")):
        raise HTTPException(status_code=400, detail="Only WAV, MP3, or M4A audio files are allowed.")

    file_path = os.path.join("data", file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Store audio information
    uploaded_files["audio"] = {
        "filename": file.filename,
        "path": file_path,
        "mode": "audio"
    }

    try:
        processed_path = file_path
        if file.filename.lower().endswith(".m4a"):
            wav_path = file_path.rsplit(".", 1)[0] + ".wav"
            convert_m4a_to_wav(file_path, wav_path)
            processed_path = wav_path

        emb = embed_audio(processed_path)
        doc = Document(
            page_content=f"[Audio: {file.filename}]",
            metadata={
                "type": "audio", 
                "source": "audio", 
                "filename": file.filename,
                "mode": "audio"
            }
        )
        logger.info(f"Embedding shape: {emb.shape}")
        emb = emb.reshape(1, -1)
        audio_index_vector_store.add_embeddings([(doc.page_content, emb[0])], metadatas=[doc.metadata])

        text = transcriber(processed_path)
        transcription_text = text["text"]
        doc_text = Document(
            page_content=transcription_text,
            metadata={
                "type": "text", 
                "source": "audio_transcription", 
                "filename": file.filename,
                "mode": "audio"
            }
        )
        text_emb = embed_text(transcription_text)
        audio_query_vector_store.add_embeddings([(transcription_text, text_emb)], metadatas=[doc_text.metadata])

        logger.info(f"Audio '{file.filename}' embedded and indexed.")
        return {
            "status": "success", 
            "message": f"Audio '{file.filename}' processed and indexed.",
            "media": {
                "type": "audio",
                "filename": file.filename,
                "base64": None,  # We won't encode full audio for size reasons
                "size": len(await file.read())
            }
        }
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

class QueryRequest(BaseModel):
    query: str
    mode: str

@app.post("/query")
async def query(request: QueryRequest):
    """Enhanced query endpoint that returns media information for display."""
    logger.info(f"Received query: {request.query} for mode: {request.mode}")

    context_docs = None
    media_response = {}

    if request.mode == "pdf":
        if len(pdf_vector_store.index_to_docstore_id) <= 1:
            raise HTTPException(status_code=400, detail="No PDF uploaded yet. Please upload a PDF first.")
        context_docs = pdf_vector_store.similarity_search(request.query, k=10)
        content = create_multimodal_message(request.query, context_docs, {})
        media_response = {
            "type": "pdf",
            "filename": uploaded_files.get("pdf", {}).get("filename", ""),
            "file_size": uploaded_files.get("pdf", {}).get("size", 0),
            "base64": None  # PDF too large for base64, use URL instead
        }
    elif request.mode == "image":
        if len(image_vector_store.index_to_docstore_id) <= 1:
            raise HTTPException(status_code=400, detail="No image uploaded yet. Please upload an image first.")
        context_docs = image_vector_store.similarity_search(request.query, k=10)
        content = create_multimodal_message(request.query, context_docs, image_data_store)
        media_response = {
            "type": "image",
            "filename": uploaded_files.get("image", {}).get("filename", ""),
            "base64": image_data_store.get("image_0", ""),  # Get first image base64
            "file_size": uploaded_files.get("image", {}).get("size", 0)
        }
    elif request.mode == "audio":
        if len(audio_query_vector_store.index_to_docstore_id) <= 1:
            raise HTTPException(status_code=400, detail="No audio uploaded yet. Please upload an audio file first.")
        context_docs = audio_query_vector_store.similarity_search(request.query, k=10)
        content = create_multimodal_message(request.query, context_docs, {})
        media_response = {
            "type": "audio",
            "filename": uploaded_files.get("audio", {}).get("filename", ""),
            "base64": None,  # Audio files are typically too large for base64
            "file_size": uploaded_files.get("audio", {}).get("size", 0)
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'pdf', 'image', or 'audio'.")

    answer = query_llm(llm, content)
    logger.info(f"Generated answer: {answer}")
    
    return {
        "answer": answer,
        "context": context_docs,
        "media": media_response
    }

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "FastAPI server is running"}
