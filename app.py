"""
Multimodal FastAPI Application for RAG with PDF, Image, and Audio Support.

- Handles file uploads (PDF, image, audio)
- Indexes into separate FAISS vector stores per modality
- Serves uploaded media files statically for frontend display
- Returns media URLs along with query answers for rich UI responses
"""

import io, base64, os
from PIL import Image
import logging
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain.embeddings.base import Embeddings

from pydub import AudioSegment
import glob

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

# ✅ Serve uploaded files statically from /data endpoint
app.mount("/data", StaticFiles(directory="data"), name="data")

os.makedirs("data", exist_ok=True)

# Global variables for tracking uploaded files
uploaded_files = {
    "pdf": None,
    "image": None, 
    "audio": None
}

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
    """Upload PDF and track filename for later retrieval."""
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
    
    # Store PDF filename for later retrieval
    uploaded_files["pdf"] = file.filename

    try:
        docs, embeddings, _ = process_pdf(file_path)
        pdf_vector_store.add_embeddings(
            [(doc.page_content, emb) for doc, emb in zip(docs, embeddings)],
            metadatas=[doc.metadata for doc in docs]
        )
        logger.info(f"PDF '{file.filename}' processed and indexed.")
        return {
            "status": "success", 
            "message": f"PDF '{file.filename}' processed and indexed.",
            "media": {
                "pdf_filename": file.filename  # Store for query endpoint
            }
        }
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """Upload image and track filename for later retrieval."""
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
    
    # Store image filename for later retrieval
    uploaded_files["image"] = file.filename

    try:
        pil_image = Image.open(file_path).convert("RGB")
        image_id = f"image_{len(image_data_store)}"
        buf = io.BytesIO()
        pil_image.save(buf, format='PNG')
        image_data_store[image_id] = base64.b64encode(buf.getvalue()).decode()
        emb = embed_image(pil_image)

        doc = Document(
            page_content=f"[Image: {image_id}]",
            metadata={"type": "image", "image_id": image_id, "source": "image", "filename": file.filename}
        )
        image_vector_store.add_embeddings([(doc.page_content, emb)], metadatas=[doc.metadata])
        logger.info(f"Image '{file.filename}' embedded and indexed.")
        return {
            "status": "success", 
            "message": f"Image '{file.filename}' processed and indexed.",
            "media": {
                "image_filename": file.filename  # Store for query endpoint
            }
        }
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload audio and track filename for later retrieval."""
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
    
    # Store audio filename for later retrieval
    uploaded_files["audio"] = file.filename

    try:
        processed_path = file_path
        if file.filename.lower().endswith(".m4a"):
            wav_path = file_path.rsplit(".", 1)[0] + ".wav"
            convert_m4a_to_wav(file_path, wav_path)
            processed_path = wav_path

        emb = embed_audio(processed_path)
        doc = Document(
            page_content=f"[Audio: {file.filename}]",
            metadata={"type": "audio", "source": "audio", "filename": file.filename}
        )
        logger.info(f"Embedding shape: {emb.shape}")
        emb = emb.reshape(1, -1)
        audio_index_vector_store.add_embeddings([(doc.page_content, emb[0])], metadatas=[doc.metadata])

        text = transcriber(processed_path)
        transcription_text = text["text"]
        doc_text = Document(
            page_content=transcription_text,
            metadata={"type": "text", "source": "audio_transcription", "filename": file.filename}
        )
        text_emb = embed_text(transcription_text)
        audio_query_vector_store.add_embeddings([(transcription_text, text_emb)], metadatas=[doc_text.metadata])

        logger.info(f"Audio '{file.filename}' embedded and indexed.")
        return {
            "status": "success", 
            "message": f"Audio '{file.filename}' processed and indexed.",
            "media": {
                "audio_filename": file.filename  # Store for query endpoint
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
    """Enhanced query endpoint that returns media information."""
    logger.info(f"Received query: {request.query} for mode: {request.mode}")

    if request.mode == "pdf":
        if len(pdf_vector_store.index_to_docstore_id) <= 1:
            raise HTTPException(status_code=400, detail="No PDF uploaded yet. Please upload a PDF first.")
        context_docs = pdf_vector_store.similarity_search(request.query, k=10)
        content = create_multimodal_message(request.query, context_docs, {})
        media = {
            "type": "pdf",
            "urls": [uploaded_files.get("pdf", "")] if uploaded_files.get("pdf") else [],
            "count": 1 if uploaded_files.get("pdf") else 0
        }
    elif request.mode == "image":
        if len(image_vector_store.index_to_docstore_id) <= 1:
            raise HTTPException(status_code=400, detail="No image uploaded yet. Please upload an image first.")
        context_docs = image_vector_store.similarity_search(request.query, k=10)
        content = create_multimodal_message(request.query, context_docs, image_data_store)
        media = {
            "type": "image",
            "urls": [uploaded_files.get("image", "")] if uploaded_files.get("image") else [],
            "count": 1 if uploaded_files.get("image") else 0
        }
    elif request.mode == "audio":
        if len(audio_query_vector_store.index_to_docstore_id) <= 1:
            raise HTTPException(status_code=400, detail="No audio uploaded yet. Please upload an audio file first.")
        context_docs = audio_query_vector_store.similarity_search(request.query, k=10)
        content = create_multimodal_message(request.query, context_docs, {})
        media = {
            "type": "audio",
            "urls": [uploaded_files.get("audio", "")] if uploaded_files.get("audio") else [],
            "count": 1 if uploaded_files.get("audio") else 0
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'pdf', 'image', or 'audio'.")

    answer = query_llm(llm, content)
    logger.info(f"Generated answer: {answer}")
    
    return {
        "answer": answer,
        "media": media,
        "context": context_docs
    }
