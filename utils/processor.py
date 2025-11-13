"""
Processor module to handle PDF, image, and audio embedding operations.

- Uses OpenAI CLIP for text and image embeddings.
- Uses Facebook Wav2Vec2 for audio embeddings and speech recognition.
- Processes PDFs supporting text and embedded images.
- Creates multimodal messages for query context.
"""

import fitz
from PIL import Image
import torch
import numpy as np
import base64
import io
import logging
import librosa
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import CLIPProcessor, CLIPModel, Wav2Vec2Processor, Wav2Vec2Model, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize CLIP model & processor
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# Initialize Wav2Vec2 audio embedding model & processor
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec2_model.eval()

# Initialize speech-to-text pipeline
transcriber = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")


def embed_image(image_data):
    """
    Generate CLIP image embedding for given image data (PIL image or path).

    Returns normalized NumPy vector.
    """
    logger.debug("Embedding image with CLIP")
    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().cpu().numpy()


def embed_text(text):
    """
    Generate CLIP text embedding for given string.

    Returns normalized NumPy vector.
    """
    logger.debug(f"Embedding text with CLIP: {text[:60]}...")
    inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().cpu().numpy()


def embed_audio(audio_path):
    """
    Generate Wav2Vec2 audio embedding for audio file at audio_path.

    Returns mean-pooled embedding vector (numpy).
    """
    try:
        logger.info(f"Loading audio file: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=16000)
        logger.info(f"Audio loaded with sample rate: {sr}")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert to mono if stereo
        inputs = wav2vec2_processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        logger.info("Processing audio with Wav2Vec2")
        with torch.no_grad():
            outputs = wav2vec2_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        logger.info("Audio embedding generated")
        return embeddings
    except Exception as e:
        logger.error(f"Error in embed_audio: {type(e).__name__}: {e}")
        raise


def process_pdf(pdf_path):
    """
    Processes a PDF file and returns lists of:
    - Document chunks (text and images as Document instances)
    - Corresponding embeddings
    - Image data dict for use in frontend/image retrieval
    """
    logger.info(f"Processing PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    all_docs, all_embeddings, image_data_store = [], [], {}
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for i, page in enumerate(doc):
        logger.debug(f"Processing page {i}")
        text = page.get_text().strip()
        if text:
            temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
            text_chunks = splitter.split_documents([temp_doc])
            for chunk in text_chunks:
                emb = embed_text(chunk.page_content)
                all_embeddings.append(emb)
                all_docs.append(chunk)
        for img_idx, img in enumerate(page.get_images(full=True)):
            try:
                xref, base_image = img[0], doc.extract_image(img[0])
                raw_img = base_image['image']
                pil_image = Image.open(io.BytesIO(raw_img)).convert("RGB")
                image_id = f"page_{i}_img_{img_idx}"
                buf = io.BytesIO()
                pil_image.save(buf, format='PNG')
                image_data_store[image_id] = base64.b64encode(buf.getvalue()).decode()
                emb = embed_image(pil_image)
                all_embeddings.append(emb)
                all_docs.append(Document(
                    page_content=f"[Image: {image_id}]",
                    metadata={"page": i, "type": "image", "image_id": image_id}
                ))
                logger.debug(f"Embedded image {image_id}")
            except Exception as e:
                logger.error(f"Failed on page {i} image {img_idx}: {e}")
    doc.close()
    logger.info(f"PDF {pdf_path} processed: {len(all_docs)} chunks")
    return all_docs, all_embeddings, image_data_store


def create_multimodal_message(query, retrieved_docs, image_data_store):
    """
    Constructs a message payload containing text, images, and audio transcription for LLM consumption.
    """
    content = []
    content.append({"type": "text", "text": f"Question: {query}\n\nRelevant context:"})
    for doc in retrieved_docs:
        if doc.metadata.get("type") == "image":
            image_id = doc.metadata.get("image_id")
            if image_id and image_id in image_data_store:
                content.append({"type": "text", "text": f"\n[Uploaded Image]:\n"})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data_store[image_id]}"}})
        
        elif doc.metadata.get("type") == "text":
            page = doc.metadata.get("page", "N/A")
            content.append({"type": "text", "text": f"[PDF Page {page}]: {doc.page_content}"})
        
        elif doc.metadata.get("type") == "audio_transcription":
            content.append({"type": "text", "text": f"[Audio Transcription]: {doc.page_content}"})
    
    content.append({"type": "text", "text": "\nPlease answer the question based on both the above images and text. Give your answer in proper markdown format."})
    return content
