# Complete Technical Breakdown: Multimodal RAG System

> **[Expanded Definition] Multimodal RAG:** This system's name breaks down as:
> * **Multimodal:** It can understand and process multiple *types* (modes) of information—specifically text, images, and audio.
> * **RAG (Retrieval-Augmented Generation):** This is a two-step AI process. Instead of just *generating* an answer from its own memory, the system first *Retrieves* relevant, specific information from a knowledge base (your uploaded files) and then *Augments* its answer by using that information as context. This makes the answers grounded, accurate, and specific to your data.

Here's a detailed, step-by-step explanation of how this multimodal RAG system works internally, covering embedding creation, storage, model usage, dimension handling, and the full processing pipeline.

## Architecture Overview Diagram

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                          MULTIMODAL RAG SYSTEM                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │    INPUT      │    │  PROCESSING  │    │   OUTPUT      │               │
│  │  (User)       │───▶│  (Backend)   │───▶│  (LLM Answer) │               │
│  │              │    │              │    │               │               │
│  │  Upload:     │    │  1. Extract   │    │   Query LLM   │               │
│  │  - PDF       │    │  2. Embed    │    │   w/ Context  │               │
│  │  - Image     │    │  3. Store    │    │   (Gemini)    │               │
│  │  - Audio     │    │              │    │               │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────┐
│        STORAGE LAYER          │
│  ┌─────────────┐ ┌─────────┐ │
│  │ PDF Vector  │ │Image    │ │Audio    │
│  │ Store (512D)│ │Store    │ │Stores   │
│  │             │ │(512D)   │ │(768D+   │
│  │             │ │         │ │512D)    │
│  └─────────────┘ └─────────┘ │└─────────┘│
└──────────────────────────────┘
1. Embedding Creation Process
[Expanded Definition] Embedding: An embedding is a vector—a long list of numbers—that represents the semantic meaning of a piece of data (like text, an image, or audio). The AI model creates these vectors in a way that similar concepts (e.g., an image of a dog and the text "a furry pet") are placed close together in a high-dimensional mathematical space.

Step 1: User Uploads Data
User uploads a file through the Streamlit UI.

[Expanded Definition] Streamlit: A Python library used to quickly build and deploy simple, interactive web applications (like the one you use) without needing complex web development knowledge.

The file is saved temporarily to the backend via a POST request to /upload_pdf, /upload_image, or /upload_audio.

[Expanded Definition] POST Request: A standard method used in web development to send data from a client (your web browser) to a server (the backend). In this case, it's used to send the entire uploaded file.

The backend validates the file type and begins processing it immediately.

Step 2: Data Type Specific Processing
A. PDF Processing (in processor.py)
Python

def process_pdf(pdf_path):
Library: Uses PyMuPDF (fitz) to parse PDF documents.

[Expanded Definition] PyMuPDF (fitz): A high-performance Python library for accessing and manipulating PDF files. It's used here to extract all the raw text and any embedded images from each page.

What happens:

Opens the PDF using fitz.open(pdf_path).

For each page: page.get_text() extracts all raw text content.

Text is chunked using LangChain's RecursiveCharacterTextSplitter (500 characters per chunk, 100 character overlap).

[Expanded Definition] Chunking: This process breaks down large documents into smaller, digestible pieces. This is crucial because AI models have a limited "context window" (a maximum amount of text they can read at once). The overlap ensures that a complete idea isn't split awkwardly between two chunks.

For each text chunk: Creates a Document object (a LangChain data structure) with metadata (page number, type="text").

For embedded images: page.get_images() extracts images, converts them to PIL (Python Imaging Library) format, and embeds them using CLIP.

Returns: A list of Document objects, their corresponding embeddings (vectors), and base64 data for the images (for the UI).

[Expanded Definition] Base64: A text-based encoding a for binary data. It's a way to represent an image (which is binary) as a long string of text, making it easy to embed directly into a web page or send in a data message.

B. Image Processing
Python

def embed_image(image_data):
Model: OpenAI CLIP (openai/clip-vit-base-patch32)

Process:

Input: A PIL Image object or a file path.

Preprocessing: Converts the image to RGB format (if it's not already).

CLIP Processing:

CLIP (Contrastive Language-Image Pre-Training) is a model trained on millions of (image, text) pairs from the internet. Its "magic" is that it learns to map both images and text descriptions into the same semantic space. A text vector for "a yellow banana" will be mathematically close to an image vector of a yellow banana.

It tokenizes the image by breaking it into a grid of 32x32 pixel patches. This is the ViT-B/32 (Vision Transformer - Base size / 32px patches) part.

It passes these patches through its Vision Transformer backbone (the main feature-extraction part of the model).

Outputs: A 512-dimensional vector (a list of 512 numbers) that is the semantic "summary" of the image.

Normalization: The vector is divided by its L2 norm.

[Expanded Definition] Normalization (L2 Norm): A mathematical step that scales the vector so its "length" is 1. This is crucial for Cosine Similarity (used later in search) because it ensures the search only measures the angle (direction/meaning) between vectors, not their magnitude (which can be arbitrary).

Returns: A (1, 512) numpy array.

C. Audio Processing
Python

def embed_audio(audio_path):
Preprocessing:

librosa.load(audio_path, sr=16000): Loads the audio file using the librosa library (a standard for audio analysis in Python) and sets the sample rate to 16,000 Hz.

[Expanded Definition] Sample Rate (sr=16000): This means the audio is measured 16,000 times per second. This is a common standard for speech recognition models, as it captures the full range of human speech frequencies.

Converts stereo (two channels) to mono (one channel) if needed (audio.mean(axis=1)), as most speech models expect a single audio stream.

Feeds the processed audio into the Wav2Vec2 processor.

Model: Facebook Wav2Vec2 (facebook/wav2vec2-base-960h)

Two stages:

Audio Embedding: The Wav2Vec2 model generates 768-dimensional feature vectors that represent the raw audio characteristics (timbre, pitch, phonetics) for the entire clip.

Speech Recognition: A separate ASR (Automatic Speech Recognition) "head" of the model, transcriber(audio_path), is used to generate a text transcription.

Storage Strategy: This is a key design choice. Both forms of data are stored.

Raw audio embeddings (768D) → audio_index_vector_store: These vectors represent how the audio sounds. (This isn't fully utilized in the text-query path but could be used for "find similar-sounding audio" queries).

Transcribed text embeddings (512D) → audio_query_vector_store: The transcription text is embedded using the same CLIP text model as the PDFs. This is what allows you to find audio clips by asking questions about their content.

2. How Storage Works
Vector Store Architecture
[Expanded Definition] Vector Store (FAISS): A Vector Store (or Vector Database) is a database specifically designed to store and efficiently search through millions of high-dimensional vectors. FAISS (Facebook AI Similarity Search) is an open-source library that does this extremely fast, acting as the "memory" for the RAG system.

Each modality has a separate FAISS index to handle its specific embedding dimension:

Python

# Text embeddings (CLIP) - 512 dimensions
pdf_vector_store = FAISS.from_embeddings(
    text_embeddings=[("", np.zeros(512))],  # 512D text embeddings
    embedding=text_embeddings_obj,
    metadatas=[{"type": "dummy"}]
)

# Image embeddings (CLIP) - 512 dimensions  
image_vector_store = FAISS.from_embeddings(
    text_embeddings=[("", np.zeros(512))],  # 512D image embeddings
    embedding=text_embeddings_obj,
    metadatas=[{"type": "dummy"}]
)

# Audio index embeddings (Wav2Vec2) - 768 dimensions
audio_index_vector_store = FAISS.from_embeddings(
    text_embeddings=[("", np.zeros(768))],  # 768D audio embeddings
    embedding=audio_embeddings_obj,
    metadatas=[{"type": "dummy"}]
)

# Audio query embeddings (text from transcription) - 512 dimensions
audio_query_vector_store = FAISS.from_embeddings(
    text_embeddings=[("", np.zeros(512))],  # 512D text embeddings
    embedding=text_embeddings_obj,
    metadatas=[{"type": "dummy"}]
)
What FAISS Stores
Vectors: The normalized embedding arrays (512D for text/image, 768D for audio).

Metadata: Additional information "tagged" to each vector, which is returned with the search result. This is how the system knows where the data came from.

type: "text", "image", "audio"

source: "pdf", "image", "audio_transcription"

page: PDF page number

image_id: Unique identifier for embedded images

Storage Process
Python

# Example: Adding embeddings to FAISS
pdf_vector_store.add_embeddings(
    [(doc.page_content, emb) for doc, emb in zip(docs, embeddings)],  # (text, vector) pairs
    metadatas=[doc.metadata for doc in docs]  # Store metadata alongside
)
Key Point: FAISS doesn't store the actual PDF/image/audio files. It stores their semantic representations as vectors, which allows for near-instantaneous search based on meaning, not just keywords.

3. How User Query Is Compared (Similarity Search)
Step 1: Query Embedding
When a user asks, "What are the main findings?" in PDF mode, the system first embeds the user's query using the exact same CLIP text model.

Python

# When user asks "What are the main findings?" in PDF mode
query_embedding = embed_text("What are the main findings?")  # 512D vector
Step 2: FAISS Similarity Search
The backend then takes this 512D query vector and uses it to search the relevant vector store.

Python

# Backend query processing
context_docs = pdf_vector_store.similarity_search(query_embedding, k=10)
FAISS Algorithm: Uses cosine similarity (or its mathematical equivalent, the dot product on normalized vectors) to compare the query vector against all stored vectors in the index.

Cosine Similarity Formula:

Plaintext

similarity(A, B) = (A · B) / (||A|| × ||B||)
[Expanded Definition] Cosine Similarity: This formula essentially measures the angle between two vectors in high-dimensional space.

A value of 1 means the vectors point in the exact same "semantic direction" (identical meaning).

A value of 0 means they are unrelated (a 90-degree angle).

A value of -1 means they have opposite meaning. This is the standard way to measure "closeness" in a vector space.

Top-K Retrieval: The k=10 parameter tells FAISS to return the 10 vectors (and their associated metadata) that have the highest similarity score—the 10 most relevant chunks.

Step 3: Result Ranking
FAISS uses IndexFlatIP (Inner Product) for this process. "Flat" means it does an exhaustive, brute-force search comparing the query to every vector in the index (which is fast for this app's scale). "IP" (Inner Product) is the mathematical operation that, on normalized vectors, is equivalent to Cosine Similarity.

4. How Each Data Type Is Handled (Summary)
PDF Handling

Input: Raw PDF file.

Processing: Extract text + embedded images → chunk text → embed each chunk.

Storage: Text chunks (512D) and Image chunks (512D) all go into pdf_vector_store.

Query: User query (text, 512D) searches pdf_vector_store → returns relevant text and image chunks.

Context for LLM: A mix of text chunks and image references/data.

Image Handling

Input: JPG/PNG file.

Processing: Single image → CLIP image embedding (512D).

Storage: Image embedding (512D) + metadata in image_vector_store.

Query: User query (text, 512D) searches image_vector_store → finds most semantically similar images.

Context for LLM: Image metadata + base64 encoded image data.

Audio Handling

Input: WAV/MP3/M4A file.

Processing:

Raw audio → Wav2Vec2 → 768D embedding.

Audio → ASR transcription (text) → CLIP text embedding (512D).

Storage:

Raw audio embedding (768D) → audio_index_vector_store.

Transcription text embedding (512D) → audio_query_vector_store.

Query: User query (text, 512D) searches audio_query_vector_store → returns relevant transcription text.

Context for LLM: The text transcription of the most relevant audio segments.

5. Model Usage Breakdown
CLIP (openai/clip-vit-base-patch32)
Purpose: Text-image alignment and multimodal understanding.

Architecture:

Vision Transformer (ViT-B/32) for images.

Transformer text encoder for text.

Both are trained to map their outputs into a shared 512-dimensional semantic space.

Dimensions: 512D for both text and image embeddings.

Usage:

Embedding PDF text chunks.

Embedding PDF embedded images.

Embedding user text queries.

Embedding audio transcriptions.

Embedding uploaded images.

Wav2Vec2 (facebook/wav2vec2-base-960h)
Purpose: Audio understanding and speech-to-text.

Architecture:

A CNN + Transformer encoder that learns from raw audio waveforms.

It's self-supervised, meaning it first learned the structure of audio from 960 hours of unlabeled audio, before being fine-tuned for a specific task like transcription.

Outputs contextualized audio representations.

Dimensions: 768D for raw audio embeddings, but its transcription text is later embedded at 512D by CLIP.

Usage:

Generating raw audio embeddings (768D) for audio_index_vector_store.

Transcribing audio to text (ASR), which is then fed to CLIP.

Gemini LLM
Purpose: Final answer generation using the retrieved multimodal context.

Input: A multimodal message that combines:

The user's original query (text).

The retrieved context documents (text chunks, image data, audio transcriptions).

Output: A natural language answer that synthesizes the retrieved information to directly address the user's query.

6. Dimension Handling
Why Different Dimensions?
CLIP (512D): The designers of CLIP found 512 dimensions to be an effective balance for representing both text and image features in a shared space.

Wav2Vec2 (768D): The designers of Wav2Vec2 (Base model) found that 768 dimensions were needed to capture the complex, rich features of a raw audio signal.

Audio Query Store (512D): This store doesn't hold audio vectors. It holds text vectors (from CLIP) that represent the transcription of the audio. This is a critical design choice.

How FAISS Handles Different Dimensions
A single FAISS index must have a fixed dimension. You cannot store 512D and 768D vectors in the same index. This is why the system requires separate vector stores:

Python

# PDF and Image use 512D (CLIP text/image embeddings)
FAISS.from_embeddings(text_embeddings=[("", np.zeros(512))], ...)

# Audio Index uses 768D (Wav2Vec2 raw audio embeddings)  
FAISS.from_embeddings(text_embeddings=[("", np.zeros(768))], ...)
The Unified Search Space
This is the most "genius" part of the design:

A user's query is always text.

This text query is always embedded using CLIP into a 512D vector.

This 512D query vector can then be directly compared against:

PDF text chunks (also 512D CLIP vectors).

Embedded images (also 512D CLIP vectors).

Audio transcriptions (which are text, also 512D CLIP vectors).

This creates a unified 512D semantic space for all queryable content, allowing for powerful cross-modal search (e.g., a text query finding a relevant image). The 768D audio store is a specialized index for "audio-feature" similarity, separate from this main query-response loop.

7. Complete Processing Flow (Example)
USER UPLOADS DATA

├─ PDF: Extract text + images → Embed all as 512D vectors → Store in pdf_vector_store.

├─ Image: Process with CLIP → Embed as 512D vector → Store in image_vector_store.

└─ Audio: [M4A→WAV] → Embed raw audio (768D) + Transcribe text → Embed text (512D) → Store both in their separate stores.

USER ASKS QUERY

└─ (In PDF Mode) "What are the main findings?" → Embed query text with CLIP → 512D query vector.

SIMILARITY SEARCH

├─ Compare Query vector (512D) ↔ pdf_vector_store vectors (512D).

├─ Cosine similarity calculation.

├─ Top-10 most similar documents (text chunks or images) are retrieved.

└─ Return: A list of Document objects + their metadata.

CONTEXT BUILDING

├─ The system gathers the content from the retrieved Document objects.

├─ It builds a multimodal message for the Gemini LLM.

└─ Format: [{"type": "text", "text": "...user query..."}, {"type": "text", "text": "...retrieved chunk 1..."}, {"type": "image_url", "image_url": "data:image/png;base64,..."}, {"type": "text", "text": "...retrieved chunk 2..."}]

LLM GENERATION

├─ Gemini LLM receives this large, context-filled multimodal prompt.

├─ It generates a new answer based only on the retrieved content.

└─ Returns the final natural language response.

FRONTEND DISPLAY

├─ The answer text is shown in a styled card.

├─ The retrieved context (the "sources") is shown in expanders for user reference.

└─ The query is added to the chat history.

8. Key Technical Decisions & Rationale
Why Separate Vector Stores?
Different Dimensions: As noted, FAISS requires fixed dimensions. 512D (CLIP) and 768D (Wav2Vec2) cannot coexist in one index.

Specialized Processing: Each modality needs its own preprocessing pipeline (PDF parsing, audio transcription).

Query-ability: The "query" (search) logic is different. You search audio transcripts with text, but you might want to search raw audio features with another audio clip (a feature not in this text-query loop). Separating them keeps this clean.

Why Reset Vector Stores on Each Upload?
Python

# On each upload, recreate empty store
vector_store = FAISS.from_embeddings(
    text_embeddings=[("", np.zeros(DIM))],  # A fresh index
    embedding=embeddings_obj,
    metadatas=[{"type": "dummy"}]  # Empty metadata
)
Prevents Stale Data: This ensures that data from a previously uploaded PDF doesn't "contaminate" the search results for a new PDF.

Single Document Context: The app is designed as a "chat with a document" (or a single image/audio file) tool, not a "chat with your entire library" tool. Resetting enforces this "single-session" model.

Simplicity: It's far simpler to implement than managing a persistent, multi-user database with complex data versioning.

Why Use Base64 for Images?
Immediate Availability: The image data is passed directly in the prompt to the LLM.

No File Serving: This avoids the complexity of saving the image to disk, creating a public URL for it, and then having the LLM fetch it. The base64 string is the image, as far as the API is concerned.

UI Display: The frontend can also instantly display the retrieved source images from this same base64 string.

Summary
This multimodal RAG system elegantly combines:

Text Processing: PyMuPDF for PDF parsing + CLIP text embeddings (512D).

Image Understanding: CLIP ViT-B/32 for visual embeddings (512D).

Audio Processing: Wav2Vec2 for both raw embedding (768D) AND transcription, with the resulting text also embedded by CLIP (512D).

Vector Storage: Separate, specialized FAISS indexes for efficient similarity search based on meaning.

Query Pipeline: A unified text query → 512D embedding → similarity search → context building → LLM generation.

Frontend: A clean Streamlit UI with a tabbed interface and rich response display.

The core design principle is the creation of a unified 512D semantic space (powered by CLIP) where text queries can seamlessly retrieve relevant information from text chunks, images, and audio transcriptions. This architecture enables sophisticated, cross-modal semantic search and retrieval-augmented generation while maintaining a clean separation of concerns (a design principle where each part of the system has its own distinct responsibility) and efficient processing.
