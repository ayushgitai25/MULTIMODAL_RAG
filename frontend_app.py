import streamlit as st
import requests
import os
import time

# Configure Streamlit for internal use
st.set_page_config(
    page_title="🎨 Multimodal RAG", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title with emoticon and color
st.markdown('<h1 style="color:#6A5ACD; font-weight:bold; text-align:center;">🤖 Multimodal RAG with Gemini</h1>', unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    st.markdown("### 📋 Quick Start")
    st.markdown("""
    1. **Choose a mode** (PDF/Image/Audio)
    2. **Upload a file** in the selected format
    3. **Enter your query** and click Ask
    4. **Get AI responses** with retrieved context
    """)
    st.markdown("---")
    st.markdown("### 🔧 Powered by")
    st.markdown("- 🔥 CLIP (Image embeddings)")
    st.markdown("- 🎵 Wav2Vec2 (Audio transcription)")
    st.markdown("- 📚 LangChain + FAISS (Vector stores)")
    st.markdown("- 🧠 Gemini 1.5 Flash (LLM)")

# Radio buttons for mode selection
mode = st.radio(
    "Choose mode:",
    ["PDF", "Image", "Audio"],
    index=0,
    horizontal=True,
    label_visibility="visible",
    help="Select the type of content you want to query"
)

# File uploader based on mode
uploaded_file = None
file_info = None

if mode == "PDF":
    uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf", help="Upload a PDF document")
    if uploaded_file:
        file_info = f"📄 **{uploaded_file.name}** (PDF) - {uploaded_file.size/1024:.1f} KB"
elif mode == "Image":
    uploaded_file = st.file_uploader(
        "🖼️ Upload Image (JPG/PNG)", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image file for visual search"
    )
    if uploaded_file:
        file_info = f"🖼️ **{uploaded_file.name}** (Image) - {uploaded_file.size/1024:.1f} KB"
elif mode == "Audio":
    uploaded_file = st.file_uploader(
        "🎵 Upload Audio (WAV/MP3)", 
        type=["wav", "mp3"],
        help="Upload an audio file for speech-to-text processing"
    )
    if uploaded_file:
        file_info = f"🎵 **{uploaded_file.name}** (Audio) - {uploaded_file.size/1024:.1f} KB"

# Display file info if uploaded
if file_info:
    st.success(file_info)

# API configuration - internal calls to FastAPI
API_BASE_URL = "http://127.0.0.1:8000"  # Internal FastAPI server
API_TIMEOUT = 120  # 2 minutes for processing heavy files

# File upload processing
if uploaded_file is not None:
    with st.spinner(f"🔄 Processing {uploaded_file.name}... This may take a moment for large files."):
        try:
            # Prepare file data
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            endpoint_dict = {
                "PDF": "upload_pdf",
                "Image": "upload_image",
                "Audio": "upload_audio"
            }
            
            # Make API call with timeout
            response = requests.post(
                f"{API_BASE_URL}/api/{endpoint_dict[mode]}",  # Note: /api prefix from startup.py
                files=files, 
                timeout=API_TIMEOUT,
                headers={"Connection": "close"}
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    st.success(f"✅ {uploaded_file.name} processed and indexed successfully!")
                    st.balloons()  # Fun animation
                    st.session_state.uploaded_file = uploaded_file.name
                    st.session_state.mode = mode
                else:
                    st.error(f"❌ Processing failed: {result.get('message', 'Unknown error')}")
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            st.error(f"❌ Processing timed out after {API_TIMEOUT}s. Try a smaller file.")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend API. Ensure FastAPI server is running.")
        except Exception as e:
            st.error(f"❌ Unexpected error during upload: {str(e)}")
            st.exception(e)

# Query input
st.markdown("---")
query = st.text_input(
    "Enter your query:", 
    placeholder="Ask questions about your uploaded content...",
    help="Type your question related to the uploaded PDF/image/audio"
)

# Query processing
if st.button("💡 Ask", type="primary", disabled=not query) and query:
    if "uploaded_file" not in st.session_state:
        st.warning("⚠️ Please upload a file first before asking questions.")
    else:
        with st.spinner("🤔 Generating answer... This may take up to 2 minutes."):
            try:
                # Prepare query payload
                payload = {
                    "query": query.strip(),
                    "mode": st.session_state.mode.lower()
                }
                
                # Make API call
                response = requests.post(
                    f"{API_BASE_URL}/api/query",
                    json=payload,
                    timeout=API_TIMEOUT,
                    headers={"Content-Type": "application/json", "Connection": "close"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("answer", "No answer generated.")
                    
                    # Styled answer display
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(
                            f"""
                            <div style="
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                color: white;
                                padding: 24px;
                                border-radius: 16px;
                                font-size: 16px;
                                line-height: 1.6;
                                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                                margin-bottom: 20px;
                            ">
                                <h3 style="margin-top:0; font-size: 20px;">💡 Answer</h3>
                                <p style="margin: 12px 0;">{answer}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.markdown("⭐")
                        if st.button("⭐", key="star"):
                            st.balloons()
                            
                elif response.status_code == 400:
                    error_detail = response.json().get("detail", "Bad request")
                    st.error(f"❌ Query error: {error_detail}")
                    if "No PDF uploaded" in error_detail or "No image uploaded" in error_detail:
                        st.info("💡 Upload a file first, then try your query again.")
                else:
                    st.error(f"❌ API Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                st.error(f"❌ Query timed out after {API_TIMEOUT}s. Try a simpler question.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Please refresh the page.")
            except Exception as e:
                st.error(f"❌ Unexpected query error: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="
        text-align: center;
        padding: 20px;
        color: #666;
        font-size: 14px;
        border-top: 1px solid #eee;
    ">
        <strong>Powered by:</strong> 
        <span style='color:#4A90E2;'>CLIP</span> | 
        <span style='color:#F39C12;'>Wav2Vec2</span> | 
        <span style='color:#8E44AD;'>LangChain</span> | 
        <span style='color:#E74C3C;'>Gemini 1.5 Flash</span>
        <br>
        <small>Deployed on Hugging Face Spaces</small>
    </div>
    """,
    unsafe_allow_html=True
)

# Initial info message
if not st.session_state.get("initialized", False):
    st.info(
        "🎉 **Welcome to Multimodal RAG!** Upload a PDF, image, or audio file to get started. "
        "The system will process your content and enable smart queries across all modalities."
    )
    st.session_state.initialized = True
