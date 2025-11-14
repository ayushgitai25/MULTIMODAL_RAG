import streamlit as st
import requests
import os

st.set_page_config(page_title="🎨 Multimodal RAG", page_icon="🤖", layout="centered")

# Title with emoticon and color
st.markdown('<h1 style="color:#6A5ACD; font-weight:bold;">🤖 Multimodal RAG with Gemini</h1>', unsafe_allow_html=True)

# Radio buttons for mode selection
mode = st.radio(
    "Choose mode:",
    ["PDF", "Image", "Audio"],
    index=0,
    horizontal=True,
    label_visibility="visible"
)

uploaded_file = None
if mode == "PDF":
    uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")
elif mode == "Image":
    uploaded_file = st.file_uploader("🖼️ Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
elif mode == "Audio":
    uploaded_file = st.file_uploader("🎵 Upload Audio (WAV/MP3)", type=["wav", "mp3"])

# Use RELATIVE URLs for Nginx proxy - NO localhost:8000
API_BASE_URL = "/api"  # Nginx proxies /api/* to FastAPI on port 8000

if uploaded_file:
    with st.spinner(f"Uploading {uploaded_file.name}..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        endpoint_dict = {
            "PDF": "upload_pdf",
            "Image": "upload_image",
            "Audio": "upload_audio"
        }
        try:
            # Use relative URL - Nginx handles proxying to FastAPI
            resp = requests.post(f"{API_BASE_URL}/{endpoint_dict[mode]}", files=files, timeout=60)
            if resp.ok:
                result = resp.json()
                if result.get("status") == "success":
                    st.success(f"✅ {uploaded_file.name} processed successfully!")
                else:
                    st.error(f"❌ Processing failed: {result.get('message', 'Unknown error')}")
            else:
                st.error(f"❌ API Error {resp.status_code}: {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request failed: {e}")
            # Fallback error for proxy issues
            if "403" in str(e):
                st.info("💡 If you see 403 errors, ensure Nginx proxy is configured correctly.")

query = st.text_input("Enter your query:")

if st.button("Ask") and query:
    with st.spinner("Generating answer..."):
        try:
            # Relative URL for query endpoint
            response = requests.post(
                f"{API_BASE_URL}/query", 
                json={"query": query, "mode": mode.lower()}, 
                timeout=120
            )
            result = response.json()
            # Styled answer box
            st.markdown(
                f'''
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 15px;
                    font-size: 18px;
                    box-shadow: 3px 3px 10px rgba(0,0,0,0.2);
                ">
                    <h3 style="margin-top:0;">💡 Answer</h3>
                    <p>{result.get("answer", "No answer found.")}</p>
                </div>
                ''',
                unsafe_allow_html=True
            )
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Query failed: {e}")
            if "403" in str(e):
                st.info("🔧 403 errors often mean proxy misconfiguration. Check Nginx logs.")

else:
    st.info("Please upload a file and enter a query to get started.")

# Footer
st.markdown(
    """
    <div style="
        position: fixed; bottom: 10px; width: 100%; text-align: left; font-size: 14px;
        color: #666666; font-style: normal; font-weight: normal; user-select: none; z-index: 1000;
    ">
        Powered by 🔥 <span style='color:#4A90E2;'>CLIP</span> | <span style='color:#F39C12;'>Wav2Vec2</span> | 
        <span style='color:#8E44AD;'>LangChain</span> | <span style='color:#E74C3C;'>Gemini 2.5 Flash</span>
    </div>
    """,
    unsafe_allow_html=True
)
