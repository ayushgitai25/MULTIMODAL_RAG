import streamlit as st
import requests

st.set_page_config(
    page_title="Multimodal RAG", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    margin-bottom: 2rem;
}
.tech-pills {
    display: flex;
    justify-content: center;
    gap: 15px;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.tech-pill {
    background: rgba(255,255,255,0.2);
    padding: 8px 16px;
    border-radius: 25px;
    font-size: 14px;
    font-weight: 500;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.3);
}
.content-card {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    margin: 1rem 0;
}
.upload-section {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    margin-bottom: 1rem;
    border-left: 5px solid #667eea;
}
.answer-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
}
.context-card {
    background: #fff9e6;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 6px solid #f39c12;
    margin: 1rem 0;
}
.stExpander > div > label > div > span {
    font-weight: bold;
    color: #2c3e50;
}
.footer {
    position: fixed;
    bottom: 10px;
    width: 100%;
    text-align: center;
    font-size: 14px;
    color: #7f8c8d;
    font-style: normal;
    z-index: 1000;
}
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 3rem;">🤖 Multimodal RAG</h1>
    <p style="margin: 0; font-size: 1.1rem; opacity: 0.9;">Advanced Retrieval-Augmented Generation for Multiple Data Types</p>
    <div class="tech-pills">
        <span class="tech-pill">🧠 LangChain</span>
        <span class="tech-pill">🖼️ CLIP</span>
        <span class="tech-pill">🎵 Wav2Vec2</span>
        <span class="tech-pill">📊 FAISS</span>
        <span class="tech-pill">✨ Gemini</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main content in columns for better layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 🚀 Quick Start")
    
    # Tabbed interface for modes
    tab1, tab2, tab3 = st.tabs(["📄 PDF", "🖼️ Image", "🎵 Audio"])
    
    with tab1:
        st.markdown("**Upload PDF Document**")
        uploaded_file_pdf = st.file_uploader(
            "📄 Choose a PDF file",
            type="pdf",
            help="Upload PDF documents for text extraction and semantic search"
        )
        if uploaded_file_pdf:
            st.info(f"📊 File: **{uploaded_file_pdf.name}** | Size: {uploaded_file_pdf.size/1024:.1f} KB")
    
    with tab2:
        st.markdown("**Upload Image File**")
        uploaded_file_image = st.file_uploader(
            "🖼️ Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload images for visual semantic search with CLIP embeddings"
        )
        if uploaded_file_image:
            st.info(f"🖼️ File: **{uploaded_file_image.name}** | Size: {uploaded_file_image.size/1024:.1f} KB")
    
    with tab3:
        st.markdown("**Upload Audio File**")
        uploaded_file_audio = st.file_uploader(
            "🎵 Choose an audio file",
            type=["wav", "mp3"],
            help="Upload audio files for speech-to-text and audio embedding"
        )
        if uploaded_file_audio:
            st.info(f"🎵 File: **{uploaded_file_audio.name}** | Size: {uploaded_file_audio.size/1024:.1f} KB")
    
    # Process upload button
    if (uploaded_file_pdf or uploaded_file_image or uploaded_file_audio):
        col_up1, col_up2 = st.columns([3, 1])
        with col_up1:
            if st.button("📤 **Process & Index**", type="primary", use_container_width=True):
                pass  # Will be handled in main logic
        with col_up2:
            st.markdown("**Status:** Ready to process")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### ❓ Query Interface")
    
    # Query input with enhanced styling
    query = st.text_area(
        "💭 **Enter your query here:**",
        placeholder="Ask questions about your uploaded PDF, describe what you see in images, or inquire about audio content...",
        height=100,
        help="Be specific about what you're looking for. The system will retrieve relevant context from your uploaded files."
    )
    
    # Mode selection (simplified since we have tabs)
    col_mode1, col_mode2, col_mode3, col_mode4 = st.columns(4)
    with col_mode1:
        mode_pdf = st.radio("Mode:", ["PDF", "Image", "Audio"], key="mode_select", horizontal=True, label_visibility="collapsed")
    
    # Ask button with status
    col_ask1, col_ask2 = st.columns([3, 1])
    with col_ask1:
        ask_button = st.button("🔍 **Generate Answer**", type="primary", use_container_width=True, disabled=not query)
    with col_ask2:
        status_placeholder = st.empty()
    
    st.markdown('</div>', unsafe_allow_html=True)

# File processing logic (consolidated)
uploaded_file = None
mode = None
if uploaded_file_pdf:
    uploaded_file = uploaded_file_pdf
    mode = "PDF"
elif uploaded_file_image:
    uploaded_file = uploaded_file_image
    mode = "Image"
elif uploaded_file_audio:
    uploaded_file = uploaded_file_audio
    mode = "Audio"

# Process file upload
if uploaded_file and st.session_state.get('file_processed', False) != uploaded_file.name:
    with st.spinner(f"🔄 Processing {uploaded_file.name}... This may take a moment."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        endpoint_dict = {
            "PDF": "upload_pdf",
            "Image": "upload_image", 
            "Audio": "upload_audio"
        }
        
        status_placeholder.info("📤 Uploading and processing...")
        resp = requests.post(f"http://localhost:8000/{endpoint_dict[mode]}", files=files, timeout=60)
        
        if resp.ok:
            st.session_state['file_processed'] = uploaded_file.name
            st.success(f"✅ **{uploaded_file.name} processed and indexed successfully!**")
            status_placeholder.success("✅ Processing complete!")
            st.balloons()
        else:
            status_placeholder.error(f"❌ Processing failed: {resp.text}")
            st.error(f"Failed to process {uploaded_file.name}: {resp.text}")

# Query processing with enhanced UI
if ask_button and query and mode:
    status_placeholder.info("🧠 Generating intelligent response...")
    
    with st.spinner("🤔 Analyzing your query and retrieving relevant context..."):
        try:
            response = requests.post(
                "http://localhost:8000/query", 
                json={"query": query, "mode": mode.lower()},
                timeout=90
            )
            result = response.json()
            
            # Enhanced Answer Card
            st.markdown(f'''
            <div class="answer-card">
                <h3 style="margin-top: 0; display: flex; align-items: center;">
                    💡 AI Response
                    <span style="margin-left: auto; font-size: 0.9em; opacity: 0.8;">Generated by Gemini</span>
                </h3>
                <div style="font-size: 1.1em; line-height: 1.6;">
                    {result.get("answer", "No answer generated. Please try rephrasing your query.")}
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Enhanced Context Section
            context = result.get("context", None)
            if context:
                st.markdown(f'''
                <div class="context-card">
                    <h4 style="margin: 0 0 1rem 0; display: flex; align-items: center;">
                        📚 Retrieved Context <span style="font-size: 0.9em; color: #7f8c8d; margin-left: 10px;">({len(context if isinstance(context, list) else [context])} sources)</span>
                    </h4>
                </div>
                ''', unsafe_allow_html=True)
                
                # Context expanders with better styling
                for idx, ctx in enumerate(context if isinstance(context, list) else [context]):
                    with st.expander(f"📖 Context {idx+1}: {'Image' if isinstance(ctx, dict) and 'image' in str(ctx).lower() else 'Text'}", expanded=False):
                        if isinstance(ctx, dict) and 'text' in ctx:
                            st.markdown(f"**Source:** {ctx.get('source', 'Document')}")
                            st.write(ctx['text'])
                        else:
                            st.markdown(f"**Source:** Retrieved Document")
                            st.write(str(ctx))
            
            status_placeholder.success("✅ Response generated successfully!")
            
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Please try a shorter query or check the backend service.")
            status_placeholder.error("⏱️ Generation timeout")
        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")
            status_placeholder.error("⚠️ Processing error")

# Instructions when no file is uploaded
if not (uploaded_file_pdf or uploaded_file_image or uploaded_file_audio):
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("""
    ### 👋 Welcome to Multimodal RAG
    
    **How it works:**
    1. **Upload** a file using the tabs on the left (PDF, Image, or Audio)
    2. **Wait** for processing and indexing to complete  
    3. **Ask** questions in natural language about your uploaded content
    4. **Get** intelligent, context-aware responses powered by advanced AI
    
    **Supported formats:**
    - 📄 **PDF**: Text extraction and semantic search
    - 🖼️ **Images**: Visual understanding and description  
    - 🎵 **Audio**: Speech-to-text transcription and analysis
    
    **Examples:**
    - *PDF*: "What are the main points discussed in this document?"
    - *Image*: "What's happening in this picture?" 
    - *Audio*: "What did the speaker say about the main topic?"
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Enhanced Footer
st.markdown("""
<div class="footer">
    <div style="max-width: 800px; margin: 0 auto; padding: 1rem;">
        <p style="margin: 0; text-align: center;">
            🔬 Built with cutting-edge AI technologies | 
            <span style="color: #3498db;">Production-ready RAG pipeline</span> | 
            🚀 Deployed on Hugging Face Spaces
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9em; opacity: 0.7;">
            © 2025 | Advanced Multimodal AI Assistant
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
