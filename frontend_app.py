import streamlit as st
import requests
import time

st.set_page_config(
    page_title="🎨 Multimodal RAG", 
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
.status-processing {
    background: #fff3cd;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #ffc107;
}
.status-success {
    background: #d4edda;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #28a745;
}
.status-error {
    background: #f8d7da;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #dc3545;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'queries' not in st.session_state:
    st.session_state.queries = []

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

# Main content in columns
col1, col2 = st.columns([1, 2])

# Left column: Upload tabs (auto-processing)
with col1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 🚀 File Upload")
    
    # Tabbed interface for uploads
    tab1, tab2, tab3 = st.tabs(["📄 PDF Analysis", "🖼️ Image Understanding", "🎵 Audio Processing"])
    
    uploaded_file = None
    mode = None
    
    with tab1:
        st.markdown("**Upload PDF Document for Text Analysis**")
        uploaded_file_pdf = st.file_uploader(
            "📄 Choose a PDF file (max 50MB)",
            type="pdf",
            key="pdf_uploader",
            help="Upload PDF documents for text extraction and semantic search"
        )
        
        if uploaded_file_pdf:
            # Auto-process PDF
            if uploaded_file_pdf.name not in st.session_state.processed_files:
                with st.spinner(f"🔄 Processing {uploaded_file_pdf.name}..."):
                    try:
                        files = {"file": (uploaded_file_pdf.name, uploaded_file_pdf.getvalue())}
                        resp = requests.post("http://localhost:8000/upload_pdf", files=files, timeout=120)
                        
                        if resp.ok:
                            st.session_state.processed_files.add(uploaded_file_pdf.name)
                            st.session_state.uploaded_files[uploaded_file_pdf.name] = {
                                "mode": "PDF",
                                "timestamp": time.time()
                            }
                            st.success(f"✅ **{uploaded_file_pdf.name} processed successfully!**")
                            st.info(f"📊 File Size: {uploaded_file_pdf.size / 1024:.1f} KB | **Ready for querying!**")
                            uploaded_file = uploaded_file_pdf
                            mode = "PDF"
                        else:
                            st.error(f"❌ Failed to process PDF: {resp.text}")
                    except Exception as e:
                        st.error(f"⚠️ Processing error: {str(e)}")
            
            # Show file status
            if uploaded_file_pdf.name in st.session_state.processed_files:
                st.markdown(f'<div class="status-success">✅ **{uploaded_file_pdf.name} processed successfully!**</div>', unsafe_allow_html=True)
                st.info(f"📊 File Size: {uploaded_file_pdf.size / 1024:.1f} KB | **Ready for querying!**")
                uploaded_file = uploaded_file_pdf
                mode = "PDF"
    
    with tab2:
        st.markdown("**Upload Image for Visual Analysis**")
        uploaded_file_image = st.file_uploader(
            "🖼️ Choose an image file (max 10MB)",
            type=["jpg", "jpeg", "png"],
            key="image_uploader",
            help="Upload images for visual semantic search with CLIP embeddings"
        )
        
        if uploaded_file_image:
            # Auto-process Image
            if uploaded_file_image.name not in st.session_state.processed_files:
                with st.spinner(f"🔄 Processing {uploaded_file_image.name}..."):
                    try:
                        files = {"file": (uploaded_file_image.name, uploaded_file_image.getvalue())}
                        resp = requests.post("http://localhost:8000/upload_image", files=files, timeout=60)
                        
                        if resp.ok:
                            st.session_state.processed_files.add(uploaded_file_image.name)
                            st.session_state.uploaded_files[uploaded_file_image.name] = {
                                "mode": "Image",
                                "timestamp": time.time()
                            }
                            st.success(f"✅ **{uploaded_file_image.name} processed successfully!**")
                            st.info(f"🖼️ File Size: {uploaded_file_image.size / 1024:.1f} KB | **Ready for visual queries!**")
                            uploaded_file = uploaded_file_image
                            mode = "Image"
                        else:
                            st.error(f"❌ Failed to process image: {resp.text}")
                    except Exception as e:
                        st.error(f"⚠️ Processing error: {str(e)}")
            
            # Show file status
            if uploaded_file_image.name in st.session_state.processed_files:
                st.markdown(f'<div class="status-success">✅ **{uploaded_file_image.name} processed successfully!**</div>', unsafe_allow_html=True)
                st.info(f"🖼️ File Size: {uploaded_file_image.size / 1024:.1f} KB | **Ready for visual queries!**")
                uploaded_file = uploaded_file_image
                mode = "Image"
    
    with tab3:
        st.markdown("**Upload Audio for Speech Analysis**")
        uploaded_file_audio = st.file_uploader(
            "🎵 Choose an audio file (max 25MB)",
            type=["wav", "mp3", "m4a"],  # Full audio format support
            key="audio_uploader",
            help="Upload audio files (WAV, MP3, M4A) for speech-to-text transcription and semantic analysis"
        )
        
        if uploaded_file_audio:
            # Auto-process Audio
            if uploaded_file_audio.name not in st.session_state.processed_files:
                with st.spinner(f"🔄 Processing {uploaded_file_audio.name}... This may take longer for audio files."):
                    try:
                        files = {"file": (uploaded_file_audio.name, uploaded_file_audio.getvalue())}
                        resp = requests.post("http://localhost:8000/upload_audio", files=files, timeout=180)
                        
                        if resp.ok:
                            st.session_state.processed_files.add(uploaded_file_audio.name)
                            st.session_state.uploaded_files[uploaded_file_audio.name] = {
                                "mode": "Audio",
                                "timestamp": time.time()
                            }
                            st.success(f"✅ **{uploaded_file_audio.name} processed successfully!**")
                            st.info(f"🎵 File Size: {uploaded_file_audio.size / 1024:.1f} KB | **Ready for audio queries!**")
                            uploaded_file = uploaded_file_audio
                            mode = "Audio"
                        else:
                            st.error(f"❌ Failed to process audio: {resp.text}")
                    except Exception as e:
                        st.error(f"⚠️ Processing error: {str(e)}")
            
            # Show file status
            if uploaded_file_audio.name in st.session_state.processed_files:
                st.markdown(f'<div class="status-success">✅ **{uploaded_file_audio.name} processed successfully!**</div>', unsafe_allow_html=True)
                st.info(f"🎵 File Size: {uploaded_file_audio.size / 1024:.1f} KB | **Ready for audio queries!**")
                uploaded_file = uploaded_file_audio
                mode = "Audio"
    
    st.markdown('</div>', unsafe_allow_html=True)

# Right column: Query interface
with col2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### ❓ Ask Questions")
    
    # Show current active mode and status
    if mode and st.session_state.processed_files:
        current_mode = mode
        current_file = list(st.session_state.processed_files)[-1]
        st.markdown(f"**Current Mode:** {current_mode}")
        st.markdown(f'<div class="status-success">📁 **{current_file} loaded** ({current_mode} mode)</div>', unsafe_allow_html=True)
        
        # Show recent files if multiple
        if len(st.session_state.processed_files) > 1:
            recent_files = sorted(st.session_state.uploaded_files.items(), 
                                key=lambda x: x[1]['timestamp'], reverse=True)[:3]
            with st.expander(f"📁 Recent Files ({len(recent_files)})", expanded=False):
                for filename, info in recent_files:
                    st.write(f"• **{info['mode']}**: {filename}")
    else:
        st.warning("👆 **Please upload a file first** to enable querying")
    
    # Query input (disabled if no mode selected)
    query = st.text_area(
        "💭 **Enter your question about the uploaded content:**",
        placeholder="Examples:\n• PDF: 'What are the main findings of this document?'\n• Image: 'What's happening in this picture?'\n• Audio: 'What was the main topic discussed?'",
        height=120,
        disabled=not mode,
        help="Ask natural language questions about your uploaded file. The AI will search through the content to provide relevant answers."
    )
    
    # Generate Answer button
    col_ask1, col_ask2 = st.columns([3, 1])
    with col_ask1:
        ask_button = st.button("🔍 **Generate AI Answer**", type="primary", use_container_width=True, disabled=not (query and mode))
    with col_ask2:
        if mode:
            st.success("✅ **Ready to query!**")
        else:
            st.info("📤 **Upload file first**")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Results section (spans full width)
if ask_button and query and mode:
    result_container = st.container()
    with result_container:
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.info("🧠 **Analyzing your query and searching relevant context...**")
        progress_bar.progress(30)
        
        try:
            # Make query request
            progress_bar.progress(70)
            response = requests.post(
                "http://localhost:8000/query", 
                json={"query": query, "mode": mode.lower()},
                timeout=120
            )
            progress_bar.progress(100)
            
            result = response.json()
            
            # Clear progress
            status_text.success("✅ **Answer generated successfully!**")
            progress_bar.empty()
            
            # Enhanced Answer Display
            st.markdown(f'''
            <div class="answer-card">
                <h3 style="margin-top: 0; display: flex; align-items: center;">
                    💡 **AI-Powered Answer**
                    <span style="margin-left: auto; font-size: 0.9em; opacity: 0.8; background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 15px;">
                        {mode} Mode
                    </span>
                </h3>
                <div style="font-size: 1.1em; line-height: 1.7; margin-top: 1rem;">
                    {result.get("answer", "No answer generated. Please try rephrasing your query or check the backend.")}
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Context/Retrieved Information
            context = result.get("context", None)
            if context:
                context_items = context if isinstance(context, list) else [context]
                st.markdown(f'''
                <div class="context-card">
                    <h4 style="margin: 0 0 1rem 0; display: flex; align-items: center;">
                        📚 **Retrieved Context** 
                        <span style="font-size: 0.9em; color: #7f8c8d; margin-left: 10px;">
                            ({len([c for c in context_items if c])} sources found)
                        </span>
                    </h4>
                </div>
                ''', unsafe_allow_html=True)
                
                # Individual context items
                for idx, ctx in enumerate(context_items):
                    with st.expander(f"📖 Context {idx+1}: {mode} Source", expanded=(idx == 0)):
                        if isinstance(ctx, dict) and 'text' in ctx:
                            st.markdown(f"**Source Type:** {ctx.get('source', 'Document').title()}")
                            if 'image_id' in ctx:
                                st.markdown(f"**Image ID:** `{ctx['image_id']}`")
                            st.write(ctx['text'])
                        else:
                            st.markdown(f"**Content:**")
                            st.write(str(ctx))
            
            # Query history (simple)
            st.session_state.queries.append({
                "query": query,
                "mode": mode,
                "timestamp": time.time()
            })
            
            if len(st.session_state.queries) > 1:
                with st.expander(f"📋 Recent Queries ({len(st.session_state.queries)})", expanded=False):
                    for i, q in enumerate(st.session_state.queries[-3:]):  # Last 3
                        st.markdown(f"**Q{i+1}:** *{q['mode']}* - {q['query'][:100]}{'...' if len(q['query']) > 100 else ''}")
            
        except requests.exceptions.Timeout:
            progress_bar.empty()
            status_text.error("⏱️ **Request timed out.** Please try a shorter query or check the backend service.")
            st.error("The query took too long to process. Try simplifying your question or uploading a smaller file.")
        except Exception as e:
            progress_bar.empty()
            status_text.error("⚠️ **Processing error occurred.**")
            st.error(f"An unexpected error occurred: {str(e)}")

# Welcome/Instructions when no files uploaded
if not st.session_state.processed_files:
    st.markdown('<div class="content-card" style="text-align: center;">', unsafe_allow_html=True)
    st.markdown("""
    ### 👋 Welcome to Multimodal RAG
    
    **Simple 3-Step Process:**
    1. **Choose** a tab (PDF, Image, or Audio) and upload your file
    2. **Wait** for automatic processing (happens instantly in the background)
    3. **Ask** natural language questions about your uploaded content
    
    **What each mode does:**
    - 📄 **PDF Analysis**: Extracts text and enables document Q&A
    - 🖼️ **Image Understanding**: Analyzes visual content and describes scenes  
    - 🎵 **Audio Processing**: Transcribes speech and answers about conversations (WAV, MP3, M4A supported)
    
    **Example Questions:**
    - *PDF*: "What are the key recommendations in this report?"
    - *Image*: "What objects can you identify in this photo?"
    - *Audio*: "What was the main topic of this recording?"
    """)
    
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    with col_ex1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e3f2fd, #bbdefb); padding: 1rem; border-radius: 10px; text-align: center;">
            <h4 style="color: #1976d2; margin-top: 0;">📄 Document Q&A</h4>
            <p style="color: #1565c0;">Upload reports, papers, or manuals</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_ex2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f3e5f5, #e1bee7); padding: 1rem; border-radius: 10px; text-align: center;">
            <h4 style="color: #7b1fa2; margin-top: 0;">🖼️ Visual AI</h4>
            <p style="color: #4a148c;">Analyze photos, diagrams, charts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_ex3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e8f5e8, #c8e6c9); padding: 1rem; border-radius: 10px; text-align: center;">
            <h4 style="color: #2e7d32; margin-top: 0;">🎵 Speech Analysis</h4>
            <p style="color: #1b5e20;">Process interviews, meetings, voice notes (WAV/MP3/M4A)</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# File management (if multiple files uploaded)
if len(st.session_state.processed_files) > 1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 📁 Manage Uploaded Files")
    
    col_manage1, col_manage2 = st.columns(2)
    with col_manage1:
        st.markdown("**Currently Active Files:**")
        recent_files = sorted(st.session_state.uploaded_files.items(), 
                            key=lambda x: x[1]['timestamp'], reverse=True)[:5]
        for filename, info in recent_files:
            mode_icon = "📄" if info["mode"] == "PDF" else "🖼️" if info["mode"] == "Image" else "🎵"
            st.write(f"{mode_icon} **{filename}** ({info['mode']})")
    
    with col_manage2:
        if st.button("🗑️ Clear All Files", type="secondary"):
            st.session_state.processed_files.clear()
            st.session_state.uploaded_files.clear()
            st.session_state.queries.clear()
            st.rerun()
        st.info("**Note:** Clearing files removes them from memory. Upload new files to query different content.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Enhanced Footer
st.markdown("""
<div class="footer">
    <div style="max-width: 800px; margin: 0 auto; padding: 1rem;">
        <p style="margin: 0; text-align: center;">
            🔬 Built with cutting-edge multimodal AI | 
            <span style="color: #3498db;">Production RAG Pipeline</span> | 
            🚀 Deployed on Hugging Face Spaces
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9em; opacity: 0.7;">
            © 2025 | Advanced AI Content Analysis Assistant | Stable & Optimized
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
