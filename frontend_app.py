import streamlit as st
import requests

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

if uploaded_file:
    with st.spinner(f"Uploading {uploaded_file.name}..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        endpoint_dict = {
            "PDF": "upload_pdf",
            "Image": "upload_image",
            "Audio": "upload_audio"
        }
        resp = requests.post(f"http://localhost:8000/{endpoint_dict[mode]}", files=files)
        if resp.ok:
            st.success(f"✅ {uploaded_file.name} processed successfully!")
        else:
            st.error(f"❌ Failed to process {uploaded_file.name}: {resp.text}")

query = st.text_input("Enter your query:")

if st.button("Ask") and query:
    with st.spinner("Generating answer..."):
        response = requests.post("http://localhost:8000/query", json={"query": query, "mode": mode.lower()})
        result = response.json()
    
    # Styled answer box with emoji and background
    st.markdown(
        f'''
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            font-size: 18px;
            box-shadow: 3px 3px 10px rgba(0,0,0,0.2);
            margin-bottom: 15px;
        ">
            <h3 style="margin-top:0;">💡 Answer</h3>
            <p>{result.get("answer", "No answer found.")}</p>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Context section with separate styling and emoji
    context = result.get("context", None)
    if context:
        st.markdown(
            f'''
            <div style="
                background-color: #FFEFB0;
                padding: 15px;
                border-radius: 10px;
                border-left: 6px solid #FFCB05;
                font-size: 16px;
                max-height: 300px;
                overflow-y: auto;
            ">
                <h4>📚 Retrieved Context:</h4>
            </div>
            ''',
            unsafe_allow_html=True
        )
        # Display individual context entries in collapsible boxes
        for idx, ctx in enumerate(context if isinstance(context, list) else [context]):
            with st.expander(f"Context snippet {idx+1}"):
                if isinstance(ctx, dict) and 'text' in ctx:
                    st.write(ctx['text'])
                else:
                    st.write(ctx)
else:
    st.info("Please upload a file and enter a query to get started.")

st.markdown(
    """
    <div style="
        position: fixed;
        bottom: 10px;
        width: 100%;
        text-align: left;
        font-size: 14px;
        color: #666666;
        font-style: normal;
        font-weight: normal;
        user-select: none;
        z-index: 1000;
    ">
        Powered by 🔥 <span style='color:#4A90E2;'>CLIP</span> | <span style='color:#F39C12;'>Wav2Vec2</span> | 
        <span style='color:#8E44AD;'>LangChain</span> | <span style='color:#E74C3C;'>Gemini 2.5 Flash</span>
    </div>
    """,
    unsafe_allow_html=True
)
