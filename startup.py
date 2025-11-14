"""
Uvicorn server startup for FastAPI + Streamlit integration.
Imports existing app.py and runs on port 7860 with Streamlit subprocess.
"""

import os
import subprocess
import signal
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from app import app as fastapi_app  # Import your existing FastAPI app

# --------- Streamlit Subprocess Management --------- #
streamlit_process = None

def start_streamlit():
    """Start Streamlit as subprocess on internal port 8501."""
    global streamlit_process
    try:
        # Create temporary Streamlit config for internal use
        streamlit_config = """
[server]
port = 8501
address = "127.0.0.1"
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
"""
        config_path = os.path.join(os.getcwd(), ".streamlit", "config.toml")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, "w") as f:
            f.write(streamlit_config)
        
        # Set environment for Streamlit config
        os.environ["STREAMLIT_SERVER_PORT"] = "8501"
        os.environ["STREAMLIT_SERVER_ADDRESS"] = "127.0.0.1"
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        
        # Start Streamlit subprocess
        streamlit_process = subprocess.Popen([
            "streamlit", "run", "frontend_app.py",
            "--server.headless=true",
            "--server.port=8501",
            "--server.address=127.0.0.1",
            "--browser.gatherUsageStats=false"
        ], 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL,
        cwd=os.getcwd(),
        preexec_fn=os.setsid  # For proper signal handling
        )
        
        print("Streamlit subprocess started on port 8501 (PID: {})".format(streamlit_process.pid))
        return True
        
    except Exception as e:
        print(f"Failed to start Streamlit subprocess: {e}")
        return False

def stop_streamlit():
    """Gracefully stop Streamlit subprocess."""
    global streamlit_process
    if streamlit_process and streamlit_process.poll() is None:
        try:
            # Send SIGTERM to process group
            os.killpg(os.getpgid(streamlit_process.pid), signal.SIGTERM)
            streamlit_process.wait(timeout=5)
            print("Streamlit subprocess stopped gracefully")
        except subprocess.TimeoutExpired:
            print("Force killing Streamlit subprocess")
            streamlit_process.kill()
        except Exception as e:
            print(f"Error stopping Streamlit: {e}")
    streamlit_process = None

# --------- Enhanced FastAPI with Lifespan --------- #
# Wrap your existing app with lifespan events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    print("🚀 Starting Multimodal RAG Application...")
    
    # Startup: Initialize Streamlit subprocess
    streamlit_started = start_streamlit()
    if not streamlit_started:
        print("⚠️  Warning: Streamlit failed to start. API will work, but UI may be limited.")
    
    # Additional startup logging
    print("✅ FastAPI application ready on port 7860")
    print("📡 API endpoints available at /upload_pdf, /upload_image, /upload_audio, /query")
    if streamlit_started:
        print("🌐 Streamlit UI available internally on port 8501")
    
    yield
    
    # Shutdown: Clean up Streamlit subprocess
    print("🛑 Shutting down application...")
    stop_streamlit()
    print("✅ Application shutdown complete")

# Create enhanced app with lifespan
enhanced_app = FastAPI(
    title="Multimodal RAG API",
    description="FastAPI backend for multimodal RAG with PDF, Image, and Audio support",
    version="1.0.0",
    lifespan=lifespan
)

# Mount your existing app under /api prefix to avoid conflicts
enhanced_app.mount("/api", fastapi_app)

# Add root endpoint for health check and UI proxy
@enhanced_app.get("/")
async def root():
    """Root endpoint - health check and UI redirect."""
    if streamlit_process and streamlit_process.poll() is None:
        return {
            "message": "Multimodal RAG API is running",
            "status": "healthy",
            "streamlit_ui": "available internally",
            "api_endpoints": ["/api/upload_pdf", "/api/upload_image", "/api/upload_audio", "/api/query"],
            "docs": "/api/docs"
        }
    return {
        "message": "Multimodal RAG API is running",
        "status": "healthy - Streamlit UI unavailable",
        "api_endpoints": ["/api/upload_pdf", "/api/upload_image", "/api/upload_audio", "/api/query"],
        "docs": "/api/docs"
    }

@enhanced_app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "multimodal-rag-api"}

# --------- Main Entry Point --------- #
if __name__ == "__main__":
    # Run with uvicorn on port 7860 for Hugging Face Spaces
    uvicorn.run(
        "startup:enhanced_app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
        reload=False,  # Disable reload in production/Docker
        workers=1,     # Single worker for HF Spaces
        access_log=True
    )
