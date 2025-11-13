from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your-default-gemini-key")
