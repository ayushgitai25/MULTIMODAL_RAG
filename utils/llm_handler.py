import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from config import GOOGLE_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_llm():
    logger.info("Initializing Gemini LLM...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY
    )
    logger.info("Gemini LLM initialized.")
    return llm

def query_llm(llm, content):
    logger.info(f"Invoking Gemini LLM with context of length {len(content)}")
    response = llm.invoke([HumanMessage(content=content)])
    logger.info("LLM response received.")
    return response.content
