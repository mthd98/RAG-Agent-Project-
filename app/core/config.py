
import os
from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)
# Load the .env file from the 'app' directory
# This is important so the app finds it when run from the root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
class Settings:
    # --- Database Settings ---
    DB_USER = os.getenv("DB_USER", "crewuser")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "strongpassword")
    DB_NAME = os.getenv("DB_NAME", "crewai")
    DB_HOST = os.getenv("DB_HOST", "postgres")
    DB_PORT = int(os.getenv("DB_PORT", 5432))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "default_collection")

    # --- Ollama Settings ---
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
    LLM_MODEL = os.getenv("LLM_MODEL", "ollama/qwen3:0.6b")

    # --- API Security ---
    API_KEY = os.getenv("API_KEY", "default_unsafe_key")
    # --- Phoenix (Arize Phoenix) Settings ---
    PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006")
    # --- Embedding Model ---
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "ollama/qwen3:0.6b")

# Instantiate settings
settings = Settings()

logger.info(f"Loading config: DB_HOST={settings.DB_HOST}, OLLAMA_URL={settings.OLLAMA_URL}")