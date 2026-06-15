import os
from pathlib import Path
from dotenv import load_dotenv

# Locate and load .env from project root (two levels up from this file)
_project_root = Path(__file__).resolve().parent.parent.parent
_env_file = _project_root / ".env"
if _env_file.exists():
    load_dotenv(_env_file)

class Config:
    # Read the current environment, default to local
    ENV = os.getenv("ENV", "local").lower()
    
    # Dev environment settings (Ollama)
    DEV_API_URL = os.getenv("DEV_API_URL", "")
    DEV_API_KEY = os.getenv("DEV_API_KEY", "")
    DEV_MODEL = os.getenv("DEV_MODEL", "llama3.2-vision")
    
    # Local sandbox environment settings (OpenRouter)
    LOCAL_API_URL = os.getenv("LOCAL_API_URL", "")
    LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "")
    LOCAL_MODEL = os.getenv("LOCAL_MODEL", "qwen/qwen3.7-plus")

# Instantiate a singleton object for global access
config = Config()
