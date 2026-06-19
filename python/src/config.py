import os
from pathlib import Path
from dotenv import load_dotenv

# Locate and load .env from project root (two levels up from this file)
_project_root = Path(__file__).resolve().parent.parent.parent
_env_file = _project_root / ".env"
if _env_file.exists():
    load_dotenv(_env_file)

# Each environment maps to exactly one inference backend.
#   test — local dev machine, OpenRouter cloud API        (served by python/src)
#   dev  — EC2, HF + transformers (eval pipeline)          (run_baseline.py, NOT this proxy)
#   demo — EC2, Ollama (weights downloaded onto EC2 GPU)   (served by python/src)
ENV_BACKEND = {
    "test": "openrouter",
    "dev": "hf",
    "demo": "ollama",
}

# Single source of truth for model identities.
#   role -> backend -> identifier   (None = role not available on that backend)
MODELS = {
    "main": {
        # `test` is local-dev streaming/plumbing convenience only (see two-track
        # separation: demo/test output never enters the research comparison table),
        # so the OpenRouter id here doesn't need to BE Qwen3-VL — it needs to reliably
        # accept a real head-CT slice. qwen/qwen3-vl-30b-a3b-instruct was tried and
        # falsely rejected a valid CT-ICH slice (twice, with two different excuses);
        # minimax/minimax-m3 has been verified multiple times on the same slice.
        "openrouter": os.getenv("OPENROUTER_MAIN_MODEL", "minimax/minimax-m3"),
        "ollama": os.getenv("OLLAMA_MAIN_MODEL", "qwen3-vl:4b"),
        "hf": os.getenv("HF_MAIN_MODEL", "Qwen/Qwen3-VL-4B-Instruct"),
    },
    "medical_baseline": {
        "openrouter": None,  # MedGemma is not hosted on OpenRouter
        "ollama": os.getenv("OLLAMA_MEDICAL_MODEL", "medgemma:4b"),
        "hf": os.getenv("HF_MEDICAL_MODEL", "google/medgemma-4b-it"),
    },
}


class Config:
    # Current environment and which model role to serve
    ENV = os.getenv("ENV", "test").lower()
    MODEL_ROLE = os.getenv("MODEL_ROLE", "main").lower()

    # OpenRouter (test)
    OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

    # Ollama (demo)
    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "")

    @property
    def backend(self) -> str:
        """The inference backend for the current ENV (openrouter / hf / ollama)."""
        return ENV_BACKEND.get(self.ENV)

    @property
    def model(self):
        """The model id for the current MODEL_ROLE on the current backend (or None)."""
        return MODELS.get(self.MODEL_ROLE, {}).get(self.backend)


# Instantiate a singleton object for global access
config = Config()
