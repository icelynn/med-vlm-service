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
#   demo — EC2, HF + transformers (same engine as eval)    (served by python/src)
#          Ollama dropped 2026-06-21: upstream CUDA kernel bug on T4/sm_75 in
#          Ollama 0.30.x (ollama/ollama#16449), see docs/Environments/aws_setup_guide.md
ENV_BACKEND = {
    "test": "openrouter",
    "dev": "hf",
    "demo": "hf",
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
        "hf": os.getenv("HF_MAIN_MODEL", "Qwen/Qwen3-VL-4B-Instruct"),
    },
    "medical_baseline": {
        "openrouter": None,  # MedGemma is not hosted on OpenRouter
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

    # Findings-conditioned two-stage RAG (opt-in, HF backend only). Default off:
    # the demo's validated refusal behavior was tested without this, so it must
    # not change anything unless explicitly enabled.
    TWO_STAGE_RAG = os.getenv("TWO_STAGE_RAG", "false").lower() == "true"

    # Image-retrieval few-shot RAG (R2, opt-in, HF backend only). Default off,
    # same reason as TWO_STAGE_RAG. Reproduces the eval-measured R2-B config
    # (RSNA pool, histogram-matched to CT-ICH) -- see python/rag/providers.py
    # and python/eval/run_baseline.py's --context image.
    IMAGE_RETRIEVAL_RAG = os.getenv("IMAGE_RETRIEVAL_RAG", "false").lower() == "true"
    R2_INDEX_DIR = os.getenv("R2_INDEX_DIR",
                             str(_project_root / "data" / "rag" / "image_index_b_harmonized"))
    R2_POOL_DIR = os.getenv("R2_POOL_DIR",
                            str(_project_root / "data" / "rsna" / "pool_images_harmonized"))

    @property
    def backend(self) -> str:
        """The inference backend for the current ENV (openrouter / hf)."""
        return ENV_BACKEND.get(self.ENV)

    @property
    def model(self):
        """The model id for the current MODEL_ROLE on the current backend (or None)."""
        return MODELS.get(self.MODEL_ROLE, {}).get(self.backend)


# Instantiate a singleton object for global access
config = Config()
