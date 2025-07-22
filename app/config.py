import os

DATABASE = os.environ.get("DATABASE", "jobs.db")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")
_default_model = os.environ.get("OLLAMA_MODEL", "llama3")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", _default_model)
OLLAMA_REPHRASE_MODEL = os.environ.get("OLLAMA_REPHRASE_MODEL", _default_model)
OLLAMA_ENABLED = bool(OLLAMA_BASE_URL)
BUILD_NUMBER = os.environ.get("GITHUB_RUN_NUMBER") or os.environ.get("BUILD_NUMBER", "dev")
MIN_FEEDBACK_FOR_TRAINING = int(os.environ.get("MIN_FEEDBACK_FOR_TRAINING", "200"))
