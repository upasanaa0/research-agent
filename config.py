import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"

EXA_API_KEY    = os.getenv("EXA_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# rest of config unchanged...

LLM_PROVIDER = "ollama"  # "ollama" or "nvidia"

# Ollama settings
OLLAMA_MODEL = "minimax-m2.5:cloud"
OLLAMA_TEMPERATURE = 0

# NVIDIA NIM / AI Endpoints settings. API key is read from NVIDIA_API_KEY env var.
NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
NVIDIA_TEMPERATURE = 1
NVIDIA_TOP_P = 0.95
NVIDIA_MAX_TOKENS = 16384
NVIDIA_REASONING_BUDGET = 16384
NVIDIA_ENABLE_THINKING = True

MAX_VERIFY_ATTEMPTS = 2
MAX_PDFS = 4

# Vector memory settings
CHROMA_PATH = "chroma_memory"
MEMORY_COLLECTION = "research_memory"
DIRECT_ANSWER_THRESHOLD = 0.85         # coarse filter before LLM coverage check
MEMORY_CONTEXT_THRESHOLD = 0.5           # include memory in planner/final context
MEMORY_CANDIDATE_TOP_K = 3             # candidate memories for fetch/save decisions
MEMORY_MAX_SNAPSHOT_CHARS = 6000         # hard cap on stored snapshot size
