"""Central LLM configuration.

Default provider is Ollama. Set LLM_PROVIDER=nvidia and NVIDIA_API_KEY to use
NVIDIA AI Endpoints / NIM via langchain_nvidia_ai_endpoints.ChatNVIDIA.
"""
from dotenv import load_dotenv
load_dotenv()

import os

from config import (
    LLM_PROVIDER,
    NVIDIA_ENABLE_THINKING,
    NVIDIA_MAX_TOKENS,
    NVIDIA_MODEL,
    NVIDIA_REASONING_BUDGET,
    NVIDIA_TEMPERATURE,
    NVIDIA_TOP_P,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _build_ollama_llm():
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", OLLAMA_MODEL),
        temperature=_env_float("OLLAMA_TEMPERATURE", OLLAMA_TEMPERATURE),
    )


def _build_nvidia_llm():
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY is required when LLM_PROVIDER=nvidia. "
            "Set it in your environment before running the app."
        )

    return ChatNVIDIA(
        model=os.getenv("NVIDIA_MODEL", NVIDIA_MODEL),
        api_key=api_key,
        temperature=_env_float("NVIDIA_TEMPERATURE", NVIDIA_TEMPERATURE),
        top_p=_env_float("NVIDIA_TOP_P", NVIDIA_TOP_P),
        max_tokens=_env_int("NVIDIA_MAX_TOKENS", NVIDIA_MAX_TOKENS),
        reasoning_budget=_env_int("NVIDIA_REASONING_BUDGET", NVIDIA_REASONING_BUDGET),
        chat_template_kwargs={
            "enable_thinking": _env_bool("NVIDIA_ENABLE_THINKING", NVIDIA_ENABLE_THINKING)
        },
    )


def build_llm():
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    if provider == "ollama":
        return _build_ollama_llm()
    if provider == "nvidia":
        return _build_nvidia_llm()
    raise ValueError(f"Unsupported LLM_PROVIDER={provider!r}. Use 'ollama' or 'nvidia'.")


llm = build_llm()