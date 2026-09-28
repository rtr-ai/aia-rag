import os

from agent.llm_client import LlmClient
from agent.ollama_client import OllamaLlmClient
from agent.openai_client import OpenAiLlmClient

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or (
    f"http://{OLLAMA_HOST}:11434" if LLM_PROVIDER == "ollama" and OLLAMA_HOST else ""
)
LLM_API_KEY = os.getenv("LLM_API_KEY") or None
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "600"))
LLM_MODEL = os.getenv("LLM_MODELS", "llama3.1:8b-instruct-fp16").split(",")[0]
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "8000"))


def create_llm_client() -> LlmClient:
    if LLM_PROVIDER == "ollama":
        return OllamaLlmClient(
            host=LLM_BASE_URL,
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            context_window=CONTEXT_WINDOW,
            timeout=LLM_TIMEOUT,
        )
    if LLM_PROVIDER == "openai":
        return OpenAiLlmClient(
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            timeout=LLM_TIMEOUT,
            api_key=LLM_API_KEY,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
