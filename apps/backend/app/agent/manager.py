import os
import logging
from typing import Dict, Any

from .exceptions import ProviderError
from .strategies.wrapper import JSONWrapper, MDWrapper
# from .providers.ollama import OllamaProvider, OllamaEmbeddingProvider
# from .providers.openai import OpenAIProvider, OpenAIEmbeddingProvider
from .providers.groq import GroqProvider, GroqEmbeddingProvider


class AgentManager:
    def __init__(self, strategy: str | None = None, model: str = "gemma3:4b") -> None:
        match strategy:
            case "md":
                self.strategy = MDWrapper()
            case "json":
                self.strategy = JSONWrapper()
            case _:
                self.strategy = JSONWrapper()
        self.model = model

    async def _get_provider(self, **kwargs: Any) -> GroqProvider:
        # Check for Groq API key first
        logging.debug("Checking for Groq API key...")
        groq_api_key = kwargs.get("groq_api_key", os.getenv("GROQ_API_KEY"))
        
        if groq_api_key:
            logging.debug("Using GroqProvider")
            model = kwargs.get("model", "llama-3.3-70b-versatile")
            return GroqProvider(api_key=groq_api_key, model=model)
        
        # Check for OpenAI API key second
        # openai_api_key = kwargs.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
        # if openai_api_key:
        #     logging.debug("Using OpenAIProvider")
        #     return OpenAIProvider(api_key=openai_api_key)

        # # Fallback to Ollama
        # model = kwargs.get("model", self.model)
        # installed_ollama_models = await OllamaProvider.get_installed_models()
        # if model not in installed_ollama_models:
        #     raise ProviderError(
        #         f"Ollama Model '{model}' is not found. Run `ollama pull {model} or pick from any available models {installed_ollama_models}"
        #     )
        # return OllamaProvider(model_name=model)

    async def run(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Run the agent with the given prompt and generation arguments.
        """
        provider = await self._get_provider(**kwargs)
        return await self.strategy(prompt, provider, **kwargs)


class EmbeddingManager:
    def __init__(self, model: str = "nomic-embed-text:137m-v1.5-fp16") -> None:
        self._model = model

    async def _get_embedding_provider(
        self, **kwargs: Any
    ) -> GroqEmbeddingProvider:
        # Check for Groq API key first for embeddings
        logging.debug("Checking for Groq API key for embeddings...")
        groq_api_key = kwargs.get("groq_api_key", os.getenv("GROQ_API_KEY"))
        if groq_api_key:
            logging.debug("Using GroqEmbeddingProvider")
            embedding_model = kwargs.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            return GroqEmbeddingProvider(api_key=groq_api_key, embedding_model=embedding_model)
        
        # Check for OpenAI API key second for embeddings
        # openai_api_key = kwargs.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
        # if openai_api_key:
        #     logging.debug("Using OpenAIEmbeddingProvider")
        #     return OpenAIEmbeddingProvider(api_key=openai_api_key)
        
        # # Fallback to Ollama for embeddings
        # model = kwargs.get("embedding_model", self._model)
        # installed_ollama_models = await OllamaProvider.get_installed_models()
        # if model not in installed_ollama_models:
        #     raise ProviderError(
        #         f"Ollama Model '{model}' is not found. Run `ollama pull {model} or pick from any available models {installed_ollama_models}"
        #     )
        # return OllamaEmbeddingProvider(embedding_model=model)

    async def embed(self, text: str, **kwargs: Any) -> list[float]:
        """
        Get the embedding for the given text.
        """
        provider = await self._get_embedding_provider(**kwargs)
        return await provider.embed(text)
