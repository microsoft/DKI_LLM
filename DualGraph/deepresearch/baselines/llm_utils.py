import json as _json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

from openai import AzureOpenAI, OpenAI

logger = logging.getLogger(__name__)

@dataclass
class ModelResponse:
    """Minimal mirror of aworld.models.model_response.ModelResponse.

    Only the fields actually read by the baselines code are included:
      - content   (str)
      - usage     (dict with 'prompt_tokens', 'completion_tokens', 'total_tokens')
    The private attribute ``_call_elapsed_time`` is set by ``call_llm_model``.
    """
    content: str = ""
    usage: Dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    })
    _call_elapsed_time: float = 0.0

    @classmethod
    def from_openai_response(cls, response: Any) -> "ModelResponse":
        message = response.choices[0].message
        usage_obj = response.usage
        return cls(
            content=message.content or "",
            usage={
                "prompt_tokens": usage_obj.prompt_tokens if usage_obj else 0,
                "completion_tokens": usage_obj.completion_tokens if usage_obj else 0,
                "total_tokens": usage_obj.total_tokens if usage_obj else 0,
            },
        )


# ---------------------------------------------------------------------------
# AgentConfig — thin config object
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Minimal replacement for aworld.config.conf.AgentConfig.

    Accepts exactly the keyword arguments that the baselines code passes
    when constructing an AgentConfig (see main.py, outline_module.py, etc.).
    """
    llm_provider: str = "openai"
    llm_model_name: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_temperature: float = 0.0

    # Azure-specific (optional)
    llm_use_aad: Optional[bool] = None
    azure_client_id: Optional[str] = None
    azure_client_secret: Optional[str] = None
    azure_endpoint: Optional[str] = None

    # Custom provider (optional)
    llm_api_version: Optional[str] = None
    llm_extra_headers: Optional[Dict[str, str]] = None


# ---------------------------------------------------------------------------
# LLMModel — wraps an OpenAI / AzureOpenAI client
# ---------------------------------------------------------------------------

class LLMModel:
    """Wraps an ``openai.OpenAI`` or ``openai.AzureOpenAI`` client."""

    def __init__(self, client: Union[OpenAI, AzureOpenAI], model_name: str):
        self.client = client
        self.model_name = model_name

    def completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> ModelResponse:
        params: Dict[str, Any] = {
            "model": kwargs.get("model_name", self.model_name) or self.model_name,
            "messages": messages,
        }
        if temperature:
            params["temperature"] = temperature
        if max_tokens:
            params["max_tokens"] = max_tokens
        if stop:
            params["stop"] = stop

        # Forward any extra OpenAI-supported params
        supported = {
            "max_completion_tokens", "reasoning_effort", "response_format",
            "seed", "top_p", "tools", "tool_choice", "stream",
            "frequency_penalty", "presence_penalty",
        }
        for k, v in kwargs.items():
            if k in supported and v is not None:
                params[k] = v

        raw = self.client.chat.completions.create(**params)
        return ModelResponse.from_openai_response(raw)

    def stream_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> Generator[ModelResponse, None, None]:
        params: Dict[str, Any] = {
            "model": kwargs.get("model_name", self.model_name) or self.model_name,
            "messages": messages,
            "stream": True,
        }
        if temperature:
            params["temperature"] = temperature
        if max_tokens:
            params["max_tokens"] = max_tokens
        if stop:
            params["stop"] = stop

        stream = self.client.chat.completions.create(**params)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield ModelResponse(content=chunk.choices[0].delta.content)


# ---------------------------------------------------------------------------
# get_llm_model
# ---------------------------------------------------------------------------

def get_llm_model(
    conf: Optional[AgentConfig] = None,
    custom_provider=None,
    **kwargs,
) -> LLMModel:
    """Create an LLMModel from an AgentConfig."""
    if conf is None:
        conf = AgentConfig(
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            llm_api_key=os.getenv("LLM_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

    provider = conf.llm_provider or "openai"
    base_url = kwargs.get("base_url") or conf.llm_base_url
    api_key = kwargs.get("api_key") or conf.llm_api_key
    model_name = kwargs.get("model_name") or conf.llm_model_name

    if provider == "azure_openai":
        # Normalise endpoint
        endpoint = base_url or conf.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        endpoint = endpoint.rstrip("/")
        if endpoint.lower().endswith("/openai"):
            endpoint = endpoint[: -len("/openai")]
        sdk_base_url = f"{endpoint}/openai/"

        api_version = os.getenv("LLM_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION") or "2025-04-01-preview"
        if conf.llm_use_aad is not None:
            use_aad = conf.llm_use_aad
        else:
            use_aad = os.getenv("AZURE_OPENAI_USE_AAD", "").lower() in ("1", "true", "yes")

        if use_aad or not (api_key or os.getenv("AZURE_OPENAI_API_KEY")):
            # AAD token-based auth — use CloudGPT helper if available,
            # otherwise fall back to generic azure-identity.
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
                from cloudgpt_aoai import get_openai_token_provider
                token_provider = get_openai_token_provider(
                    client_id=conf.azure_client_id,
                    client_secret=conf.azure_client_secret,
                )
                token_provider()  # warm up / validate
            except ImportError:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
                credential = DefaultAzureCredential()
                token_provider = get_bearer_token_provider(
                    credential, "https://cognitiveservices.azure.com/.default"
                )
            client = AzureOpenAI(
                api_version=api_version,
                base_url=sdk_base_url,
                azure_ad_token_provider=token_provider,
            )
        else:
            resolved_key = api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("LLM_API_KEY", "")
            client = AzureOpenAI(
                api_key=resolved_key,
                api_version=api_version,
                base_url=sdk_base_url,
            )
    elif provider == "custom":
        # Third-party OpenAI-compatible endpoint (vLLM, LiteLLM, Ollama, DeepSeek, …)
        resolved_key = api_key or os.getenv("LLM_API_KEY") or "EMPTY"
        resolved_url = base_url or os.getenv("LLM_BASE_URL")
        client_kwargs: Dict[str, Any] = {"api_key": resolved_key, "timeout": 180}
        if resolved_url:
            client_kwargs["base_url"] = resolved_url

        api_version = conf.llm_api_version or os.getenv("LLM_API_VERSION")
        if api_version:
            client_kwargs["default_query"] = {"api-version": api_version}

        extra_headers = conf.llm_extra_headers
        if not extra_headers:
            raw_hdr = os.getenv("LLM_EXTRA_HEADERS")
            if raw_hdr:
                try:
                    extra_headers = _json.loads(raw_hdr)
                except Exception:
                    logger.warning("Failed to parse LLM_EXTRA_HEADERS as JSON; ignoring.")
        if extra_headers:
            client_kwargs["default_headers"] = extra_headers

        client = OpenAI(**client_kwargs)
    else:
        # Generic OpenAI-compatible endpoint
        resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
        resolved_url = base_url or os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
        client = OpenAI(api_key=resolved_key, base_url=resolved_url, timeout=180)

    return LLMModel(client=client, model_name=model_name or "")


# ---------------------------------------------------------------------------
# call_llm_model
# ---------------------------------------------------------------------------

_RETRY_AFTER_RE = re.compile(r"retry after\s+(\d+(?:\.\d+)?)\s+second", re.IGNORECASE)


def _is_rate_limit_error(err: Exception) -> bool:
    s = str(err).lower()
    return (
        "ratelimitreached" in s
        or "rate limit" in s
        or "error code: 429" in s
        or "status code: 429" in s
        or "http 429" in s
    )


def _extract_retry_after(err: Exception) -> Optional[float]:
    m = _RETRY_AFTER_RE.search(str(err))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def call_llm_model(
    llm_model: LLMModel,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    stop: Optional[List[str]] = None,
    stream: bool = False,
    **kwargs,
) -> Union[ModelResponse, Generator[ModelResponse, None, None]]:
    """Call the LLM with automatic retry on rate-limit errors."""
    num_retry = kwargs.pop("num_retry", 5)
    backoff_base = float(kwargs.pop("rate_limit_backoff_base", 1.0))
    backoff_cap = float(kwargs.pop("rate_limit_backoff_cap", 30.0))

    for attempt in range(num_retry):
        try:
            call_start = time.time()
            if stream:
                response = llm_model.stream_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    **kwargs,
                )
            else:
                response = llm_model.completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    **kwargs,
                )
            elapsed = time.time() - call_start

            if response and response.content and response.content.strip():
                response._call_elapsed_time = elapsed
                return response
            else:
                logger.warning("LLM returned empty content on attempt %d/%d", attempt + 1, num_retry)
        except Exception as e:
            logger.warning("LLM call failed on attempt %d/%d: %s", attempt + 1, num_retry, e)
            if _is_rate_limit_error(e) and attempt < num_retry - 1:
                retry_after = _extract_retry_after(e)
                high = min(backoff_cap, backoff_base * (2 ** attempt))
                low = max(0.0, retry_after) if retry_after is not None else 0.0
                high = max(high, low)
                sleep_s = random.uniform(low, high)
                logger.warning("Rate-limit (429). Backing off %.2fs before retry.", sleep_s)
                time.sleep(sleep_s)
            if attempt == num_retry - 1:
                raise

    return None  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# embed_texts — provider-aware embedding helper
# ---------------------------------------------------------------------------

def _get_embedding_client(
    conf: Optional[AgentConfig] = None,
) -> Union[OpenAI, AzureOpenAI]:
    """Return an OpenAI/AzureOpenAI client for embedding calls, reusing get_llm_model auth logic."""
    if conf is None:
        conf = AgentConfig(
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            llm_api_key=os.getenv("LLM_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
    provider = conf.llm_provider or "openai"

    if provider == "azure_openai":
        endpoint = conf.llm_base_url or conf.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        endpoint = endpoint.rstrip("/")
        if endpoint.lower().endswith("/openai"):
            endpoint = endpoint[: -len("/openai")]
        sdk_base_url = f"{endpoint}/openai/"

        api_version = os.getenv("LLM_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION") or "2025-04-01-preview"
        api_key = conf.llm_api_key

        if conf.llm_use_aad is not None:
            use_aad = conf.llm_use_aad
        else:
            use_aad = os.getenv("AZURE_OPENAI_USE_AAD", "").lower() in ("1", "true", "yes")

        if use_aad or not (api_key or os.getenv("AZURE_OPENAI_API_KEY")):
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
                from cloudgpt_aoai import get_openai_token_provider
                token_provider = get_openai_token_provider(
                    client_id=conf.azure_client_id,
                    client_secret=conf.azure_client_secret,
                )
                token_provider()
            except ImportError:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
                credential = DefaultAzureCredential()
                token_provider = get_bearer_token_provider(
                    credential, "https://cognitiveservices.azure.com/.default"
                )
            client = AzureOpenAI(
                api_version=api_version,
                base_url=sdk_base_url,
                azure_ad_token_provider=token_provider,
            )
        else:
            resolved_key = api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("LLM_API_KEY", "")
            client = AzureOpenAI(
                api_key=resolved_key,
                api_version=api_version,
                base_url=sdk_base_url,
            )
    elif provider == "custom":
        resolved_key = conf.llm_api_key or os.getenv("LLM_API_KEY") or "EMPTY"
        resolved_url = conf.llm_base_url or os.getenv("LLM_BASE_URL")
        client_kwargs: Dict[str, Any] = {"api_key": resolved_key, "timeout": 180}
        if resolved_url:
            client_kwargs["base_url"] = resolved_url
        api_version = conf.llm_api_version or os.getenv("LLM_API_VERSION")
        if api_version:
            client_kwargs["default_query"] = {"api-version": api_version}
        extra_headers = conf.llm_extra_headers
        if not extra_headers:
            raw_hdr = os.getenv("LLM_EXTRA_HEADERS")
            if raw_hdr:
                try:
                    extra_headers = _json.loads(raw_hdr)
                except Exception:
                    pass
        if extra_headers:
            client_kwargs["default_headers"] = extra_headers
        client = OpenAI(**client_kwargs)
    else:
        resolved_key = conf.llm_api_key or os.getenv("OPENAI_API_KEY", "")
        resolved_url = conf.llm_base_url or os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
        client = OpenAI(api_key=resolved_key, base_url=resolved_url, timeout=180)

    return client


def embed_texts(
    texts: List[str],
    embedding_model: str = "text-embedding-3-large",
    batch_size: int = 96,
    conf: Optional[AgentConfig] = None,
) -> List[List[float]]:
    """Compute embeddings for a list of texts.

    Uses the same provider/auth logic as ``get_llm_model``:
      - ``azure_openai``: Azure OpenAI (AAD or API-key)
      - ``openai``: generic OpenAI-compatible endpoint

    The provider is determined by ``conf.llm_provider`` (defaults to env /
    AgentConfig defaults).
    """
    if not texts:
        return []

    client = _get_embedding_client(conf)
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        resp = client.embeddings.create(
            model=embedding_model,
            input=chunk,
        )
        vectors.extend([item.embedding for item in resp.data])
    return vectors
