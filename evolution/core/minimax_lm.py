"""Custom MiniMax DSPy LM — supports MiniMax-M2.7 and other MiniMax models.

Usage:
    from evolution.core.minimax_lm import MiniMaxLM

    lm = MiniMaxLM(model="MiniMax-M2.7", quality="highspeed")
    with dspy.context(lm=lm):
        ...

Environment variables (checked in order):
    MINIMAX_API_KEY       — International API key (api.minimax.io)
    MINIMAX_CN_API_KEY    — China API key (api.minimaxi.com)

    MINIMAX_BASE_URL      — Override base URL (default: https://api.minimaxi.com/v1)
"""

import os
from typing import Optional

import dspy
from dspy.clients.base_lm import BaseLM


class MiniMaxLM(BaseLM):
    """MiniMax language model compatible with DSPy's BaseLM interface.

    Uses the OpenAI-compatible API endpoint. Supports the MiniMax quality
    parameter for models that offer quality/speed tradeoffs (e.g. MiniMax-M2.7-highspeed).

    Args:
        model: MiniMax model name, e.g. "MiniMax-M2.7", "MiniMax-M2.5".
              Use "MiniMax-M2.7-highspeed" as the model string and set
              quality="highspeed" to activate highspeed mode.
        quality: Quality preset — "highspeed" for faster/cheaper inference,
                 None for standard quality. Passed as extra_body["quality"].
        api_key: MiniMax API key. If None, reads from MINIMAX_API_KEY or
                 MINIMAX_CN_API_KEY environment variables.
        base_url: API base URL. If None, reads from MINIMAX_BASE_URL env or
                  defaults to https://api.minimaxi.com/v1 (China).
        temperature: Sampling temperature (default 0.0 for deterministic).
        max_tokens: Max tokens to generate (default 2048).
        cache: Whether to use DSPy caching (default True).
    """

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        quality: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        cache: bool = True,
        **kwargs,
    ):
        super().__init__(
            model=model,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            **kwargs,
        )
        self.quality = quality

        # Resolve API key — prefer MINIMAX_API_KEY, fall back to MINIMAX_CN_API_KEY
        self.api_key = (
            api_key
            or os.getenv("MINIMAX_API_KEY")
            or os.getenv("MINIMAX_CN_API_KEY")
            or ""
        )

        # Resolve base URL
        # Priority: explicit > MINIMAX_CN_BASE_URL > default CN endpoint
        # NOTE: MINIMAX_BASE_URL (international) often points to a proxy that
        # doesn't route MiniMax requests correctly. Prefer the explicit CN
        # endpoint unless user overrides with base_url parameter.
        self.base_url = (
            base_url
            or os.getenv("MINIMAX_CN_BASE_URL")
            or "https://api.minimaxi.com/v1"
        )

        # Lazy-init the OpenAI client (import is cheap)
        self._client = None

    # ── Pickle support ───────────────────────────────────────────────────────

    def __getstate__(self):
        """Exclude the non-picklable OpenAI client from the state."""
        state = self.__dict__.copy()
        state["_client"] = None
        state["_async_client"] = None
        return state

    def __setstate__(self, state):
        """Restore state and lazily recreate the OpenAI client."""
        self.__dict__.update(state)
        self._client = None
        self._async_client = None

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def _openai_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=120.0,
                max_retries=3,
            )
        return self._client

    @property
    def supports_function_calling(self) -> bool:
        """MiniMax supports function calling."""
        return True

    @property
    def supports_reasoning(self) -> bool:
        """MiniMax-M2.7 has reasoning capability."""
        return True

    @property
    def supports_response_schema(self) -> bool:
        """MiniMax supports structured output via response_format."""
        return True

    @property
    def supported_params(self) -> set:
        return {"response_format", "quality"}

    # ── Forward pass ───────────────────────────────────────────────────────

    # DSPy-internal keys that must not be forwarded to the OpenAI API
    _FORBIDDEN_KWARGS = {
        "signature_kwargs", "lm_kwargs", "rollout_id", "run_id", "metric_name",
        "track_from_child", "original_completion", "compiled_instruction",
    }

    # OpenAI-compatible kwargs that should be forwarded
    _FORWARD_KWARGS = {
        "temperature", "max_tokens", "top_p", "n", "stop", "presence_penalty",
        "frequency_penalty", "logit_bias", "user", "response_format", "seed",
        "tools", "tool_choice", "parallel_tool_calls", "prediction", "extra_headers",
        "extra_query", "extra_body", "timeout", "stream", "stream_options",
    }

    def forward(self, prompt=None, messages=None, **kwargs):
        """Call MiniMax chat completion API.

        Returns an OpenAI-format ChatCompletion response object.
        DSPy's _process_lm_response handles the rest.
        """
        # Build messages list
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        # Extract lm_kwargs (passed as dict by DSPy adapters) and merge
        lm_kwargs = kwargs.pop("lm_kwargs", {})
        merged = {**self.kwargs, **lm_kwargs, **kwargs}

        # Filter out DSPy-internal keys
        for key in self._FORBIDDEN_KWARGS:
            merged.pop(key, None)

        # Build request kwargs
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            # quality goes in extra_body for MiniMax
            "extra_body": {"quality": self.quality} if self.quality else {},
        }

        # Merge only OpenAI-compatible kwargs
        for key in self._FORWARD_KWARGS:
            if key in merged:
                val = merged.pop(key)
                if key == "extra_body":
                    # Merge with existing extra_body
                    request_kwargs["extra_body"].update(val)
                else:
                    request_kwargs[key] = val

        # Discard any remaining unknown kwargs (these are DSPy internals)
        # Do NOT forward them to the API — only whitelist above is safe.
        return self._openai_client.chat.completions.create(**request_kwargs)

    async def aforward(self, prompt=None, messages=None, **kwargs):
        """Async forward pass — uses openai.AsyncOpenAI."""
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        # Extract lm_kwargs (passed as dict by DSPy adapters) and merge
        lm_kwargs = kwargs.pop("lm_kwargs", {})
        merged = {**self.kwargs, **lm_kwargs, **kwargs}

        # Filter out DSPy-internal keys
        for key in self._FORBIDDEN_KWARGS:
            merged.pop(key, None)

        # Build request kwargs
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "extra_body": {"quality": self.quality} if self.quality else {},
        }

        # Merge only OpenAI-compatible kwargs
        for key in self._FORWARD_KWARGS:
            if key in merged:
                val = merged.pop(key)
                if key == "extra_body":
                    request_kwargs["extra_body"].update(val)
                else:
                    request_kwargs[key] = val

        # Discard any remaining unknown kwargs (DSPy internals)
        if not hasattr(self, "_async_client") or self._async_client is None:
            import openai
            self._async_client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=120.0,
                max_retries=3,
            )
        return await self._async_client.chat.completions.create(**request_kwargs)


# ── Convenience factory ────────────────────────────────────────────────────────

def get_minimax_lm(
    model: str = "MiniMax-M2.7",
    quality: str = "highspeed",
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> "MiniMaxLM":
    """Factory function with sensible defaults for hermes-agent-self-evolution.

    Default quality="highspeed" matches MiniMax-M2.7-highspeed behavior.
    Set quality=None for standard MiniMax-M2.7 quality.
    """
    return MiniMaxLM(
        model=model,
        quality=quality,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_lm(
    model_name: str,
    quality: str | None = "highspeed",
    **kwargs,
) -> "BaseLM":
    """Factory that returns the right DSPy LM for any model name.

    - If model_name starts with "minimax/" → MiniMaxLM (quality param respected)
    - Otherwise → standard dspy.LM (OpenAI, Anthropic, Ollama, etc.)

    Examples:
        get_lm("minimax/MiniMax-M2.7", quality="highspeed")
        get_lm("openai/gpt-4.1")
        get_lm("anthropic/claude-sonnet-4-20250514")
    """
    if model_name.startswith("minimax/"):
        # Strip "minimax/" prefix — MiniMaxLM takes bare model name
        bare_model = model_name.split("/", 1)[1]
        return MiniMaxLM(
            model=bare_model,
            quality=quality,
            **kwargs,
        )
    else:
        return dspy.LM(model_name, **kwargs)


# ── Convenience shortcuts for hermes-agent-self-evolution configs ──────────────

def make_config_lm(
    config: "EvolutionConfig",
) -> "BaseLM":
    """Create the eval/judge LM from an EvolutionConfig.

    Respects config.minimax_quality for minimax models.
    """
    return get_lm(config.eval_model, quality=config.minimax_quality)


def make_judge_lm(
    config: "EvolutionConfig",
) -> "BaseLM":
    """Create the judge LM from an EvolutionConfig (for dataset generation)."""
    return get_lm(config.judge_model, quality=config.minimax_quality)
