"""Model registry: name -> constructor.

Model ids are pinned here so a swap is a one-line change and every script,
result file and report agrees on what was run. Closed-source ids follow the
verifier comparison the pilot is replicating; open-weight ids are the
HuggingFace repo names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .base import VerifierClient


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    model_id: str
    group: str                      # "closed" | "open" | "open_sft"
    note: str = ""
    kwargs: dict = field(default_factory=dict)


REGISTRY: dict[str, ModelSpec] = {
    # ---- closed-source, prompting only ----
    "gpt-5.1": ModelSpec(
        "gpt-5.1", "openai", "gpt-5.1", "closed",
        "frontier closed model",
    ),
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", "gemini", "gemini-2.5-flash", "closed",
        "thinking disabled for deterministic classification",
    ),
    "claude-sonnet-4.6": ModelSpec(
        "claude-sonnet-4.6", "anthropic", "claude-sonnet-4-6", "closed",
        "forced tool use for structured output",
    ),
    # ---- open-weight, zero-shot ----
    "qwen2.5-coder-7b": ModelSpec(
        "qwen2.5-coder-7b", "local", "Qwen/Qwen2.5-Coder-7B-Instruct", "open",
        "code-specialised; same backbone family as the student-code generator",
    ),
    "deepseek-coder-v2-lite": ModelSpec(
        "deepseek-coder-v2-lite", "local",
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "open",
        "code-specialised MoE",
    ),
    "qwen2.5-7b": ModelSpec(
        "qwen2.5-7b", "local", "Qwen/Qwen2.5-7B-Instruct", "open",
        "control: same family, no coding specialisation",
    ),
}

#: convenience groups for --models
GROUPS: dict[str, list[str]] = {
    "closed": [k for k, v in REGISTRY.items() if v.group == "closed"],
    "open": [k for k, v in REGISTRY.items() if v.group == "open"],
    "all": list(REGISTRY),
}


def resolve(names: list[str]) -> list[str]:
    """Expand group aliases and validate model names."""
    out: list[str] = []
    for name in names:
        if name in GROUPS:
            out.extend(GROUPS[name])
        elif name in REGISTRY:
            out.append(name)
        else:
            raise SystemExit(
                f"unknown model {name!r}. Known models: {sorted(REGISTRY)}; "
                f"groups: {sorted(GROUPS)}"
            )
    return list(dict.fromkeys(out))


def build(name: str, **overrides) -> VerifierClient:
    """Instantiate the client for ``name``. Imports the SDK lazily."""
    if name not in REGISTRY:
        raise SystemExit(f"unknown model {name!r}; known: {sorted(REGISTRY)}")
    spec = REGISTRY[name]
    kwargs = {**spec.kwargs, **overrides}

    if spec.provider == "gemini":
        from .gemini import GeminiClient
        return GeminiClient(model=spec.model_id, name=spec.name, **kwargs)
    if spec.provider == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(model=spec.model_id, name=spec.name, **kwargs)
    if spec.provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(model=spec.model_id, name=spec.name, **kwargs)
    if spec.provider == "local":
        from .hf_local import HFLocalClient
        return HFLocalClient(model_id=spec.model_id, name=spec.name, **kwargs)
    raise SystemExit(f"unknown provider {spec.provider!r} for model {name!r}")
