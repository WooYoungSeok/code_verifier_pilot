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

#: Models excluded from the convenience groups, with the reason.
#: They remain runnable by naming them explicitly on --models.
EXCLUDED_FROM_GROUPS: dict[str, str] = {
    "deepseek-coder-v2-lite": (
        "16B MoE, ~31GB in bf16: does not fit a single 24GB card, so it would have "
        "to run 4-bit while the Qwen models run bf16. That precision mismatch "
        "confounds RQ2. Run it explicitly with --models deepseek-coder-v2-lite "
        "--load-in-4bit and record the deviation in EXPERIMENT_LOG.md."
    ),
}

#: convenience groups for --models. A group never silently pulls in a model the
#: experiment plan excluded -- an earlier version did, and started a 31GB
#: download in the middle of a run.
GROUPS: dict[str, list[str]] = {
    "closed": [
        k for k, v in REGISTRY.items()
        if v.group == "closed" and k not in EXCLUDED_FROM_GROUPS
    ],
    "open": [
        k for k, v in REGISTRY.items()
        if v.group == "open" and k not in EXCLUDED_FROM_GROUPS
    ],
    "all": [k for k in REGISTRY if k not in EXCLUDED_FROM_GROUPS],
}


def resolve(names: list[str]) -> list[str]:
    """Expand group aliases and validate model names."""
    out: list[str] = []
    for name in names:
        if name in GROUPS:
            out.extend(GROUPS[name])
            for excluded, reason in EXCLUDED_FROM_GROUPS.items():
                if REGISTRY[excluded].group in (name, "all") or name == "all":
                    print(f"  note: {excluded!r} is not in group {name!r} -- {reason}")
        elif name in REGISTRY:
            if name in EXCLUDED_FROM_GROUPS:
                print(f"  note: running excluded model {name!r} -- {EXCLUDED_FROM_GROUPS[name]}")
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
        # greedy decoding has no sampling parameters to reject
        kwargs.pop("strict_params", None)
        return HFLocalClient(model_id=spec.model_id, name=spec.name, **kwargs)
    raise SystemExit(f"unknown provider {spec.provider!r} for model {name!r}")
