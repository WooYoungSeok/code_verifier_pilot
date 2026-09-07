"""Anthropic verifier client (Claude Sonnet 4.6).

Structured output is obtained with **forced tool use** rather than
``output_config.format``: the pinned SDK in requirements.txt (anthropic 0.71)
predates the ``output_config`` / ``messages.parse`` surface, and forced tool use
works on every SDK version this pilot supports. If the API rejects ``strict`` or
the forced ``tool_choice``, the client degrades to plain text plus
:func:`parse_label` and records that it did.

``thinking`` is deliberately omitted. On Sonnet 4.6 omitting it means no
extended thinking, which is what a temperature-0 binary classifier wants, and it
keeps ``temperature`` usable (adaptive thinking would forbid it).

``stop_reason == "refusal"`` is checked before reading content: it returns
HTTP 200 with no usable answer, and would otherwise look like an empty response.
"""

from __future__ import annotations

from ..env import get_api_key
from .base import (
    ERROR, MAX_ANSWER_TOKENS, UNSUPPORTED_BY_PROVIDER, ParameterRejectedError,
    Prediction, RetryingClient, parse_label,
)

MODEL = "claude-sonnet-4-6"

TOOL_NAME = "submit_verdict"
TOOL = {
    "name": TOOL_NAME,
    "description": "Report whether the student's code matches the given error category.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": ["aligned", "not_aligned"]}},
        "required": ["label"],
        "additionalProperties": False,
    },
}

_TOOL_REJECTION_MARKERS = (
    "strict", "tool_choice", "not supported for this model", "unsupported",
)


class AnthropicClient(RetryingClient):
    provider = "anthropic"

    def __init__(
        self,
        model: str = MODEL,
        name: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int = MAX_ANSWER_TOKENS,
        max_retries: int = 5,
        strict_params: bool = True,
    ) -> None:
        import anthropic

        self.model = model
        self.name = name or model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._temperature = temperature
        self._client = anthropic.Anthropic(api_key=get_api_key("anthropic"))
        self._use_tool = True
        self._use_strict = True
        self._init_params({
            "temperature": temperature,
            # The Messages API exposes no seed. Recorded rather than left blank so
            # the write-up cannot claim all three models were seed-pinned: Gemini
            # and GPT-5.1 take seed=42, Claude cannot.
            "seed": UNSUPPORTED_BY_PROVIDER,
            "max_tokens": max_tokens,
            "thinking": "omitted (no extended thinking; keeps temperature usable)",
            "structured_output": "forced tool use (strict)",
        }, strict_params=strict_params)
        self.record_param_event(
            "seed", "unsupported_by_provider", "",
            "anthropic.messages.create has no seed parameter; Claude runs are not "
            "seed-pinned and are expected to vary slightly between runs",
        )

    def _request_kwargs(self, system: str, user: str) -> dict:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._use_tool:
            tool = dict(TOOL)
            if not self._use_strict:
                tool.pop("strict", None)
            kwargs["tools"] = [tool]
            kwargs["tool_choice"] = {"type": "tool", "name": TOOL_NAME}
        return kwargs

    def _degrade(self, message: str) -> bool:
        lowered = message.lower()
        if not any(marker in lowered for marker in _TOOL_REJECTION_MARKERS):
            return False
        if self._use_strict:
            self._use_strict = False
            self.effective_params["structured_output"] = "forced tool use (non-strict)"
            self.record_param_event(
                "structured_output", "changed", message,
                "strict tool schema rejected; retrying without strict",
            )
            return True
        if self._use_tool:
            self._use_tool = False
            self.effective_params["structured_output"] = "free text + regex parse"
            self.record_param_event(
                "structured_output", "changed", message,
                "forced tool use rejected; falling back to text parsing",
            )
            return True
        return False

    def _call(self, system: str, user: str) -> Prediction:
        import anthropic

        try:
            response = self._client.messages.create(**self._request_kwargs(system, user))
        except anthropic.BadRequestError as exc:
            message = str(exc)
            if "temperature" in message.lower():
                if self.strict_params:
                    raise ParameterRejectedError(
                        self.name, "temperature", message, self.requested_params
                    ) from exc
                if self._temperature is not None:
                    self._temperature = None
                    self.record_param_event(
                        "temperature", "dropped", message,
                        "strict_params=False: continuing at the API default",
                    )
                    return self._call(system, user)
            if self._degrade(message):
                return self._call(system, user)
            raise

        stop_reason = getattr(response, "stop_reason", None)
        usage = _usage(response)

        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            return Prediction(
                label=ERROR, raw="", ok=False, error_kind="Refusal",
                error_detail=f"category={getattr(details, 'category', None)}",
                usage=usage, finish_reason=stop_reason,
            )

        label, texts = None, []
        for block in getattr(response, "content", None) or []:
            block_type = getattr(block, "type", None)
            if block_type == "tool_use" and getattr(block, "name", None) == TOOL_NAME:
                value = (getattr(block, "input", None) or {}).get("label")
                if value in ("aligned", "not_aligned"):
                    label = value
            elif block_type == "text":
                texts.append(getattr(block, "text", "") or "")

        raw = (" ".join(texts)).strip()
        if label is None:
            label = parse_label(raw)
        if label is None:
            kind = "EmptyResponse" if not raw else "ParseFailure"
            hint = (
                f"stop_reason={stop_reason}; if 'max_tokens', raise max_tokens "
                f"(now {self.max_tokens})"
            )
            return Prediction(
                label=ERROR, raw=raw, ok=False, error_kind=kind,
                error_detail=raw[:200] or hint, usage=usage, finish_reason=stop_reason,
            )
        return Prediction(
            label=label, raw=raw or f'{{"label": "{label}"}}',
            usage=usage, finish_reason=stop_reason,
        )


def _usage(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    out = {
        "prompt_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", None),
    }
    return {k: v for k, v in out.items() if v is not None}
