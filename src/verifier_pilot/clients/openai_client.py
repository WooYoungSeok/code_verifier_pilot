"""OpenAI verifier client (GPT-5.1).

Reasoning models reject some parameters that non-reasoning models accept, and
which ones they reject varies by snapshot. Rather than hard-code a guess, the
client starts with the strictest useful request and permanently drops whichever
parameter the API rejects with a 400:

* ``temperature`` -- reasoning models often allow only the default. Dropped on
  "unsupported parameter" / "does not support" 400s. The task is a
  single-token classification, so losing temperature=0 costs little; the run
  log records that it happened.
* ``reasoning_effort`` -- dropped if the snapshot does not accept the requested
  level.
* ``response_format`` json_schema -- falls back to plain text plus
  :func:`parse_label`.

Structured output uses ``json_schema`` with ``strict: true`` so the answer is a
validated enum rather than prose to be regex'd.
"""

from __future__ import annotations

import json

from ..env import get_api_key
from .base import ERROR, Prediction, RetryingClient, parse_label

MODEL = "gpt-5.1"

JSON_SCHEMA = {
    "name": "verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": ["aligned", "not_aligned"]}},
        "required": ["label"],
        "additionalProperties": False,
    },
}

_UNSUPPORTED_MARKERS = (
    "unsupported parameter", "unsupported_parameter", "does not support",
    "unknown parameter", "not supported with", "unsupported value",
)


class OpenAIClient(RetryingClient):
    provider = "openai"

    def __init__(
        self,
        model: str = MODEL,
        name: str | None = None,
        temperature: float | None = 0.0,
        reasoning_effort: str | None = "low",
        max_completion_tokens: int = 2048,
        max_retries: int = 5,
    ) -> None:
        import openai

        self.model = model
        self.name = name or model
        self.max_retries = max_retries
        self.max_completion_tokens = max_completion_tokens
        self._client = openai.OpenAI(api_key=get_api_key("openai"))
        self._use = {
            "temperature": temperature is not None,
            "reasoning_effort": bool(reasoning_effort),
            "response_format": True,
        }
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort

    def _request_kwargs(self, system: str, user: str) -> dict:
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": self.max_completion_tokens,
        }
        if self._use["temperature"]:
            kwargs["temperature"] = self._temperature
        if self._use["reasoning_effort"]:
            kwargs["reasoning_effort"] = self._reasoning_effort
        if self._use["response_format"]:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": JSON_SCHEMA}
        return kwargs

    def _drop_unsupported(self, message: str) -> bool:
        """Disable whichever optional parameter the 400 complained about."""
        lowered = message.lower()
        if not any(marker in lowered for marker in _UNSUPPORTED_MARKERS):
            return False
        for parameter in ("temperature", "reasoning_effort", "response_format"):
            if parameter in lowered and self._use[parameter]:
                self._use[parameter] = False
                print(
                    f"    [{self.name}] API rejected {parameter!r}; retrying without it",
                    flush=True,
                )
                return True
        return False

    def _call(self, system: str, user: str) -> Prediction:
        import openai

        try:
            response = self._client.chat.completions.create(**self._request_kwargs(system, user))
        except openai.BadRequestError as exc:
            if self._drop_unsupported(str(exc)):
                return self._call(system, user)
            raise

        choice = response.choices[0]
        raw = (choice.message.content or "").strip()
        finish = getattr(choice, "finish_reason", None)
        usage = _usage(response)

        if not raw:
            return Prediction(
                label=ERROR, raw="", ok=False, error_kind="EmptyResponse",
                error_detail=(
                    f"finish_reason={finish}; if 'length', reasoning consumed the "
                    f"budget -- raise max_completion_tokens (now {self.max_completion_tokens}) "
                    f"or lower reasoning_effort"
                ),
                usage=usage, finish_reason=finish,
            )

        label = None
        try:
            label = json.loads(raw).get("label")
        except (json.JSONDecodeError, AttributeError):
            pass
        if label not in ("aligned", "not_aligned"):
            label = parse_label(raw)
        if label is None:
            return Prediction(
                label=ERROR, raw=raw, ok=False, error_kind="ParseFailure",
                error_detail=raw[:200], usage=usage, finish_reason=finish,
            )
        return Prediction(label=label, raw=raw, usage=usage, finish_reason=finish)


def _usage(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    details = getattr(usage, "completion_tokens_details", None)
    out = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
    }
    return {k: v for k, v in out.items() if v is not None}
