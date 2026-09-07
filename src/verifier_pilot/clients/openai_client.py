"""OpenAI verifier client (GPT-5.1).

**``reasoning_effort`` is deliberately unset.** Probed against the live API
(2026-09-07):

    temperature=0 alone                     -> OK
    reasoning_effort="low" alone            -> OK
    temperature=0 + reasoning_effort="low"  -> 400 unsupported_value on
                                               'temperature': "does not support 0
                                               with this model. Only the default
                                               (1) value is supported."
    temperature=0 + seed=42                 -> OK

Setting ``reasoning_effort`` makes GPT-5.1 validate the call as a reasoning
request, which locks ``temperature`` to its default of 1. Leaving it unset keeps
the non-reasoning path, where ``temperature=0`` is accepted. The pilot holds
every model at temperature 0, so temperature wins and reasoning_effort stays
off.

To enable reasoning you must also pass ``temperature=None``, and you must record
it in EXPERIMENT_LOG.md -- it changes what the GPT-5.1 numbers mean and makes
them non-comparable with the runs already collected.

A parameter the API still refuses is **not** dropped silently: by default the
run aborts with the raw provider error (:class:`ParameterRejectedError`). With
``strict_params=False`` the drop proceeds but is recorded in ``param_events``
and written into the run manifest. The rejected parameter is identified from the
400's ``error.param`` field rather than guessed from the message text -- the
earlier substring-matching version mis-attributed the reasoning_effort conflict
above to ``temperature``.

Structured output uses ``json_schema`` with ``strict: true`` so the answer is a
validated enum rather than prose to be regex'd.
"""

from __future__ import annotations

import json

from ..env import get_api_key
from .base import (
    ERROR, MAX_ANSWER_TOKENS, ParameterRejectedError, Prediction, RetryingClient,
    parse_label,
)

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

#: parameters that may be dropped when strict_params is off, in priority order
_DROPPABLE = ("reasoning_effort", "seed", "temperature", "response_format")


class OpenAIClient(RetryingClient):
    provider = "openai"

    def __init__(
        self,
        model: str = MODEL,
        name: str | None = None,
        temperature: float | None = 0.0,
        reasoning_effort: str | None = None,
        seed: int | None = 42,
        max_completion_tokens: int = MAX_ANSWER_TOKENS,
        max_retries: int = 5,
        strict_params: bool = True,
    ) -> None:
        import openai

        self.model = model
        self.name = name or model
        self.max_retries = max_retries
        self.max_completion_tokens = max_completion_tokens
        self._client = openai.OpenAI(api_key=get_api_key("openai"))

        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._seed = seed
        self._use = {
            "temperature": temperature is not None,
            "reasoning_effort": reasoning_effort is not None,
            "seed": seed is not None,
            "response_format": True,
        }
        self._init_params({
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "seed": seed,
            "max_completion_tokens": max_completion_tokens,
            "response_format": "json_schema(strict)",
            "decoding": "api_default",
        }, strict_params=strict_params)

        if temperature is not None and reasoning_effort is not None:
            print(
                f"    [{self.name}] WARNING: temperature={temperature} together with "
                f"reasoning_effort={reasoning_effort!r} is rejected by gpt-5.1. "
                f"Set one of them to None.",
                flush=True,
            )

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
        if self._use["seed"]:
            kwargs["seed"] = self._seed
        if self._use["response_format"]:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": JSON_SCHEMA}
        return kwargs

    @staticmethod
    def _rejected_parameter(exc) -> str | None:
        """The offending parameter, from the 400 body rather than the message text."""
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            parameter = (body.get("error") or {}).get("param")
            if parameter:
                return str(parameter).split(".")[0]
        lowered = str(exc).lower()
        for parameter in _DROPPABLE:
            if parameter in lowered:
                return parameter
        return None

    def _handle_rejection(self, exc) -> bool:
        """Abort in strict mode; otherwise record the drop and report retryable."""
        parameter = self._rejected_parameter(exc)
        raw_error = str(exc)
        if self.strict_params:
            raise ParameterRejectedError(
                self.name, parameter, raw_error, self.requested_params
            ) from exc
        if parameter in _DROPPABLE and self._use.get(parameter):
            self._use[parameter] = False
            self.record_param_event(
                parameter, "dropped", raw_error,
                "strict_params=False: continuing without it",
            )
            return True
        return False

    def _call(self, system: str, user: str) -> Prediction:
        import openai

        try:
            response = self._client.chat.completions.create(**self._request_kwargs(system, user))
        except openai.BadRequestError as exc:
            if self._handle_rejection(exc):
                return self._call(system, user)
            raise

        choice = response.choices[0]
        raw = (choice.message.content or "").strip()
        finish = getattr(choice, "finish_reason", None)
        usage = _usage(response)
        fingerprint = getattr(response, "system_fingerprint", None)

        if not raw:
            return Prediction(
                label=ERROR, raw="", ok=False, error_kind="EmptyResponse",
                error_detail=(
                    f"finish_reason={finish}; if 'length', the budget ran out -- "
                    f"raise max_completion_tokens (now {self.max_completion_tokens})"
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
        if fingerprint:
            usage = {**usage, "system_fingerprint": fingerprint}
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
