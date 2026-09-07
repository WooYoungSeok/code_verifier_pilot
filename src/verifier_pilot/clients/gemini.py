"""Gemini verifier client (google-genai).

The straightforward implementation of this call fails on gemini-2.5-flash. Five
distinct causes, all fixed here:

1. **Thinking eats the entire output budget.** Gemini 2.5 Flash has thinking
   enabled by default with a dynamic budget. Combined with a response schema the
   model can spend every output token on thinking, return ``finish_reason
   MAX_TOKENS`` and *zero* text parts. ``response.text`` is then ``None`` and
   ``response.text.strip()`` raises ``AttributeError``. Because that is not a
   recognisably transient error, a naive retry classifier gives up immediately
   and records "error" for the whole run. Fixed by ``ThinkingConfig(
   thinking_budget=0)`` -- the classifier needs no reasoning tokens -- plus an
   explicit ``max_output_tokens``.

2. **``BLOCK_NONE`` is not always grantable.** On projects without the relevant
   allowlist, ``HarmBlockThreshold.BLOCK_NONE`` is rejected with a 400. The 2.5
   models take ``OFF`` instead. We try ``OFF``, fall back to ``BLOCK_NONE``, then
   to sending no safety settings at all, and cache whichever worked.

3. **``response.text`` is ``None`` on a safety block too.** Student code is
   routinely flagged. We inspect ``prompt_feedback`` / ``candidates[0]
   .finish_reason`` and surface a typed reason instead of an AttributeError.

4. **The key may simply be missing.** The reference script loaded ``.env`` from
   the script's own directory, so a repo-root ``.env`` was never read. See
   ``verifier_pilot.env``.

5. **Transient 429/503/500.** Retried by ``RetryingClient`` with jittered
   backoff, honouring ``retry_after`` when the SDK reports one.
"""

from __future__ import annotations

import json

from ..env import get_api_key
from .base import ERROR, Prediction, RetryingClient, parse_label

MODEL = "gemini-2.5-flash"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["aligned", "not_aligned"]}},
    "required": ["label"],
}

_HARM_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)

#: safety-setting strategies, tried in order until one is accepted
_SAFETY_MODES = ("OFF", "BLOCK_NONE", None)


class GeminiClient(RetryingClient):
    provider = "gemini"

    def __init__(
        self,
        model: str = MODEL,
        name: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
        thinking_budget: int = 0,
        max_retries: int = 5,
    ) -> None:
        from google import genai  # imported lazily so other providers work without it

        self.model = model
        self.name = name or model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.thinking_budget = thinking_budget
        self.max_retries = max_retries
        self._client = genai.Client(api_key=get_api_key("gemini"))
        self._safety_mode_index = 0

    # ---- config -----------------------------------------------------------
    def _safety_settings(self):
        from google.genai import types

        mode = _SAFETY_MODES[self._safety_mode_index]
        if mode is None:
            return None
        return [
            types.SafetySetting(category=category, threshold=mode)
            for category in _HARM_CATEGORIES
        ]

    def _config(self, system: str):
        from google.genai import types

        kwargs = dict(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        safety = self._safety_settings()
        if safety is not None:
            kwargs["safety_settings"] = safety
        if self.thinking_budget is not None:
            # Cause (1): 2.5 Flash thinks by default and can burn the whole budget.
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
        return types.GenerateContentConfig(**kwargs)

    def _degrade_safety(self) -> bool:
        """Advance to the next safety strategy. False when none are left."""
        if self._safety_mode_index + 1 < len(_SAFETY_MODES):
            self._safety_mode_index += 1
            print(
                f"    [{self.name}] safety setting rejected; falling back to "
                f"{_SAFETY_MODES[self._safety_mode_index]!r}",
                flush=True,
            )
            return True
        return False

    # ---- response handling -------------------------------------------------
    @staticmethod
    def _extract_text(response) -> str:
        """Pull text out of a response without assuming ``.text`` is populated."""
        text = getattr(response, "text", None)
        if text:
            return text.strip()
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    return part_text.strip()
        return ""

    @staticmethod
    def _blocked_reason(response) -> str | None:
        feedback = getattr(response, "prompt_feedback", None)
        blocked = getattr(feedback, "block_reason", None)
        if blocked:
            return f"prompt_blocked:{blocked}"
        for candidate in getattr(response, "candidates", None) or []:
            finish = getattr(candidate, "finish_reason", None)
            if finish is not None and str(finish).upper().split(".")[-1] not in ("STOP", "NONE"):
                return f"finish_reason:{str(finish).split('.')[-1]}"
        return None

    def _call(self, system: str, user: str) -> Prediction:
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=user, config=self._config(system)
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            # Cause (2): the safety threshold itself was rejected -- retune, retry.
            if ("safety" in message or "harmblockthreshold" in message
                    or "block_none" in message) and self._degrade_safety():
                return self._call(system, user)
            raise

        raw = self._extract_text(response)
        finish = self._blocked_reason(response)
        usage = _usage(response)

        if not raw:
            # Cause (1)/(3): no text came back. Report *why*, do not crash on None.
            return Prediction(
                label=ERROR, raw="", ok=False,
                error_kind="EmptyResponse",
                error_detail=(
                    finish or "no text parts returned"
                ) + " (thinking_budget=%s, max_output_tokens=%s)" % (
                    self.thinking_budget, self.max_output_tokens
                ),
                usage=usage, finish_reason=finish,
            )

        label = None
        try:
            label = json.loads(raw).get("label")
        except (json.JSONDecodeError, AttributeError):
            label = parse_label(raw)
        if label not in ("aligned", "not_aligned"):
            label = parse_label(raw)

        if label is None:
            return Prediction(
                label=ERROR, raw=raw, ok=False, error_kind="ParseFailure",
                error_detail=raw[:200], usage=usage, finish_reason=finish,
            )
        return Prediction(label=label, raw=raw, usage=usage, finish_reason=finish)


def _usage(response) -> dict:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return {}
    return {
        key: value
        for key, value in (
            ("prompt_tokens", getattr(metadata, "prompt_token_count", None)),
            ("output_tokens", getattr(metadata, "candidates_token_count", None)),
            ("thinking_tokens", getattr(metadata, "thoughts_token_count", None)),
            ("total_tokens", getattr(metadata, "total_token_count", None)),
        )
        if value is not None
    }
