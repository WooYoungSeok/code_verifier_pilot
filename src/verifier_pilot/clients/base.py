"""Verifier client contract, shared retry policy, and answer parsing."""

from __future__ import annotations

import abc
import random
import re
import time
from dataclasses import dataclass, field, asdict

ALIGNED = "aligned"
NOT_ALIGNED = "not_aligned"
ERROR = "error"
VALID_LABELS = (ALIGNED, NOT_ALIGNED)


@dataclass
class Prediction:
    """One verifier judgement.

    ``label`` is ``aligned`` / ``not_aligned`` / ``error``. ``error`` is kept
    distinct from a wrong answer so that API failures are never silently scored
    as incorrect predictions -- they are reported separately and, by default,
    excluded from the metrics.
    """

    label: str
    raw: str = ""
    ok: bool = True
    error_kind: str | None = None
    error_detail: str | None = None
    attempts: int = 1
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def failure(cls, kind: str, detail: str, attempts: int = 1) -> "Prediction":
        return cls(
            label=ERROR, raw="", ok=False, error_kind=kind,
            error_detail=detail[:500], attempts=attempts,
        )


_LABEL_PATTERN = re.compile(r"\b(not[_\s-]?aligned|aligned)\b", re.IGNORECASE)


def parse_label(text: str) -> str | None:
    """Recover a label from free text.

    Used as a fallback when a structured-output channel is unavailable or the
    model wraps its answer in prose. ``not_aligned`` is matched before
    ``aligned`` because the latter is a substring of the former; when both a
    'not aligned' and a bare 'aligned' occur, the first match in the string wins.
    """
    if not text:
        return None
    match = _LABEL_PATTERN.search(text)
    if match is None:
        return None
    token = match.group(1).lower().replace(" ", "_").replace("-", "_")
    return NOT_ALIGNED if token == "not_aligned" else ALIGNED


#: Substrings that mark a transient, worth-retrying failure across all providers.
RETRYABLE_MARKERS: tuple[str, ...] = (
    "429", "resource_exhausted", "resourceexhausted", "quota", "rate limit", "rate_limit",
    "500", "502", "503", "504", "internal error", "internalservererror",
    "unavailable", "overloaded", "high demand", "server had an error",
    "timeout", "timed out", "deadline", "connection", "connection reset",
    "temporarily", "try again",
)

#: Substrings that mark a permanent failure -- retrying only wastes quota.
FATAL_MARKERS: tuple[str, ...] = (
    "api key not valid", "invalid api key", "unauthenticated", "401", "403",
    "permission denied", "authentication", "billing", "not found", "404",
    "model not found", "does not exist", "unsupported",
)


def classify_error(exc: BaseException) -> tuple[str, bool]:
    """Return ``(kind, retryable)`` for a provider exception."""
    text = f"{type(exc).__name__}: {exc}".lower()
    for marker in FATAL_MARKERS:
        if marker in text:
            return type(exc).__name__, False
    for marker in RETRYABLE_MARKERS:
        if marker in text:
            return type(exc).__name__, True
    return type(exc).__name__, False


def backoff_sleep(attempt: int, base: float = 4.0, cap: float = 90.0, rng: random.Random | None = None) -> float:
    """Exponential backoff with full jitter. Returns the seconds slept."""
    rng = rng or random
    delay = min(cap, base * (2 ** attempt))
    delay = rng.uniform(delay * 0.5, delay)
    time.sleep(delay)
    return delay


class VerifierClient(abc.ABC):
    """A model that answers one verifier query."""

    #: identifier used in output filenames and result tables
    name: str
    #: "gemini" | "openai" | "anthropic" | "local"
    provider: str
    max_retries: int = 5

    @abc.abstractmethod
    def predict(self, system: str, user: str) -> Prediction:
        """Answer one query. Must never raise -- return Prediction.failure instead."""

    def close(self) -> None:
        """Release resources (GPU memory for local models). Optional."""

    def __enter__(self) -> "VerifierClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class RetryingClient(VerifierClient):
    """Base class implementing the shared retry loop.

    Subclasses implement :meth:`_call`, which either returns a Prediction or
    raises. Transient provider failures are retried with jittered exponential
    backoff; permanent ones fail fast so a bad key does not burn the run.
    """

    def _call(self, system: str, user: str) -> Prediction:  # pragma: no cover - abstract
        raise NotImplementedError

    def predict(self, system: str, user: str) -> Prediction:
        last_kind = last_detail = ""
        for attempt in range(self.max_retries):
            started = time.monotonic()
            try:
                prediction = self._call(system, user)
                prediction.attempts = attempt + 1
                prediction.latency_s = round(time.monotonic() - started, 3)
                return prediction
            except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
                kind, retryable = classify_error(exc)
                last_kind, last_detail = kind, f"{type(exc).__name__}: {exc}"
                if retryable and attempt < self.max_retries - 1:
                    slept = backoff_sleep(attempt)
                    print(
                        f"    [{self.name}] retry {attempt + 1}/{self.max_retries} "
                        f"in {slept:.1f}s ({kind})",
                        flush=True,
                    )
                    continue
                break
        return Prediction.failure(last_kind or "UnknownError", last_detail, self.max_retries)
