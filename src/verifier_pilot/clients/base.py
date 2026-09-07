"""Verifier client contract, shared retry policy, and answer parsing.

Sampling parameters are treated as part of the experimental record. A parameter
the API refuses is never dropped silently: in the default ``strict_params`` mode
the run aborts with the raw provider error, and with ``strict_params=False`` the
drop is logged loudly *and* recorded in ``param_events``, which the runner writes
into the run manifest alongside the results. A verifier scored under a
temperature the write-up does not know about is a silently invalid experiment.
"""

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

#: recorded when a provider has no equivalent of a parameter we standardise on
UNSUPPORTED_BY_PROVIDER = "<unsupported_by_provider>"

#: Output-token cap, identical for every model so the budget is not a
#: per-model variable. The answer is one short label -- measured maxima are 20
#: tokens (gpt-5.1, JSON schema), 35 (claude, tool_use block), 6 (gemini), and a
#: bare word for the open-weight models -- so 64 leaves headroom without ever
#: binding. A cap that never binds does not affect the output; if a run starts
#: reporting finish_reason 'length' / 'max_tokens', raise it here for all models
#: at once and record it in EXPERIMENT_LOG.md.
MAX_ANSWER_TOKENS = 64


class ParameterRejectedError(RuntimeError):
    """The API refused a sampling parameter and strict_params is on.

    Carries the raw provider error so the decision -- change the parameter, or
    accept the fallback and say so in the write-up -- is made by a person.
    """

    def __init__(self, model: str, parameter: str | None, raw_error: str, requested: dict):
        self.model = model
        self.parameter = parameter
        self.raw_error = raw_error
        self.requested = dict(requested)
        super().__init__(
            f"\n{'=' * 72}\n"
            f"{model}: the API rejected parameter {parameter!r}.\n"
            f"{'=' * 72}\n"
            f"requested parameters : {requested}\n\n"
            f"raw provider error:\n  {raw_error}\n\n"
            f"This is NOT dropped automatically, because running with different\n"
            f"sampling parameters than the ones you recorded invalidates the\n"
            f"comparison. Choose one and record it in EXPERIMENT_LOG.md:\n"
            f"  1. change the parameter (e.g. drop reasoning_effort to keep\n"
            f"     temperature=0 on gpt-5.1), or\n"
            f"  2. re-run with --allow-param-fallback to let the client drop it;\n"
            f"     the drop is then written into the run manifest.\n"
            f"{'=' * 72}"
        )


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
    #: abort rather than silently drop a rejected sampling parameter
    strict_params: bool = True

    #: what we asked for, and what the API actually honoured
    requested_params: dict
    effective_params: dict
    #: every rejection/fallback, with the raw provider error
    param_events: list[dict]

    def _init_params(self, requested: dict, strict_params: bool = True) -> None:
        """Call from __init__ once the requested sampling parameters are known."""
        self.strict_params = strict_params
        self.requested_params = dict(requested)
        self.effective_params = dict(requested)
        self.param_events = []

    def record_param_event(
        self, parameter: str | None, action: str, raw_error: str = "", detail: str = ""
    ) -> None:
        """Log a parameter change so it reaches the run manifest, not just stdout."""
        self.param_events.append({
            "parameter": parameter,
            "action": action,               # "dropped" | "unsupported_by_provider" | "changed"
            "raw_error": raw_error[:1000],
            "detail": detail,
        })
        if action == "dropped":
            self.effective_params[parameter] = f"<dropped: {raw_error[:120]}>"
        print(
            f"    [{self.name}] PARAMETER {action.upper()}: {parameter!r}"
            + (f" -- {detail}" if detail else "")
            + (f"\n      raw error: {raw_error[:300]}" if raw_error else ""),
            flush=True,
        )

    def params_manifest(self) -> dict:
        """The sampling provenance for this client, for the run manifest.

        The model *identifier* must be a string: on local clients ``self.model``
        is the loaded nn.Module, so ``model_id`` is preferred and anything
        non-string falls back to the client name. The manifest is written with
        json.dump and a non-serialisable value here kills the run before the
        first request.
        """
        identifier = getattr(self, "model_id", None)
        if not isinstance(identifier, str):
            identifier = getattr(self, "model", None)
        if not isinstance(identifier, str):
            identifier = self.name
        return {
            "model": identifier,
            "provider": self.provider,
            "strict_params": self.strict_params,
            "requested": getattr(self, "requested_params", {}),
            "effective": getattr(self, "effective_params", {}),
            "param_events": getattr(self, "param_events", []),
        }

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
            except ParameterRejectedError:
                # Never swallowed into a Prediction.failure: a rejected sampling
                # parameter must stop the run, not become one more error row.
                raise
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
