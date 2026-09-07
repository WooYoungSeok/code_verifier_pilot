"""Environment / API-key resolution.

The reference script this pilot started from called::

    load_dotenv(Path(__file__).resolve().parent / ".env")

which looks for ``.env`` *next to the script*. When the file actually lives at
the repository root (the usual layout) nothing is loaded, ``GEMINI_API_KEY``
stays unset, and every Gemini call fails -- one of the ways the original run
died. We instead walk upwards from this file to the repository root and load the
first ``.env`` we find, then fall back to the CWD.
"""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def repo_root() -> Path:
    """Repository root: the first ancestor holding a .git dir or requirements.txt."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists() or (parent / "requirements.txt").exists():
            return parent
    return here.parents[2]


def load_env(override: bool = False) -> Path | None:
    """Load the repository ``.env`` exactly once. Returns the file used, if any."""
    global _LOADED
    if _LOADED and not override:
        return None
    try:
        from dotenv import load_dotenv
    except ImportError:  # dotenv is optional; real env vars still work
        _LOADED = True
        return None

    candidates = [repo_root() / ".env", Path.cwd() / ".env"]
    for candidate in candidates:
        if candidate.is_file():
            load_dotenv(candidate, override=override)
            _LOADED = True
            return candidate
    _LOADED = True
    return None


#: provider -> env var names, in priority order
_KEY_NAMES: dict[str, tuple[str, ...]] = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}


def get_api_key(provider: str, required: bool = True) -> str | None:
    """Fetch the API key for ``provider``, loading ``.env`` first.

    Raises a message that names the exact variable and file to fix, instead of
    letting an empty key surface later as an opaque 400 from the provider.
    """
    load_env()
    names = _KEY_NAMES.get(provider)
    if names is None:
        raise ValueError(f"unknown provider {provider!r}; expected one of {sorted(_KEY_NAMES)}")
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    if not required:
        return None
    raise RuntimeError(
        f"No API key for provider {provider!r}. Set one of {' / '.join(names)} in "
        f"{repo_root() / '.env'} (copy .env.example) or export it in your shell."
    )
