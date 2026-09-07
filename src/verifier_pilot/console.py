"""UTF-8 console setup.

PyMETA problem statements are Chinese and COJ2022 code contains non-ASCII
comments. On a Korean/Japanese/Chinese Windows console the default code page is
cp949/cp932/cp936, so a bare ``print`` of a problem statement dies with
``UnicodeEncodeError`` mid-run. Every CLI entry point calls :func:`setup` first.
"""

from __future__ import annotations

import sys


def setup() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
