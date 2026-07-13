"""Vercel entrypoint. Vercel's Python runtime serves the ASGI variable named `app`."""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root (parent of api/) importable regardless of runtime cwd.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.web.server import app  # noqa: E402,F401
