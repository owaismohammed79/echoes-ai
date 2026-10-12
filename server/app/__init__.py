"""Echoes backend application package.

Layout:

* ``app.api``        -- HTTP routes. Parse and shape only.
* ``app.services``   -- business logic: context aggregation, narration, QA,
                        speech providers, sessions.
* ``app.validation`` -- sentence assembly and citation rules (the factual moat).
* ``app.models``     -- shared request/response contracts.
* ``app.config``     -- environment-driven settings.

Construct the app with :func:`create_app`; ``server/main.py`` is the ASGI
entrypoint that does so.
"""

from __future__ import annotations

from app.main import create_app

__all__ = ["create_app"]
