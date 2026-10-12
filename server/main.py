"""ASGI entrypoint.

Run from ``server/``::

    uvicorn main:app --reload

This file is intentionally a shim. The application is assembled by
:func:`app.main.create_app`, and everything it wires lives under ``app/``:
routes in ``app.api``, behaviour in ``app.services``, and the citation rules in
``app.validation``.

Rationale for keeping the shim: ``uvicorn main:app`` and Render's default start
command both target this module, so the entrypoint stays put while the
implementation is free to be organised into packages.
"""

from __future__ import annotations

import os

from app.main import create_app

app = create_app()

__all__ = ["app"]


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
