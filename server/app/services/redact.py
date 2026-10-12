"""Keeping provider keys out of log messages.

Two providers take their credential as a *query parameter* -- SerpApi's
``api_key`` and Gemini's ``key`` -- and ``httpx`` puts the whole URL into the
exception it raises. So the obvious ``logger.warning("lookup failed: %s", exc)``
writes a live credential into the log, where it stays: in a terminal scrollback,
in a Render log, in a pasted bug report. That is exactly how the SerpApi key
reached the console during the debugging that found the unsupported-engine bug.

Every message that may contain a request URL goes through :func:`redact` first.
Keys sent as *headers* (ElevenLabs, Groq) do not need it, because httpx does not
put headers in its error text.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = ["redact"]

#: A credential in a URL: ``api_key=...`` or ``key=...``, up to the next
#: separator. Deliberately broad on the left (``[?&;]``) so a key in the middle
#: of a query string is caught, not just the first parameter.
_CREDENTIAL: Final[re.Pattern[str]] = re.compile(
    r"([?&;](?:api_key|key|access_token|token)=)[^&\s]+",
    re.IGNORECASE,
)


def redact(value: object) -> str:
    """A string with any credential in it replaced by ``<redacted>``.

    Takes ``object`` rather than ``str`` so an exception can be passed straight
    through -- ``redact(exc)`` -- at the call site that logs it.
    """
    return _CREDENTIAL.sub(r"\1<redacted>", str(value))
