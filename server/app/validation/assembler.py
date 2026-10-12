"""Streaming sentence assembly.

Implements section 7.1 of docs/llm-orchestration.md.

The assembler turns a token stream into complete sentences. It exists as its own
module because sentence splitting is load-bearing rather than plumbing: the
validator judges *sentences*, so a split made in the wrong place changes what is
judged. Two cases break naive splitters and are handled here:

* abbreviations and initials (``"St. James's"``, ``"J. Smith"``) -- splitting
  there would let half a sentence pass validation on its own;
* a marker sitting *directly* after the terminal period
  (``"Nelson rises.[cite:abc1]"``, ``"...the buzz is real.[news]"``) -- a
  "terminator followed by whitespace" rule never fires there, gluing a whole
  response into one sentence.
"""

from __future__ import annotations

from app.validation.citations import match_tag

_TERMINATORS = ".!?\u2026"

#: Tokens after which a period is not a sentence boundary.
_ABBREVIATIONS = frozenset(
    {
        "st", "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc",
        "no", "approx", "ca", "cf", "eg", "ie", "al", "inc", "ltd", "co",
        "fig", "vol", "ch", "pt", "sq", "ave", "rd", "blvd", "mt",
    }
)


def _sentence_boundary_end(text: str, index: int) -> int | None:
    """End index (exclusive) of the sentence terminated at ``index``, else None.

    Handles the case that matters most: a tag normally sits *directly* after the
    terminal period, e.g. ``"Nelson rises.[cite:abc1]"`` or ``"...the buzz is
    real.[news]"``. A plain "terminator followed by whitespace" rule would
    never split there, gluing every sentence in a response into one.
    """
    j = index + 1
    while True:
        k = j
        while k < len(text) and text[k] in " \t":
            k += 1
        end = match_tag(text, k)
        if end is None:
            break
        j = end

    if j >= len(text) or text[j].isspace():
        return j
    return None


def _is_abbreviation(text: str, index: int) -> bool:
    """True when the period at ``index`` ends an abbreviation or an initial."""
    start = index
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "."):
        start -= 1
    token = text[start:index].strip(".")
    if not token:
        return False
    # single-letter initial, e.g. "J. Smith"
    if len(token) == 1 and token.isalpha() and token.isupper():
        return True
    return token.lower() in _ABBREVIATIONS


def _is_decimal(text: str, index: int) -> bool:
    """True when the period at ``index`` is a decimal point or a version dot."""
    if text[index] != ".":
        return False
    if index == 0 or not text[index - 1].isdigit():
        return False
    following = text[index + 1 : index + 2]
    return bool(following) and following.isdigit()


class SentenceAssembler:
    """Accumulates stream chunks and yields complete sentences.

    Only the final fragment is withheld until :meth:`flush`, so the last
    sentence of a response is validated exactly like the others.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def push(self, chunk: str) -> list[str]:
        """Add stream chunk(s); return any sentences that are now complete."""
        self._buffer += chunk
        sentences: list[str] = []
        cursor = 0

        while cursor < len(self._buffer):
            char = self._buffer[cursor]
            if char in _TERMINATORS:
                if char == "." and (
                    _is_decimal(self._buffer, cursor)
                    or _is_abbreviation(self._buffer, cursor)
                ):
                    cursor += 1
                    continue
                end = _sentence_boundary_end(self._buffer, cursor)
                if end is None:
                    cursor += 1
                    continue
                candidate = self._buffer[:end].strip()
                if candidate:
                    sentences.append(candidate)
                self._buffer = self._buffer[end:]
                cursor = 0
                continue
            cursor += 1

        return sentences

    def flush(self) -> list[str]:
        """Return the unterminated tail, if any, and reset."""
        tail = self._buffer.strip()
        self._buffer = ""
        return [tail] if tail else []


__all__ = ["SentenceAssembler"]
