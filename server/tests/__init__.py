"""Echoes backend test suite.

Run from ``server/``:  python -m pytest -q

No test in this suite requires network access or an API key. Every external
service is either driven through ``httpx.MockTransport`` or replaced by a stub
that records what it was asked to do, so the fallback paths are genuinely
executed rather than assumed.
"""
