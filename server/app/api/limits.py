"""Who is asking, and what they are allowed.

The service layer decides; this module only turns a request into an identity and
a decision into headers.

**On identity, which is the part that goes wrong.** Behind a proxy the socket
address is the proxy, so the limit has to key off ``X-Forwarded-For``. That
header is client-writable, so trusting it blindly hands every abuser a fresh
allowance: send a random address each time and the limiter never trips. The rule
here is hop-counting, which is the only version of this that survives contact
with an attacker -- ``ECHOES_TRUSTED_PROXY_HOPS`` says how many proxies append
to the chain, and the client is the entry that many from the *right*. Render's
edge is one hop, so the default reads the last entry, and a forged prefix is
harmless because the trusted hop appended the real address after it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from app.config import TRUSTED_PROXY_HOPS
from app.dependencies import ContainerDep
from app.services.budgets import Admission, Guardrails

logger = logging.getLogger("echoes.api.limits")

__all__ = [
    "AdmittedWalk",
    "AdmittedWalkDep",
    "DEMO_HEADER",
    "DemoAdmissionDep",
    "admit_qa",
    "admit_walk",
    "client_identity",
    "demo_headers",
    "rate_headers",
]

#: The kill-switch flag. Value is capitalised exactly as specified, and the name
#: is hyphenated because nginx -- and therefore Render -- drops headers
#: containing underscores. Same reasoning as ``X-Use-Native-Tts``.
DEMO_HEADER = "X-Demo-Mode"


def client_identity(request: Request) -> str:
    """The client's address, resolved through the trusted proxy chain."""
    socket_host = request.client.host if request.client else "unknown"
    hops = TRUSTED_PROXY_HOPS
    if hops <= 0:
        # No proxy in front: the socket address is the client, and the forwarded
        # header is just something a client made up.
        return socket_host

    forwarded = request.headers.get("x-forwarded-for", "")
    chain = [entry.strip() for entry in forwarded.split(",") if entry.strip()]
    if len(chain) >= hops:
        return chain[-hops]
    # Fewer entries than trusted hops: the request did not come through the
    # expected chain, so the socket address is the honest answer.
    return socket_host


@dataclass(frozen=True)
class AdmittedWalk:
    """A walk request that has been identified and decided on, not yet charged."""

    identity: str
    admission: Admission
    guardrails: Guardrails


def admit_qa(container: ContainerDep) -> Admission:
    """The decision for push-to-talk: the cap applies, the walk limit does not.

    PTT is a paid path (ElevenLabs and the model both answer it), so it must
    honour the kill-switch. It is not a walk *generation*, so it must not consume
    the walker's hourly allowance -- someone who has asked five questions has not
    started five walks.
    """
    return container.guardrails.admit("qa", enforce_walk_limit=False)


#: Handler signature shorthand: ``admission: DemoAdmissionDep``.
DemoAdmissionDep = Annotated[Admission, Depends(admit_qa)]


def admit_walk(request: Request, container: ContainerDep) -> AdmittedWalk:
    """FastAPI dependency: identify the caller and decide how to serve them.

    Deliberately does not consume quota. The charge is taken by the handler once
    it knows a walk is really being generated, so a request that 422s on its body
    -- or is served from the demo -- never spends a walk.
    """
    identity = client_identity(request)
    admission = container.guardrails.admit(identity)
    if not admission.allowed:
        logger.info("walk limit reached for %s", identity)
    return AdmittedWalk(
        identity=identity,
        admission=admission,
        guardrails=container.guardrails,
    )


#: Handler signature shorthand: ``admitted: AdmittedWalkDep``.
AdmittedWalkDep = Annotated[AdmittedWalk, Depends(admit_walk)]


def demo_headers(demo: bool, *, reason: str | None = None) -> dict[str, str]:
    """The demo flag, and why it fired, so an operator can act on it.

    ``X-Demo-Mode`` is emitted even when false. The absence of a header is
    indistinguishable from a proxy stripping it, and a client that cannot tell
    those apart cannot label the mode honestly.
    """
    headers = {DEMO_HEADER: "True" if demo else "False"}
    if demo and reason:
        headers["X-Demo-Reason"] = reason
    return headers


def rate_headers(admission: Admission, *, limit: int) -> dict[str, str]:
    """Quota headers. Sent on success and refusal alike."""
    headers = {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(max(0, admission.walks_remaining)),
    }
    if not admission.allowed and admission.retry_after_s:
        headers["Retry-After"] = str(admission.retry_after_s)
    return headers
