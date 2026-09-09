"""API Gateway -- single entry point for external/operator requests (system
doc §3, service #1), routing to the right internal service by name instead
of every external client needing to know every service's address/port.

Scope kept minimal: pure reverse-proxy routing, no auth/rate-limiting yet
-- those are natural follow-ups but not required to satisfy the "single
entry point" requirement, and this is a student project, not a production
deployment.
"""
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response

from neurotrust_common import get_logger, get_settings

SERVICE_NAME = "api_gateway"

# Mirrors docker-compose.yml's service names/ports -- every other service's
# internal address, keyed by the path prefix external clients use.
SERVICE_MAP = {
    "agent_registry": "http://agent_registry:8000",
    "cdt_service": "http://cdt_service:8000",
    "behavior_analysis": "http://behavior_analysis:8000",
    "trust_prediction": "http://trust_prediction:8000",
    "decision_engine": "http://decision_engine:8000",
    "security_monitoring": "http://security_monitoring:8000",
    "anomaly_detection": "http://anomaly_detection:8000",
    "trust_forecast": "http://trust_forecast:8000",
    "collusion_detection": "http://collusion_detection:8000",
    "federated_learning": "http://federated_learning:8000",
    "agent_isolation": "http://agent_isolation:8000",
    "dashboard": "http://dashboard:8000",
}

# Headers that don't make sense to forward across a proxy hop.
_HOP_BY_HOP = {"host", "content-length", "connection", "transfer-encoding"}

settings = get_settings(SERVICE_NAME)
log = get_logger(SERVICE_NAME, settings.log_level)


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("api_gateway started")
    yield


app = FastAPI(title="NeuroTrust Mesh API Gateway", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.api_route("/{service}/{path:path}", methods=["GET", "POST", "PATCH", "PUT", "DELETE"])
async def proxy(service: str, path: str, request: Request):
    base_url = SERVICE_MAP.get(service)
    if base_url is None:
        return Response(content=f'{{"error": "unknown service \\"{service}\\""}}', status_code=404, media_type="application/json")

    target_url = f"{base_url}/{path}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
    body = await request.body()

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.request(
                request.method, target_url, params=request.query_params, headers=headers, content=body
            )
        except httpx.HTTPError as exc:
            log.warning("upstream request failed", extra={"trace": {"service": service, "error": str(exc)}})
            return Response(content='{"error": "upstream service unavailable"}', status_code=502, media_type="application/json")

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )
