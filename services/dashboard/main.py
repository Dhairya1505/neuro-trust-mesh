"""Dashboard -- a read-only aggregator + UI over every other service's
REST API. Not part of the decision-making pipeline at all (no event bus
connection, no writes anywhere): it exists purely to make the system's
live state legible, matching the explainability principle the rest of
this system is built around.

Aggregates server-side (not client-side fetches from the browser) so the
frontend never needs CORS configured on nine other services and gets one
fast, parallel round of requests per refresh instead of dozens.
"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from neurotrust_common import get_logger, get_settings

SERVICE_NAME = "dashboard"

SERVICE_URLS = {
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
}

settings = get_settings(SERVICE_NAME)
log = get_logger(SERVICE_NAME, settings.log_level)

STATIC_DIR = Path(__file__).parent / "static"


async def _get(client: httpx.AsyncClient, url: str) -> dict | list | None:
    try:
        resp = await client.get(url)
        if resp.status_code == 200:
            return resp.json()
        return None
    except httpx.HTTPError:
        return None


async def _agent_detail(client: httpx.AsyncClient, agent: dict, recovery_map: dict) -> dict:
    agent_id = agent["agent_id"]
    trust, forecast, anomaly, security, features, twin = await asyncio.gather(
        _get(client, f"{SERVICE_URLS['trust_prediction']}/trust/{agent_id}"),
        _get(client, f"{SERVICE_URLS['trust_forecast']}/forecast/{agent_id}"),
        _get(client, f"{SERVICE_URLS['anomaly_detection']}/anomaly/{agent_id}"),
        _get(client, f"{SERVICE_URLS['security_monitoring']}/security/{agent_id}"),
        _get(client, f"{SERVICE_URLS['behavior_analysis']}/features/{agent_id}"),
        _get(client, f"{SERVICE_URLS['cdt_service']}/twins/{agent_id}"),
    )
    return {
        **agent,
        "trust": trust,
        "forecast": forecast,
        "anomaly": anomaly,
        "security": security,
        "features": features,
        "twin": twin,
        # Recovery/ETA info (system doc §4.10) -- only present for
        # currently-isolated agents; None otherwise.
        "recovery": recovery_map.get(agent_id),
    }


def _summarize(agents: list[dict]) -> dict:
    total = len(agents)
    active = sum(1 for a in agents if a["status"] == "active")
    isolated = sum(1 for a in agents if a["status"] == "isolated")
    leaders = sum(1 for a in agents if a["role"] == "leader")
    trust_scores = [a["trust"]["trust_score"] for a in agents if a.get("trust")]
    avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else None
    return {
        "total_agents": total,
        "active": active,
        "isolated": isolated,
        "leaders": leaders,
        "avg_trust": avg_trust,
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("dashboard started")
    yield


app = FastAPI(title="NeuroTrust Mesh Dashboard", lifespan=lifespan)


@app.get("/api/overview")
async def overview():
    async with httpx.AsyncClient(timeout=8) as client:
        agents_raw, pairs, reviews, federated, decisions, isolated = await asyncio.gather(
            _get(client, f"{SERVICE_URLS['agent_registry']}/agents"),
            _get(client, f"{SERVICE_URLS['collusion_detection']}/pairs"),
            _get(client, f"{SERVICE_URLS['collusion_detection']}/reviews"),
            _get(client, f"{SERVICE_URLS['federated_learning']}/status"),
            _get(client, f"{SERVICE_URLS['decision_engine']}/recent_decisions?limit=30"),
            _get(client, f"{SERVICE_URLS['agent_isolation']}/isolated"),
        )
        agents_raw = agents_raw or []
        recovery_map = {r["agent_id"]: r for r in (isolated or [])}
        agents = await asyncio.gather(*(_agent_detail(client, a, recovery_map) for a in agents_raw))
        agents = sorted(agents, key=lambda a: a["agent_id"])

    return JSONResponse(
        {
            "summary": _summarize(agents),
            "agents": agents,
            "collusion": {"pairs": pairs or [], "pending_reviews": reviews or []},
            "federated": federated,
            "recent_decisions": decisions or [],
        }
    )


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}
