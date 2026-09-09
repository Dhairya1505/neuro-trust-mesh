from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from neurotrust_common import EventBus, get_logger, get_settings
from neurotrust_common.schemas import AgentRoleChangeRequested, DecisionAction

SERVICE_NAME = "leader_reassignment"
AGENT_REGISTRY_URL = "http://agent_registry:8000"
TRUST_PREDICTION_URL = "http://trust_prediction:8000"

settings = get_settings(SERVICE_NAME)
log = get_logger(SERVICE_NAME, settings.log_level)
bus = EventBus(settings.rabbitmq_url, SERVICE_NAME)


async def _best_candidate(client: httpx.AsyncClient, exclude_agent_id: str) -> str | None:
    """Rank remaining team agents by trust score (system doc §4.8).
    Capability matching is deferred -- M1 has a single flat team.
    """
    resp = await client.get(f"{AGENT_REGISTRY_URL}/agents", params={"role": "worker", "status": "active"})
    resp.raise_for_status()
    candidates = [a for a in resp.json() if a["agent_id"] != exclude_agent_id]
    if not candidates:
        return None

    best_id, best_score = None, -1.0
    for candidate in candidates:
        trust_resp = await client.get(f"{TRUST_PREDICTION_URL}/trust/{candidate['agent_id']}")
        score = trust_resp.json()["trust_score"] if trust_resp.status_code == 200 else 0.5
        if score > best_score:
            best_id, best_score = candidate["agent_id"], score
    return best_id


async def on_decision(event: DecisionAction) -> None:
    if event.action != "reassign_leader":
        return

    async with httpx.AsyncClient(timeout=10) as client:
        promoted = await _best_candidate(client, exclude_agent_id=event.agent_id)
        if promoted is None:
            log.warning("no candidate available for leader reassignment", extra={"trace": {"agent_id": event.agent_id}})
            return

    # Role change is a write/trigger, not a query -- flows through the
    # event bus (system doc §2/§12) rather than a direct PATCH; the
    # candidate lookups above stay REST since they're read-only queries.
    await bus.publish(AgentRoleChangeRequested(agent_id=event.agent_id, role="worker"))
    await bus.publish(AgentRoleChangeRequested(agent_id=promoted, role="leader"))

    log.info(
        "leader reassigned",
        extra={"trace": {"demoted": event.agent_id, "promoted": promoted, "rule": event.rule, "inputs": event.inputs}},
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    await bus.connect()
    await bus.subscribe(["decision.action"], DecisionAction, on_decision)
    log.info("leader_reassignment started")
    yield
    await bus.close()


app = FastAPI(title="Leader Reassignment Service", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}
