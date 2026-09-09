from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from neurotrust_common import EventBus, get_logger, get_settings
from neurotrust_common.schemas import AgentTaskChangeRequested, DecisionAction

SERVICE_NAME = "task_redistribution"
AGENT_REGISTRY_URL = "http://agent_registry:8000"

settings = get_settings(SERVICE_NAME)
log = get_logger(SERVICE_NAME, settings.log_level)
bus = EventBus(settings.rabbitmq_url, SERVICE_NAME)


async def on_decision(event: DecisionAction) -> None:
    if event.action != "redistribute_tasks":
        return

    async with httpx.AsyncClient(timeout=10) as client:
        failing = await client.get(f"{AGENT_REGISTRY_URL}/agents/{event.agent_id}")
        failing.raise_for_status()
        task = failing.json()["current_task"]
        if task is None:
            return  # nothing queued on this agent to move

        candidates_resp = await client.get(
            f"{AGENT_REGISTRY_URL}/agents", params={"role": "worker", "status": "active"}
        )
        candidates = [a for a in candidates_resp.json() if a["agent_id"] != event.agent_id]
        # Greedy reassignment to nearest trusted agent with available capacity
        # (system doc §4.9) -- M1 has no capacity model yet, so pick the
        # first agent with no task currently assigned, else the first candidate.
        target = next((a for a in candidates if a["current_task"] is None), None) or (
            candidates[0] if candidates else None
        )
        if target is None:
            log.warning("no candidate available for task redistribution", extra={"trace": {"agent_id": event.agent_id}})
            return

    # Task reassignment is a write/trigger, not a query -- flows through
    # the event bus (system doc §2/§12); the candidate lookups above stay
    # REST since they're read-only queries.
    await bus.publish(AgentTaskChangeRequested(agent_id=target["agent_id"], current_task=task))
    await bus.publish(AgentTaskChangeRequested(agent_id=event.agent_id, current_task=None))

    log.info(
        "task redistributed",
        extra={
            "trace": {
                "from_agent": event.agent_id,
                "to_agent": target["agent_id"],
                "task": task,
                "rule": event.rule,
            }
        },
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    await bus.connect()
    await bus.subscribe(["decision.action"], DecisionAction, on_decision)
    log.info("task_redistribution started")
    yield
    await bus.close()


app = FastAPI(title="Task Redistribution Service", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}
