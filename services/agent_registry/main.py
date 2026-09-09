import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from models import Agent
from neurotrust_common import EventBus, get_logger, get_settings
from neurotrust_common.db import Database
from neurotrust_common.schemas import (
    AgentHeartbeat,
    AgentIsolationRequested,
    AgentMissedHeartbeat,
    AgentReinstateRequested,
    AgentRoleChangeRequested,
    AgentTaskChangeRequested,
)

SERVICE_NAME = "agent_registry"
HEARTBEAT_INTERVAL_S = 5
MISSED_BEATS_THRESHOLD = 3

settings = get_settings(SERVICE_NAME)
log = get_logger(SERVICE_NAME, settings.log_level)
db = Database(settings.postgres_dsn)
bus = EventBus(settings.rabbitmq_url, SERVICE_NAME)


async def _missed_heartbeat_monitor() -> None:
    """Every HEARTBEAT_INTERVAL_S, flag agents that have gone quiet.

    Missed heartbeats are the first signal fed into Behavior Analysis
    (system doc §4.1) -- this loop is the source of truth for that signal.
    """
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)
        now = datetime.now(timezone.utc)
        async with db.session() as session:
            result = await session.execute(select(Agent).where(Agent.status == "active"))
            agents = result.scalars().all()
            for agent in agents:
                gap = (now - agent.last_heartbeat).total_seconds()
                if gap > HEARTBEAT_INTERVAL_S:
                    agent.missed_beats += 1
                    if agent.missed_beats > MISSED_BEATS_THRESHOLD:
                        await bus.publish(
                            AgentMissedHeartbeat(
                                agent_id=agent.agent_id, missed_beats=agent.missed_beats
                            )
                        )
                        log.warning(
                            "agent silent",
                            extra={"trace": {"agent_id": agent.agent_id, "missed_beats": agent.missed_beats}},
                        )
            await session.commit()


async def _set_role(agent_id: str, role: str) -> bool:
    async with db.session() as session:
        agent = await session.get(Agent, agent_id)
        if agent is None:
            return False
        agent.role = role
        await session.commit()
    log.info("agent role updated", extra={"trace": {"agent_id": agent_id, "role": role}})
    return True


async def _set_task(agent_id: str, current_task: str | None) -> bool:
    async with db.session() as session:
        agent = await session.get(Agent, agent_id)
        if agent is None:
            return False
        agent.current_task = current_task
        await session.commit()
    log.info("agent task updated", extra={"trace": {"agent_id": agent_id, "current_task": current_task}})
    return True


async def _set_isolated(agent_id: str) -> bool:
    async with db.session() as session:
        agent = await session.get(Agent, agent_id)
        if agent is None:
            return False
        agent.status = "isolated"
        await session.commit()
    log.info("agent isolated", extra={"trace": {"agent_id": agent_id}})
    return True


async def _set_reinstated(agent_id: str) -> bool:
    async with db.session() as session:
        agent = await session.get(Agent, agent_id)
        if agent is None:
            return False
        agent.status = "active"
        agent.missed_beats = 0
        await session.commit()
    log.info("agent reinstated", extra={"trace": {"agent_id": agent_id}})
    return True


async def on_role_change_requested(event: AgentRoleChangeRequested) -> None:
    if not await _set_role(event.agent_id, event.role):
        log.warning("role change requested for unknown agent", extra={"trace": {"agent_id": event.agent_id}})


async def on_task_change_requested(event: AgentTaskChangeRequested) -> None:
    if not await _set_task(event.agent_id, event.current_task):
        log.warning("task change requested for unknown agent", extra={"trace": {"agent_id": event.agent_id}})


async def on_isolation_requested(event: AgentIsolationRequested) -> None:
    if not await _set_isolated(event.agent_id):
        log.warning("isolation requested for unknown agent", extra={"trace": {"agent_id": event.agent_id}})


async def on_reinstate_requested(event: AgentReinstateRequested) -> None:
    if not await _set_reinstated(event.agent_id):
        log.warning("reinstate requested for unknown agent", extra={"trace": {"agent_id": event.agent_id}})


@asynccontextmanager
async def lifespan(_: FastAPI):
    await db.create_all()
    await bus.connect()
    await bus.subscribe(["agent.role_change_requested"], AgentRoleChangeRequested, on_role_change_requested)
    await bus.subscribe(["agent.task_change_requested"], AgentTaskChangeRequested, on_task_change_requested)
    await bus.subscribe(["agent.isolation_requested"], AgentIsolationRequested, on_isolation_requested)
    await bus.subscribe(["agent.reinstate_requested"], AgentReinstateRequested, on_reinstate_requested)
    monitor_task = asyncio.create_task(_missed_heartbeat_monitor())
    log.info("agent_registry started")
    yield
    monitor_task.cancel()
    await bus.close()
    await db.dispose()


app = FastAPI(title="Agent Registry", lifespan=lifespan)


class RegisterRequest(BaseModel):
    agent_id: str
    role: str
    capabilities: list[str] = []
    current_task: str | None = None


class HeartbeatRequest(BaseModel):
    current_task: str | None = None


class RoleUpdateRequest(BaseModel):
    role: str


class TaskUpdateRequest(BaseModel):
    current_task: str | None


@app.post("/agents/register", status_code=201)
async def register_agent(body: RegisterRequest):
    async with db.session() as session:
        existing = await session.get(Agent, body.agent_id)
        if existing is None:
            existing = Agent(agent_id=body.agent_id)
            session.add(existing)
        existing.role = body.role
        existing.capabilities = body.capabilities
        existing.current_task = body.current_task
        existing.status = "active"
        existing.missed_beats = 0
        existing.last_heartbeat = datetime.now(timezone.utc)
        await session.commit()
    log.info("agent registered", extra={"trace": {"agent_id": body.agent_id, "role": body.role}})
    return {"agent_id": body.agent_id, "status": "registered"}


@app.post("/agents/{agent_id}/heartbeat")
async def heartbeat(agent_id: str, body: HeartbeatRequest):
    async with db.session() as session:
        agent = await session.get(Agent, agent_id)
        if agent is None:
            raise HTTPException(404, f"agent {agent_id} not registered")
        agent.last_heartbeat = datetime.now(timezone.utc)
        agent.missed_beats = 0
        # An isolated agent is removed from the active task pool (system
        # doc §4.10) -- it can still heartbeat (CDT observation mode), but
        # must not be allowed to self-assign a new current_task.
        if body.current_task is not None and agent.status != "isolated":
            agent.current_task = body.current_task
        role, capabilities, current_task, status = agent.role, agent.capabilities, agent.current_task, agent.status
        await session.commit()

    await bus.publish(
        AgentHeartbeat(
            agent_id=agent_id, role=role, capabilities=capabilities, current_task=current_task
        )
    )
    return {"agent_id": agent_id, "status": status}


@app.get("/agents")
async def list_agents(role: str | None = None, status: str | None = None):
    async with db.session() as session:
        stmt = select(Agent)
        if role:
            stmt = stmt.where(Agent.role == role)
        if status:
            stmt = stmt.where(Agent.status == status)
        result = await session.execute(stmt)
        agents = result.scalars().all()
        return [
            {
                "agent_id": a.agent_id,
                "role": a.role,
                "capabilities": a.capabilities,
                "current_task": a.current_task,
                "status": a.status,
                "missed_beats": a.missed_beats,
            }
            for a in agents
        ]


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    async with db.session() as session:
        agent = await session.get(Agent, agent_id)
        if agent is None:
            raise HTTPException(404, f"agent {agent_id} not registered")
        return {
            "agent_id": agent.agent_id,
            "role": agent.role,
            "capabilities": agent.capabilities,
            "current_task": agent.current_task,
            "status": agent.status,
            "missed_beats": agent.missed_beats,
        }


@app.patch("/agents/{agent_id}/role")
async def update_role(agent_id: str, body: RoleUpdateRequest):
    """Kept for manual/admin use -- the decision pipeline itself now
    triggers role changes via the agent.role_change_requested event
    (system doc §2/§12: writes flow through the bus, not direct
    request-response) rather than calling this endpoint.
    """
    if not await _set_role(agent_id, body.role):
        raise HTTPException(404, f"agent {agent_id} not registered")
    return {"agent_id": agent_id, "role": body.role}


@app.patch("/agents/{agent_id}/task")
async def update_task(agent_id: str, body: TaskUpdateRequest):
    """Kept for manual/admin use -- see update_role's note."""
    if not await _set_task(agent_id, body.current_task):
        raise HTTPException(404, f"agent {agent_id} not registered")
    return {"agent_id": agent_id, "current_task": body.current_task}


@app.patch("/agents/{agent_id}/isolate")
async def isolate_agent(agent_id: str):
    """Kept for manual/admin use -- see update_role's note."""
    if not await _set_isolated(agent_id):
        raise HTTPException(404, f"agent {agent_id} not registered")
    return {"agent_id": agent_id, "status": "isolated"}


@app.patch("/agents/{agent_id}/reinstate")
async def reinstate_agent(agent_id: str):
    """Auto-reinstatement (system doc §4.10's 'recovery tracking'):
    isolation was a one-way door until now -- an agent whose risk was a
    transient/false-positive spike stayed isolated forever, so any nonzero
    false-positive rate compounded into "everyone eventually isolated"
    over a long enough run. Kept for manual/admin use -- see update_role's
    note about the event-driven trigger path.
    """
    if not await _set_reinstated(agent_id):
        raise HTTPException(404, f"agent {agent_id} not registered")
    return {"agent_id": agent_id, "status": "active"}


@app.get("/health")
async def health():
    return {"status": "ok"}
