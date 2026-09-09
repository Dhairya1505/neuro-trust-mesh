from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from sqlalchemy.exc import IntegrityError

from models import Twin
from neurotrust_common import EventBus, get_logger, get_settings
from neurotrust_common.db import Database
from neurotrust_common.locks import KeyedLock
from neurotrust_common.schemas import (
    AgentHeartbeat,
    AgentMetricsSample,
    AgentTaskOutcome,
    TrustScore,
)

SERVICE_NAME = "cdt_service"
SHORT_TERM_WINDOW_SIZE = 50

settings = get_settings(SERVICE_NAME)
log = get_logger(SERVICE_NAME, settings.log_level)
db = Database(settings.postgres_dsn)
bus = EventBus(settings.rabbitmq_url, SERVICE_NAME)
_agent_locks = KeyedLock()


async def _get_or_create(session, agent_id: str) -> Twin:
    twin = await session.get(Twin, agent_id)
    if twin is None:
        # See trust_prediction/main.py::_get_or_create -- mapped_column
        # defaults aren't applied until flush, so pass them explicitly to
        # avoid crashing on this agent's first event.
        twin = Twin(agent_id=agent_id, state_mirror={}, short_term_window=[], long_term_summary={})
        session.add(twin)
        try:
            await session.flush()
        except IntegrityError:
            # See trust_prediction/main.py::_get_or_create -- concurrent
            # handlers (heartbeat/task_outcome/metrics_sample/trust_score
            # each consumed independently) can race to insert the same
            # brand-new agent. Recover instead of dropping this event.
            await session.rollback()
            twin = await session.get(Twin, agent_id)
    return twin


def _push_window(twin: Twin, entry: dict) -> None:
    window = list(twin.short_term_window)
    window.append(entry)
    twin.short_term_window = window[-SHORT_TERM_WINDOW_SIZE:]


def _bump_summary(twin: Twin, key: str) -> None:
    summary = dict(twin.long_term_summary)
    summary[key] = summary.get(key, 0) + 1
    twin.long_term_summary = summary


async def on_heartbeat(event: AgentHeartbeat) -> None:
    async with _agent_locks.get(event.agent_id):
        async with db.session() as session:
            twin = await _get_or_create(session, event.agent_id)
            twin.state_mirror = {
                "role": event.role,
                "current_task": event.current_task,
                "capabilities": event.capabilities,
                "last_heartbeat": event.ts.isoformat(),
            }
            twin.updated_at = datetime.now(timezone.utc)
            await session.commit()


async def on_task_outcome(event: AgentTaskOutcome) -> None:
    async with _agent_locks.get(event.agent_id):
        async with db.session() as session:
            twin = await _get_or_create(session, event.agent_id)
            _push_window(twin, {"type": "task_outcome", "outcome": event.outcome, "ts": event.ts.isoformat()})
            _bump_summary(twin, f"outcome_{event.outcome}")
            twin.updated_at = datetime.now(timezone.utc)
            await session.commit()


async def on_metrics_sample(event: AgentMetricsSample) -> None:
    async with _agent_locks.get(event.agent_id):
        async with db.session() as session:
            twin = await _get_or_create(session, event.agent_id)
            _push_window(
                twin,
                {
                    "type": "metrics_sample",
                    "response_latency_ms": event.response_latency_ms,
                    "resource_utilization_pct": event.resource_utilization_pct,
                    "comms_count": event.comms_count,
                    "ts": event.ts.isoformat(),
                },
            )
            twin.updated_at = datetime.now(timezone.utc)
            await session.commit()


async def on_trust_score(event: TrustScore) -> None:
    async with _agent_locks.get(event.agent_id):
        async with db.session() as session:
            twin = await _get_or_create(session, event.agent_id)
            mirror = dict(twin.state_mirror)
            mirror["trust_score"] = event.trust_score
            mirror["trust_confidence"] = event.confidence
            twin.state_mirror = mirror
            twin.updated_at = datetime.now(timezone.utc)
            await session.commit()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await db.create_all()
    await bus.connect()
    await bus.subscribe(["agent.heartbeat"], AgentHeartbeat, on_heartbeat)
    await bus.subscribe(["agent.task_outcome"], AgentTaskOutcome, on_task_outcome)
    await bus.subscribe(["agent.metrics_sample"], AgentMetricsSample, on_metrics_sample)
    await bus.subscribe(["trust.score"], TrustScore, on_trust_score)
    log.info("cdt_service started")
    yield
    await bus.close()
    await db.dispose()


app = FastAPI(title="Cognitive Digital Twin Service", lifespan=lifespan)


@app.get("/twins/{agent_id}")
async def get_twin(agent_id: str):
    async with db.session() as session:
        twin = await session.get(Twin, agent_id)
        if twin is None:
            raise HTTPException(404, f"no twin for agent {agent_id}")
        return {
            "agent_id": twin.agent_id,
            "state_mirror": twin.state_mirror,
            "short_term_window": twin.short_term_window,
            "long_term_summary": twin.long_term_summary,
            "updated_at": twin.updated_at.isoformat(),
        }


@app.get("/health")
async def health():
    return {"status": "ok"}
