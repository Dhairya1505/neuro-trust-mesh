from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from sqlalchemy.exc import IntegrityError

import features
from models import BehaviorState
from neurotrust_common import EventBus, get_logger, get_settings
from neurotrust_common.db import Database
from neurotrust_common.locks import KeyedLock
from neurotrust_common.schemas import (
    AgentHeartbeat,
    AgentMetricsSample,
    AgentTaskOutcome,
    BehaviorFeatures,
    TrustScore,
)

SERVICE_NAME = "behavior_analysis"

settings = get_settings(SERVICE_NAME)
log = get_logger(SERVICE_NAME, settings.log_level)
db = Database(settings.postgres_dsn)
bus = EventBus(settings.rabbitmq_url, SERVICE_NAME)
_agent_locks = KeyedLock()


async def _get_or_create(session, agent_id: str, role: str = "worker") -> BehaviorState:
    state = await session.get(BehaviorState, agent_id)
    if state is None:
        # See trust_prediction/main.py::_get_or_create -- mapped_column
        # defaults aren't applied until flush, so pass them explicitly to
        # avoid crashing on this agent's first event.
        state = BehaviorState(
            agent_id=agent_id,
            role=role,
            task_success_count=0,
            task_total_count=0,
            latency_samples=[],
            resource_samples=[],
            comms_timestamps=[],
        )
        session.add(state)
        try:
            await session.flush()
        except IntegrityError:
            # See trust_prediction/main.py::_get_or_create -- concurrent
            # handlers (heartbeat/task_outcome/metrics_sample/trust_score
            # each consumed independently) can race to insert the same
            # brand-new agent. Recover instead of dropping this event.
            await session.rollback()
            state = await session.get(BehaviorState, agent_id)
    return state


async def _publish_features(state: BehaviorState) -> None:
    vector = features.build_feature_vector(
        task_success_count=state.task_success_count,
        task_total_count=state.task_total_count,
        latency_samples=state.latency_samples,
        resource_samples=state.resource_samples,
        comms_timestamps=state.comms_timestamps,
        role_flag=1 if state.role == "leader" else 0,
        bayesian_trust_score=state.bayesian_trust_score,
        bayesian_confidence=state.bayesian_confidence,
    )
    await bus.publish(BehaviorFeatures(agent_id=state.agent_id, **vector))


async def on_heartbeat(event: AgentHeartbeat) -> None:
    async with _agent_locks.get(event.agent_id):
        async with db.session() as session:
            state = await _get_or_create(session, event.agent_id, event.role)
            state.role = event.role
            state.comms_timestamps = features.push_capped(
                state.comms_timestamps, event.ts.isoformat(), cap=200
            )
            state.updated_at = datetime.now(timezone.utc)
            await session.commit()
            await _publish_features(state)


async def on_task_outcome(event: AgentTaskOutcome) -> None:
    async with _agent_locks.get(event.agent_id):
        async with db.session() as session:
            state = await _get_or_create(session, event.agent_id)
            state.task_total_count += 1
            if event.outcome == "success":
                state.task_success_count += 1
            state.updated_at = datetime.now(timezone.utc)
            await session.commit()
            await _publish_features(state)


async def on_metrics_sample(event: AgentMetricsSample) -> None:
    async with _agent_locks.get(event.agent_id):
        async with db.session() as session:
            state = await _get_or_create(session, event.agent_id)
            state.latency_samples = features.push_capped(state.latency_samples, event.response_latency_ms)
            state.resource_samples = features.push_capped(state.resource_samples, event.resource_utilization_pct)
            state.updated_at = datetime.now(timezone.utc)
            await session.commit()
            await _publish_features(state)


async def on_trust_score(event: TrustScore) -> None:
    async with _agent_locks.get(event.agent_id):
        async with db.session() as session:
            state = await _get_or_create(session, event.agent_id, event.role)
            state.bayesian_trust_score = event.trust_score
            state.bayesian_confidence = event.confidence
            state.updated_at = datetime.now(timezone.utc)
            await session.commit()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await db.create_all()
    await bus.connect()
    await bus.subscribe(["agent.heartbeat"], AgentHeartbeat, on_heartbeat)
    await bus.subscribe(["agent.task_outcome"], AgentTaskOutcome, on_task_outcome)
    await bus.subscribe(["agent.metrics_sample"], AgentMetricsSample, on_metrics_sample)
    await bus.subscribe(["trust.score"], TrustScore, on_trust_score)
    log.info("behavior_analysis started")
    yield
    await bus.close()
    await db.dispose()


app = FastAPI(title="Behavior Analysis Service", lifespan=lifespan)


@app.get("/features/{agent_id}")
async def get_features(agent_id: str):
    async with db.session() as session:
        state = await session.get(BehaviorState, agent_id)
        if state is None:
            raise HTTPException(404, f"no behavior state for agent {agent_id}")
        vector = features.build_feature_vector(
            task_success_count=state.task_success_count,
            task_total_count=state.task_total_count,
            latency_samples=state.latency_samples,
            resource_samples=state.resource_samples,
            comms_timestamps=state.comms_timestamps,
            role_flag=1 if state.role == "leader" else 0,
            bayesian_trust_score=state.bayesian_trust_score,
            bayesian_confidence=state.bayesian_confidence,
        )
        return {"agent_id": agent_id, **vector}


@app.get("/health")
async def health():
    return {"status": "ok"}
