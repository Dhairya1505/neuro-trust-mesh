from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from sqlalchemy.exc import IntegrityError

import rules
from models import SecurityState
from neurotrust_common import EventBus, get_logger, get_settings
from neurotrust_common.db import Database
from neurotrust_common.locks import KeyedLock
from neurotrust_common.schemas import (
    AgentHeartbeat,
    AgentMetricsSample,
    AgentMissedHeartbeat,
    AgentTaskOutcome,
    AnomalyScore,
    SecurityRiskUpdate,
    SecurityRuleHit,
)

SERVICE_NAME = "security_monitoring"
EVENTS_CAP = 50
RESOURCE_SAMPLES_CAP = 30
COMMS_SAMPLES_CAP = 60

settings = get_settings(SERVICE_NAME)
log = get_logger(SERVICE_NAME, settings.log_level)
db = Database(settings.postgres_dsn)
bus = EventBus(settings.rabbitmq_url, SERVICE_NAME)
_agent_locks = KeyedLock()


async def _get_or_create(session, agent_id: str, role: str = "worker") -> SecurityState:
    state = await session.get(SecurityState, agent_id)
    if state is None:
        state = SecurityState(
            agent_id=agent_id,
            role=role,
            events=[],
            resource_samples=[],
            comms_samples=[],
            anomaly_risk=0.0,
            message_hashes={},
            last_task_state=None,
            task_outcomes={},
        )
        session.add(state)
        try:
            await session.flush()
        except IntegrityError:
            await session.rollback()
            state = await session.get(SecurityState, agent_id)
    return state


def _record_hit(state: SecurityState, rule_type: str, now: datetime) -> None:
    events = list(state.events)
    events.append({"rule_type": rule_type, "ts": now.isoformat()})
    state.events = events[-EVENTS_CAP:]


async def _publish_risk(state: SecurityState) -> None:
    now = datetime.now(timezone.utc)
    score = rules.security_score(state.events, now)
    combined = rules.combined_security_risk(score, state.anomaly_risk)
    contributing = [
        SecurityRuleHit(rule_type=e["rule_type"], severity=rules.severity(e["rule_type"], (now - datetime.fromisoformat(e["ts"])).total_seconds()), ts=e["ts"])
        for e in state.events
        if (now - datetime.fromisoformat(e["ts"])).total_seconds() <= rules.RECENCY_WINDOW_S
    ]
    await bus.publish(
        SecurityRiskUpdate(
            agent_id=state.agent_id,
            role=state.role,
            security_score=score,
            combined_risk=combined,
            contributing_rules=contributing,
        )
    )


async def on_heartbeat(event: AgentHeartbeat) -> None:
    now = datetime.now(timezone.utc)
    async with _agent_locks.get(event.agent_id):
        async with db.session() as session:
            state = await _get_or_create(session, event.agent_id, event.role)
            state.role = event.role

            hashes = dict(state.message_hashes)
            replay = rules.check_replay_attack(
                event.message_hash, datetime.fromisoformat(hashes[event.message_hash]) if event.message_hash in hashes else None, now
            )
            if event.message_hash is not None:
                hashes[event.message_hash] = now.isoformat()
                state.message_hashes = hashes

            spoofed = rules.check_spoofed_identity(event.agent_id, event.sender_id)

            impossible = rules.check_impossible_transition(state.last_task_state, event.task_state)
            if event.task_state is not None:
                state.last_task_state = event.task_state

            if replay:
                _record_hit(state, "replay_attack", now)
            if spoofed:
                _record_hit(state, "spoofed_identity", now)
            if impossible:
                _record_hit(state, "impossible_transition", now)

            state.updated_at = now
            await session.commit()
            if replay or spoofed or impossible:
                await _publish_risk(state)
    for hit, rule_type in ((replay, "replay_attack"), (spoofed, "spoofed_identity"), (impossible, "impossible_transition")):
        if hit:
            log.info("security rule hit", extra={"trace": {"agent_id": event.agent_id, "rule": rule_type}})


async def on_anomaly_score(event: AnomalyScore) -> None:
    """ML doc §6: combined_security_risk = max(rule-engine score,
    autoencoder reconstruction error) -- fold in the M3 anomaly signal.

    Uses event.combined_anomaly_risk directly -- anomaly_detection already
    folds its own Autoencoder + Isolation Forest ensemble into this single
    normalized value via the same max() fusion operator, so this service
    doesn't need to know that ensemble exists on the other side of the bus.
    """
    if not event.trained:
        return
    async with _agent_locks.get(event.agent_id):
        async with db.session() as session:
            state = await _get_or_create(session, event.agent_id, event.role)
            state.anomaly_risk = event.combined_anomaly_risk
            state.updated_at = datetime.now(timezone.utc)
            await session.commit()
            await _publish_risk(state)


async def on_missed_heartbeat(event: AgentMissedHeartbeat) -> None:
    now = datetime.now(timezone.utc)
    async with _agent_locks.get(event.agent_id):
        async with db.session() as session:
            state = await _get_or_create(session, event.agent_id)
            _record_hit(state, "silent_agent", now)
            state.updated_at = now
            await session.commit()
            await _publish_risk(state)
    log.info("security rule hit", extra={"trace": {"agent_id": event.agent_id, "rule": "silent_agent"}})


async def on_task_outcome(event: AgentTaskOutcome) -> None:
    now = datetime.now(timezone.utc)
    async with _agent_locks.get(event.agent_id):
        async with db.session() as session:
            state = await _get_or_create(session, event.agent_id)

            outcomes = dict(state.task_outcomes)
            contradiction = False
            if event.task_id is not None:
                contradiction = rules.check_task_contradiction(outcomes.get(event.task_id), event.outcome)
                outcomes[event.task_id] = event.outcome
                state.task_outcomes = outcomes

            flagged = event.outcome == "security_flagged"
            if flagged:
                _record_hit(state, "security_flagged_outcome", now)
            if contradiction:
                _record_hit(state, "task_contradiction", now)

            state.updated_at = now
            await session.commit()
            if flagged or contradiction:
                await _publish_risk(state)
    if flagged:
        log.info("security rule hit", extra={"trace": {"agent_id": event.agent_id, "rule": "security_flagged_outcome"}})
    if contradiction:
        log.info("security rule hit", extra={"trace": {"agent_id": event.agent_id, "rule": "task_contradiction"}})


async def on_metrics_sample(event: AgentMetricsSample) -> None:
    now = datetime.now(timezone.utc)
    async with _agent_locks.get(event.agent_id):
        async with db.session() as session:
            state = await _get_or_create(session, event.agent_id)

            prior_resource_samples = list(state.resource_samples)
            spike = rules.check_resource_spike(event.resource_utilization_pct, prior_resource_samples)
            state.resource_samples = (prior_resource_samples + [event.resource_utilization_pct])[-RESOURCE_SAMPLES_CAP:]

            comms_samples = list(state.comms_samples) + [{"ts": event.ts.isoformat(), "count": event.comms_count}]
            state.comms_samples = comms_samples[-COMMS_SAMPLES_CAP:]
            flooding = rules.check_message_flooding(
                [(datetime.fromisoformat(s["ts"]), s["count"]) for s in state.comms_samples], state.role, now
            )

            if spike:
                _record_hit(state, "resource_spike", now)
            if flooding:
                _record_hit(state, "message_flooding", now)

            state.updated_at = now
            await session.commit()
            if spike or flooding:
                await _publish_risk(state)
    if spike:
        log.info("security rule hit", extra={"trace": {"agent_id": event.agent_id, "rule": "resource_spike"}})
    if flooding:
        log.info("security rule hit", extra={"trace": {"agent_id": event.agent_id, "rule": "message_flooding"}})


@asynccontextmanager
async def lifespan(_: FastAPI):
    await db.create_all()
    await bus.connect()
    await bus.subscribe(["agent.heartbeat"], AgentHeartbeat, on_heartbeat)
    await bus.subscribe(["agent.missed_heartbeat"], AgentMissedHeartbeat, on_missed_heartbeat)
    await bus.subscribe(["agent.task_outcome"], AgentTaskOutcome, on_task_outcome)
    await bus.subscribe(["agent.metrics_sample"], AgentMetricsSample, on_metrics_sample)
    await bus.subscribe(["anomaly.score"], AnomalyScore, on_anomaly_score)
    log.info("security_monitoring started")
    yield
    await bus.close()
    await db.dispose()


app = FastAPI(title="Security Monitoring Service", lifespan=lifespan)


@app.get("/security/{agent_id}")
async def get_security_state(agent_id: str):
    async with db.session() as session:
        state = await session.get(SecurityState, agent_id)
        if state is None:
            raise HTTPException(404, f"no security state for agent {agent_id}")
        now = datetime.now(timezone.utc)
        score = rules.security_score(state.events, now)
        return {
            "agent_id": agent_id,
            "role": state.role,
            "security_score": score,
            "anomaly_risk": state.anomaly_risk,
            "combined_risk": rules.combined_security_risk(score, state.anomaly_risk),
            "recent_events": state.events[-10:],
        }


@app.get("/health")
async def health():
    return {"status": "ok"}
