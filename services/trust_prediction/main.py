import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

import bayesian
from models import TrustState
from neurotrust_common import EventBus, get_logger, get_settings
from neurotrust_common.db import Database
from neurotrust_common.locks import KeyedLock
from neurotrust_common.schemas import (
    AgentHeartbeat,
    AgentMissedHeartbeat,
    AgentTaskOutcome,
    SecurityRiskUpdate,
    TrustEvidenceEvent,
    TrustScore,
)

SERVICE_NAME = "trust_prediction"
DECAY_INTERVAL_S = 30

settings = get_settings(SERVICE_NAME)
log = get_logger(SERVICE_NAME, settings.log_level)
db = Database(settings.postgres_dsn)
bus = EventBus(settings.rabbitmq_url, SERVICE_NAME)
_agent_locks = KeyedLock()


async def _get_or_create(session, agent_id: str, role: str = "worker") -> TrustState:
    state = await session.get(TrustState, agent_id)
    if state is None:
        # Pass defaults explicitly -- mapped_column(default=...) is only
        # applied at flush/INSERT time, so a freshly constructed row would
        # otherwise have alpha/beta/recent_events as None until commit,
        # crashing the very first evidence update for a brand-new agent.
        state = TrustState(agent_id=agent_id, role=role, alpha=bayesian.ALPHA0, beta=bayesian.BETA0, recent_events=[])
        session.add(state)
        try:
            await session.flush()
        except IntegrityError:
            # Another handler (e.g. on_heartbeat vs on_task_outcome, fired
            # from separate concurrent queue consumers) inserted this same
            # brand-new agent first. Recover instead of crashing/dropping
            # this event: roll back the failed insert and pick up the row
            # the other transaction just committed.
            await session.rollback()
            state = await session.get(TrustState, agent_id)
    return state


async def _publish_score(state: TrustState) -> None:
    result = bayesian.score(state.alpha, state.beta)
    explanation = bayesian.explain(state.alpha, state.beta, state.recent_events)
    await bus.publish(
        TrustScore(
            agent_id=state.agent_id,
            role=state.role,
            alpha=result.alpha,
            beta=result.beta,
            trust_score=result.trust_score,
            trust_variance=result.trust_variance,
            confidence=result.confidence,
            explanation=explanation,
            recent_events=[TrustEvidenceEvent(**e) for e in state.recent_events],
        )
    )


async def _record_evidence(agent_id: str, outcome: str, role: str | None = None) -> None:
    async with _agent_locks.get(agent_id):
        async with db.session() as session:
            state = await _get_or_create(session, agent_id, role or "worker")
            if role:
                state.role = role
            state.alpha, state.beta = bayesian.apply_evidence(state.alpha, state.beta, outcome)
            events = list(state.recent_events)
            events.append(
                {"kind": outcome, "weight": bayesian.EVIDENCE_WEIGHTS[outcome], "ts": datetime.now(timezone.utc).isoformat()}
            )
            state.recent_events = events[-bayesian.RECENT_EVENTS_KEPT:]
            state.updated_at = datetime.now(timezone.utc)
            await session.commit()
            await _publish_score(state)
    log.info(
        "trust evidence applied",
        extra={"trace": {"agent_id": agent_id, "outcome": outcome, "alpha": state.alpha, "beta": state.beta}},
    )


async def on_heartbeat(event: AgentHeartbeat) -> None:
    async with _agent_locks.get(event.agent_id):
        async with db.session() as session:
            state = await _get_or_create(session, event.agent_id, event.role)
            state.role = event.role
            await session.commit()


async def on_task_outcome(event: AgentTaskOutcome) -> None:
    if event.outcome == "missed_heartbeat":
        return  # handled via AgentMissedHeartbeat instead
    await _record_evidence(event.agent_id, event.outcome)


async def on_missed_heartbeat(event: AgentMissedHeartbeat) -> None:
    await _record_evidence(event.agent_id, "missed_heartbeat")


async def on_security_risk(event: SecurityRiskUpdate) -> None:
    """ML doc §6: combined_security_risk in the watch band tightens this
    agent's trust decay rate instead of isolating outright -- recent
    evidence should age out faster while the agent looks suspicious.
    """
    if event.combined_risk <= bayesian.SECURITY_WATCH_THRESHOLD:
        return
    async with _agent_locks.get(event.agent_id):
        async with db.session() as session:
            state = await _get_or_create(session, event.agent_id, event.role)
            state.watch_until = datetime.now(timezone.utc) + timedelta(seconds=bayesian.WATCH_DURATION_S)
            await session.commit()


async def _decay_tick() -> None:
    """Recency weighting (ML doc §2, closes Gap 1): every window, decay all
    agents' counters before new evidence accumulates, and re-publish scores
    so trend consumers (fusion/decision engine later) see the drift even
    with no new events.
    """
    while True:
        await asyncio.sleep(DECAY_INTERVAL_S)
        async with db.session() as session:
            result = await session.execute(select(TrustState.agent_id))
            agent_ids = result.scalars().all()

        # Decay each agent under its own lock/session rather than one big
        # batch commit -- otherwise this loop can race with a concurrent
        # evidence-update task for the same agent and lose one side's write.
        for agent_id in agent_ids:
            async with _agent_locks.get(agent_id):
                async with db.session() as session:
                    state = await session.get(TrustState, agent_id)
                    if state is None:
                        continue
                    now = datetime.now(timezone.utc)
                    on_watch = state.watch_until is not None and state.watch_until > now
                    lam = bayesian.TIGHTENED_DECAY_LAMBDA if on_watch else bayesian.DECAY_LAMBDA
                    state.alpha, state.beta = bayesian.decay(state.alpha, state.beta, lam)
                    state.updated_at = now
                    await session.commit()
                    await _publish_score(state)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await db.create_all()
    await bus.connect()
    await bus.subscribe(["agent.heartbeat"], AgentHeartbeat, on_heartbeat)
    await bus.subscribe(["agent.task_outcome"], AgentTaskOutcome, on_task_outcome)
    await bus.subscribe(["agent.missed_heartbeat"], AgentMissedHeartbeat, on_missed_heartbeat)
    await bus.subscribe(["security.risk_update"], SecurityRiskUpdate, on_security_risk)
    decay_task = asyncio.create_task(_decay_tick())
    log.info("trust_prediction started")
    yield
    decay_task.cancel()
    await bus.close()
    await db.dispose()


app = FastAPI(title="Trust Prediction Service (Bayesian)", lifespan=lifespan)


@app.get("/trust/{agent_id}")
async def get_trust(agent_id: str):
    async with db.session() as session:
        state = await session.get(TrustState, agent_id)
        if state is None:
            raise HTTPException(404, f"no trust state for agent {agent_id}")
        result = bayesian.score(state.alpha, state.beta)
        return {
            "agent_id": state.agent_id,
            "role": state.role,
            "alpha": state.alpha,
            "beta": state.beta,
            "trust_score": result.trust_score,
            "trust_variance": result.trust_variance,
            "confidence": result.confidence,
            "explanation": bayesian.explain(state.alpha, state.beta, state.recent_events),
            "recent_events": state.recent_events,
            "watch_until": state.watch_until.isoformat() if state.watch_until else None,
        }


@app.get("/health")
async def health():
    return {"status": "ok"}
