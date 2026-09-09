import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI

import recovery
from neurotrust_common import EventBus, get_logger, get_settings
from neurotrust_common.locks import KeyedLock
from neurotrust_common.schemas import AgentIsolationRequested, AgentReinstateRequested, DecisionAction, SecurityRiskUpdate

SERVICE_NAME = "agent_isolation"
AGENT_REGISTRY_URL = "http://agent_registry:8000"

settings = get_settings(SERVICE_NAME)
log = get_logger(SERVICE_NAME, settings.log_level)
bus = EventBus(settings.rabbitmq_url, SERVICE_NAME)

# Tracks only CURRENTLY isolated agents -- in-memory, same tradeoff as
# decision_engine's cooldown dict. Seeded from the registry on startup so
# a service restart doesn't lose track of who's isolated.
_isolated: dict[str, recovery.RecoveryState] = {}
# When each agent's isolation started -- separate from RecoveryState (which
# only tracks the low-risk clock) so /isolated can show "isolated since" and
# an ETA together. On a restart-seeded agent the exact original isolation
# time is unknown, so it's set to "now" (best-effort, matches the same
# tradeoff _seed_isolated_agents already accepts for RecoveryState).
_isolated_since: dict[str, datetime] = {}
_agent_locks = KeyedLock()


async def on_decision(event: DecisionAction) -> None:
    if event.action != "isolate":
        return

    # Isolation is a write/trigger, not a query -- flows through the event
    # bus (system doc §2/§12) instead of a synchronous PATCH. This trades
    # the old immediate failure check for eventual consistency: if
    # agent_registry is briefly unreachable the event still lands once it
    # recovers (durable queue), whereas the old PATCH would just drop the
    # isolation on a transient failure.
    await bus.publish(AgentIsolationRequested(agent_id=event.agent_id))

    async with _agent_locks.get(event.agent_id):
        # decision_engine re-fires "isolate" every ISOLATE_COOLDOWN_S while
        # risk stays high (not just once) -- resetting the recovery clock
        # each time is correct (risk is still bad), but isolated_since must
        # NOT reset on a repeat, or the dashboard's "isolated since" would
        # keep jumping to "just now" for an agent that's been isolated for
        # a while.
        _isolated[event.agent_id] = recovery.RecoveryState()
        _isolated_since.setdefault(event.agent_id, datetime.now(timezone.utc))
    log.info(
        "agent isolated",
        extra={"trace": {"agent_id": event.agent_id, "rule": event.rule, "inputs": event.inputs}},
    )

    # Removes agent from active task pool (system doc §4.10) -- reuse
    # task_redistribution's existing logic rather than duplicating it by
    # republishing the same decision it already knows how to act on. The
    # CDT keeps observing the isolated agent (no change needed there);
    # only its eligibility as a task/leadership candidate is affected,
    # which agent_registry's status="isolated" already excludes it from
    # everywhere candidates are queried (role=worker, status=active).
    await bus.publish(
        DecisionAction(
            agent_id=event.agent_id,
            role=event.role,
            action="redistribute_tasks",
            rule="agent isolated -- pulling its active task",
            inputs=event.inputs,
        )
    )

    if event.role == "leader":
        # An isolated leader also leaves the leader role vacant -- reuse
        # leader_reassignment the same way, for the same reason.
        await bus.publish(
            DecisionAction(
                agent_id=event.agent_id,
                role=event.role,
                action="reassign_leader",
                rule="agent isolated -- leader role vacated",
                inputs=event.inputs,
            )
        )


async def on_security_risk(event: SecurityRiskUpdate) -> None:
    async with _agent_locks.get(event.agent_id):
        if event.agent_id not in _isolated:
            return  # only track agents we know are currently isolated

        state, should_reinstate = recovery.update(
            _isolated[event.agent_id], event.combined_risk, datetime.now(timezone.utc)
        )
        if not should_reinstate:
            _isolated[event.agent_id] = state
            return

        # Same write-via-event shift as on_decision above.
        await bus.publish(AgentReinstateRequested(agent_id=event.agent_id))
        del _isolated[event.agent_id]
        _isolated_since.pop(event.agent_id, None)

    log.info(
        "agent reinstated",
        extra={"trace": {"agent_id": event.agent_id, "combined_risk": event.combined_risk}},
    )


async def _seed_isolated_agents() -> None:
    """A restart loses the in-memory tracking dict -- reload it from the
    registry so currently-isolated agents stay tracked for recovery
    instead of getting stuck isolated forever until the next fresh flag.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        for attempt in range(30):
            try:
                resp = await client.get(f"{AGENT_REGISTRY_URL}/agents", params={"status": "isolated"})
                if resp.status_code == 200:
                    now = datetime.now(timezone.utc)
                    for agent in resp.json():
                        _isolated[agent["agent_id"]] = recovery.RecoveryState()
                        _isolated_since[agent["agent_id"]] = now
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2.0)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await _seed_isolated_agents()
    await bus.connect()
    await bus.subscribe(["decision.action"], DecisionAction, on_decision)
    await bus.subscribe(["security.risk_update"], SecurityRiskUpdate, on_security_risk)
    log.info("agent_isolation started", extra={"trace": {"seeded_isolated": list(_isolated)}})
    yield
    await bus.close()


app = FastAPI(title="Agent Isolation Service", lifespan=lifespan)


@app.get("/isolated")
async def list_isolated():
    """Per-agent recovery status for the dashboard (system doc §4.10's
    'recovery tracking') -- without this there was no way to see WHEN an
    isolated agent might come back, only that it's isolated. `eta_seconds`
    is null until the agent's risk has dropped below
    recovery.RECOVERY_RISK_THRESHOLD at least once (the recovery clock
    hasn't started yet, so there's nothing to count down from).
    """
    now = datetime.now(timezone.utc)
    result = []
    for agent_id, state in _isolated.items():
        eta_seconds = None
        if state.low_risk_since is not None:
            elapsed = (now - state.low_risk_since).total_seconds()
            eta_seconds = max(0.0, recovery.RECOVERY_DURATION_S - elapsed)
        isolated_since = _isolated_since.get(agent_id)
        result.append(
            {
                "agent_id": agent_id,
                "isolated_since": isolated_since.isoformat() if isolated_since else None,
                "risk_improving": state.low_risk_since is not None,
                "eta_seconds": eta_seconds,
            }
        )
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}
