import asyncio
import os

import httpx

from model import NeuroTrustSimModel
from neurotrust_common import EventBus, get_logger, get_settings
from neurotrust_common.schemas import AgentHeartbeat, AgentMetricsSample, AgentTaskOutcome

SERVICE_NAME = "mesa_sim"
AGENT_REGISTRY_URL = "http://agent_registry:8000"
TICK_INTERVAL_S = 5  # matches the heartbeat cadence in system doc §4.1
FAULT_INJECTION_DELAY_S = 90  # normal warm-up before a faulty agent's fault activates
COLLUSION_DELAY_S = 90  # normal warm-up before colluding agents start correlating

settings = get_settings(SERVICE_NAME)
log = get_logger(SERVICE_NAME, settings.log_level)
bus = EventBus(settings.rabbitmq_url, SERVICE_NAME)


def _faulty_ids() -> set[str]:
    raw = os.environ.get("FAULTY_AGENT_IDS", "agent-worker-0")
    return {a.strip() for a in raw.split(",") if a.strip()}


def _colluding_ids() -> set[str]:
    raw = os.environ.get("COLLUDING_AGENT_IDS", "")
    return {a.strip() for a in raw.split(",") if a.strip()}


async def _wait_for_registry(client: httpx.AsyncClient, retries: int = 30, delay_s: float = 2.0) -> None:
    for attempt in range(retries):
        try:
            resp = await client.get(f"{AGENT_REGISTRY_URL}/health")
            if resp.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        log.info("waiting for agent_registry", extra={"trace": {"attempt": attempt}})
        await asyncio.sleep(delay_s)
    raise RuntimeError("agent_registry never became healthy")


async def _register_all(client: httpx.AsyncClient, model: NeuroTrustSimModel) -> None:
    await _wait_for_registry(client)
    for agent in model.schedule.agents:
        await client.post(
            f"{AGENT_REGISTRY_URL}/agents/register",
            json={
                "agent_id": agent.unique_id,
                "role": agent.role,
                "capabilities": ["generic"],
                "current_task": agent.current_task,
            },
        )
    log.info("simulation agents registered", extra={"trace": {"count": len(model.schedule.agents)}})


async def _publish_tick(client: httpx.AsyncClient, model: NeuroTrustSimModel) -> None:
    for agent in model.schedule.agents:
        if agent.will_be_faulty and agent.faulty and not getattr(agent, "_logged_fault_injection", False):
            agent._logged_fault_injection = True
            log.warning("fault injected", extra={"trace": {"agent_id": agent.unique_id}})

        # Only report a new current_task when this tick actually completed
        # one (pending_outcome == "success"). Otherwise omit it (defaults
        # to null, which agent_registry's heartbeat handler treats as
        # "leave current_task alone") -- sending the agent's own stale
        # local task on every tick would silently overwrite whatever
        # task_redistribution just assigned it, undoing that action a few
        # seconds later.
        heartbeat_body = {}
        if not agent.pending_send_heartbeat:
            # a genuinely silent agent doesn't communicate at all this tick
            # (feeds security_monitoring's silent_agent rule) -- skip
            # heartbeat, outcome, and metrics together
            continue

        if agent.pending_outcome == "success":
            heartbeat_body["current_task"] = agent.current_task
        await client.post(
            f"{AGENT_REGISTRY_URL}/agents/{agent.unique_id}/heartbeat",
            json=heartbeat_body,
        )
        # agent_registry re-publishes AgentHeartbeat itself on that call,
        # so we only need to publish the outcome/metrics events here.
        await bus.publish(AgentTaskOutcome(agent_id=agent.unique_id, outcome=agent.pending_outcome))
        m = agent.pending_metrics
        await bus.publish(
            AgentMetricsSample(
                agent_id=agent.unique_id,
                response_latency_ms=m["response_latency_ms"],
                resource_utilization_pct=m["resource_utilization_pct"],
                comms_count=m["comms_count"],
            )
        )


async def main() -> None:
    n_workers = int(os.environ.get("N_WORKERS", "5"))
    delay_ticks = int(os.environ.get("FAULT_INJECTION_DELAY_S", str(FAULT_INJECTION_DELAY_S))) // TICK_INTERVAL_S
    collusion_delay_ticks = int(os.environ.get("COLLUSION_DELAY_S", str(COLLUSION_DELAY_S))) // TICK_INTERVAL_S
    model = NeuroTrustSimModel(
        n_workers=n_workers,
        faulty_agent_ids=_faulty_ids(),
        fault_injection_delay_ticks=delay_ticks,
        colluding_agent_ids=_colluding_ids(),
        collusion_delay_ticks=collusion_delay_ticks,
    )

    await bus.connect()
    async with httpx.AsyncClient(timeout=10) as client:
        await _register_all(client, model)
        log.info(
            "simulation loop starting",
            extra={
                "trace": {
                    "faulty_agents": sorted(_faulty_ids()),
                    "fault_injection_delay_ticks": delay_ticks,
                    "colluding_agents": sorted(_colluding_ids()),
                    "collusion_delay_ticks": collusion_delay_ticks,
                }
            },
        )
        while True:
            model.step()
            await _publish_tick(client, model)
            await asyncio.sleep(TICK_INTERVAL_S)


if __name__ == "__main__":
    asyncio.run(main())
