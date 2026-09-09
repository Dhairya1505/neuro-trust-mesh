import random

from mesa import Model
from mesa.time import RandomActivation

from agents import SimAgent


class NeuroTrustSimModel(Model):
    """N workers + 1 leader. `faulty_agent_ids` lets the driver mark
    specific agents as compromised/unreliable to demonstrate trust decay,
    security flags, and the resulting reassignment/redistribution loop.
    `fault_injection_delay_ticks` gives each faulty agent a normal
    warm-up period before its fault activates (see agents.py::SimAgent).

    `colluding_agent_ids` (exactly 2 agent_ids) makes those two agents'
    outcomes/comms/resource bumps move together after
    `collusion_delay_ticks` -- fault injection for the collusion_detection
    service (ML doc §7), the same warm-up-then-activate pattern as
    single-agent faults.
    """

    def __init__(
        self,
        n_workers: int = 5,
        faulty_agent_ids: set[str] | None = None,
        fault_injection_delay_ticks: int = 0,
        colluding_agent_ids: set[str] | None = None,
        collusion_delay_ticks: int = 0,
    ):
        super().__init__()
        self.schedule = RandomActivation(self)
        faulty_agent_ids = faulty_agent_ids or set()
        self.colluding_agent_ids = colluding_agent_ids or set()
        self.collusion_delay_ticks = collusion_delay_ticks
        self.tick_count = 0

        leader = SimAgent(
            "agent-leader",
            self,
            role="leader",
            will_be_faulty="agent-leader" in faulty_agent_ids,
            fault_injection_delay_ticks=fault_injection_delay_ticks,
        )
        self.schedule.add(leader)

        for i in range(n_workers):
            agent_id = f"agent-worker-{i}"
            worker = SimAgent(
                agent_id,
                self,
                role="worker",
                will_be_faulty=agent_id in faulty_agent_ids,
                fault_injection_delay_ticks=fault_injection_delay_ticks,
            )
            self.schedule.add(worker)

    def step(self) -> None:
        self.tick_count += 1
        self.schedule.step()
        if self.tick_count >= self.collusion_delay_ticks and len(self.colluding_agent_ids) == 2:
            self._apply_collusion()

    def _apply_collusion(self) -> None:
        colluders = [a for a in self.schedule.agents if a.unique_id in self.colluding_agent_ids]
        if len(colluders) != 2:
            return
        a, b = colluders

        # synchronized comms burst -- feeds collusion_detection's
        # comms_anomaly signal
        comms = random.randint(12, 20) if random.random() < 0.4 else random.randint(0, 2)
        a.pending_metrics["comms_count"] = comms
        b.pending_metrics["comms_count"] = comms

        # synchronized outcome -- feeds outcome_corroboration
        shared_outcome = "success" if random.random() < 0.75 else "failed_critical"
        a.pending_outcome = shared_outcome
        b.pending_outcome = shared_outcome

        # shared resource perturbation -- feeds timing/behavior correlation
        shift = random.gauss(0, 12)
        for agent in (a, b):
            agent.pending_metrics["resource_utilization_pct"] = max(
                0.0, min(100.0, agent.pending_metrics["resource_utilization_pct"] + shift)
            )
