import random

from mesa import Agent


class SimAgent(Agent):
    """A simulated multi-agent-system participant. Each step decides a task
    outcome and a metrics sample; the async driver (run.py) does the actual
    HTTP/event-bus I/O since Mesa's scheduler is synchronous.

    `will_be_faulty` marks agents scheduled to go bad -- this is the
    fault-injection hook referenced in the ML doc §9 (labels from synthetic
    fault injection) and §11 (build order step 2). The fault doesn't start
    on tick 1: it activates after `fault_injection_delay_ticks`, so this
    agent contributes genuine normal-behavior data first. Without that
    warm-up, the Autoencoder (Model C, ML doc §4) -- which learns each
    agent's own "normal" baseline -- would be trained on nothing but faulty
    data for a permanently-faulty agent, making it unable to detect
    anything as a deviation.
    """

    def __init__(self, unique_id: str, model, role: str, will_be_faulty: bool = False, fault_injection_delay_ticks: int = 0):
        super().__init__(unique_id, model)
        self.role = role
        self.will_be_faulty = will_be_faulty
        self.fault_injection_delay_ticks = fault_injection_delay_ticks
        self.tick_count = 0
        self.faulty = False  # becomes True once the delay elapses
        self.current_task = f"task-{unique_id}-0"
        self.task_counter = 0
        self.pending_outcome: str | None = None
        self.pending_metrics: dict | None = None
        self.pending_send_heartbeat = True

    def _roll_outcome(self) -> str:
        if not self.faulty:
            return "success" if random.random() < 0.92 else "failed_critical"
        # faulty agents drift toward failure/security-flagged behavior so
        # the Bayesian trust score decays and eventually crosses threshold
        roll = random.random()
        if roll < 0.5:
            return "failed_critical"
        if roll < 0.65:
            return "security_flagged"
        return "success"

    def _roll_metrics(self) -> dict:
        if not self.faulty:
            latency = random.gauss(120, 15)
            resource = random.gauss(40, 8)
            comms_count = random.randint(0, 3)
        else:
            latency = random.gauss(400, 60)
            resource = random.gauss(85, 10)
            # occasional message-flooding burst -- feeds security_monitoring's
            # message_flooding rule (ML doc §6), which otherwise never sees
            # enough comms volume to fire under normal simulated traffic
            comms_count = random.randint(15, 25) if random.random() < 0.10 else random.randint(0, 3)

        # rare, brief resource spike for ANY agent (not just faulty ones) --
        # a genuine point anomaly distinct from a faulty agent's sustained
        # elevated baseline, demonstrating security_monitoring's
        # resource_spike rule independently of Bayesian trust decay
        if random.random() < 0.02:
            resource = min(100.0, resource * 3)

        return {
            "response_latency_ms": max(1.0, latency),
            "resource_utilization_pct": max(0.0, min(100.0, resource)),
            "comms_count": comms_count,
        }

    def step(self) -> None:
        self.tick_count += 1
        if self.will_be_faulty and not self.faulty and self.tick_count >= self.fault_injection_delay_ticks:
            self.faulty = True

        self.pending_outcome = self._roll_outcome()
        self.pending_metrics = self._roll_metrics()
        # faulty agents occasionally go silent entirely -- feeds
        # security_monitoring's silent_agent rule via agent_registry's
        # existing missed-heartbeat monitor
        self.pending_send_heartbeat = not (self.faulty and random.random() < 0.15)
        if self.pending_outcome == "success":
            self.task_counter += 1
            self.current_task = f"task-{self.unique_id}-{self.task_counter}"
