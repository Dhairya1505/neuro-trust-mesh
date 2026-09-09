"""Rule engine, ML doc §6. Pure functions so they're cheap to unit test.

Rules backed by data the event schema can produce:
  - silent_agent       <- AgentMissedHeartbeat (agent_registry's own monitor)
  - message_flooding   <- AgentMetricsSample.comms_count, summed over a
                           short window
  - resource_spike     <- AgentMetricsSample.resource_utilization_pct vs.
                           this agent's own recent baseline
  - security_flagged_outcome <- AgentTaskOutcome(outcome="security_flagged"),
                           standing in for a generic high-confidence flag
  - replay_attack      <- AgentHeartbeat.message_hash reseen within
                           REPLAY_WINDOW_S
  - spoofed_identity    <- AgentHeartbeat.sender_id != agent_id
  - task_contradiction  <- AgentTaskOutcome.task_id previously "success",
                           later reported failed_critical/security_flagged
  - impossible_transition <- AgentHeartbeat.task_state jump not in
                           ALLOWED_TRANSITIONS

sender_id/message_hash/task_state/task_id are all optional on the schema --
agents/sims that don't produce them simply never trigger these rules rather
than faking data. leader_override_abuse needs command-authorization data
this system doesn't produce yet -- deliberately not implemented.
collusion_signal is handled by the separate collusion_detection service.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta

# base_weight table, ML doc §6
BASE_WEIGHTS = {
    "message_flooding": 2.0,
    "silent_agent": 1.0,
    "resource_spike": 2.0,
    "security_flagged_outcome": 5.0,
    "replay_attack": 4.0,
    "spoofed_identity": 5.0,
    "task_contradiction": 3.0,
    "impossible_transition": 3.0,
}

REPLAY_WINDOW_S = 120.0

# Task state machine an AgentHeartbeat.task_state is allowed to move
# through -- None (never reported) is the implicit starting state.
ALLOWED_TRANSITIONS: dict[str | None, set[str]] = {
    None: {"assigned", "in_progress", "complete", "failed"},
    "assigned": {"in_progress", "complete", "failed"},
    "in_progress": {"complete", "failed"},
    "complete": set(),
    "failed": set(),
}

RECENCY_WINDOW_S = 60.0
SECURITY_SCORE_CAP = 10.0

MESSAGE_FLOOD_WINDOW_S = 10.0
ROLE_MSG_THRESHOLD = {"leader": 30, "worker": 20}

RESOURCE_SPIKE_MULTIPLIER = 2.0
RESOURCE_SPIKE_MIN_SAMPLES = 5

ISOLATION_THRESHOLD = 0.7
WATCH_THRESHOLD = 0.4


def recency_multiplier(age_s: float, window_s: float = RECENCY_WINDOW_S) -> float:
    """Older events within the window count for less -- a rule hit from 55
    seconds ago shouldn't weigh the same as one from this instant.
    """
    return max(0.3, 1.0 - age_s / window_s)


def severity(rule_type: str, age_s: float) -> float:
    return BASE_WEIGHTS[rule_type] * recency_multiplier(age_s)


def security_score(events: list[dict], now: datetime, window_s: float = RECENCY_WINDOW_S) -> float:
    total = 0.0
    for e in events:
        ts = e["ts"] if isinstance(e["ts"], datetime) else datetime.fromisoformat(e["ts"])
        age = (now - ts).total_seconds()
        if age > window_s or age < 0:
            continue
        total += severity(e["rule_type"], age)
    return min(total, SECURITY_SCORE_CAP)


def combined_security_risk(security_score_value: float, reconstruction_error_normalized: float = 0.0) -> float:
    """ML doc §6: max(normalize(security_score), normalize(reconstruction_error)).
    The autoencoder term (from anomaly_detection, M3) is 0.0 for an agent
    whose autoencoder hasn't trained yet.
    """
    return max(security_score_value / SECURITY_SCORE_CAP, reconstruction_error_normalized)


def check_resource_spike(latest: float, prior_samples: list[float]) -> bool:
    """Sudden deviation from THIS agent's own recent baseline, not a global
    threshold (system doc §4.3) -- and 'sudden', not a sustained shift, so
    we compare against the max of samples BEFORE this one.
    """
    if len(prior_samples) < RESOURCE_SPIKE_MIN_SAMPLES:
        return False
    baseline_max = max(prior_samples)
    return baseline_max > 0 and latest > baseline_max * RESOURCE_SPIKE_MULTIPLIER


def check_message_flooding(comms_samples: list[tuple[datetime, int]], role: str, now: datetime) -> bool:
    cutoff = now - timedelta(seconds=MESSAGE_FLOOD_WINDOW_S)
    total = sum(count for ts, count in comms_samples if ts > cutoff)
    threshold = ROLE_MSG_THRESHOLD.get(role, ROLE_MSG_THRESHOLD["worker"])
    return total > threshold


def check_replay_attack(message_hash: str | None, last_seen: datetime | None, now: datetime) -> bool:
    """A message_hash reseen within REPLAY_WINDOW_S of its last sighting is
    a replay (ML doc §6). `last_seen` is this agent's prior timestamp for
    the SAME hash, looked up by the caller before recording the new one.
    """
    if message_hash is None or last_seen is None:
        return False
    return (now - last_seen).total_seconds() <= REPLAY_WINDOW_S


def check_spoofed_identity(agent_id: str, sender_id: str | None) -> bool:
    """Lightweight stand-in for full signature verification (ML doc §6):
    a heartbeat whose sender_id doesn't match the agent_id it claims to be
    is spoofed.
    """
    return sender_id is not None and sender_id != agent_id


def check_task_contradiction(prior_outcome: str | None, new_outcome: str) -> bool:
    """"complete" reported but downstream verification fails (ML doc §6):
    a task previously reported successful that's now reported failed or
    security-flagged contradicts itself.
    """
    return prior_outcome == "success" and new_outcome in ("failed_critical", "security_flagged")


def check_impossible_transition(prev_state: str | None, new_state: str | None) -> bool:
    """A task_state jump that violates ALLOWED_TRANSITIONS (ML doc §6)."""
    if new_state is None:
        return False
    return new_state not in ALLOWED_TRANSITIONS.get(prev_state, set())


@dataclass
class RiskBand:
    isolate: bool
    watch: bool


def classify_risk(combined_risk: float) -> RiskBand:
    return RiskBand(
        isolate=combined_risk > ISOLATION_THRESHOLD,
        watch=not (combined_risk > ISOLATION_THRESHOLD) and combined_risk > WATCH_THRESHOLD,
    )
