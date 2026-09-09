"""Pydantic schemas for every event on the bus.

Routing key convention: "<domain>.<event>", e.g. "agent.heartbeat".
Each schema's `routing_key` classmethod is the canonical key to publish
under -- keep publishers and subscribers importing the same constant
instead of hand-typing strings.
"""
from datetime import datetime, timezone
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

Role = Literal["leader", "worker"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AgentHeartbeat(BaseModel):
    routing_key: ClassVar[str] = "agent.heartbeat"

    agent_id: str
    role: Role
    capabilities: list[str] = Field(default_factory=list)
    current_task: str | None = None
    # Optional -- absent for agents/sims that don't produce them yet. Feed
    # security_monitoring's replay_attack/spoofed_identity/impossible_transition
    # rules (ML doc §6) when present.
    sender_id: str | None = None
    message_hash: str | None = None
    task_state: Literal["assigned", "in_progress", "complete", "failed"] | None = None
    ts: datetime = Field(default_factory=_utcnow)


class AgentMissedHeartbeat(BaseModel):
    routing_key: ClassVar[str] = "agent.missed_heartbeat"

    agent_id: str
    missed_beats: int
    ts: datetime = Field(default_factory=_utcnow)


class AgentTaskOutcome(BaseModel):
    routing_key: ClassVar[str] = "agent.task_outcome"

    agent_id: str
    outcome: Literal["success", "failed_critical", "security_flagged", "missed_heartbeat"]
    # Optional -- lets security_monitoring's task_contradiction rule
    # correlate a later contradicting outcome to an earlier "success" on
    # the same task (ML doc §6).
    task_id: str | None = None
    ts: datetime = Field(default_factory=_utcnow)


class AgentMetricsSample(BaseModel):
    routing_key: ClassVar[str] = "agent.metrics_sample"

    agent_id: str
    response_latency_ms: float
    resource_utilization_pct: float
    comms_count: int
    ts: datetime = Field(default_factory=_utcnow)


class BehaviorFeatures(BaseModel):
    routing_key: ClassVar[str] = "behavior.features"

    agent_id: str
    task_completion_rate: float
    avg_response_latency: float
    comms_frequency: float
    resource_utilization: float
    bayesian_trust_score: float | None = None
    bayesian_confidence: float | None = None
    role_flag: int
    ts: datetime = Field(default_factory=_utcnow)


class TrustEvidenceEvent(BaseModel):
    kind: str
    weight: float
    ts: datetime


class TrustScore(BaseModel):
    routing_key: ClassVar[str] = "trust.score"

    agent_id: str
    role: Role
    alpha: float
    beta: float
    trust_score: float
    trust_variance: float
    confidence: float
    explanation: str
    recent_events: list[TrustEvidenceEvent] = Field(default_factory=list)
    ts: datetime = Field(default_factory=_utcnow)


class AnomalyScore(BaseModel):
    routing_key: ClassVar[str] = "anomaly.score"

    agent_id: str
    role: Role
    reconstruction_error: float  # Autoencoder-specific (Model C), kept for debugging/display
    threshold: float  # Autoencoder-specific
    combined_anomaly_risk: float  # max() of Autoencoder + Isolation Forest normalized scores, 0-1
    is_anomalous: bool  # ensemble decision (either model), persistence-smoothed
    trained: bool
    ts: datetime = Field(default_factory=_utcnow)


class TrustForecast(BaseModel):
    routing_key: ClassVar[str] = "trust.forecast"

    agent_id: str
    role: Role
    predicted_mean: float
    predicted_std: float
    horizon_k: int
    trained: bool
    ts: datetime = Field(default_factory=_utcnow)


class SecurityRuleHit(BaseModel):
    rule_type: str
    severity: float
    ts: datetime


class SecurityRiskUpdate(BaseModel):
    routing_key: ClassVar[str] = "security.risk_update"

    agent_id: str
    role: Role
    security_score: float
    combined_risk: float
    contributing_rules: list[SecurityRuleHit] = Field(default_factory=list)
    ts: datetime = Field(default_factory=_utcnow)


class DecisionAction(BaseModel):
    routing_key: ClassVar[str] = "decision.action"

    agent_id: str
    role: Role
    action: Literal["reassign_leader", "redistribute_tasks", "isolate", "flag_for_review", "none"]
    rule: str
    inputs: dict
    ts: datetime = Field(default_factory=_utcnow)


class CollusionSignalScores(BaseModel):
    timing_correlation: float
    behavior_similarity: float
    comms_anomaly: float
    outcome_corroboration: float


class CollusionFlag(BaseModel):
    routing_key: ClassVar[str] = "collusion.flag"

    group: list[str]  # agent_ids, connected component of >=2 flagged pairs
    roles: dict[str, Role]  # agent_id -> role, for downstream per-agent actions
    group_severity: float
    pair_scores: dict[str, CollusionSignalScores]  # "agent_a|agent_b" -> per-signal scores
    windows_persisted: int
    ts: datetime = Field(default_factory=_utcnow)


class FinalRiskSignal(BaseModel):
    """ML doc §5 fusion layer -- one object combining the outputs of Models
    A/B/C plus the routing decision, for the XAI dashboard and for anything
    that wants a single per-agent risk view instead of subscribing to three
    separate topics.
    """

    routing_key: ClassVar[str] = "risk.final_signal"

    agent_id: str
    role: Role
    trust_now: float | None = None
    trust_confidence: float | None = None
    trust_trend: float | None = None
    trend_confidence: float | None = None
    is_anomalous: bool = False
    top_factors: list[TrustEvidenceEvent] = Field(default_factory=list)
    route: Literal["security", "decision", "log"] = "log"
    ts: datetime = Field(default_factory=_utcnow)


class AgentRoleChangeRequested(BaseModel):
    routing_key: ClassVar[str] = "agent.role_change_requested"

    agent_id: str
    role: Role
    ts: datetime = Field(default_factory=_utcnow)


class AgentTaskChangeRequested(BaseModel):
    routing_key: ClassVar[str] = "agent.task_change_requested"

    agent_id: str
    current_task: str | None
    ts: datetime = Field(default_factory=_utcnow)


class AgentIsolationRequested(BaseModel):
    routing_key: ClassVar[str] = "agent.isolation_requested"

    agent_id: str
    ts: datetime = Field(default_factory=_utcnow)


class AgentReinstateRequested(BaseModel):
    routing_key: ClassVar[str] = "agent.reinstate_requested"

    agent_id: str
    ts: datetime = Field(default_factory=_utcnow)
