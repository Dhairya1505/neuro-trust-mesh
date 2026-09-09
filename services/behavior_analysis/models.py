from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from neurotrust_common.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BehaviorState(Base):
    """Rolling raw samples per agent, used to build the 7-dim feature
    vector (ML doc §3). Compared against the agent's OWN history, not a
    global average (system doc §4.3, closes Gap 6) -- hence min/max
    normalization is computed from this agent's own capped sample lists,
    never across agents.
    """

    __tablename__ = "behavior_state"

    agent_id: Mapped[str] = mapped_column(String, primary_key=True)
    role: Mapped[str] = mapped_column(String, default="worker")
    task_success_count: Mapped[int] = mapped_column(Integer, default=0)
    task_total_count: Mapped[int] = mapped_column(Integer, default=0)
    latency_samples: Mapped[list] = mapped_column(JSON, default=list)
    resource_samples: Mapped[list] = mapped_column(JSON, default=list)
    comms_timestamps: Mapped[list] = mapped_column(JSON, default=list)
    bayesian_trust_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    bayesian_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
