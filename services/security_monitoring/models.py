from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from neurotrust_common.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SecurityState(Base):
    """Rolling rule-hit history + raw samples per agent, feeding the
    security_score / combined_security_risk calculation (ML doc §6).
    """

    __tablename__ = "security_state"

    agent_id: Mapped[str] = mapped_column(String, primary_key=True)
    role: Mapped[str] = mapped_column(String, default="worker")
    events: Mapped[list] = mapped_column(JSON, default=list)  # [{rule_type, ts}]
    resource_samples: Mapped[list] = mapped_column(JSON, default=list)
    comms_samples: Mapped[list] = mapped_column(JSON, default=list)  # [{ts, count}]
    anomaly_risk: Mapped[float] = mapped_column(Float, default=0.0)  # normalized reconstruction error, from M3
    message_hashes: Mapped[dict] = mapped_column(JSON, default=dict)  # message_hash -> last-seen ISO ts (replay_attack)
    last_task_state: Mapped[str | None] = mapped_column(String, nullable=True)  # impossible_transition
    task_outcomes: Mapped[dict] = mapped_column(JSON, default=dict)  # task_id -> last outcome (task_contradiction)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
