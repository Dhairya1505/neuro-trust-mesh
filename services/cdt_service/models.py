from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from neurotrust_common.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Twin(Base):
    """One Cognitive Digital Twin per agent (system doc §4.2).

    - state_mirror: latest known snapshot (task, comms, resources)
    - short_term_window: rolling list of recent raw events, capped, used
      for anomaly detection later (M3)
    - long_term_summary: compressed running aggregates, used for trust
      model training later
    """

    __tablename__ = "twins"

    agent_id: Mapped[str] = mapped_column(String, primary_key=True)
    state_mirror: Mapped[dict] = mapped_column(JSON, default=dict)
    short_term_window: Mapped[list] = mapped_column(JSON, default=list)
    long_term_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
