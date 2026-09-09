from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from neurotrust_common.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TrustState(Base):
    """Model A (Bayesian Beta) state per agent -- ML doc §2."""

    __tablename__ = "trust_state"

    agent_id: Mapped[str] = mapped_column(String, primary_key=True)
    role: Mapped[str] = mapped_column(String, default="worker")
    alpha: Mapped[float] = mapped_column(Float, default=1.0)
    beta: Mapped[float] = mapped_column(Float, default=1.0)
    recent_events: Mapped[list] = mapped_column(JSON, default=list)
    watch_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
