from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from neurotrust_common.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Agent(Base):
    __tablename__ = "agents"

    agent_id: Mapped[str] = mapped_column(String, primary_key=True)
    role: Mapped[str] = mapped_column(String, nullable=False)
    capabilities: Mapped[list] = mapped_column(JSON, default=list)
    current_task: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="active")
    missed_beats: Mapped[int] = mapped_column(Integer, default=0)
    last_heartbeat: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
