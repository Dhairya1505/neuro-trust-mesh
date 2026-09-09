from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from neurotrust_common.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AnomalyState(Base):
    """Per-agent training buffer + fitted-profile stats for Model C (ML doc
    §4). The torch model itself is kept in-memory (retrained from this
    buffer on demand/startup) -- not persisted, since re-fitting a tiny
    7-4-7 net from a <=200-row buffer is cheap.
    """

    __tablename__ = "anomaly_state"

    agent_id: Mapped[str] = mapped_column(String, primary_key=True)
    role: Mapped[str] = mapped_column(String, default="worker")
    training_buffer: Mapped[list] = mapped_column(JSON, default=list)  # list[list[float]]
    samples_since_retrain: Mapped[int] = mapped_column(Integer, default=0)
    trained: Mapped[bool] = mapped_column(Boolean, default=False)
    mean_error: Mapped[float] = mapped_column(Float, default=0.0)
    std_error: Mapped[float] = mapped_column(Float, default=0.0)
    threshold: Mapped[float] = mapped_column(Float, default=0.0)
    contaminated_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
