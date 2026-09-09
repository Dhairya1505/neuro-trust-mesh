from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from neurotrust_common.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ForecastState(Base):
    """Per-agent regularly-sampled timestep buffer feeding Model B (ML doc
    §3). The torch model itself is kept in-memory (retrained/fine-tuned
    from this buffer) -- not persisted, same tradeoff as anomaly_detection.
    """

    __tablename__ = "forecast_state"

    agent_id: Mapped[str] = mapped_column(String, primary_key=True)
    role: Mapped[str] = mapped_column(String, default="worker")
    timestep_buffer: Mapped[list] = mapped_column(JSON, default=list)  # list[list[float]], one per sample tick
    samples_since_finetune: Mapped[int] = mapped_column(Integer, default=0)
    trained: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
