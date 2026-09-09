"""Pure functions for building the 7-dim feature vector (ML doc §3).
Kept dependency-free so they're cheap to unit test.
"""
import math
import statistics
from datetime import datetime, timedelta, timezone

SAMPLE_CAP = 60
COMMS_WINDOW_S = 60


def relative_normalize(value: float, samples: list[float]) -> float:
    """Normalize against THIS agent's own observed distribution -- an
    agent's 'normal' latency may be another agent's anomaly (ML doc §3
    training practicals), so normalization must never be cross-agent.

    Uses a z-score run through a logistic squash, not min-max. Latency and
    resource usage are unbounded, roughly-gaussian signals -- min-max
    against a small rolling window is dominated by whichever single sample
    happens to be most extreme, so the "normalized" value swings wildly
    from one sample to the next even for a genuinely steady agent (seen
    live: 0.75 -> 0.47 three seconds apart, no fault involved). A z-score
    changes gradually as new points shift the running mean/std, giving a
    stable notion of "normal" for downstream models (Autoencoder, GRU) to
    actually learn from.
    """
    if len(samples) < 2:
        return 0.5
    mean = statistics.fmean(samples)
    std = statistics.pstdev(samples)
    if std == 0:
        return 0.5
    z = (value - mean) / std
    return 1.0 / (1.0 + math.exp(-z))


def push_capped(items: list, value, cap: int = SAMPLE_CAP) -> list:
    out = list(items)
    out.append(value)
    return out[-cap:]


def comms_frequency(timestamps: list[str], now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=COMMS_WINDOW_S)
    recent = [t for t in timestamps if datetime.fromisoformat(t) > cutoff]
    return len(recent) / (COMMS_WINDOW_S / 60.0)  # messages per minute


def task_completion_rate(success_count: int, total_count: int) -> float:
    if total_count == 0:
        return 0.5
    return success_count / total_count


def build_feature_vector(
    *,
    task_success_count: int,
    task_total_count: int,
    latency_samples: list[float],
    resource_samples: list[float],
    comms_timestamps: list[str],
    role_flag: int,
    bayesian_trust_score: float | None,
    bayesian_confidence: float | None,
) -> dict:
    latest_latency = latency_samples[-1] if latency_samples else 0.0
    latest_resource = resource_samples[-1] if resource_samples else 0.0
    return {
        "task_completion_rate": task_completion_rate(task_success_count, task_total_count),
        "avg_response_latency": relative_normalize(latest_latency, latency_samples),
        "comms_frequency": min(1.0, comms_frequency(comms_timestamps) / 10.0),
        "resource_utilization": relative_normalize(latest_resource, resource_samples),
        "bayesian_trust_score": bayesian_trust_score,
        "bayesian_confidence": bayesian_confidence,
        "role_flag": role_flag,
    }
