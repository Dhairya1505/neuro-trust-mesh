"""Auto-reinstatement for isolated agents (system doc §4.10's 'recovery
tracking'). Pure state-transition logic kept separate from the event bus
wiring so it's cheap to unit test.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta

# reuse the same "agent looks fine" boundary security_monitoring/
# trust_prediction/anomaly_detection already use as their watch threshold
RECOVERY_RISK_THRESHOLD = 0.4
RECOVERY_DURATION_S = 60


@dataclass
class RecoveryState:
    low_risk_since: datetime | None = None


def update(state: RecoveryState, combined_risk: float, now: datetime) -> tuple[RecoveryState, bool]:
    """Returns (new_state, should_reinstate). A risk reading above the
    threshold resets the clock -- reinstatement requires a SUSTAINED low
    reading, not just one good sample in the middle of a noisy stretch.
    """
    if combined_risk > RECOVERY_RISK_THRESHOLD:
        return RecoveryState(low_risk_since=None), False

    since = state.low_risk_since or now
    if now - since >= timedelta(seconds=RECOVERY_DURATION_S):
        return RecoveryState(low_risk_since=None), True
    return RecoveryState(low_risk_since=since), False
