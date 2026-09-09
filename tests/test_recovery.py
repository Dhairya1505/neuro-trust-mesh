from datetime import datetime, timedelta, timezone

import recovery


def test_high_risk_never_starts_the_clock():
    state, reinstate = recovery.update(recovery.RecoveryState(), combined_risk=0.9, now=datetime.now(timezone.utc))
    assert state.low_risk_since is None
    assert reinstate is False


def test_low_risk_starts_the_clock_but_does_not_reinstate_immediately():
    now = datetime.now(timezone.utc)
    state, reinstate = recovery.update(recovery.RecoveryState(), combined_risk=0.1, now=now)
    assert state.low_risk_since == now
    assert reinstate is False


def test_sustained_low_risk_triggers_reinstatement():
    start = datetime.now(timezone.utc)
    state = recovery.RecoveryState(low_risk_since=start)
    later = start + timedelta(seconds=recovery.RECOVERY_DURATION_S + 1)
    state, reinstate = recovery.update(state, combined_risk=0.1, now=later)
    assert reinstate is True


def test_risk_spike_mid_recovery_resets_the_clock():
    start = datetime.now(timezone.utc)
    state = recovery.RecoveryState(low_risk_since=start)
    almost_there = start + timedelta(seconds=recovery.RECOVERY_DURATION_S - 1)
    # a spike right before recovery would have completed resets it --
    # reinstatement requires a SUSTAINED low reading, not a lucky average
    state, reinstate = recovery.update(state, combined_risk=0.9, now=almost_there)
    assert reinstate is False
    assert state.low_risk_since is None


def test_not_quite_sustained_long_enough_does_not_reinstate():
    start = datetime.now(timezone.utc)
    state = recovery.RecoveryState(low_risk_since=start)
    too_soon = start + timedelta(seconds=recovery.RECOVERY_DURATION_S - 1)
    state, reinstate = recovery.update(state, combined_risk=0.1, now=too_soon)
    assert reinstate is False
    assert state.low_risk_since == start
