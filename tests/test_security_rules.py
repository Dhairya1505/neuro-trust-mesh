"""Imported by explicit file path (not sys.path) because
services/security_monitoring/rules.py and services/decision_engine/rules.py
share a module name -- adding both service dirs to sys.path would make
`import rules` ambiguous.
"""
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "security_rules", Path(__file__).resolve().parent.parent / "services" / "security_monitoring" / "rules.py"
)
security_rules = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(security_rules)


def test_security_score_empty_events():
    assert security_rules.security_score([], datetime.now(timezone.utc)) == 0.0


def test_security_score_ignores_events_outside_window():
    now = datetime.now(timezone.utc)
    stale = (now - timedelta(seconds=security_rules.RECENCY_WINDOW_S + 5)).isoformat()
    score = security_rules.security_score([{"rule_type": "silent_agent", "ts": stale}], now)
    assert score == 0.0


def test_security_score_caps_at_ten():
    now = datetime.now(timezone.utc)
    events = [{"rule_type": "security_flagged_outcome", "ts": now.isoformat()} for _ in range(10)]
    assert security_rules.security_score(events, now) == security_rules.SECURITY_SCORE_CAP


def test_recency_multiplier_decreases_with_age():
    fresh = security_rules.recency_multiplier(0)
    stale = security_rules.recency_multiplier(50)
    assert fresh > stale


def test_combined_security_risk_takes_max_of_both_signals():
    # security_score contributes 5/10 = 0.5, reconstruction error contributes 0.8
    risk = security_rules.combined_security_risk(5.0, 0.8)
    assert risk == 0.8


def test_check_resource_spike_needs_minimum_samples():
    assert security_rules.check_resource_spike(200.0, [10.0, 10.0]) is False


def test_check_resource_spike_detects_sudden_jump():
    baseline = [40.0, 42.0, 38.0, 41.0, 39.0]
    assert security_rules.check_resource_spike(100.0, baseline) is True


def test_check_resource_spike_no_false_positive_on_sustained_high_baseline():
    # a consistently elevated agent isn't a "spike" -- it's already its own baseline
    baseline = [85.0, 88.0, 82.0, 90.0, 86.0]
    assert security_rules.check_resource_spike(95.0, baseline) is False


def test_check_message_flooding_below_threshold():
    now = datetime.now(timezone.utc)
    samples = [(now, 3), (now, 2)]
    assert security_rules.check_message_flooding(samples, "worker", now) is False


def test_check_message_flooding_above_threshold():
    now = datetime.now(timezone.utc)
    samples = [(now, 25)]
    assert security_rules.check_message_flooding(samples, "worker", now) is True


def test_check_message_flooding_ignores_stale_samples():
    now = datetime.now(timezone.utc)
    stale = now - timedelta(seconds=security_rules.MESSAGE_FLOOD_WINDOW_S + 5)
    samples = [(stale, 25)]
    assert security_rules.check_message_flooding(samples, "worker", now) is False


def test_combined_security_risk_folds_in_anomaly_term():
    # security_score alone gives 0.2; anomaly term pushes it to 0.6
    risk = security_rules.combined_security_risk(2.0, 0.6)
    assert risk == 0.6


def test_classify_risk_bands():
    assert security_rules.classify_risk(0.9) == security_rules.RiskBand(isolate=True, watch=False)
    assert security_rules.classify_risk(0.5) == security_rules.RiskBand(isolate=False, watch=True)
    assert security_rules.classify_risk(0.1) == security_rules.RiskBand(isolate=False, watch=False)


def test_check_replay_attack_no_hash_or_no_prior_sighting():
    now = datetime.now(timezone.utc)
    assert security_rules.check_replay_attack(None, None, now) is False
    assert security_rules.check_replay_attack("abc", None, now) is False


def test_check_replay_attack_reseen_within_window():
    now = datetime.now(timezone.utc)
    last_seen = now - timedelta(seconds=5)
    assert security_rules.check_replay_attack("abc", last_seen, now) is True


def test_check_replay_attack_reseen_outside_window():
    now = datetime.now(timezone.utc)
    last_seen = now - timedelta(seconds=security_rules.REPLAY_WINDOW_S + 5)
    assert security_rules.check_replay_attack("abc", last_seen, now) is False


def test_check_spoofed_identity_matching_sender_is_fine():
    assert security_rules.check_spoofed_identity("agent-1", "agent-1") is False
    assert security_rules.check_spoofed_identity("agent-1", None) is False


def test_check_spoofed_identity_mismatched_sender_is_flagged():
    assert security_rules.check_spoofed_identity("agent-1", "agent-2") is True


def test_check_task_contradiction_success_then_failure():
    assert security_rules.check_task_contradiction("success", "failed_critical") is True
    assert security_rules.check_task_contradiction("success", "security_flagged") is True


def test_check_task_contradiction_no_prior_or_consistent_outcome():
    assert security_rules.check_task_contradiction(None, "failed_critical") is False
    assert security_rules.check_task_contradiction("failed_critical", "failed_critical") is False


def test_check_impossible_transition_valid_sequence():
    assert security_rules.check_impossible_transition(None, "assigned") is False
    assert security_rules.check_impossible_transition("assigned", "in_progress") is False
    assert security_rules.check_impossible_transition("in_progress", "complete") is False


def test_check_impossible_transition_skips_or_reverses_are_flagged():
    assert security_rules.check_impossible_transition("complete", "in_progress") is True
    assert security_rules.check_impossible_transition("failed", "assigned") is True
