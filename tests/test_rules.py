import rules


def test_leader_below_threshold_triggers_reassignment():
    decision = rules.evaluate("leader", 0.5)
    assert decision.action == "reassign_leader"


def test_leader_above_threshold_no_action():
    decision = rules.evaluate("leader", 0.9)
    assert decision.action == "none"


def test_worker_below_threshold_triggers_redistribution():
    decision = rules.evaluate("worker", 0.3)
    assert decision.action == "redistribute_tasks"


def test_worker_above_threshold_no_action():
    decision = rules.evaluate("worker", 0.9)
    assert decision.action == "none"


def test_worker_between_worker_and_leader_threshold_is_fine():
    # 0.5 would fail a leader but a worker is only held to 0.45
    decision = rules.evaluate("worker", 0.5)
    assert decision.action == "none"


def test_security_risk_above_isolation_threshold_isolates():
    decision = rules.evaluate_security(0.9)
    assert decision.action == "isolate"


def test_security_risk_below_isolation_threshold_no_action():
    decision = rules.evaluate_security(0.3)
    assert decision.action == "none"


def test_forecast_low_confidence_no_action_even_below_threshold():
    decision = rules.evaluate_forecast("worker", 0.1, trend_confidence=1.0)
    assert decision.action == "none"


def test_forecast_confident_declining_leader_triggers_reassignment():
    decision = rules.evaluate_forecast("leader", 0.5, trend_confidence=5.0)
    assert decision.action == "reassign_leader"


def test_forecast_confident_declining_worker_triggers_redistribution():
    decision = rules.evaluate_forecast("worker", 0.3, trend_confidence=5.0)
    assert decision.action == "redistribute_tasks"


def test_forecast_confident_but_above_threshold_no_action():
    decision = rules.evaluate_forecast("worker", 0.9, trend_confidence=5.0)
    assert decision.action == "none"
