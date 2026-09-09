from datetime import datetime, timezone

from neurotrust_common import fusion
from neurotrust_common.schemas import AnomalyScore, TrustForecast, TrustScore


def _trust(role="worker", trust_score=0.8, confidence=0.9):
    return TrustScore(
        agent_id="agent-1",
        role=role,
        alpha=8,
        beta=2,
        trust_score=trust_score,
        trust_variance=0.01,
        confidence=confidence,
        explanation="test",
        recent_events=[],
    )


def _forecast(role="worker", predicted_mean=0.8, predicted_std=0.1, trained=True):
    return TrustForecast(
        agent_id="agent-1", role=role, predicted_mean=predicted_mean, predicted_std=predicted_std, horizon_k=1, trained=trained
    )


def _anomaly(is_anomalous=False, trained=True):
    return AnomalyScore(
        agent_id="agent-1",
        role="worker",
        reconstruction_error=0.1,
        threshold=0.5,
        combined_anomaly_risk=0.1,
        is_anomalous=is_anomalous,
        trained=trained,
    )


def test_compute_final_risk_signal_requires_trust():
    assert fusion.compute_final_risk_signal(None, _forecast(), _anomaly()) is None


def test_compute_final_risk_signal_trust_only():
    signal = fusion.compute_final_risk_signal(_trust(), None, None)
    assert signal is not None
    assert signal.trust_now == 0.8
    assert signal.trust_trend is None
    assert signal.is_anomalous is False
    assert signal.route == "log"


def test_route_signal_anomalous_routes_to_security():
    route = fusion.route_signal_inputs("worker", _forecast(), _anomaly(is_anomalous=True))
    assert route == "security"


def test_route_signal_confident_declining_forecast_routes_to_decision():
    route = fusion.route_signal_inputs("worker", _forecast(predicted_mean=0.2, predicted_std=0.1), None)
    assert route == "decision"


def test_route_signal_low_confidence_forecast_routes_to_log():
    route = fusion.route_signal_inputs("worker", _forecast(predicted_mean=0.2, predicted_std=5.0), None)
    assert route == "log"


def test_route_signal_no_forecast_no_anomaly_routes_to_log():
    assert fusion.route_signal_inputs("worker", None, None) == "log"


def test_route_signal_anomalous_takes_priority_over_decision():
    route = fusion.route_signal_inputs("worker", _forecast(predicted_mean=0.2, predicted_std=0.1), _anomaly(is_anomalous=True))
    assert route == "security"
