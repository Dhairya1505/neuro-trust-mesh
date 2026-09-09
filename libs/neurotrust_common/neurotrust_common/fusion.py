"""ML doc §5 fusion layer -- combines Model A/B/C outputs (trust score,
trust forecast, anomaly score) into one FinalRiskSignal plus a routing
decision. Kept as pure functions in the shared lib so any service can
compute one without owning the "cache latest event per agent" plumbing;
decision_engine is the one that actually calls it, since it already
subscribes to all three inputs.

This is additive: the existing per-event decision rules in
decision_engine/rules.py (evaluate/evaluate_security/evaluate_forecast)
keep making the actual isolate/reassign/redistribute calls -- this module
gives the XAI dashboard and anything else that wants it a single combined
view of the same inputs, per ML doc §5's final_risk_signal shape.
"""
from neurotrust_common.schemas import AnomalyScore, FinalRiskSignal, TrustForecast, TrustScore

# Must match decision_engine/rules.py -- same role thresholds and forecast
# confidence gate the routing logic reacts to.
LEADER_THRESHOLD = 0.65
WORKER_THRESHOLD = 0.45
MIN_TREND_CONFIDENCE = 2.0


def compute_final_risk_signal(
    trust: TrustScore | None, forecast: TrustForecast | None, anomaly: AnomalyScore | None
) -> FinalRiskSignal | None:
    if trust is None:
        return None  # need at least a role + current trust_now to build a meaningful signal
    return FinalRiskSignal(
        agent_id=trust.agent_id,
        role=trust.role,
        trust_now=trust.trust_score,
        trust_confidence=trust.confidence,
        trust_trend=forecast.predicted_mean if forecast and forecast.trained else None,
        trend_confidence=(1.0 / max(forecast.predicted_std, 1e-3)) if forecast and forecast.trained else None,
        is_anomalous=anomaly.is_anomalous if anomaly else False,
        top_factors=trust.recent_events[:3],
        route=route_signal_inputs(trust.role, forecast, anomaly),
    )


def route_signal_inputs(role: str, forecast: TrustForecast | None, anomaly: AnomalyScore | None) -> str:
    """ML doc §5:
    is_anomalous -> security; trust_trend below role threshold AND
    confident -> decision; else -> log.
    """
    if anomaly is not None and anomaly.is_anomalous:
        return "security"
    if forecast is not None and forecast.trained:
        threshold = LEADER_THRESHOLD if role == "leader" else WORKER_THRESHOLD
        trend_confidence = 1.0 / max(forecast.predicted_std, 1e-3)
        if forecast.predicted_mean < threshold and trend_confidence > MIN_TREND_CONFIDENCE:
            return "decision"
    return "log"
