"""Explicit, inspectable rule table (system doc §4.7) -- deliberately not a
black box. M1 only implements the two trust-threshold rules; security_flag
and collusion_flag branches are added in M2/M5 per the build plan.
"""
from dataclasses import dataclass

LEADER_THRESHOLD = 0.65
WORKER_THRESHOLD = 0.45

# must match security_monitoring/rules.py::ISOLATION_THRESHOLD -- that
# service is the source of truth for the security scoring, this is just
# the decision boundary the engine reacts to
SECURITY_ISOLATION_THRESHOLD = 0.7

# ML doc §5 fusion layer: trend_confidence = 1/predicted_std, gated so a
# wide/uncertain forecast never fires the predictive path on its own
MIN_TREND_CONFIDENCE = 2.0


@dataclass
class Decision:
    action: str
    rule: str


def evaluate(role: str, trust_score: float) -> Decision:
    if role == "leader" and trust_score < LEADER_THRESHOLD:
        return Decision(
            action="reassign_leader",
            rule=f"trust < leader_threshold({LEADER_THRESHOLD}) AND role == leader",
        )
    if role == "worker" and trust_score < WORKER_THRESHOLD:
        return Decision(
            action="redistribute_tasks",
            rule=f"trust < worker_threshold({WORKER_THRESHOLD}) AND role == worker",
        )
    return Decision(action="none", rule="no threshold breached")


def evaluate_security(combined_risk: float) -> Decision:
    """security_flag == True -> isolate immediately (system doc §4.7).
    Security flags bypass the slower trust-decay path entirely -- this is
    evaluated independently of trust score, not merged into evaluate().
    """
    if combined_risk > SECURITY_ISOLATION_THRESHOLD:
        return Decision(
            action="isolate",
            rule=f"combined_security_risk > isolation_threshold({SECURITY_ISOLATION_THRESHOLD})",
        )
    return Decision(action="none", rule="security risk below isolation threshold")


def evaluate_forecast(role: str, predicted_mean: float, trend_confidence: float) -> Decision:
    """ML doc §5 fusion layer: predicted trust trend crossing the same
    role threshold as evaluate() -- but only acted on when the forecast is
    confident enough (trend_confidence > MIN_TREND_CONFIDENCE), so the
    Decision Engine doesn't act on a low-confidence forecast (ML doc §3).
    This is a predictive, EARLIER-warning path alongside evaluate()'s
    reactive one -- both can fire the same actions.
    """
    if trend_confidence <= MIN_TREND_CONFIDENCE:
        return Decision(action="none", rule="forecast confidence below min_confidence")
    if role == "leader" and predicted_mean < LEADER_THRESHOLD:
        return Decision(
            action="reassign_leader",
            rule=f"predicted trust trend < leader_threshold({LEADER_THRESHOLD}) AND role == leader AND trend_confidence > {MIN_TREND_CONFIDENCE}",
        )
    if role == "worker" and predicted_mean < WORKER_THRESHOLD:
        return Decision(
            action="redistribute_tasks",
            rule=f"predicted trust trend < worker_threshold({WORKER_THRESHOLD}) AND role == worker AND trend_confidence > {MIN_TREND_CONFIDENCE}",
        )
    return Decision(action="none", rule="predicted trust trend above threshold")
