"""Model A -- Bayesian (Beta) trust scoring, ML doc §2. Pure functions so
they're cheap to unit test independently of the event bus / DB.
"""
from dataclasses import dataclass

ALPHA0 = 1.0
BETA0 = 1.0
DECAY_LAMBDA = 0.97  # recency weighting, closes Gap 1
TIGHTENED_DECAY_LAMBDA = 0.90  # applied while an agent is under security watch (ML doc §6)
WATCH_DURATION_S = 120

# must match security_monitoring/rules.py::WATCH_THRESHOLD -- that service
# is the source of truth for the security scoring
SECURITY_WATCH_THRESHOLD = 0.4

# severity weights (ML doc §2)
EVIDENCE_WEIGHTS = {
    "success": 1.0,
    "missed_heartbeat": 1.0,
    "failed_critical": 3.0,
    "security_flagged": 5.0,
}

# closes Gap 5 -- leaders held to a stricter threshold than workers
ROLE_THRESHOLDS = {"leader": 0.65, "worker": 0.45}

RECENT_EVENTS_KEPT = 5


@dataclass
class TrustResult:
    alpha: float
    beta: float
    trust_score: float
    trust_variance: float
    confidence: float


def decay(alpha: float, beta: float, lam: float = DECAY_LAMBDA) -> tuple[float, float]:
    return alpha * lam, beta * lam


def apply_evidence(alpha: float, beta: float, outcome: str) -> tuple[float, float]:
    weight = EVIDENCE_WEIGHTS[outcome]
    if outcome == "success":
        return alpha + weight, beta
    return alpha, beta + weight


def score(alpha: float, beta: float) -> TrustResult:
    total = alpha + beta
    trust_score = alpha / total
    trust_variance = (alpha * beta) / (total**2 * (total + 1))
    confidence = 1 - trust_variance
    return TrustResult(alpha, beta, trust_score, trust_variance, confidence)


def explain(alpha: float, beta: float, recent_events: list[dict]) -> str:
    successes = round(alpha - ALPHA0, 1)
    failures = round(beta - BETA0, 1)
    trust = round(score(alpha, beta).trust_score, 2)
    tail = ""
    if recent_events:
        last_kinds = [e["kind"] for e in recent_events[-3:]]
        tail = f", most recent events: {', '.join(last_kinds)}"
    return (
        f"trust = {trust} — {successes} weighted successes, {failures} weighted "
        f"failures over the decayed window{tail}."
    )
