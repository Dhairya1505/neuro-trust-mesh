"""Collusion detection, ML doc §7 -- the hardest rule, highest
false-positive risk in the system per the docs, so kept deliberately
conservative: every signal, the persistence guardrail, and group
formation are pure functions here, unit-tested against both synthetic
collusion and synthetic independent-noise scenarios before ever touching
the event bus.

Core reframe (ML doc §7): collusion_signal = observed_correlation -
expected_correlation, where expected_correlation comes from the task
graph/hierarchy. This system doesn't model an explicit task graph (mesa_sim
runs a flat leader+workers structure with no inter-agent task
dependencies) -- so expected_correlation is a lightweight baseline off the
data agent_registry/heartbeats already carry (same current_task ->
SAME_TASK_BASELINE, same role -> SAME_ROLE_BASELINE, else 0), not raw
correlation. Two agents legitimately collaborating on the same task are
expected to correlate somewhat; only correlation IN EXCESS of that baseline
is suspicious.
"""
import math
from dataclasses import dataclass, field

FEATURE_ORDER = [
    "task_completion_rate",
    "avg_response_latency",
    "resource_utilization",
    "bayesian_trust_score",
]
# role_flag excluded -- shared role is structural, not a collusion signal.
# comms_frequency and bayesian_confidence excluded too -- confirmed live,
# both sit at nearly the SAME value for every agent in this system
# (comms_frequency is pinned at 1.0 by the heartbeat cadence math in
# behavior_analysis/features.py; bayesian_confidence converges to ~0.99+
# for any agent with enough evidence, colluding or not). Two dimensions
# with an identical deviation direction for every agent inflate cosine
# similarity across EVERY pair regardless of genuine correlation --
# confirmed live: every pair, including totally unrelated agents, showed
# behavior_similarity in the 0.7-0.9 range.

MIN_OVERLAP_SAMPLES = 10

# ML doc §7: "weak individually, combined for strength" -- deliberately
# strict per-signal bars, since these are meant to be conservative gates,
# not the whole story (the >=2-of-4 requirement carries most of the weight)
TIMING_ELEVATED_THRESHOLD = 0.6
BEHAVIOR_ELEVATED_THRESHOLD = 0.75
COMMS_ELEVATED_THRESHOLD = 0.6
OUTCOME_ELEVATED_THRESHOLD = 0.6
MIN_SIGNALS_ELEVATED = 2  # ML doc §7: "Require >=2 of 4 signals elevated simultaneously"

SIGNAL_WEIGHTS = {
    "timing_correlation": 0.25,
    "behavior_similarity": 0.25,
    "comms_anomaly": 0.25,
    "outcome_corroboration": 0.25,
}

PERSISTENCE_WINDOWS = 3  # ML doc §7: "Require persistence across >=3 consecutive windows"

# expected_correlation baseline (ML doc §7) -- agents that share structure
# (same task, same role) are expected to correlate somewhat even when
# innocent, so only correlation beyond this baseline counts as excess.
SAME_TASK_BASELINE = 0.2
SAME_ROLE_BASELINE = 0.1


def expected_correlation(role_a: str | None, role_b: str | None, task_a: str | None, task_b: str | None) -> float:
    if task_a is not None and task_a == task_b:
        return SAME_TASK_BASELINE
    if role_a is not None and role_a == role_b:
        return SAME_ROLE_BASELINE
    return 0.0


def _excess(value: float, expected: float = 0.0) -> float:
    """collusion_signal = observed - expected (ML doc §7) -- only positive
    correlation beyond the expected baseline is suspicious; anti-correlated
    or merely-as-expected behavior is not collusion.
    """
    return max(0.0, value - expected)


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < MIN_OVERLAP_SAMPLES or n != len(ys):
        return 0.0
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return 0.0
    return cov / denom


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class AgentSample:
    ts: float  # monotonic seconds, for ordering only
    anomaly_error: float
    feature_vector: dict  # BehaviorFeatures-shaped dict, may have None trust fields
    comms_count: int
    outcome_success: float  # 1.0 success, 0.0 failure, 0.5 unknown/no data this tick
    role: str | None = None  # for expected_correlation's same-role baseline
    current_task: str | None = None  # for expected_correlation's same-task baseline


@dataclass
class CollusionScores:
    timing_correlation: float
    behavior_similarity: float
    comms_anomaly: float
    outcome_corroboration: float

    def elevated_count(self) -> int:
        return sum(
            [
                self.timing_correlation > TIMING_ELEVATED_THRESHOLD,
                self.behavior_similarity > BEHAVIOR_ELEVATED_THRESHOLD,
                self.comms_anomaly > COMMS_ELEVATED_THRESHOLD,
                self.outcome_corroboration > OUTCOME_ELEVATED_THRESHOLD,
            ]
        )

    def is_elevated(self) -> bool:
        return self.elevated_count() >= MIN_SIGNALS_ELEVATED

    def weighted_score(self) -> float:
        return (
            SIGNAL_WEIGHTS["timing_correlation"] * self.timing_correlation
            + SIGNAL_WEIGHTS["behavior_similarity"] * self.behavior_similarity
            + SIGNAL_WEIGHTS["comms_anomaly"] * self.comms_anomaly
            + SIGNAL_WEIGHTS["outcome_corroboration"] * self.outcome_corroboration
        )


def _feature_deviation_vector(feature_vector: dict) -> list[float]:
    """'behavioral-similarity vector' (ML doc §7 signal B) -- how far each
    dimension sits from this system's neutral midpoint (0.5, since every
    feature here is already normalized to roughly that range). Two agents
    deviating from normal in the SAME direction/shape is the suspicious
    pattern, not merely having similar raw values.
    """
    return [(feature_vector.get(key) if feature_vector.get(key) is not None else 0.5) - 0.5 for key in FEATURE_ORDER]


def compute_pair_signals(history_a: list[AgentSample], history_b: list[AgentSample]) -> CollusionScores | None:
    n = min(len(history_a), len(history_b))
    if n < MIN_OVERLAP_SAMPLES:
        return None
    a, b = history_a[-n:], history_b[-n:]

    latest_a, latest_b = a[-1], b[-1]
    expected = expected_correlation(latest_a.role, latest_b.role, latest_a.current_task, latest_b.current_task)

    timing = _excess(pearson_correlation([s.anomaly_error for s in a], [s.anomaly_error for s in b]), expected)
    comms = _excess(pearson_correlation([float(s.comms_count) for s in a], [float(s.comms_count) for s in b]), expected)
    outcome = _excess(pearson_correlation([s.outcome_success for s in a], [s.outcome_success for s in b]), expected)

    # behavior similarity: average cosine similarity of deviation vectors
    # across the overlap window, not just the latest sample -- one
    # coincidental match shouldn't count as "behaviorally similar"
    similarities = [
        cosine_similarity(_feature_deviation_vector(sa.feature_vector), _feature_deviation_vector(sb.feature_vector))
        for sa, sb in zip(a, b)
    ]
    behavior = _excess(sum(similarities) / len(similarities), expected)

    return CollusionScores(
        timing_correlation=timing, behavior_similarity=behavior, comms_anomaly=comms, outcome_corroboration=outcome
    )


@dataclass
class PairTracker:
    consecutive_hits: int = 0
    last_flagged_group: tuple[str, ...] | None = None


def update_persistence(tracker: PairTracker, elevated_this_window: bool) -> PairTracker:
    """ML doc §7: persistence across >=3 CONSECUTIVE windows -- a gap
    resets progress, so a one-off coincidence can't slip through by
    accumulating hits across unrelated, non-consecutive windows.
    """
    if elevated_this_window:
        return PairTracker(consecutive_hits=tracker.consecutive_hits + 1, last_flagged_group=tracker.last_flagged_group)
    return PairTracker(consecutive_hits=0, last_flagged_group=None)


def is_persisted(tracker: PairTracker) -> bool:
    return tracker.consecutive_hits >= PERSISTENCE_WINDOWS


def connected_components(pairs: list[tuple[str, str]]) -> list[list[str]]:
    """ML doc §7: 'From pairs to groups' -- collusion_graph of flagged
    pairs, suspected_groups = connected_components(collusion_graph).
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for a, b in pairs:
        union(a, b)

    groups: dict[str, list[str]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    return [sorted(members) for members in groups.values() if len(members) >= 2]
