"""Imported by explicit file path, same reasoning as test_security_rules.py
(collusion_detection/signals.py could collide with another service's
module name in sys.path if added there)."""
import importlib.util
import random
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "collusion_signals", Path(__file__).resolve().parent.parent / "services" / "collusion_detection" / "signals.py"
)
signals = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(signals)


def _feature_vector(base: float, jitter: float = 0.0) -> dict:
    """comms_frequency and bayesian_confidence deliberately DON'T vary with
    base/jitter here -- confirmed live, both sit at nearly the same value
    for every agent in this system regardless of individual behavior
    (comms_frequency pinned at 1.0 by heartbeat-cadence math,
    bayesian_confidence converges to ~0.99+ for any evidenced agent).
    A test helper that varies them independently per-agent would hide
    exactly the bug this shape of data caused live.
    """
    return {
        "task_completion_rate": base + random.gauss(0, jitter),
        "avg_response_latency": base + random.gauss(0, jitter),
        "comms_frequency": 1.0,
        "resource_utilization": base + random.gauss(0, jitter),
        "bayesian_trust_score": base + random.gauss(0, jitter),
        "bayesian_confidence": 0.99,
        "role_flag": 0,
    }


def test_pearson_correlation_perfect_positive():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ys = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    assert signals.pearson_correlation(xs, ys) > 0.99


def test_pearson_correlation_too_few_samples_is_zero():
    assert signals.pearson_correlation([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_pearson_correlation_zero_variance_is_safe():
    xs = [1.0] * 12
    ys = [random.random() for _ in range(12)]
    assert signals.pearson_correlation(xs, ys) == 0.0


def test_cosine_similarity_identical_vectors():
    assert signals.cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) > 0.99


def test_cosine_similarity_orthogonal_vectors():
    assert abs(signals.cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9


def _independent_history(n: int, seed: int) -> list:
    random.seed(seed)
    return [
        signals.AgentSample(
            ts=float(i),
            anomaly_error=random.gauss(0.1, 0.05),
            feature_vector=_feature_vector(random.gauss(0.5, 0.1)),
            comms_count=random.randint(0, 3),
            outcome_success=1.0 if random.random() < 0.9 else 0.0,
        )
        for i in range(n)
    ]


def _colluding_histories(n: int, seed: int) -> tuple[list, list]:
    """Two agents whose anomaly error, comms bursts, and outcomes move
    together -- a synthetic stand-in for coordinated behavior.
    """
    random.seed(seed)
    hist_a, hist_b = [], []
    for i in range(n):
        shared_anomaly = random.gauss(0.3, 0.1)
        shared_comms = random.randint(10, 20) if random.random() < 0.3 else random.randint(0, 2)
        shared_outcome = 1.0 if random.random() < 0.5 else 0.0
        shared_base = random.gauss(0.5, 0.15)
        hist_a.append(
            signals.AgentSample(
                ts=float(i),
                anomaly_error=shared_anomaly + random.gauss(0, 0.01),
                feature_vector=_feature_vector(shared_base, jitter=0.02),
                comms_count=shared_comms,
                outcome_success=shared_outcome,
            )
        )
        hist_b.append(
            signals.AgentSample(
                ts=float(i),
                anomaly_error=shared_anomaly + random.gauss(0, 0.01),
                feature_vector=_feature_vector(shared_base, jitter=0.02),
                comms_count=shared_comms,
                outcome_success=shared_outcome,
            )
        )
    return hist_a, hist_b


def test_compute_pair_signals_none_below_min_overlap():
    hist = _independent_history(5, seed=0)
    assert signals.compute_pair_signals(hist, hist) is None


def test_independent_agents_are_not_elevated():
    hist_a = _independent_history(40, seed=1)
    hist_b = _independent_history(40, seed=2)
    scores = signals.compute_pair_signals(hist_a, hist_b)
    assert scores is not None
    assert scores.is_elevated() is False


def test_colluding_agents_are_elevated():
    hist_a, hist_b = _colluding_histories(40, seed=3)
    scores = signals.compute_pair_signals(hist_a, hist_b)
    assert scores is not None
    assert scores.elevated_count() >= signals.MIN_SIGNALS_ELEVATED
    assert scores.is_elevated() is True


def test_shared_constant_dimensions_do_not_inflate_behavior_similarity():
    """Regression test for a live incident: comms_frequency and
    bayesian_confidence sit at nearly the same value for every agent in
    this system (unrelated to any real correlation), which inflated
    behavior_similarity into the 0.7-0.9 range for EVERY pair, including
    totally unrelated agents -- confirmed live before FEATURE_ORDER
    dropped those two dimensions.
    """
    random.seed(0)
    hist_a = [
        signals.AgentSample(
            ts=float(i),
            anomaly_error=random.gauss(0.1, 0.05),
            feature_vector=_feature_vector(random.gauss(0.3, 0.1)),
            comms_count=random.randint(0, 3),
            outcome_success=1.0 if random.random() < 0.9 else 0.0,
        )
        for i in range(40)
    ]
    random.seed(1)
    hist_b = [
        signals.AgentSample(
            ts=float(i),
            anomaly_error=random.gauss(0.1, 0.05),
            feature_vector=_feature_vector(random.gauss(0.8, 0.1)),  # very different baseline
            comms_count=random.randint(0, 3),
            outcome_success=1.0 if random.random() < 0.9 else 0.0,
        )
        for i in range(40)
    ]
    scores = signals.compute_pair_signals(hist_a, hist_b)
    assert scores is not None
    assert scores.behavior_similarity < signals.BEHAVIOR_ELEVATED_THRESHOLD


def test_false_positive_rate_across_seeds_for_independent_agents():
    false_positives = 0
    trials = 15
    for seed in range(trials):
        hist_a = _independent_history(40, seed=seed * 2)
        hist_b = _independent_history(40, seed=seed * 2 + 1)
        scores = signals.compute_pair_signals(hist_a, hist_b)
        if scores and scores.is_elevated():
            false_positives += 1
    assert false_positives / trials < 0.2


def test_update_persistence_accumulates_on_consecutive_hits():
    tracker = signals.PairTracker()
    for _ in range(signals.PERSISTENCE_WINDOWS):
        tracker = signals.update_persistence(tracker, elevated_this_window=True)
    assert signals.is_persisted(tracker) is True


def test_update_persistence_resets_on_gap():
    tracker = signals.PairTracker()
    tracker = signals.update_persistence(tracker, elevated_this_window=True)
    tracker = signals.update_persistence(tracker, elevated_this_window=True)
    tracker = signals.update_persistence(tracker, elevated_this_window=False)  # gap
    tracker = signals.update_persistence(tracker, elevated_this_window=True)
    assert signals.is_persisted(tracker) is False  # only 1 consecutive hit after the gap


def test_update_persistence_not_yet_reached():
    tracker = signals.PairTracker()
    tracker = signals.update_persistence(tracker, elevated_this_window=True)
    assert signals.is_persisted(tracker) is False


def test_connected_components_merges_overlapping_pairs():
    pairs = [("a", "b"), ("b", "c"), ("d", "e")]
    groups = signals.connected_components(pairs)
    assert sorted(groups) == [["a", "b", "c"], ["d", "e"]]


def test_connected_components_empty_input():
    assert signals.connected_components([]) == []


def test_connected_components_single_pair():
    assert signals.connected_components([("x", "y")]) == [["x", "y"]]


def test_expected_correlation_same_task_beats_same_role():
    assert signals.expected_correlation("worker", "worker", "task-1", "task-1") == signals.SAME_TASK_BASELINE


def test_expected_correlation_same_role_different_task():
    assert signals.expected_correlation("worker", "worker", "task-1", "task-2") == signals.SAME_ROLE_BASELINE


def test_expected_correlation_unrelated_agents_is_zero():
    assert signals.expected_correlation("worker", "leader", "task-1", "task-2") == 0.0
    assert signals.expected_correlation(None, None, None, None) == 0.0


def test_excess_subtracts_expected_baseline():
    assert abs(signals._excess(0.5, expected=0.2) - 0.3) < 1e-9
    assert signals._excess(0.1, expected=0.2) == 0.0


def test_compute_pair_signals_uses_role_baseline_for_same_role_agents():
    """Two independent agents sharing a role/task get a nonzero expected
    baseline subtracted, so their excess correlation should be <= what it
    would be with no baseline (expected=0) -- confirms the pair's role is
    actually threaded through into the excess calculation.
    """
    hist_a = _independent_history(40, seed=10)
    hist_b = _independent_history(40, seed=11)
    for s in hist_a + hist_b:
        s.role = "worker"
    scores_with_baseline = signals.compute_pair_signals(hist_a, hist_b)

    for s in hist_a + hist_b:
        s.role = None
    scores_without_baseline = signals.compute_pair_signals(hist_a, hist_b)

    assert scores_with_baseline is not None and scores_without_baseline is not None
    assert scores_with_baseline.timing_correlation <= scores_without_baseline.timing_correlation
