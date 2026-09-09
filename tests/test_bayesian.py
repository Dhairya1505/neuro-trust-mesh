import bayesian


def test_initial_prior_is_uniform():
    result = bayesian.score(bayesian.ALPHA0, bayesian.BETA0)
    assert result.trust_score == 0.5


def test_success_increases_trust():
    a, b = bayesian.apply_evidence(1.0, 1.0, "success")
    assert bayesian.score(a, b).trust_score > 0.5


def test_failed_critical_outweighs_missed_heartbeat():
    a1, b1 = bayesian.apply_evidence(1.0, 1.0, "missed_heartbeat")
    a2, b2 = bayesian.apply_evidence(1.0, 1.0, "failed_critical")
    assert bayesian.score(a2, b2).trust_score < bayesian.score(a1, b1).trust_score


def test_decay_pulls_counters_toward_prior_ratio_unchanged_but_shrinks_confidence():
    a, b = 10.0, 2.0
    da, db = bayesian.decay(a, b, lam=0.9)
    assert da < a and db < b
    # decay shrinks both proportionally, so the score itself is unchanged...
    assert abs(bayesian.score(a, b).trust_score - bayesian.score(da, db).trust_score) < 1e-9
    # ...but variance (uncertainty) increases as evidence "ages out"
    assert bayesian.score(da, db).trust_variance > bayesian.score(a, b).trust_variance


def test_role_thresholds_leader_stricter_than_worker():
    assert bayesian.ROLE_THRESHOLDS["leader"] > bayesian.ROLE_THRESHOLDS["worker"]


def test_repeated_failures_eventually_cross_worker_threshold():
    a, b = bayesian.ALPHA0, bayesian.BETA0
    for _ in range(10):
        a, b = bayesian.apply_evidence(a, b, "failed_critical")
    assert bayesian.score(a, b).trust_score < bayesian.ROLE_THRESHOLDS["worker"]
