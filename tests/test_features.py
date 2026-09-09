from datetime import datetime, timedelta, timezone

import features


def test_relative_normalize_too_few_samples_defaults_to_midpoint():
    assert features.relative_normalize(50.0, []) == 0.5
    assert features.relative_normalize(50.0, [10.0]) == 0.5


def test_relative_normalize_flat_samples_defaults_to_midpoint():
    assert features.relative_normalize(10.0, [10.0, 10.0, 10.0]) == 0.5


def test_relative_normalize_uses_own_agent_distribution_not_global():
    # an agent whose own history centers on 150ms sees a 150ms sample as
    # "normal" (0.5), even though 150ms would be extreme for a faster agent
    samples = [140.0, 150.0, 160.0, 145.0, 155.0]
    assert abs(features.relative_normalize(150.0, samples) - 0.5) < 0.05


def test_relative_normalize_is_stable_across_one_new_sample():
    # regression: min-max over a small window let ONE new extreme sample
    # swing the normalized value wildly (seen live: 0.75 -> 0.47 in three
    # seconds for a genuinely steady agent) -- a z-score-based measure
    # should barely move when one more typical sample arrives
    samples = [120.0 + i * 0.3 for i in range(40)]  # tight, steady distribution
    before = features.relative_normalize(samples[-1], samples)
    after_one_more = features.relative_normalize(samples[-1], samples + [121.0])
    assert abs(before - after_one_more) < 0.05


def test_relative_normalize_still_flags_a_genuine_outlier():
    samples = [120.0 + i * 0.3 for i in range(40)]
    outlier_score = features.relative_normalize(400.0, samples)  # far outside the steady range
    assert outlier_score > 0.9


def test_push_capped_respects_cap():
    items = list(range(80))
    out = features.push_capped(items, 999, cap=60)
    assert len(out) == 60
    assert out[-1] == 999


def test_comms_frequency_ignores_stale_timestamps():
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(seconds=10)).isoformat()
    stale = (now - timedelta(seconds=300)).isoformat()
    freq = features.comms_frequency([stale, recent, recent], now=now)
    assert freq == 2 / (features.COMMS_WINDOW_S / 60.0)


def test_task_completion_rate_no_tasks_defaults_to_midpoint():
    assert features.task_completion_rate(0, 0) == 0.5


def test_task_completion_rate_basic():
    assert features.task_completion_rate(3, 4) == 0.75
