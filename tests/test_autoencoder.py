import random

import torch
import sys
from pathlib import Path

import autoencoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "behavior_analysis"))
import features as behavior_features  # noqa: E402


def test_vector_from_feature_dict_preserves_order():
    features = {
        "task_completion_rate": 0.9,
        "avg_response_latency": 0.1,
        "comms_frequency": 0.2,
        "resource_utilization": 0.3,
        "bayesian_trust_score": 0.8,
        "bayesian_confidence": 0.95,
        "role_flag": 1,
    }
    assert autoencoder.vector_from_feature_dict(features) == [0.9, 0.1, 0.2, 0.3, 0.8, 0.95, 1]


def test_vector_from_feature_dict_defaults_none_trust_fields():
    features = {
        "task_completion_rate": 0.5,
        "avg_response_latency": 0.5,
        "comms_frequency": 0.5,
        "resource_utilization": 0.5,
        "bayesian_trust_score": None,
        "bayesian_confidence": None,
        "role_flag": 0,
    }
    vector = autoencoder.vector_from_feature_dict(features)
    assert vector[4] == 0.5 and vector[5] == 0.5


def _normal_vector():
    return [
        random.gauss(0.9, 0.02),
        random.gauss(0.2, 0.02),
        random.gauss(0.1, 0.02),
        random.gauss(0.3, 0.02),
        random.gauss(0.8, 0.02),
        random.gauss(0.9, 0.02),
        0,
    ]


def test_train_produces_threshold_above_mean_error():
    random.seed(0)
    torch.manual_seed(0)
    vectors = [_normal_vector() for _ in range(autoencoder.MIN_TRAINING_SAMPLES)]
    profile = autoencoder.train(vectors, epochs=60)
    assert profile.threshold == profile.mean_error + autoencoder.ANOMALY_STD_MULTIPLIER * profile.std_error
    assert profile.threshold >= profile.mean_error


def test_trained_model_flags_a_clear_outlier_as_anomalous():
    random.seed(1)
    torch.manual_seed(1)
    vectors = [_normal_vector() for _ in range(autoencoder.MIN_TRAINING_SAMPLES)]
    profile = autoencoder.train(vectors, epochs=150)

    outlier = [0.0, 0.95, 0.9, 0.99, 0.05, 0.05, 1]  # nothing like the training distribution
    error = autoencoder.reconstruction_error(profile.model, outlier)
    assert autoencoder.is_anomalous(error, profile.threshold) is True


def test_trained_model_rarely_flags_typical_samples():
    # a single draw right at a 3-sigma boundary can land either side by
    # chance -- assert the aggregate rate, not one specific draw
    random.seed(2)
    torch.manual_seed(2)
    vectors = [_normal_vector() for _ in range(autoencoder.MIN_TRAINING_SAMPLES)]
    profile = autoencoder.train(vectors, epochs=150)

    typical_samples = [_normal_vector() for _ in range(30)]
    flagged = sum(
        1 for v in typical_samples if autoencoder.is_anomalous(autoencoder.reconstruction_error(profile.model, v), profile.threshold)
    )
    assert flagged / len(typical_samples) < 0.2


def test_false_positive_rate_is_low_and_stable_across_seeds():
    """Regression test: a single random holdout split (the first attempt at
    threshold calibration) was statistically fragile at this data scale --
    it passed its own unit test but then, on a live rebuild with different
    random data, produced a razor-thin threshold that flagged nearly every
    healthy agent in the fleet as anomalous. Check the false-positive rate
    directly, across several independent seeds, so a regression to that
    fragility fails deterministically instead of only "sometimes" in prod.
    """
    max_fpr_seen = 0.0
    for seed in range(5):
        random.seed(seed)
        torch.manual_seed(seed)
        training_vectors = [_normal_vector() for _ in range(40)]
        profile = autoencoder.train(training_vectors, epochs=80)

        fresh_normal_samples = [_normal_vector() for _ in range(100)]
        false_positives = sum(
            1
            for v in fresh_normal_samples
            if autoencoder.is_anomalous(autoencoder.reconstruction_error(profile.model, v), profile.threshold)
        )
        fpr = false_positives / len(fresh_normal_samples)
        max_fpr_seen = max(max_fpr_seen, fpr)

    # a healthy 3-sigma threshold should sit in the low single digits % --
    # generous bound to allow small-sample noise, while still catching a
    # gross regression like the ~80-100% FPR seen before this fix
    assert max_fpr_seen < 0.2, f"false positive rate too high: {max_fpr_seen:.0%}"


def _realistic_normal_vector(latency_hist: list[float], resource_hist: list[float]) -> list[float]:
    """Uses the ACTUAL behavior_analysis normalization function against
    realistic Gaussian sensor noise, not an idealized tight distribution --
    this is what caught the live false-positive cascade that the simpler
    synthetic test above did not.
    """
    latency = random.gauss(120, 15)
    resource = random.gauss(40, 8)
    latency_hist.append(latency)
    latency_hist[:] = latency_hist[-behavior_features.SAMPLE_CAP:]
    resource_hist.append(resource)
    resource_hist[:] = resource_hist[-behavior_features.SAMPLE_CAP:]
    return [
        random.gauss(0.9, 0.03),
        behavior_features.relative_normalize(latency, latency_hist),
        1.0,
        behavior_features.relative_normalize(resource, resource_hist),
        random.gauss(0.8, 0.02),
        random.gauss(0.95, 0.01),
        0,
    ]


def test_false_positive_rate_with_realistic_normalized_noise():
    """Regression test for the live incident: the simpler idealized-gaussian
    FPR test above passed throughout, but real feature vectors (run through
    behavior_analysis's actual per-agent normalization) still produced a
    severe false-positive cascade -- multiple healthy agents isolated in a
    single run. This test reproduces that data path directly.
    """
    max_fpr_seen = 0.0
    for seed in range(5):
        random.seed(seed)
        torch.manual_seed(seed)
        latency_hist, resource_hist = [], []
        training = [_realistic_normal_vector(latency_hist, resource_hist) for _ in range(autoencoder.MIN_TRAINING_SAMPLES)]
        profile = autoencoder.train(training, epochs=80)

        fresh = [_realistic_normal_vector(latency_hist, resource_hist) for _ in range(100)]
        false_positives = sum(
            1 for v in fresh if autoencoder.is_anomalous(autoencoder.reconstruction_error(profile.model, v), profile.threshold)
        )
        max_fpr_seen = max(max_fpr_seen, false_positives / len(fresh))

    assert max_fpr_seen < 0.2, f"false positive rate too high on realistic data: {max_fpr_seen:.0%}"


def test_persistence_single_hit_does_not_flag():
    state, flagged = autoencoder.update_anomaly_persistence(autoencoder.AnomalyPersistence(), True)
    assert flagged is False
    assert state.consecutive_hits == 1


def test_persistence_required_consecutive_hits_flags():
    state = autoencoder.AnomalyPersistence()
    for _ in range(autoencoder.ANOMALY_PERSISTENCE_REQUIRED - 1):
        state, flagged = autoencoder.update_anomaly_persistence(state, True)
        assert flagged is False
    state, flagged = autoencoder.update_anomaly_persistence(state, True)
    assert flagged is True


def test_persistence_gap_resets_progress():
    state = autoencoder.AnomalyPersistence()
    state, _ = autoencoder.update_anomaly_persistence(state, True)
    state, flagged = autoencoder.update_anomaly_persistence(state, False)  # gap
    assert flagged is False
    assert state.consecutive_hits == 0


def test_normalize_for_fusion_clips_at_one():
    assert autoencoder.normalize_for_fusion(error=10.0, threshold=1.0) == 1.0


def test_normalize_for_fusion_zero_threshold_is_safe():
    assert autoencoder.normalize_for_fusion(error=0.5, threshold=0.0) == 0.0


def test_normalize_for_fusion_proportional_below_threshold():
    assert autoencoder.normalize_for_fusion(error=0.5, threshold=1.0) == 0.5
