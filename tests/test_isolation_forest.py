import random

import isolation_forest


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


def test_train_returns_fitted_model():
    random.seed(0)
    vectors = [_normal_vector() for _ in range(40)]
    model = isolation_forest.train(vectors, random_state=0)
    assert model is not None


def test_flags_a_clear_outlier_as_anomalous():
    random.seed(1)
    vectors = [_normal_vector() for _ in range(40)]
    model = isolation_forest.train(vectors, random_state=1)

    outlier = [0.05, 0.99, 1.0, 0.98, 0.05, 0.05, 1]
    assert isolation_forest.is_anomalous(model, outlier) is True


def test_does_not_flag_typical_samples_at_high_rate():
    random.seed(2)
    vectors = [_normal_vector() for _ in range(40)]
    model = isolation_forest.train(vectors, random_state=2)

    typical_samples = [_normal_vector() for _ in range(30)]
    flagged = sum(1 for v in typical_samples if isolation_forest.is_anomalous(model, v))
    assert flagged / len(typical_samples) < 0.3


def test_normalize_for_fusion_is_bounded_zero_to_one():
    random.seed(3)
    vectors = [_normal_vector() for _ in range(40)]
    model = isolation_forest.train(vectors, random_state=3)

    for v in [_normal_vector() for _ in range(10)] + [[0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1]]:
        score = isolation_forest.normalize_for_fusion(model, v)
        assert 0.0 <= score <= 1.0


def test_normalize_for_fusion_higher_for_outlier_than_typical():
    random.seed(4)
    vectors = [_normal_vector() for _ in range(40)]
    model = isolation_forest.train(vectors, random_state=4)

    typical_score = isolation_forest.normalize_for_fusion(model, _normal_vector())
    outlier_score = isolation_forest.normalize_for_fusion(model, [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1])
    assert outlier_score > typical_score
