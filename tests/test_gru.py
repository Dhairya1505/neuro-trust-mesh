import random

import torch

import gru


def _buffer(trust_trajectory: list[float]) -> list[list[float]]:
    """One synthetic timestep per trust value, other fields held roughly
    steady so the trust trajectory is the dominant signal to learn.
    """
    return [
        [0.9, 0.2, 0.1, 0.3, trust, 0.9, 0]
        for trust in trust_trajectory
    ]


def test_build_windows_shapes():
    buffer = _buffer([0.5] * 40)
    X, y = gru.build_windows(buffer, seq_len=20, k=1)
    # 40 timesteps, seq_len=20, k=1 -> windows start at 0..19 inclusive (20 windows)
    assert len(X) == 20
    assert len(X[0]) == 20
    assert len(y) == 20


def test_build_windows_label_is_trust_score_k_steps_after_window():
    # trust score (index gru.TRUST_SCORE_INDEX) == timestep index, other fields irrelevant
    buffer = [
        [0.0] * gru.TRUST_SCORE_INDEX + [float(i)] + [0.0] * (6 - gru.TRUST_SCORE_INDEX)
        for i in range(25)
    ]
    X, y = gru.build_windows(buffer, seq_len=20, k=1)
    # first window covers timesteps 0..19, label should be timestep 20's trust score
    assert y[0] == 20.0


def test_train_produces_valid_outputs():
    random.seed(0)
    torch.manual_seed(0)
    buffer = _buffer([0.8 + random.gauss(0, 0.01) for _ in range(40)])
    model = gru.train(buffer, epochs=15)
    pred = gru.predict(model, buffer[-gru.SEQUENCE_LENGTH:])
    assert 0.0 <= pred.mean <= 1.0
    assert pred.std > 0.0


def test_model_learns_a_declining_trend():
    random.seed(1)
    torch.manual_seed(1)
    # trust decays steadily from 0.9 down to 0.1 -- the next value after
    # any window should clearly continue the decline, not sit near 0.9
    trajectory = [max(0.05, 0.9 - 0.02 * i) for i in range(60)]
    buffer = _buffer(trajectory)
    model = gru.train(buffer, epochs=200, lr=1e-2)
    pred = gru.predict(model, buffer[-gru.SEQUENCE_LENGTH:])
    assert pred.mean < 0.5  # should track the decline, not the early-window high values


def test_fine_tune_does_not_reinitialize_model():
    random.seed(2)
    torch.manual_seed(2)
    buffer = _buffer([0.7 + random.gauss(0, 0.01) for _ in range(40)])
    model = gru.train(buffer, epochs=20)
    weight_before = model.mean_head.weight.clone()
    gru.fine_tune(model, buffer, epochs=5)
    weight_after = model.mean_head.weight
    # fine-tuning should move weights, but from the SAME starting point
    # (same object identity), not a freshly re-initialized model
    assert not (weight_before == weight_after).all()


def test_trend_confidence_is_inverse_of_std():
    assert gru.trend_confidence(0.5) == 2.0


def test_trend_confidence_floors_std_to_avoid_div_by_zero():
    assert gru.trend_confidence(0.0) == 1.0 / gru.MIN_STD
