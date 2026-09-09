import random

import fedavg
import model
import model_gru
import pytest
import torch


def test_tree_mean_scalars():
    assert fedavg.tree_mean([1.0, 2.0, 3.0]) == 2.0


def test_tree_mean_nested_lists():
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    assert fedavg.tree_mean([a, b]) == [[3.0, 4.0], [5.0, 6.0]]


def test_fedavg_requires_at_least_one_client():
    with pytest.raises(ValueError):
        fedavg.fedavg([])


def test_fedavg_rejects_mismatched_architectures():
    with pytest.raises(ValueError):
        fedavg.fedavg([{"a": [1.0]}, {"b": [1.0]}])


def test_fedavg_two_clients_averages_weights():
    sd_a = {"layer.weight": [[1.0, 1.0]], "layer.bias": [0.0]}
    sd_b = {"layer.weight": [[3.0, 3.0]], "layer.bias": [2.0]}
    merged = fedavg.fedavg([sd_a, sd_b])
    assert merged["layer.weight"] == [[2.0, 2.0]]
    assert merged["layer.bias"] == [1.0]


def test_fedavg_single_client_is_a_noop():
    sd = {"layer.weight": [[1.0, 2.0]], "layer.bias": [0.5]}
    assert fedavg.fedavg([sd]) == sd


def test_state_dict_roundtrip_preserves_weights():
    torch.manual_seed(0)
    original = model.Autoencoder()
    exported = {name: tensor.tolist() for name, tensor in original.state_dict().items()}
    restored = model.state_dict_from_json(exported)
    for (name_a, param_a), (name_b, param_b) in zip(original.state_dict().items(), restored.state_dict().items()):
        assert name_a == name_b
        assert torch.allclose(param_a, param_b)


def test_evaluate_returns_threshold_above_mean_error():
    torch.manual_seed(1)
    random.seed(1)
    m = model.Autoencoder()
    validation = [[random.gauss(0.5, 0.1) for _ in range(model.FEATURE_DIM)] for _ in range(30)]
    mean_error, std_error, threshold = model.evaluate(m, validation)
    assert threshold == mean_error + model.ANOMALY_STD_MULTIPLIER * std_error
    assert threshold >= mean_error


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
    vector = model.vector_from_feature_dict(features)
    assert vector[4] == 0.5 and vector[5] == 0.5


def test_gru_state_dict_roundtrip_preserves_weights():
    torch.manual_seed(0)
    original = model_gru.GRUForecaster()
    exported = {name: tensor.tolist() for name, tensor in original.state_dict().items()}
    restored = model_gru.state_dict_from_json(exported)
    for (name_a, param_a), (name_b, param_b) in zip(original.state_dict().items(), restored.state_dict().items()):
        assert name_a == name_b
        assert torch.allclose(param_a, param_b)


def test_gru_build_windows_shapes():
    buffer = [[float(i)] * model_gru.FEATURE_DIM for i in range(25)]
    X, y = model_gru.build_windows(buffer)
    assert len(X) == 25 - model_gru.SEQUENCE_LENGTH - model_gru.HORIZON_K + 1
    assert len(X[0]) == model_gru.SEQUENCE_LENGTH


def test_gru_evaluate_returns_a_float_error():
    torch.manual_seed(2)
    random.seed(2)
    m = model_gru.GRUForecaster()
    buffer = [[random.gauss(0.5, 0.1) for _ in range(model_gru.FEATURE_DIM)] for _ in range(35)]
    mae = model_gru.evaluate(m, buffer)
    assert isinstance(mae, float)
    assert mae >= 0.0
