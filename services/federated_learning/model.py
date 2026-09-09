"""Autoencoder architecture duplicated from
anomaly_detection/autoencoder.py -- MUST match exactly, since this service
loads and evaluates weights trained there. Only what's needed to load a
state_dict and score it is here; this service never trains, only
validates merged weights (ML doc §8).
"""
import torch
from torch import nn

FEATURE_DIM = 7
ANOMALY_STD_MULTIPLIER = 3.0  # must match anomaly_detection/autoencoder.py
FEATURE_ORDER = [  # must match anomaly_detection/autoencoder.py::FEATURE_ORDER
    "task_completion_rate",
    "avg_response_latency",
    "comms_frequency",
    "resource_utilization",
    "bayesian_trust_score",
    "bayesian_confidence",
    "role_flag",
]


def vector_from_feature_dict(features: dict) -> list[float]:
    return [features[key] if features.get(key) is not None else 0.5 for key in FEATURE_ORDER]


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(FEATURE_DIM, 16), nn.ReLU(), nn.Linear(16, 4), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, FEATURE_DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def state_dict_from_json(data: dict) -> Autoencoder:
    model = Autoencoder()
    model.load_state_dict({name: torch.tensor(values, dtype=torch.float32) for name, values in data.items()})
    return model


def evaluate(model: Autoencoder, validation_vectors: list[list[float]]) -> tuple[float, float, float]:
    """Returns (mean_error, std_error, threshold) over the validation set
    -- the 'held-out trust-prediction benchmark' ML doc §8 validates a
    merged model against before it's ever redistributed.
    """
    model.eval()
    with torch.no_grad():
        X = torch.tensor(validation_vectors, dtype=torch.float32)
        errors = torch.mean((model(X) - X) ** 2, dim=1)
    mean_error = errors.mean().item()
    std_error = errors.std(unbiased=False).item()
    return mean_error, std_error, mean_error + ANOMALY_STD_MULTIPLIER * std_error
