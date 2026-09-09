"""Model C -- Autoencoder anomaly detector, ML doc §4. Architecture and
training/scoring logic kept as plain functions around a small torch.nn
module so the math is unit-testable without a live event bus/DB.
"""
import random
from dataclasses import dataclass

import torch
from torch import nn

FEATURE_DIM = 7  # matches BehaviorFeatures exactly, in this field order:
FEATURE_ORDER = [
    "task_completion_rate",
    "avg_response_latency",
    "comms_frequency",
    "resource_utilization",
    "bayesian_trust_score",
    "bayesian_confidence",
    "role_flag",
]

MIN_TRAINING_SAMPLES = 40
RETRAIN_EVERY = 20
TRAINING_BUFFER_CAP = 200
TRAIN_EPOCHS = 80
TRAIN_LR = 1e-3
N_FOLDS = 5
ANOMALY_STD_MULTIPLIER = 3.0  # threshold = mean_error + 3*std_error (ML doc §4)

# A 3-sigma threshold on ~40-sample calibration data still has a real
# single-reading false-positive rate -- confirmed live, isolated healthy
# agents on a single noisy sample. Require a SUSTAINED signal before
# treating it as a real anomaly, same pattern already proven in this
# codebase (security_monitoring's rule window, collusion_detection's
# persistence guardrail): one noisy reading shouldn't isolate an agent,
# several in a row should.
ANOMALY_PERSISTENCE_REQUIRED = 2


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(FEATURE_DIM, 16), nn.ReLU(), nn.Linear(16, 4), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, FEATURE_DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def state_dict_to_json(model: Autoencoder) -> dict:
    """Weight-only export for federated_learning (ML doc §8) -- only
    model parameters cross the wire, never an agent's raw behavioral
    samples (system doc §4.6).
    """
    return {name: tensor.detach().tolist() for name, tensor in model.state_dict().items()}


def state_dict_from_json(data: dict) -> Autoencoder:
    model = Autoencoder()
    model.load_state_dict({name: torch.tensor(values, dtype=torch.float32) for name, values in data.items()})
    return model


def vector_from_feature_dict(features: dict) -> list[float]:
    """None-safe extraction in FEATURE_ORDER -- bayesian_trust_score and
    bayesian_confidence can be None before trust_prediction has scored the
    agent yet; 0.5 is a neutral stand-in (matches the Bayesian prior).
    """
    return [
        features[key] if features.get(key) is not None else 0.5
        for key in FEATURE_ORDER
    ]


@dataclass
class TrainedProfile:
    model: Autoencoder
    mean_error: float
    std_error: float
    threshold: float


def _fit(vectors: list[list[float]], epochs: int, lr: float) -> Autoencoder:
    model = Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(vectors, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.mean((model(X) - X) ** 2)
        loss.backward()
        optimizer.step()
    return model


def _per_sample_errors(model: Autoencoder, vectors: list[list[float]]) -> list[float]:
    model.eval()
    with torch.no_grad():
        X = torch.tensor(vectors, dtype=torch.float32)
        errors = torch.mean((model(X) - X) ** 2, dim=1)
    return errors.tolist()


def train(vectors: list[list[float]], epochs: int = TRAIN_EPOCHS, lr: float = TRAIN_LR, n_folds: int = N_FOLDS) -> TrainedProfile:
    """Calibrate mean_error/std_error/threshold via k-fold cross-validation,
    then fit the deployed model on the full dataset.

    A single random train/holdout split (the first attempt at this fix) is
    too noisy at this data scale (tens of samples, single-digit holdout
    size): one unlucky split can still produce a razor-thin threshold that
    flags ordinary normal behavior as anomalous -- confirmed live, where a
    rebuild's unlucky split isolated nearly every healthy agent in the
    fleet. Pooling held-out errors across k folds uses every sample as
    held-out exactly once, giving a materially more stable estimate of
    genuine generalization error without wasting data, while the final
    model is still trained on everything for the best deployed accuracy.
    """
    shuffled = list(vectors)
    random.shuffle(shuffled)
    n = len(shuffled)
    fold_size = max(1, n // n_folds)

    pooled_errors: list[float] = []
    for i in range(n_folds):
        start, end = i * fold_size, (i + 1) * fold_size if i < n_folds - 1 else n
        held_out = shuffled[start:end]
        train_split = shuffled[:start] + shuffled[end:]
        if not held_out or len(train_split) < 2:
            continue
        fold_model = _fit(train_split, epochs, lr)
        pooled_errors.extend(_per_sample_errors(fold_model, held_out))

    errors_tensor = torch.tensor(pooled_errors)
    mean_error = errors_tensor.mean().item()
    std_error = errors_tensor.std(unbiased=False).item()
    threshold = mean_error + ANOMALY_STD_MULTIPLIER * std_error

    final_model = _fit(shuffled, epochs, lr)
    return TrainedProfile(model=final_model, mean_error=mean_error, std_error=std_error, threshold=threshold)


def reconstruction_error(model: Autoencoder, vector: list[float]) -> float:
    model.eval()
    with torch.no_grad():
        x = torch.tensor([vector], dtype=torch.float32)
        recon = model(x)
        return torch.mean((recon - x) ** 2).item()


def is_anomalous(error: float, threshold: float) -> bool:
    return error > threshold


@dataclass
class AnomalyPersistence:
    consecutive_hits: int = 0


def update_anomaly_persistence(state: AnomalyPersistence, raw_is_anomalous: bool) -> tuple[AnomalyPersistence, bool]:
    """A gap resets progress, same reasoning as
    collusion_detection/signals.py::update_persistence -- a single good
    reading in between two bad ones means it wasn't sustained, so it
    shouldn't count toward flagging.
    """
    if not raw_is_anomalous:
        return AnomalyPersistence(consecutive_hits=0), False
    hits = state.consecutive_hits + 1
    return AnomalyPersistence(consecutive_hits=hits), hits >= ANOMALY_PERSISTENCE_REQUIRED


def normalize_for_fusion(error: float, threshold: float) -> float:
    """Scale reconstruction error onto ~[0,1] against this agent's own
    threshold, for security_monitoring's combined_security_risk (ML doc §6:
    max(normalize(security_score), normalize(reconstruction_error))).
    """
    if threshold <= 0:
        return 0.0
    return min(1.0, error / threshold)
