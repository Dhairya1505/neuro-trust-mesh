"""GRU architecture duplicated from trust_forecast/gru.py -- MUST match
exactly, since this service loads and evaluates weights trained there. Only
what's needed to load a state_dict and score it is here; this service never
trains, only validates merged weights (ML doc §8).
"""
import torch
import torch.nn.functional as F
from torch import nn

FEATURE_DIM = 7
TRUST_SCORE_INDEX = 4  # index of bayesian_trust_score within the shared 7-dim feature vector
SEQUENCE_LENGTH = 20  # must match trust_forecast/gru.py::SEQUENCE_LENGTH
HORIZON_K = 1  # must match trust_forecast/gru.py::HORIZON_K
MIN_STD = 1e-3


class GRUForecaster(nn.Module):
    def __init__(self, hidden1: int = 64, hidden2: int = 32, dense: int = 16, dropout: float = 0.2):
        super().__init__()
        self.gru1 = nn.GRU(FEATURE_DIM, hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.gru2 = nn.GRU(hidden1, hidden2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden2, dense)
        self.mean_head = nn.Linear(dense, 1)
        self.std_head = nn.Linear(dense, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out1, _ = self.gru1(x)
        out1 = self.dropout1(out1)
        out2, _ = self.gru2(out1)
        last = self.dropout2(out2[:, -1, :])
        d = F.relu(self.dense(last))
        mean = torch.sigmoid(self.mean_head(d)).squeeze(-1)
        std = F.softplus(self.std_head(d)).squeeze(-1) + MIN_STD
        return mean, std


def state_dict_from_json(data: dict) -> GRUForecaster:
    model = GRUForecaster()
    model.load_state_dict({name: torch.tensor(values, dtype=torch.float32) for name, values in data.items()})
    return model


def build_windows(buffer: list[list[float]], seq_len: int = SEQUENCE_LENGTH, k: int = HORIZON_K):
    X, y = [], []
    last_start = len(buffer) - seq_len - k
    for start in range(0, last_start + 1):
        window = buffer[start : start + seq_len]
        label_idx = start + seq_len + k - 1
        X.append(window)
        y.append(buffer[label_idx][TRUST_SCORE_INDEX])
    return X, y


def evaluate(model: GRUForecaster, validation_buffer: list[list[float]]) -> float:
    """Mean absolute error of the predicted mean against the trust-score
    label, over windows built from the pooled cross-agent validation buffer
    (ML doc §8's 'held-out trust-prediction benchmark') -- lower is better,
    same rollback-comparison shape as anomaly_detection's mean_error.
    """
    X_list, y_list = build_windows(validation_buffer)
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X_list, dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.float32)
        mean, _ = model(X)
        return torch.mean(torch.abs(mean - y)).item()
