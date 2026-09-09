"""Model B -- GRU trust forecaster, ML doc §3. Architecture and
training/inference logic kept as plain functions around a small torch.nn
module so the math is unit-testable without a live event bus/DB.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

FEATURE_DIM = 7  # same 7-dim vector as BehaviorFeatures / Model C
TRUST_SCORE_INDEX = 4  # index of bayesian_trust_score within that vector

SEQUENCE_LENGTH = 20  # T=20 (ML doc §3)
HORIZON_K = 1  # "Start with K=1 to validate the pipeline" (ML doc §3)

MIN_TRAINING_WINDOWS = 15
BUFFER_CAP = 300  # ~25 min of history at a 5s sampling cadence

INITIAL_TRAIN_EPOCHS = 100
INITIAL_TRAIN_LR = 1e-3
FINE_TUNE_EVERY_TIMESTEPS = 100  # "periodic small-batch updates ... every 100 new timesteps"
FINE_TUNE_EPOCHS = 20
FINE_TUNE_LR = 1e-4  # "low learning rate" (ML doc §3)
FINE_TUNE_WINDOW_COUNT = 50  # fine-tune on the most recent windows, not the whole buffer

MIN_STD = 1e-3  # floor to avoid div-by-zero in trend_confidence = 1/std and log(0) in the NLL loss


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
        out1, _ = self.gru1(x)  # (batch, T, hidden1) -- return_sequences=True
        out1 = self.dropout1(out1)
        out2, _ = self.gru2(out1)  # (batch, T, hidden2)
        last = self.dropout2(out2[:, -1, :])  # return_sequences=False -- final timestep only
        d = F.relu(self.dense(last))
        mean = torch.sigmoid(self.mean_head(d)).squeeze(-1)
        std = F.softplus(self.std_head(d)).squeeze(-1) + MIN_STD
        return mean, std


def build_windows(buffer: list[list[float]], seq_len: int = SEQUENCE_LENGTH, k: int = HORIZON_K):
    """Slide a window of length seq_len over buffer, paired with the trust
    score k steps after the window ends. buffer[t][TRUST_SCORE_INDEX] IS
    the trust score at time t, so no separate label source is needed.
    """
    X, y = [], []
    last_start = len(buffer) - seq_len - k
    for start in range(0, last_start + 1):
        window = buffer[start : start + seq_len]
        label_idx = start + seq_len + k - 1
        X.append(window)
        y.append(buffer[label_idx][TRUST_SCORE_INDEX])
    return X, y


def _nll_step(model: GRUForecaster, optimizer: torch.optim.Optimizer, X: torch.Tensor, y: torch.Tensor) -> float:
    optimizer.zero_grad()
    mean, std = model(X)
    loss = F.gaussian_nll_loss(mean, y, std**2, full=True)
    loss.backward()
    optimizer.step()
    return loss.item()


def train(buffer: list[list[float]], epochs: int = INITIAL_TRAIN_EPOCHS, lr: float = INITIAL_TRAIN_LR) -> GRUForecaster:
    X_list, y_list = build_windows(buffer)
    model = GRUForecaster()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(X_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        _nll_step(model, optimizer, X, y)
    return model


def fine_tune(
    model: GRUForecaster,
    buffer: list[list[float]],
    epochs: int = FINE_TUNE_EPOCHS,
    lr: float = FINE_TUNE_LR,
    window_count: int = FINE_TUNE_WINDOW_COUNT,
) -> GRUForecaster:
    """Online fine-tuning (ML doc §3): continue training the EXISTING model
    -- not reinitialized -- on recent windows at a low learning rate,
    instead of full retraining.
    """
    X_list, y_list = build_windows(buffer)
    X_list, y_list = X_list[-window_count:], y_list[-window_count:]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(X_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        _nll_step(model, optimizer, X, y)
    return model


@dataclass
class Prediction:
    mean: float
    std: float


def predict(model: GRUForecaster, window: list[list[float]]) -> Prediction:
    model.eval()
    with torch.no_grad():
        x = torch.tensor([window], dtype=torch.float32)
        mean, std = model(x)
        return Prediction(mean=mean.item(), std=std.item())


def trend_confidence(std: float) -> float:
    return 1.0 / max(std, MIN_STD)


def state_dict_to_json(model: GRUForecaster) -> dict:
    """Weight-only export for federated_learning (ML doc §8) -- only model
    parameters cross the wire, never an agent's raw behavioral samples
    (system doc §4.6).
    """
    return {name: tensor.detach().tolist() for name, tensor in model.state_dict().items()}


def state_dict_from_json(data: dict) -> GRUForecaster:
    model = GRUForecaster()
    model.load_state_dict({name: torch.tensor(values, dtype=torch.float32) for name, values in data.items()})
    return model
