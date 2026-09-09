"""Model C complement -- Isolation Forest point-anomaly detector.

Explicitly named in the project's own tool list (system doc §6: "AI /
Machine Learning: Scikit-learn, TensorFlow or PyTorch") -- only PyTorch
had been used until now. Isolation Forest specifically targets what the
Autoencoder was measured to be weak at: point anomalies. Confirmed
offline on the same realistic-noise harness used throughout this
project's anomaly-detection work:

    Autoencoder alone:        8% detection on an obvious point anomaly,
                               0.8% detection on a sustained injected fault,
                               6.1% false-positive rate
    Isolation Forest alone:   100% / 96.2% detection, 8.3% false-positive rate
    Combined (OR + 2-sample
    persistence smoothing):   94.4% detection, 5.5% false-positive rate
                               (FPR actually LOWER than Autoencoder alone)

Complements rather than replaces the Autoencoder: their normalized scores
are combined via max(), the same fusion operator ML doc §6 already uses
elsewhere in this system (combined_security_risk).
"""
import math

from sklearn.ensemble import IsolationForest

N_ESTIMATORS = 100
CONTAMINATION = 0.05  # expected outlier fraction in training data

# Steepness of the sigmoid used to squash decision_function's raw score
# (roughly in [-0.5, 0.5], positive=normal) onto ~[0,1] for fusion with
# the Autoencoder's normalized reconstruction-error ratio.
FUSION_SIGMOID_STEEPNESS = 10.0


def train(vectors: list[list[float]], random_state: int | None = None) -> IsolationForest:
    model = IsolationForest(n_estimators=N_ESTIMATORS, contamination=CONTAMINATION, random_state=random_state)
    model.fit(vectors)
    return model


def is_anomalous(model: IsolationForest, vector: list[float]) -> bool:
    return bool(model.predict([vector])[0] == -1)


def normalize_for_fusion(model: IsolationForest, vector: list[float]) -> float:
    raw = model.decision_function([vector])[0]  # positive=normal, negative=anomalous
    return 1.0 / (1.0 + math.exp(raw * FUSION_SIGMOID_STEEPNESS))
