import asyncio
from collections import deque
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

import fedavg
import model
import model_gru
from neurotrust_common import EventBus, get_logger, get_settings
from neurotrust_common.schemas import BehaviorFeatures

SERVICE_NAME = "federated_learning"
ANOMALY_DETECTION_URL = "http://anomaly_detection:8000"
TRUST_FORECAST_URL = "http://trust_forecast:8000"

AGGREGATION_INTERVAL_S = 90
MIN_CLIENTS = 2  # FL needs multiple contributors to be meaningful
MIN_VALIDATION_SAMPLES = 20
# GRU validation needs whole windows, not single samples -- a few windows'
# worth of slack beyond the bare minimum (T=20 + K=1) for a stable estimate.
MIN_VALIDATION_SAMPLES_GRU = model_gru.SEQUENCE_LENGTH + model_gru.HORIZON_K + 10
VALIDATION_BUFFER_CAP = 150
# ML doc §8: "if merged model degrades performance -> automatic rollback".
# A small tolerance avoids rejecting a merge over ordinary validation noise.
ROLLBACK_TOLERANCE = 0.10
ROUND_HISTORY_CAP = 20

settings = get_settings(SERVICE_NAME)
log = get_logger(SERVICE_NAME, settings.log_level)
bus = EventBus(settings.rabbitmq_url, SERVICE_NAME)

# Same pooled cross-agent vector feeds both models' validation sets --
# BehaviorFeatures uses the identical 7-dim feature order everywhere in
# this system (autoencoder.py::FEATURE_ORDER == gru.py's per-timestep
# vector layout), so one buffer serves both.
_validation_buffer: deque = deque(maxlen=VALIDATION_BUFFER_CAP)
_best_model: model.Autoencoder | None = None
_best_error: float | None = None
_version = 0
_round_history: deque = deque(maxlen=ROUND_HISTORY_CAP)

_best_error_gru: float | None = None
_version_gru = 0
_round_history_gru: deque = deque(maxlen=ROUND_HISTORY_CAP)


async def on_behavior_features(event: BehaviorFeatures) -> None:
    # A pooled, cross-agent held-out set -- this service never sees which
    # agent a sample came from beyond this one field, and never trains on
    # it, only evaluates (ML doc §8's "held-out trust-prediction benchmark").
    _validation_buffer.append(model.vector_from_feature_dict(event.model_dump()))


async def _aggregation_round() -> None:
    global _best_model, _best_error, _version

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(f"{ANOMALY_DETECTION_URL}/models/export")
        except httpx.HTTPError as exc:
            log.warning("could not reach anomaly_detection for export", extra={"trace": {"error": str(exc)}})
            return
        if resp.status_code != 200:
            return
        client_state_dicts = resp.json()

    if len(client_state_dicts) < MIN_CLIENTS:
        log.info(
            "skipping round -- not enough clients",
            extra={"trace": {"clients": len(client_state_dicts), "min_clients": MIN_CLIENTS}},
        )
        return
    if len(_validation_buffer) < MIN_VALIDATION_SAMPLES:
        log.info(
            "skipping round -- not enough validation data",
            extra={"trace": {"validation_samples": len(_validation_buffer)}},
        )
        return

    merged_state_dict = fedavg.fedavg(list(client_state_dicts.values()))
    merged_model = model.state_dict_from_json(merged_state_dict)
    mean_error, std_error, threshold = model.evaluate(merged_model, list(_validation_buffer))

    accepted = _best_error is None or mean_error <= _best_error * (1 + ROLLBACK_TOLERANCE)
    round_summary = {
        "round": _version + 1,
        "clients": sorted(client_state_dicts),
        "mean_error": mean_error,
        "std_error": std_error,
        "threshold": threshold,
        "accepted": accepted,
        "best_error_before": _best_error,
    }
    _round_history.append(round_summary)

    if not accepted:
        log.warning(
            "federated round rejected -- merged model degraded performance, rolling back to last known-good",
            extra={"trace": round_summary},
        )
        return

    _best_model = merged_model
    _best_error = mean_error
    _version += 1
    log.info("federated round accepted", extra={"trace": round_summary})

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            await client.post(
                f"{ANOMALY_DETECTION_URL}/models/import_federated",
                json={"state_dict": merged_state_dict, "threshold": threshold, "version": _version},
            )
        except httpx.HTTPError as exc:
            log.warning("could not push federated model to anomaly_detection", extra={"trace": {"error": str(exc)}})


async def _aggregation_round_gru() -> None:
    """Same FedAvg/validate/rollback shape as _aggregation_round(), for
    Model B (ML doc §8 says GRU and Autoencoder are both federatable --
    only the Autoencoder was wired up before). Tracks its own version/error
    independently since the two models train and drift on separate cycles.
    """
    global _best_error_gru, _version_gru

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(f"{TRUST_FORECAST_URL}/models/export")
        except httpx.HTTPError as exc:
            log.warning("could not reach trust_forecast for export", extra={"trace": {"error": str(exc)}})
            return
        if resp.status_code != 200:
            return
        client_state_dicts = resp.json()

    if len(client_state_dicts) < MIN_CLIENTS:
        log.info(
            "skipping GRU round -- not enough clients",
            extra={"trace": {"clients": len(client_state_dicts), "min_clients": MIN_CLIENTS}},
        )
        return
    if len(_validation_buffer) < MIN_VALIDATION_SAMPLES_GRU:
        log.info(
            "skipping GRU round -- not enough validation data",
            extra={"trace": {"validation_samples": len(_validation_buffer)}},
        )
        return

    merged_state_dict = fedavg.fedavg(list(client_state_dicts.values()))
    merged_model = model_gru.state_dict_from_json(merged_state_dict)
    mean_abs_error = model_gru.evaluate(merged_model, list(_validation_buffer))

    accepted = _best_error_gru is None or mean_abs_error <= _best_error_gru * (1 + ROLLBACK_TOLERANCE)
    round_summary = {
        "round": _version_gru + 1,
        "clients": sorted(client_state_dicts),
        "mean_abs_error": mean_abs_error,
        "accepted": accepted,
        "best_error_before": _best_error_gru,
    }
    _round_history_gru.append(round_summary)

    if not accepted:
        log.warning(
            "federated GRU round rejected -- merged model degraded performance, rolling back to last known-good",
            extra={"trace": round_summary},
        )
        return

    _best_error_gru = mean_abs_error
    _version_gru += 1
    log.info("federated GRU round accepted", extra={"trace": round_summary})

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            await client.post(
                f"{TRUST_FORECAST_URL}/models/import_federated",
                json={"state_dict": merged_state_dict, "version": _version_gru},
            )
        except httpx.HTTPError as exc:
            log.warning("could not push federated GRU model to trust_forecast", extra={"trace": {"error": str(exc)}})


async def _aggregation_tick() -> None:
    while True:
        await asyncio.sleep(AGGREGATION_INTERVAL_S)
        await _aggregation_round()
        await _aggregation_round_gru()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await bus.connect()
    await bus.subscribe(["behavior.features"], BehaviorFeatures, on_behavior_features)
    tick_task = asyncio.create_task(_aggregation_tick())
    log.info("federated_learning started")
    yield
    tick_task.cancel()
    await bus.close()


app = FastAPI(title="Federated Learning Service", lifespan=lifespan)


@app.get("/status")
async def get_status():
    return {
        "autoencoder": {
            "version": _version,
            "best_error": _best_error,
            "rounds": list(_round_history),
        },
        "gru": {
            "version": _version_gru,
            "best_error": _best_error_gru,
            "rounds": list(_round_history_gru),
        },
        "validation_buffer_size": len(_validation_buffer),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
