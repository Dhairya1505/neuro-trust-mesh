import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

import gru
from models import ForecastState
from neurotrust_common import EventBus, get_logger, get_settings
from neurotrust_common.db import Database
from neurotrust_common.locks import KeyedLock
from neurotrust_common.schemas import BehaviorFeatures, TrustForecast

SERVICE_NAME = "trust_forecast"
SNAPSHOT_INTERVAL_S = 5  # matches the heartbeat cadence (system doc §4.1) used elsewhere as the sampling tick

settings = get_settings(SERVICE_NAME)
log = get_logger(SERVICE_NAME, settings.log_level)
db = Database(settings.postgres_dsn)
bus = EventBus(settings.rabbitmq_url, SERVICE_NAME)
_agent_locks = KeyedLock()
_models: dict[str, gru.GRUForecaster] = {}

# Federated version counter (ML doc §8/system doc §4.6) -- mirrors
# anomaly_detection's pattern. federated_learning validates a merged model
# before pushing it here; the import is a version-gated accept, matching
# the autoencoder's rollback-friendly contract.
_federated_version: int = 0

# BehaviorFeatures fires several times per tick (heartbeat/task_outcome/
# metrics_sample each publish it) -- cache only the latest per agent here;
# the periodic snapshot tick below is what actually samples it into the
# regularly-spaced timestep buffer the GRU needs.
_latest_features: dict[str, dict] = {}


async def on_behavior_features(event: BehaviorFeatures) -> None:
    _latest_features[event.agent_id] = event.model_dump()


async def _get_or_create(session, agent_id: str, role: str = "worker") -> ForecastState:
    state = await session.get(ForecastState, agent_id)
    if state is None:
        state = ForecastState(agent_id=agent_id, role=role, timestep_buffer=[], samples_since_finetune=0, trained=False)
        session.add(state)
        try:
            await session.flush()
        except IntegrityError:
            await session.rollback()
            state = await session.get(ForecastState, agent_id)
    return state


def _train_or_finetune(state: ForecastState) -> None:
    buffer = state.timestep_buffer
    if not state.trained:
        _models[state.agent_id] = gru.train(buffer)
        state.trained = True
        state.samples_since_finetune = 0
        log.info("gru trained", extra={"trace": {"agent_id": state.agent_id, "samples": len(buffer)}})
    elif state.samples_since_finetune >= gru.FINE_TUNE_EVERY_TIMESTEPS:
        _models[state.agent_id] = gru.fine_tune(_models[state.agent_id], buffer)
        state.samples_since_finetune = 0
        log.info("gru fine-tuned", extra={"trace": {"agent_id": state.agent_id, "samples": len(buffer)}})


async def _snapshot_tick() -> None:
    while True:
        await asyncio.sleep(SNAPSHOT_INTERVAL_S)
        for agent_id, features in list(_latest_features.items()):
            role = "leader" if features["role_flag"] == 1 else "worker"
            vector = [
                features["task_completion_rate"],
                features["avg_response_latency"],
                features["comms_frequency"],
                features["resource_utilization"],
                features["bayesian_trust_score"] if features["bayesian_trust_score"] is not None else 0.5,
                features["bayesian_confidence"] if features["bayesian_confidence"] is not None else 0.5,
                features["role_flag"],
            ]

            async with _agent_locks.get(agent_id):
                async with db.session() as session:
                    state = await _get_or_create(session, agent_id, role)
                    state.role = role
                    buffer = list(state.timestep_buffer)
                    buffer.append(vector)
                    state.timestep_buffer = buffer[-gru.BUFFER_CAP:]
                    state.samples_since_finetune += 1

                    min_len = gru.SEQUENCE_LENGTH + gru.HORIZON_K + gru.MIN_TRAINING_WINDOWS - 1
                    if len(state.timestep_buffer) >= min_len:
                        _train_or_finetune(state)

                    state.updated_at = datetime.now(timezone.utc)
                    await session.commit()

                    trained = state.trained
                    window = state.timestep_buffer[-gru.SEQUENCE_LENGTH:]

            if trained and agent_id in _models and len(window) == gru.SEQUENCE_LENGTH:
                pred = gru.predict(_models[agent_id], window)
                await bus.publish(
                    TrustForecast(
                        agent_id=agent_id,
                        role=role,
                        predicted_mean=pred.mean,
                        predicted_std=pred.std,
                        horizon_k=gru.HORIZON_K,
                        trained=True,
                    )
                )


@asynccontextmanager
async def lifespan(_: FastAPI):
    await db.create_all()

    # Torch models aren't persisted -- rebuild them from each agent's
    # stored timestep buffer so a restart doesn't lose a fitted model.
    async with db.session() as session:
        result = await session.execute(select(ForecastState))
        min_len = gru.SEQUENCE_LENGTH + gru.HORIZON_K + gru.MIN_TRAINING_WINDOWS - 1
        for state in result.scalars().all():
            if len(state.timestep_buffer) >= min_len:
                _models[state.agent_id] = gru.train(state.timestep_buffer)
                state.trained = True
        await session.commit()

    await bus.connect()
    await bus.subscribe(["behavior.features"], BehaviorFeatures, on_behavior_features)
    snapshot_task = asyncio.create_task(_snapshot_tick())
    log.info("trust_forecast started")
    yield
    snapshot_task.cancel()
    await bus.close()
    await db.dispose()


app = FastAPI(title="Trust Forecast Service (GRU)", lifespan=lifespan)


@app.get("/forecast/{agent_id}")
async def get_forecast_state(agent_id: str):
    async with db.session() as session:
        state = await session.get(ForecastState, agent_id)
        if state is None:
            raise HTTPException(404, f"no forecast state for agent {agent_id}")
        result = {
            "agent_id": agent_id,
            "role": state.role,
            "trained": state.trained,
            "sample_count": len(state.timestep_buffer),
        }
        if state.trained and agent_id in _models and len(state.timestep_buffer) >= gru.SEQUENCE_LENGTH:
            pred = gru.predict(_models[agent_id], state.timestep_buffer[-gru.SEQUENCE_LENGTH:])
            result["predicted_mean"] = pred.mean
            result["predicted_std"] = pred.std
            result["trend_confidence"] = gru.trend_confidence(pred.std)
        return result


@app.get("/models/export")
async def export_models():
    """federated_learning's per-round pull -- weights only (system doc
    §4.6). Model B is federated alongside Model C (ML doc §8).
    """
    return {agent_id: gru.state_dict_to_json(m) for agent_id, m in _models.items()}


class ImportFederatedRequest(BaseModel):
    state_dict: dict
    version: int


@app.post("/models/import_federated")
async def import_federated_model(body: ImportFederatedRequest):
    global _federated_version
    if body.version <= _federated_version:
        return {"accepted": False, "reason": "not newer than current federated version", "current_version": _federated_version}
    _federated_version = body.version
    log.info("federated model version accepted", extra={"trace": {"version": body.version}})
    return {"accepted": True, "version": _federated_version}


@app.get("/health")
async def health():
    return {"status": "ok"}
