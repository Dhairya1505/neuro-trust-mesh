from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from sklearn.ensemble import IsolationForest

import autoencoder
import isolation_forest
from models import AnomalyState
from neurotrust_common import EventBus, get_logger, get_settings
from neurotrust_common.db import Database
from neurotrust_common.locks import KeyedLock
from neurotrust_common.schemas import AnomalyScore, BehaviorFeatures, SecurityRiskUpdate

SERVICE_NAME = "anomaly_detection"

# must match security_monitoring/rules.py::WATCH_THRESHOLD -- reuse the
# same "agent looks suspicious" boundary to decide when NOT to let new
# samples pollute this agent's "normal" training profile
CONTAMINATION_THRESHOLD = 0.4
CONTAMINATION_DURATION_S = 120
# must match security_monitoring/rules.py::SECURITY_SCORE_CAP -- used to
# normalize security_score onto the same ~[0,1] scale as CONTAMINATION_THRESHOLD
SECURITY_SCORE_CAP = 10.0

settings = get_settings(SERVICE_NAME)
log = get_logger(SERVICE_NAME, settings.log_level)
db = Database(settings.postgres_dsn)
bus = EventBus(settings.rabbitmq_url, SERVICE_NAME)
_agent_locks = KeyedLock()
_models: dict[str, autoencoder.Autoencoder] = {}
_iso_models: dict[str, IsolationForest] = {}
_anomaly_persistence: dict[str, autoencoder.AnomalyPersistence] = {}

# Federated fallback (ML doc §8/system doc §4.6): a merged model built
# from every agent's local weights (never raw data) by federated_learning,
# validated there before being pushed here. Used only for agents that
# haven't individually trained yet -- an agent with its own trained model
# always uses that instead; this just gives a data-sparse agent SOME
# anomaly coverage immediately instead of none.
_federated_model: autoencoder.Autoencoder | None = None
_federated_threshold: float = 0.0
_federated_version: int = 0


async def _get_or_create(session, agent_id: str, role: str = "worker") -> AnomalyState:
    state = await session.get(AnomalyState, agent_id)
    if state is None:
        state = AnomalyState(agent_id=agent_id, role=role, training_buffer=[], samples_since_retrain=0, trained=False)
        session.add(state)
        try:
            await session.flush()
        except IntegrityError:
            await session.rollback()
            state = await session.get(AnomalyState, agent_id)
    return state


def _retrain(state: AnomalyState) -> None:
    profile = autoencoder.train(state.training_buffer)
    _models[state.agent_id] = profile.model
    _iso_models[state.agent_id] = isolation_forest.train(state.training_buffer)
    state.trained = True
    state.mean_error = profile.mean_error
    state.std_error = profile.std_error
    state.threshold = profile.threshold
    state.samples_since_retrain = 0
    log.info(
        "autoencoder retrained",
        extra={
            "trace": {
                "agent_id": state.agent_id,
                "samples": len(state.training_buffer),
                "threshold": profile.threshold,
            }
        },
    )


async def on_behavior_features(event: BehaviorFeatures) -> None:
    role = "leader" if event.role_flag == 1 else "worker"
    now = datetime.now(timezone.utc)
    vector = autoencoder.vector_from_feature_dict(event.model_dump())

    async with _agent_locks.get(event.agent_id):
        async with db.session() as session:
            state = await _get_or_create(session, event.agent_id, role)
            state.role = role

            contaminated = state.contaminated_until is not None and state.contaminated_until > now
            if not contaminated:
                # Trained only on this agent's own historical "normal" data
                # (ML doc §4) -- skip buffering samples collected while the
                # agent is flagged suspicious by security_monitoring, so a
                # developing compromise doesn't get learned as "normal".
                buffer = list(state.training_buffer)
                buffer.append(vector)
                state.training_buffer = buffer[-autoencoder.TRAINING_BUFFER_CAP:]
                state.samples_since_retrain += 1

            should_retrain = len(state.training_buffer) >= autoencoder.MIN_TRAINING_SAMPLES and (
                not state.trained or state.samples_since_retrain >= autoencoder.RETRAIN_EVERY
            )
            if should_retrain:
                _retrain(state)

            state.updated_at = now
            await session.commit()

            own_trained = state.trained
            threshold = state.threshold

        if own_trained and event.agent_id in _models:
            error = autoencoder.reconstruction_error(_models[event.agent_id], vector)
            ae_anomalous = autoencoder.is_anomalous(error, threshold)
            combined_risk = autoencoder.normalize_for_fusion(error, threshold)
            used_model = "own"

            if event.agent_id in _iso_models:
                # Isolation Forest complement (Model C+): specifically
                # strong at point anomalies, confirmed offline to be where
                # the Autoencoder alone was weak (8% vs 100% detection on
                # an obvious outlier). Combined via max(), the same fusion
                # operator ML doc §6 uses for combined_security_risk.
                iso_model = _iso_models[event.agent_id]
                iso_anomalous = isolation_forest.is_anomalous(iso_model, vector)
                iso_risk = isolation_forest.normalize_for_fusion(iso_model, vector)
                raw_anomalous = ae_anomalous or iso_anomalous
                combined_risk = max(combined_risk, iso_risk)
            else:
                raw_anomalous = ae_anomalous
        elif _federated_model is not None:
            # Data-sparse agent: no local model yet, but federated_learning
            # has validated a merged model from other agents' shared
            # weights (never their raw data) -- use it as a fallback
            # rather than reporting no signal at all. Isolation Forest
            # isn't federated (ML doc §8 only federates Model B/C's neural
            # nets), so no ensemble is available for a fallback agent.
            error = autoencoder.reconstruction_error(_federated_model, vector)
            threshold = _federated_threshold
            raw_anomalous = autoencoder.is_anomalous(error, threshold)
            combined_risk = autoencoder.normalize_for_fusion(error, threshold)
            used_model = "federated"
        else:
            error, threshold, raw_anomalous, combined_risk, used_model = 0.0, 0.0, False, 0.0, "none"

        trained = used_model != "none"

        # Require a SUSTAINED elevated reading before actually flagging --
        # a 3-sigma threshold on this much calibration data still has a
        # real single-reading false-positive rate (confirmed live).
        persistence_state, anomalous = autoencoder.update_anomaly_persistence(
            _anomaly_persistence.get(event.agent_id, autoencoder.AnomalyPersistence()), raw_anomalous
        )
        _anomaly_persistence[event.agent_id] = persistence_state

    await bus.publish(
        AnomalyScore(
            agent_id=event.agent_id,
            role=role,
            reconstruction_error=error,
            threshold=threshold,
            combined_anomaly_risk=combined_risk,
            is_anomalous=anomalous,
            trained=trained,
        )
    )
    if anomalous:
        log.info(
            "anomaly detected",
            extra={
                "trace": {
                    "agent_id": event.agent_id,
                    "reconstruction_error": error,
                    "threshold": threshold,
                    "model": used_model,
                }
            },
        )


async def on_security_risk(event: SecurityRiskUpdate) -> None:
    # Gate on security_score (rule engine) alone, NOT combined_risk.
    # combined_risk = max(security_score, this service's OWN anomaly_risk)
    # -- gating contamination on it would let an agent's own borderline
    # anomaly output pause its own future training data, which can become
    # self-sustaining (elevated score -> contamination -> no corrective
    # data -> threshold never corrects -> score stays elevated). Confirmed
    # live: one clean agent had 74% of its events discarded this way.
    # security_score comes only from independent rule-engine signals
    # (missed heartbeats, task outcomes, message counts), so it can never
    # be driven by this service's own prior output.
    if event.security_score / SECURITY_SCORE_CAP <= CONTAMINATION_THRESHOLD:
        return
    async with _agent_locks.get(event.agent_id):
        async with db.session() as session:
            state = await _get_or_create(session, event.agent_id, event.role)
            state.contaminated_until = datetime.now(timezone.utc) + timedelta(seconds=CONTAMINATION_DURATION_S)
            await session.commit()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await db.create_all()

    # Torch models aren't persisted -- rebuild them from each agent's
    # stored training buffer so a restart doesn't lose a fitted profile.
    async with db.session() as session:
        result = await session.execute(select(AnomalyState))
        for state in result.scalars().all():
            if len(state.training_buffer) >= autoencoder.MIN_TRAINING_SAMPLES:
                _retrain(state)
        await session.commit()

    await bus.connect()
    await bus.subscribe(["behavior.features"], BehaviorFeatures, on_behavior_features)
    await bus.subscribe(["security.risk_update"], SecurityRiskUpdate, on_security_risk)
    log.info("anomaly_detection started")
    yield
    await bus.close()
    await db.dispose()


app = FastAPI(title="Anomaly Detection Service (Autoencoder)", lifespan=lifespan)


@app.get("/anomaly/{agent_id}")
async def get_anomaly_state(agent_id: str):
    async with db.session() as session:
        state = await session.get(AnomalyState, agent_id)
        if state is None:
            raise HTTPException(404, f"no anomaly state for agent {agent_id}")
        return {
            "agent_id": agent_id,
            "role": state.role,
            "trained": state.trained,
            "sample_count": len(state.training_buffer),
            "mean_error": state.mean_error,
            "std_error": state.std_error,
            "threshold": state.threshold,
            "contaminated_until": state.contaminated_until.isoformat() if state.contaminated_until else None,
        }


@app.get("/models/export")
async def export_models():
    """federated_learning's per-round pull -- weights only, matching
    system doc §4.6: 'sends only model weight updates (not raw logs)'.
    """
    return {agent_id: autoencoder.state_dict_to_json(model) for agent_id, model in _models.items()}


class ImportFederatedRequest(BaseModel):
    state_dict: dict
    threshold: float
    version: int


@app.post("/models/import_federated")
async def import_federated_model(body: ImportFederatedRequest):
    global _federated_model, _federated_threshold, _federated_version
    if body.version <= _federated_version:
        return {"accepted": False, "reason": "not newer than current federated version", "current_version": _federated_version}
    _federated_model = autoencoder.state_dict_from_json(body.state_dict)
    _federated_threshold = body.threshold
    _federated_version = body.version
    log.info("federated model imported", extra={"trace": {"version": body.version, "threshold": body.threshold}})
    return {"accepted": True, "version": _federated_version}


@app.get("/health")
async def health():
    return {"status": "ok"}
