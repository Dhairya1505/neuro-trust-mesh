import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI

import rules
from neurotrust_common import EventBus, get_logger, get_settings
from neurotrust_common import fusion
from neurotrust_common.schemas import (
    AnomalyScore,
    CollusionFlag,
    DecisionAction,
    SecurityRiskUpdate,
    TrustForecast,
    TrustScore,
)

SERVICE_NAME = "decision_engine"
COOLDOWN_S = 60  # prevents leadership flapping / repeat redistribution spam
ISOLATE_COOLDOWN_S = 20  # security flags act fast (doc §4.5) -- shorter, not zero, just anti-spam
GROUP_REVIEW_COOLDOWN_S = 120  # collusion_detection already cools down per-group; this just protects per-agent spam
RECENT_DECISIONS_CAP = 100  # dashboard's "recent activity" feed

settings = get_settings(SERVICE_NAME)
log = get_logger(SERVICE_NAME, settings.log_level)
bus = EventBus(settings.rabbitmq_url, SERVICE_NAME)

_last_action_at: dict[tuple[str, str], float] = {}
_recent_decisions: deque = deque(maxlen=RECENT_DECISIONS_CAP)

# Latest per-agent inputs for the fusion layer (ML doc §5) -- decision_engine
# already subscribes to trust/forecast; anomaly is added alongside these
# purely to feed the combined signal below, independent of the existing
# per-event decision rules.
_latest_trust: dict[str, TrustScore] = {}
_latest_forecast: dict[str, TrustForecast] = {}
_latest_anomaly: dict[str, AnomalyScore] = {}


async def _publish_fusion_signal(agent_id: str) -> None:
    signal = fusion.compute_final_risk_signal(
        _latest_trust.get(agent_id), _latest_forecast.get(agent_id), _latest_anomaly.get(agent_id)
    )
    if signal is not None:
        await bus.publish(signal)


def _cooldown_ok(agent_id: str, action: str, cooldown_s: float = COOLDOWN_S) -> bool:
    key = (agent_id, action)
    now = time.monotonic()
    last = _last_action_at.get(key)
    if last is not None and now - last < cooldown_s:
        return False
    _last_action_at[key] = now
    return True


async def _publish_decision(agent_id: str, role: str, action: str, rule: str, inputs: dict) -> None:
    await bus.publish(DecisionAction(agent_id=agent_id, role=role, action=action, rule=rule, inputs=inputs))
    if action != "none":
        # dashboard's "recent activity" feed -- only genuine actions, not
        # every routine "no threshold breached" evaluation
        _recent_decisions.appendleft(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "agent_id": agent_id,
                "role": role,
                "action": action,
                "rule": rule,
                "inputs": inputs,
            }
        )


async def on_trust_score(event: TrustScore) -> None:
    _latest_trust[event.agent_id] = event
    await _publish_fusion_signal(event.agent_id)

    decision = rules.evaluate(event.role, event.trust_score)

    if decision.action != "none" and not _cooldown_ok(event.agent_id, decision.action):
        return

    await _publish_decision(
        event.agent_id,
        event.role,
        decision.action,
        decision.rule,
        {"trust_score": event.trust_score, "confidence": event.confidence},
    )
    if decision.action != "none":
        log.info(
            "decision made",
            extra={
                "trace": {
                    "agent_id": event.agent_id,
                    "action": decision.action,
                    "rule": decision.rule,
                    "trust_score": event.trust_score,
                }
            },
        )


async def on_security_risk(event: SecurityRiskUpdate) -> None:
    decision = rules.evaluate_security(event.combined_risk)
    if decision.action == "none":
        return
    if not _cooldown_ok(event.agent_id, decision.action, ISOLATE_COOLDOWN_S):
        return

    await _publish_decision(
        event.agent_id,
        event.role,
        decision.action,
        decision.rule,
        {"security_score": event.security_score, "combined_risk": event.combined_risk},
    )
    log.info(
        "decision made",
        extra={
            "trace": {
                "agent_id": event.agent_id,
                "action": decision.action,
                "rule": decision.rule,
                "combined_risk": event.combined_risk,
            }
        },
    )


async def on_anomaly_score(event: AnomalyScore) -> None:
    _latest_anomaly[event.agent_id] = event
    await _publish_fusion_signal(event.agent_id)


async def on_trust_forecast(event: TrustForecast) -> None:
    _latest_forecast[event.agent_id] = event
    if event.trained:
        await _publish_fusion_signal(event.agent_id)

    if not event.trained:
        return
    confidence = 1.0 / max(event.predicted_std, 1e-3)
    decision = rules.evaluate_forecast(event.role, event.predicted_mean, confidence)
    if decision.action == "none":
        return
    if not _cooldown_ok(event.agent_id, decision.action):
        return

    await _publish_decision(
        event.agent_id,
        event.role,
        decision.action,
        decision.rule,
        {"predicted_mean": event.predicted_mean, "predicted_std": event.predicted_std},
    )
    log.info(
        "decision made",
        extra={
            "trace": {
                "agent_id": event.agent_id,
                "action": decision.action,
                "rule": decision.rule,
                "predicted_mean": event.predicted_mean,
            }
        },
    )


async def on_collusion_flag(event: CollusionFlag) -> None:
    """system doc §4.7: collusion_flag == True -> route to human review.
    NEVER auto-isolate on collusion alone -- highest false-positive risk
    rule in the system (ML doc §7), so this always resolves to
    'flag_for_review', never 'isolate', regardless of severity.
    """
    rule = f"collusion group persisted >= {event.windows_persisted} windows, severity={event.group_severity:.2f}"
    for agent_id in event.group:
        if not _cooldown_ok(agent_id, "flag_for_review", GROUP_REVIEW_COOLDOWN_S):
            continue
        await _publish_decision(
            agent_id,
            event.roles.get(agent_id, "worker"),
            "flag_for_review",
            rule,
            {"group": event.group, "group_severity": event.group_severity, "windows_persisted": event.windows_persisted},
        )
    log.warning(
        "decision made",
        extra={
            "trace": {
                "action": "flag_for_review",
                "group": event.group,
                "rule": rule,
                "group_severity": event.group_severity,
            }
        },
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    await bus.connect()
    await bus.subscribe(["trust.score"], TrustScore, on_trust_score)
    await bus.subscribe(["security.risk_update"], SecurityRiskUpdate, on_security_risk)
    await bus.subscribe(["trust.forecast"], TrustForecast, on_trust_forecast)
    await bus.subscribe(["anomaly.score"], AnomalyScore, on_anomaly_score)
    await bus.subscribe(["collusion.flag"], CollusionFlag, on_collusion_flag)
    log.info("decision_engine started")
    yield
    await bus.close()


app = FastAPI(title="Autonomous Decision Engine", lifespan=lifespan)


@app.get("/recent_decisions")
async def get_recent_decisions(limit: int = 50):
    return list(_recent_decisions)[:limit]


@app.get("/health")
async def health():
    return {"status": "ok"}
