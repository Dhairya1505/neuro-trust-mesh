import asyncio
import itertools
import time
from contextlib import asynccontextmanager
from collections import deque
from datetime import datetime, timezone

from fastapi import FastAPI

import signals
from neurotrust_common import EventBus, get_logger, get_settings
from neurotrust_common.schemas import (
    AgentHeartbeat,
    AgentMetricsSample,
    AgentTaskOutcome,
    AnomalyScore,
    BehaviorFeatures,
    CollusionFlag,
    CollusionSignalScores,
)

SERVICE_NAME = "collusion_detection"
SNAPSHOT_INTERVAL_S = 5  # matches the heartbeat cadence used as the sampling tick elsewhere
ANALYSIS_INTERVAL_S = 30  # one "window" (ML doc §7) = one analysis pass
HISTORY_CAP = 120
GROUP_COOLDOWN_S = 180  # don't re-flag an unchanged group every single window once persisted
PENDING_REVIEWS_CAP = 50

settings = get_settings(SERVICE_NAME)
log = get_logger(SERVICE_NAME, settings.log_level)
bus = EventBus(settings.rabbitmq_url, SERVICE_NAME)

# Everything here is in-memory only -- a restart loses recent correlation
# history and starts the persistence counters over, which is an acceptable
# tradeoff for a signal that's inherently about RECENT windows, not
# long-term memory (unlike Bayesian trust). Keeps this service free of the
# get-or-create/locking machinery every DB-backed service in this system
# has needed fixes for.
_history: dict[str, deque] = {}
_latest_anomaly: dict[str, float] = {}
_latest_features: dict[str, dict] = {}
_latest_role: dict[str, str] = {}
_latest_task: dict[str, str | None] = {}
_comms_since_snapshot: dict[str, int] = {}
_latest_outcome: dict[str, float] = {}
_pair_trackers: dict[tuple[str, str], signals.PairTracker] = {}
_group_last_flagged: dict[tuple[str, ...], float] = {}
_pending_reviews: deque = deque(maxlen=PENDING_REVIEWS_CAP)


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


async def on_heartbeat(event: AgentHeartbeat) -> None:
    _latest_role[event.agent_id] = event.role
    _latest_task[event.agent_id] = event.current_task


async def on_anomaly_score(event: AnomalyScore) -> None:
    _latest_anomaly[event.agent_id] = event.reconstruction_error
    _latest_role[event.agent_id] = event.role


async def on_behavior_features(event: BehaviorFeatures) -> None:
    _latest_features[event.agent_id] = event.model_dump()
    _latest_role[event.agent_id] = "leader" if event.role_flag == 1 else "worker"


async def on_metrics_sample(event: AgentMetricsSample) -> None:
    _comms_since_snapshot[event.agent_id] = _comms_since_snapshot.get(event.agent_id, 0) + event.comms_count


async def on_task_outcome(event: AgentTaskOutcome) -> None:
    _latest_outcome[event.agent_id] = 1.0 if event.outcome == "success" else 0.0


async def _snapshot_tick() -> None:
    while True:
        await asyncio.sleep(SNAPSHOT_INTERVAL_S)
        now = time.monotonic()
        known_agents = set(_latest_features) | set(_latest_anomaly)
        for agent_id in known_agents:
            sample = signals.AgentSample(
                ts=now,
                anomaly_error=_latest_anomaly.get(agent_id, 0.0),
                feature_vector=_latest_features.get(agent_id, {}),
                comms_count=_comms_since_snapshot.pop(agent_id, 0),
                outcome_success=_latest_outcome.get(agent_id, 0.5),
                role=_latest_role.get(agent_id),
                current_task=_latest_task.get(agent_id),
            )
            _history.setdefault(agent_id, deque(maxlen=HISTORY_CAP)).append(sample)


async def _analysis_tick() -> None:
    while True:
        await asyncio.sleep(ANALYSIS_INTERVAL_S)
        agent_ids = list(_history)
        persisted_pairs: list[tuple[str, str]] = []
        pair_scores: dict[tuple[str, str], signals.CollusionScores] = {}

        for a, b in itertools.combinations(sorted(agent_ids), 2):
            key = _pair_key(a, b)
            scores = signals.compute_pair_signals(list(_history[a]), list(_history[b]))
            if scores is None:
                continue
            tracker = _pair_trackers.get(key, signals.PairTracker())
            tracker = signals.update_persistence(tracker, scores.is_elevated())
            _pair_trackers[key] = tracker

            if signals.is_persisted(tracker):
                persisted_pairs.append(key)
                pair_scores[key] = scores
                if scores.is_elevated():
                    log.info(
                        "collusion pair persisted",
                        extra={
                            "trace": {
                                "pair": key,
                                "elevated_signals": scores.elevated_count(),
                                "score": scores.weighted_score(),
                            }
                        },
                    )

        for group in signals.connected_components(persisted_pairs):
            group_key = tuple(sorted(group))
            last_flagged = _group_last_flagged.get(group_key)
            if last_flagged is not None and time.monotonic() - last_flagged < GROUP_COOLDOWN_S:
                continue
            _group_last_flagged[group_key] = time.monotonic()

            group_pairs = [p for p in persisted_pairs if p[0] in group and p[1] in group]
            group_severity = sum(pair_scores[p].weighted_score() for p in group_pairs) / len(group_pairs)
            windows_persisted = min(_pair_trackers[p].consecutive_hits for p in group_pairs)

            flag = CollusionFlag(
                group=list(group_key),
                roles={agent_id: _latest_role.get(agent_id, "worker") for agent_id in group_key},
                group_severity=group_severity,
                pair_scores={
                    f"{p[0]}|{p[1]}": CollusionSignalScores(
                        timing_correlation=pair_scores[p].timing_correlation,
                        behavior_similarity=pair_scores[p].behavior_similarity,
                        comms_anomaly=pair_scores[p].comms_anomaly,
                        outcome_corroboration=pair_scores[p].outcome_corroboration,
                    )
                    for p in group_pairs
                },
                windows_persisted=windows_persisted,
            )
            _pending_reviews.append(flag.model_dump(mode="json"))
            await bus.publish(flag)
            log.warning(
                "collusion group flagged -- routed to human review, NOT auto-isolated",
                extra={"trace": {"group": group_key, "group_severity": group_severity, "windows_persisted": windows_persisted}},
            )


@asynccontextmanager
async def lifespan(_: FastAPI):
    await bus.connect()
    await bus.subscribe(["agent.heartbeat"], AgentHeartbeat, on_heartbeat)
    await bus.subscribe(["anomaly.score"], AnomalyScore, on_anomaly_score)
    await bus.subscribe(["behavior.features"], BehaviorFeatures, on_behavior_features)
    await bus.subscribe(["agent.metrics_sample"], AgentMetricsSample, on_metrics_sample)
    await bus.subscribe(["agent.task_outcome"], AgentTaskOutcome, on_task_outcome)
    snapshot_task = asyncio.create_task(_snapshot_tick())
    analysis_task = asyncio.create_task(_analysis_tick())
    log.info("collusion_detection started")
    yield
    snapshot_task.cancel()
    analysis_task.cancel()
    await bus.close()


app = FastAPI(title="Collusion Detection Service", lifespan=lifespan)


@app.get("/pairs")
async def get_pairs():
    """Debug/inspection surface -- current per-pair signal state."""
    result = []
    for a, b in itertools.combinations(sorted(_history), 2):
        key = _pair_key(a, b)
        scores = signals.compute_pair_signals(list(_history[a]), list(_history[b]))
        tracker = _pair_trackers.get(key, signals.PairTracker())
        result.append(
            {
                "pair": list(key),
                "scores": scores.__dict__ if scores else None,
                "elevated_this_window": scores.is_elevated() if scores else False,
                "consecutive_hits": tracker.consecutive_hits,
                "persisted": signals.is_persisted(tracker),
            }
        )
    return result


@app.get("/reviews")
async def get_pending_reviews():
    """Human-review queue (system doc §4.5: collusion routes to human
    review, never auto-isolate). A full approval/override API is Control
    Center scope -- not yet built (system doc §5, listed as future work
    alongside the XAI dashboard) -- this is the read-only surface for it.
    """
    return list(_pending_reviews)


@app.get("/health")
async def health():
    return {"status": "ok"}
