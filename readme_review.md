# NeuroTrust Mesh — Project Review Document

> A self-evolving Cognitive Digital Twin framework for predictive trust and autonomous
> collaboration in hierarchical multi-agent systems.

---

## Table of Contents

1. [Agenda](#1-agenda)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [Objectives](#4-objectives)
5. [Proposed Methodology](#5-proposed-methodology)
6. [System Architecture](#6-system-architecture)
7. [Microservices — Detailed Breakdown](#7-microservices--detailed-breakdown)
8. [Event Bus and Communication](#8-event-bus-and-communication)
9. [Machine Learning Models](#9-machine-learning-models)
10. [Data Flow — Worked Example](#10-data-flow--worked-example)
11. [Simulation and Fault Injection](#11-simulation-and-fault-injection)
12. [Deployment and Infrastructure](#12-deployment-and-infrastructure)
13. [API Reference](#13-api-reference)
14. [Testing Strategy](#14-testing-strategy)
15. [Development Milestones](#15-development-milestones)
16. [Known Limitations](#16-known-limitations)
17. [Conclusion](#17-conclusion)

---

## 1. Agenda

| # | Topic | Description |
|---|-------|-------------|
| 1 | Introduction | Background on multi-agent systems and the trust problem |
| 2 | Problem Statement | Core challenges in autonomous agent security and reliability |
| 3 | Objectives | Goals and success criteria of this project |
| 4 | Proposed Methodology | Approach, models, and architectural patterns used |
| 5 | System Architecture | Full microservices design with event-driven communication |
| 6 | How It Works | Step-by-step signal flow from heartbeat to corrective action |
| 7 | ML Models | Bayesian trust, GRU forecasting, Anomaly detection ensemble |
| 8 | Results and Demo | Live dashboard, API outputs, decision logs |
| 9 | Limitations and Future Work | Known gaps and planned extensions |
| 10 | Conclusion | Summary of achievements |

---

## 2. Introduction

### Background

Modern distributed systems increasingly rely on autonomous agents to perform tasks
independently — whether in robotics, cloud orchestration, IoT networks, or AI pipelines.
As these systems scale, a new class of challenge emerges: **how do you trust an agent
that operates autonomously, potentially fails silently, or worse, acts maliciously?**

**NeuroTrust Mesh** is a research-grade framework that answers this question by combining:

- **Cognitive Digital Twins (CDT)** — a persistent mirror of each agent's behavioral state
- **Multi-model trust scoring** — reactive (Bayesian) and predictive (GRU neural network)
- **Anomaly detection ensemble** — Autoencoder + Isolation Forest, calibrated per agent
- **Rule-based security monitoring** — explainable, no black-box decisions
- **Federated learning** — shared model improvements without centralizing raw data
- **Autonomous corrective actions** — isolate, reassign, redistribute, flag for human review

The system is built as **16 real production-grade microservices** communicating over an
asynchronous event bus (RabbitMQ), backed by PostgreSQL, and driven by a Mesa-based
multi-agent simulation that injects real faults and coordinated collusion.

### Key Design Philosophy

> **Explainability is a first-class constraint — every corrective action is logged
> with the exact rule and evidence that triggered it.**

---

## 3. Problem Statement

### Core Challenges

In hierarchical multi-agent systems, the following problems arise that existing
solutions do not adequately address:

#### 3.1 Silent Failures
Agents may fail to complete tasks, stop heartbeating, or produce incorrect outputs
without any explicit error signal. Traditional monitoring detects only binary up/down
status — not *degrading* trustworthiness.

#### 3.2 Insider Threats and Collusion
Compromised agents can behave normally in isolation but coordinate with other agents
to undermine system integrity. Detecting this requires cross-agent correlation, not
per-agent monitoring alone.

#### 3.3 Lack of Predictive Awareness
Reactive systems only act *after* a failure occurs. There is no mechanism to
predict which agents are *trending toward* failure and pre-emptively reduce their
responsibility.

#### 3.4 Black-Box Decisions
Most ML-based security systems produce a score with no explanation. In safety-critical
autonomous systems, operators need to know *why* an agent was isolated, not just that it was.

#### 3.5 Model Staleness in Distributed Environments
Per-agent ML models trained locally have limited data. Without cross-agent learning,
new agents start with no baseline and are vulnerable to false flags or blind spots.

#### 3.6 Permanent Isolation
Once isolated, most systems have no mechanism to automatically reinstate a healthy
agent — leading to compounding resource loss as false-positive isolations accumulate.

---

## 4. Objectives

### Primary Objectives

1. Build a real-time trust monitoring system for hierarchical multi-agent environments
   that continuously scores each agent's trustworthiness using multiple independent signals.

2. Detect anomalous and malicious behavior using a calibrated ensemble of ML models
   (Autoencoder + Isolation Forest) operating per-agent against that agent's own baseline.

3. Predict trust degradation before failures occur using a GRU-based time-series
   forecaster with genuine confidence intervals.

4. Detect coordinated collusion between agents using cross-agent behavioral correlation
   signals, persistence-gated, routed only to human review.

5. Automate corrective actions — leader reassignment, task redistribution, and agent
   isolation — in a fully explainable, logged decision pipeline.

6. Enable federated model improvement via FedAvg aggregation across agents without
   centralizing raw behavioral data, with automatic rollback on degradation.

7. Provide automatic recovery — isolated agents are automatically reinstated once
   their risk profile sustains below threshold for a configurable duration.

### Secondary Objectives

- Maintain full explainability: every decision logged with its triggering rule and evidence
- Achieve a production-grade microservices deployment, runnable with one command via Docker
- Provide a live dashboard aggregating all signals, decisions, and recovery ETAs
- Implement comprehensive unit tests covering all pure-function ML and logic cores

---

## 5. Proposed Methodology

### 5.1 Architectural Pattern: Event-Driven Microservices

Rather than a monolithic system, each concern is separated into an independent service.
Services communicate exclusively via a **RabbitMQ topic exchange** (`neurotrust.events`),
with routing keys like `agent.heartbeat`, `trust.score`, `decision.action`.

**Key principle:** State-mutating actions (role changes, isolation, task moves) flow
through the event bus. Read-only queries stay REST. This prevents tight coupling.

### 5.2 Trust Scoring: Multi-Model Fusion

Two independent trust models run in parallel:

| Model | Type | Signal | Strength |
|-------|------|--------|----------|
| **Model A** — Bayesian Beta | Reactive | Current trust score | Transparent, evidence-weighted |
| **Model B** — GRU Neural Net | Predictive | Future trust trend | Early warning, confidence-gated |

Their outputs are **never merged** — each triggers the decision engine independently
via its own threshold and cooldown, so neither can mask the other.

### 5.3 Anomaly Detection: Ensemble with Calibrated Threshold

**Model C** is an ensemble of:
- **Autoencoder** (7 to 16 to 4 to 16 to 7): detects behavioral drift from an agent's own history
- **Isolation Forest**: complements the autoencoder specifically for point anomalies

Threshold set via **k-fold cross-validation**. Requires **2 consecutive elevated readings**
before flagging (persistence smoothing).

### 5.4 Security Rules: Explainable Rule Engine

Eight rule types in `security_monitoring`, each producing an independent severity score.
Combined risk = `max(security_score/10, anomaly_ensemble_risk)`.

### 5.5 Collusion Detection: Conservative Multi-Signal Correlation

Four signals per agent pair. Requires >=2 of 4 signals elevated, persisted across >=3
consecutive windows. **Never triggers auto-isolation** — always routes to human review.

### 5.6 Decision Engine: Single Point of Action

Only one service ever decides an action — `decision_engine`. It maintains independent
cooldowns per trigger type and logs every decision with its literal rule string and
triggering input values.

### 5.7 Federated Learning: FedAvg with Validation and Rollback

FedAvg of autoencoder weights across all trained agents. Validates against held-out
cross-agent samples. Only pushes to production if within tolerance of last known-good model.

---

## 6. System Architecture

### 6.1 High-Level Architecture

```
SIMULATION LAYER
  mesa_sim: 5 workers + 1 leader, fault injection, collusion injection
       |
       | heartbeats, task outcomes, metrics
       v
AGENT CORE LAYER
  agent_registry --> cdt_service --> behavior_analysis
  (identity/status)  (digital twin)  (7-dim feature vector)
       |                                     |
       | agent.heartbeat             behavior.features
       v                                     v
TRUST MODELS                    SECURITY AND ANOMALY
  trust_prediction               anomaly_detection (Autoencoder + IsoForest)
  (Bayesian Beta)                security_monitoring (8 rules + fusion)
  trust_forecast                 collusion_detection (cross-agent correlation)
  (GRU Network)                  federated_learning (FedAvg + rollback)
       |                                     |
       | trust.score, trust.forecast         | security.risk_update, collusion.flag
       +-------------------------------------+
                          |
                          v
               DECISION ENGINE (single point of action)
                rule table, full audit log, cooldowns
                          |
            +-------------+-------------+
            v             v             v
  leader_reassignment  task_redistribution  agent_isolation
                                            + auto-reinstatement
                                                    |
                                           agent_registry (state update)

FRONTEND
  dashboard (port 8000) -- live UI, all signals in one view
  api_gateway (port 8011) -- single external entry point
```

### 6.2 Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.11 |
| Web Framework | FastAPI + Uvicorn | >=0.111 |
| Message Broker | RabbitMQ (topic exchange) | 3.13-management |
| Database | PostgreSQL (async SQLAlchemy) | 16-alpine |
| ML Framework | PyTorch (CPU) | 2.14+ |
| Anomaly Detection | scikit-learn (Isolation Forest) | 1.9+ |
| Agent Simulation | Mesa | >=2.3, <3.0 |
| HTTP Client | httpx (async) | >=0.27 |
| Schema Validation | Pydantic v2 | >=2.7 |
| Containerization | Docker + Docker Compose | v5.0+ |

### 6.3 Database Schema

PostgreSQL `agents` table:

| Column | Type | Description |
|--------|------|-------------|
| agent_id | String (PK) | Unique agent identifier |
| role | String | "leader" or "worker" |
| capabilities | JSON | List of capability strings |
| current_task | String (nullable) | Active task name |
| status | String | "active" or "isolated" |
| missed_beats | Integer | Consecutive missed heartbeats |
| last_heartbeat | DateTime (TZ) | Last received heartbeat |
| registered_at | DateTime (TZ) | First registration timestamp |

### 6.4 Service Port Map

| Port | Service | Role |
|------|---------|------|
| 8000 | dashboard | Live UI |
| 8001 | agent_registry | Agent identity and heartbeat source of truth |
| 8002 | cdt_service | Cognitive Digital Twin |
| 8003 | behavior_analysis | 7-dim feature vector computation |
| 8004 | trust_prediction | Bayesian trust scoring |
| 8005 | decision_engine | Action decision and audit logging |
| 8006 | security_monitoring | Rule engine + anomaly fusion |
| 8007 | anomaly_detection | Autoencoder + Isolation Forest |
| 8008 | trust_forecast | GRU trust forecaster |
| 8009 | collusion_detection | Cross-agent correlation |
| 8010 | federated_learning | FedAvg aggregation |
| 8011 | api_gateway | Single external entry point |
| 8012 | agent_isolation | Isolation execution + recovery |
| 5672 | RabbitMQ | AMQP message bus |
| 15672 | RabbitMQ Management | Admin UI (guest/guest) |
| 5432 | PostgreSQL | Relational database |

---

## 7. Microservices — Detailed Breakdown

### 7.1 agent_registry (Port 8001)
Source of truth for every agent's identity, role, status, and current task.
- Agents register via `POST /agents/register`
- Background monitor flags agents silent for >3 heartbeats
- Status changes applied via event subscriptions — never makes decisions, only stores state

### 7.2 cdt_service (Port 8002)
Maintains a Cognitive Digital Twin per agent with:
- **Latest snapshot**: current role, task, trust, feature vector
- **Short-term memory**: rolling window of the last ~50 events
- **Long-term memory**: compressed outcome counters
- Continues receiving data even for isolated agents (observation mode)

### 7.3 behavior_analysis (Port 8003)
Computes the 7-dimensional behavioral feature vector:

| Dimension | Description |
|-----------|-------------|
| task_completion_rate | Ratio of completed vs. attempted tasks |
| avg_response_latency | Z-score normalized against agent's own history |
| comms_frequency | Message rate per heartbeat interval |
| resource_utilization | Z-score normalized against agent's own history |
| bayesian_trust_score | Current Bayesian trust score |
| bayesian_confidence | Confidence interval width of the Beta distribution |
| role_flag | 1 for leader, 0 for worker |

**Key decision:** Z-score + logistic squash instead of min-max. Min-max caused
0.75->0.47 swings in 3 seconds for healthy agents, destabilizing all downstream models.

### 7.4 trust_prediction (Port 8004)
Bayesian (Beta distribution) reactive trust scoring — Model A.
- Counters: alpha (successes), beta (failures), decayed each tick (lambda=0.97)
- Under security watch: decay rate tightened to lambda=0.90
- Evidence weights: missed heartbeat=1.0, failed task=3.0, security flagged=5.0
- Trust score = alpha / (alpha + beta)
- Thresholds: leaders require >0.65, workers require >0.45
- Plain-English explanation generated from the same counters

### 7.5 decision_engine (Port 8005)
The ONLY service that decides corrective actions. Rule table:

| Trigger | Threshold | Action | Cooldown |
|---------|-----------|--------|----------|
| Reactive trust | leader <0.65 / worker <0.45 | reassign/redistribute/isolate | 30s |
| Predictive forecast | predicted mean < threshold AND confidence high | pre-emptive flag | 60s |
| Security risk | combined_risk > 0.7 | isolate (bypasses cooldown) | 10s |
| Collusion flag | persisted collusion group | flag_for_review ONLY | 120s |

Every decision logged with: agent_id, action, rule (literal string), inputs (exact numeric values).

### 7.6 security_monitoring (Port 8006)
Rule-based security engine:

| Rule | Status | Description |
|------|--------|-------------|
| silent_agent | Implemented | Agent exceeded missed heartbeat threshold |
| message_flooding | Implemented | Comms rate burst above role-based threshold |
| resource_spike | Implemented | Sudden resource deviation from own baseline |
| security_flagged_outcome | Implemented | Task completed with security flag |
| replay_attack | Not implemented | Requires message hash data |
| impossible_transition | Not implemented | Requires task state machine |
| leader_override_abuse | Not implemented | Requires command authorization data |

Combined risk = max(security_score / 10, anomaly_ensemble_risk)

### 7.7 anomaly_detection (Port 8007)
Autoencoder + Isolation Forest ensemble — Model C.
- Autoencoder: 7->16->4->16->7 architecture
- Threshold from k-fold cross-validation (not training-set fitting)
- Isolation Forest: scikit-learn, per-agent, strong on point anomalies
- Ensemble: combined_anomaly_risk = max(autoencoder_normalized, isoforest_normalized)
- Persistence: requires 2 consecutive elevated readings before flagging

### 7.8 trust_forecast (Port 8008)
GRU-based predictive trust forecaster — Model B.
- Architecture: two-layer GRU (64->32 units) over 20-timestep window
- Output: sigmoid mean (predicted trust) + softplus std (uncertainty)
- Loss: Gaussian NLL
- Online fine-tuning every 100 new timesteps at low learning rate
- Confidence gate: decision engine only acts when 1/std > min_confidence

### 7.9 collusion_detection (Port 8009)
Cross-agent behavioral correlation for coordinated attack detection.
Four signals per pair:
1. **Timing correlation** — anomaly score time series correlation
2. **Behavioral similarity** — cosine similarity of feature-deviation vectors (5 of 7 dims)
3. **Comms-graph anomaly** — structural communication pattern anomaly
4. **Outcome corroboration** — task outcome sequence correlation

Flagging: >=2 of 4 signals elevated, persisted across >=3 windows.
Routes EXCLUSIVELY to flag_for_review — never auto-isolation.

### 7.10 federated_learning (Port 8010)
FedAvg aggregation with validation and rollback.
1. Poll anomaly_detection for each agent's local model weights
2. Compute unweighted FedAvg of all weight tensors
3. Validate against held-out cross-agent sample set
4. Push only if validation error within tolerance of previous best
5. Otherwise: silently reject the round, keep previous best

### 7.11 api_gateway (Port 8011)
Single reverse-proxy entry point. Routes: /{service_name}/{path} to internal services.
Currently routing-only — no auth or rate-limiting.

### 7.12 agent_isolation (Port 8012)
Executes isolation, cascades consequences, monitors for auto-reinstatement.
- On isolation: marks agent isolated, redistributes tasks, reassigns leadership if needed
- Auto-reinstatement: combined_risk < threshold for 60 continuous seconds
- Exposes GET /isolated with per-agent recovery ETA

### 7.13 leader_reassignment (No public port)
Subscribes to decision.action(reassign_leader). Queries active workers, ranks by
trust score, promotes the highest-trust eligible candidate.

### 7.14 task_redistribution (No public port)
Subscribes to decision.action(redistribute_tasks). Queries active, available workers,
ranks by trust score, assigns pending task to best candidate.

### 7.15 dashboard (Port 8000)
Read-only live UI. Server-side aggregation of all services into one parallel async
HTTP round per page load. Shows: per-agent trust/risk/forecast, collusion signals,
pending human reviews, federated learning status, recent decisions, recovery ETAs.

### 7.16 mesa_sim (No public port)
Mesa-based multi-agent simulation with genuine fault and collusion injection.
- Default: 1 leader + 5 workers
- Fault injection: agent-worker-0 after 60s warm-up
- Collusion injection: agent-worker-2 and agent-worker-3 after 60s

---

## 8. Event Bus and Communication

### 8.1 Exchange Configuration
- **Exchange name:** neurotrust.events
- **Type:** RabbitMQ Topic Exchange
- **Durability:** Durable queues + persistent messages
- **Prefetch:** 20 messages per consumer

### 8.2 Full Event Routing Table

| Routing Key | Published By | Consumed By |
|-------------|-------------|-------------|
| agent.heartbeat | agent_registry | cdt_service, behavior_analysis, trust_prediction, security_monitoring |
| agent.missed_heartbeat | agent_registry | trust_prediction, security_monitoring |
| agent.task_outcome | mesa_sim | cdt_service, behavior_analysis, trust_prediction, security_monitoring, collusion_detection |
| agent.metrics_sample | mesa_sim | cdt_service, behavior_analysis, security_monitoring, collusion_detection |
| behavior.features | behavior_analysis | anomaly_detection, trust_forecast, collusion_detection, federated_learning |
| trust.score | trust_prediction | decision_engine, behavior_analysis, cdt_service |
| anomaly.score | anomaly_detection | security_monitoring, collusion_detection |
| trust.forecast | trust_forecast | decision_engine |
| security.risk_update | security_monitoring | decision_engine, trust_prediction, anomaly_detection, agent_isolation |
| collusion.flag | collusion_detection | decision_engine |
| decision.action | decision_engine, agent_isolation | leader_reassignment, task_redistribution, agent_isolation |
| agent.isolation_requested | agent_isolation | agent_registry |
| agent.reinstate_requested | agent_isolation | agent_registry |
| agent.role_change_requested | leader_reassignment | agent_registry |
| agent.task_change_requested | task_redistribution | agent_registry |

### 8.3 REST vs. Event Bus

| Pattern | Used For |
|---------|----------|
| Event Bus | All state-mutating actions (isolation, role changes, task moves, reinstatement) |
| REST HTTP | Read-only queries, federated model weight export/import |

---

## 9. Machine Learning Models

### 9.1 Model A — Bayesian Beta Trust (Reactive)

```
Prior: alpha0 = 1, beta0 = 1  (uniform prior)

Each tick decay:
  alpha_new = alpha * lambda
  beta_new  = beta  * lambda
  lambda = 0.97 (normal) | 0.90 (under security watch)

Evidence update:
  task_success          -> alpha += 2.0
  missed_heartbeat      -> beta  += 1.0
  failed_critical_task  -> beta  += 3.0
  security_flagged      -> beta  += 5.0

Trust score = alpha / (alpha + beta)  in range (0, 1)
Variance    = (alpha*beta) / ((alpha+beta)^2 * (alpha+beta+1))

Thresholds:
  Leader: trust_score < 0.65 -> trigger decision
  Worker: trust_score < 0.45 -> trigger decision
```

### 9.2 Model B — GRU Trust Forecaster (Predictive)

```
Input:  20-timestep window of 7-dim feature vector

Architecture:
  GRU Layer 1:   input_size=7,  hidden_size=64, batch_first=True
  GRU Layer 2:   input_size=64, hidden_size=32, batch_first=True
  Output Head 1: Linear(32->1) + Sigmoid   -> predicted trust in (0,1)
  Output Head 2: Linear(32->1) + Softplus  -> uncertainty > 0

Loss: Gaussian NLL = log(sigma) + (y - mu)^2 / (2*sigma^2)
Online fine-tuning every 100 new timesteps at lr=1e-4
Confidence gate: only acts when 1/sigma > MIN_CONFIDENCE_THRESHOLD
```

### 9.3 Model C — Anomaly Detection Ensemble

#### Autoencoder
```
Architecture: 7 -> 16 -> 4 -> 16 -> 7
Activation: ReLU (hidden), Sigmoid (output)
Loss: MSE reconstruction error

Threshold calibration (k-fold cross-validation):
  threshold = mean(held-out errors) + 3 * std(held-out errors)
  Uses genuine held-out error, NOT training-set fitting
```

#### Isolation Forest
```
Library: scikit-learn IsolationForest
Per-agent, trained on 7-dim feature vectors
Strong against point anomalies (complementary to autoencoder)
```

#### Ensemble Fusion
```
combined_anomaly_risk = max(autoencoder_normalized, isoforest_normalized)
is_anomalous = True  only if  elevated for 2 CONSECUTIVE readings
```

### 9.4 Federated Learning — FedAvg

```
Each round:
  1. Collect weights Wi from each trained agent i
  2. W_global = (1/n) * sum(Wi)   [unweighted average]
  3. Validate W_global on held-out cross-agent samples
  4. If validation_error <= best_error * (1 + tolerance):
       push W_global to all agents
     Else:
       reject round, keep previous best weights
```

---

## 10. Data Flow — Worked Example

**Scenario:** agent-worker-0 goes faulty and gets isolated.

```
Step 1: mesa_sim publishes
        agent.task_outcome(agent_id="agent-worker-0", outcome="failed_critical")

Step 2: trust_prediction receives event
        beta += 3.0 -> trust_score drops below 0.45 threshold
        publishes trust.score(trust_score=0.38, ...)

Step 3: behavior_analysis updates 7-dim feature vector
        publishes behavior.features(...)

Step 4: anomaly_detection scores the feature vector
        autoencoder: reconstruction_error > threshold
        isolation_forest: anomaly confirmed
        2nd consecutive elevated reading -> is_anomalous = True
        publishes anomaly.score(is_anomalous=True, combined_anomaly_risk=0.81)

Step 5: security_monitoring receives anomaly.score
        combined_security_risk = max(rule_score/10, 0.81) = 0.81
        publishes security.risk_update(combined_risk=0.81, ...)

Step 6: decision_engine receives trust.score AND security.risk_update
        trust trigger: score < 0.45 -> action = "redistribute_tasks"
        security trigger: combined_risk > 0.70 -> action = "isolate"
        publishes decision.action(action="isolate",
                                  rule="security_risk > 0.70",
                                  inputs={combined_risk: 0.81, trust: 0.38})

Step 7: agent_isolation receives decision.action(isolate)
        publishes AgentIsolationRequested -> registry marks status="isolated"
        republishes decision.action(redistribute_tasks) -> task moved
        agent is not a leader -> no leader reassignment needed

Step 8: task_redistribution picks agent-worker-1 (trust=0.82)
        publishes AgentTaskChangeRequested -> registry updates task owner

Step 9: agent_isolation monitors security.risk_update for agent-worker-0
        combined_risk < threshold for 60 continuous seconds
        publishes AgentReinstateRequested -> registry marks status="active"

Dashboard shows: isolated agent, recovery ETA countdown, task moved indicator
```

---

## 11. Simulation and Fault Injection

### Default Configuration

```yaml
N_WORKERS: "5"
FAULTY_AGENT_IDS: "agent-worker-0"
FAULT_INJECTION_DELAY_S: "60"
COLLUDING_AGENT_IDS: "agent-worker-2,agent-worker-3"
COLLUSION_DELAY_S: "60"
```

### Fault Behaviors (agent-worker-0)

| Behavior | Downstream Effect |
|----------|------------------|
| Task failure | trust decay, behavior feature update |
| Resource spike | resource_spike security rule trigger |
| Silent ticks | silent_agent rule after 3 missed beats |
| Security-flagged outputs | security_flagged_outcome rule trigger |

### Collusion Behaviors (agent-worker-2 and agent-worker-3)

| Behavior | Signal Affected |
|----------|----------------|
| Synchronized comms bursts | Timing correlation + comms anomaly |
| Correlated task outcomes | Outcome corroboration |
| Similar deviation patterns | Behavioral similarity |

---

## 12. Deployment and Infrastructure

### Quickstart

```bash
# First run — build all images and start all containers
docker compose up --build -d

# Already built — start fast
docker compose up -d

# View live logs
docker compose logs -f

# Stop (keep database data)
docker compose down

# Stop and wipe all data (clean slate)
docker compose down -v
```

### Container Startup Order

```
1. rabbitmq (health-checked) + postgres (health-checked)
        |
2. All business services (wait for healthy infra)
        |
3. dashboard, api_gateway, mesa_sim, agent_isolation,
   leader_reassignment, task_redistribution (wait for agent_registry started)
```

---

## 13. API Reference

```bash
# AGENT REGISTRY (8001)
GET  /agents                    # list all agents (filter: ?role=leader&status=active)
GET  /agents/{id}               # get one agent full state
POST /agents/register           # register new agent
POST /agents/{id}/heartbeat     # heartbeat + optional task update
PATCH /agents/{id}/isolate      # manual isolation
PATCH /agents/{id}/reinstate    # manual reinstatement

# TRUST PREDICTION (8004)
GET  /trust/{agent_id}          # Bayesian score + plain-English explanation

# SECURITY MONITORING (8006)
GET  /security/{agent_id}       # combined risk + contributing rule hits

# ANOMALY DETECTION (8007)
GET  /anomaly/{agent_id}        # anomaly score + is_anomalous flag

# TRUST FORECAST (8008)
GET  /forecast/{agent_id}       # GRU predicted trust + confidence interval

# COLLUSION DETECTION (8009)
GET  /pairs                     # all tracked agent pairs + per-signal scores
GET  /reviews                   # groups pending human review

# FEDERATED LEARNING (8010)
GET  /status                    # round history (accepted/rolled-back merges)

# AGENT ISOLATION (8012)
GET  /isolated                  # isolated agents + recovery ETA in seconds

# DECISION ENGINE (8005)
GET  /recent_decisions?limit=30 # last N decisions with rule + evidence

# BEHAVIOR ANALYSIS (8003)
GET  /features/{agent_id}       # current 7-dim feature vector

# CDT SERVICE (8002)
GET  /twins/{agent_id}          # digital twin state snapshot

# VIA API GATEWAY (8011) -- all routes available as /{service_name}/{path}
# Examples:
curl http://localhost:8011/trust_prediction/trust/agent-worker-0
curl http://localhost:8011/agent_registry/agents
curl http://localhost:8011/collusion_detection/pairs
```

---

## 14. Testing Strategy

Tests cover every pure-function core without requiring Docker, Postgres, or RabbitMQ:

| Test File | Coverage |
|-----------|----------|
| test_bayesian.py | Beta update math, decay, threshold crossing |
| test_features.py | Feature vector normalization, z-score stability |
| test_autoencoder.py | Training, threshold calibration, k-fold validation |
| test_isolation_forest.py | Training and scoring |
| test_gru.py | Training, online fine-tuning, confidence gating |
| test_fedavg.py | Weight aggregation, validation, rollback |
| test_collusion_signals.py | All 4 signals, persistence gating |
| test_rules.py | Decision engine rule table, cooldowns |
| test_security_rules.py | Security monitoring rules, fusion |
| test_fusion.py | Final risk signal routing |
| test_recovery.py | Isolation recovery state machine |

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -r services/trust_prediction/requirements.txt
pip install --extra-index-url https://download.pytorch.org/whl/cpu ^
            -r services/anomaly_detection/requirements.txt
pip install -r services/federated_learning/requirements.txt
pytest tests/
```

---

## 15. Development Milestones

| Milestone | Key Additions | Notable Fix |
|-----------|---------------|-------------|
| M1 | Agent registry, CDT, Bayesian trust, basic decision loop | SQLAlchemy default-field race: new agent first event silently dropped. Structured log evidence payload silently discarded by LoggerAdapter. Both fixed. |
| M2 | Rule-based security engine, real agent isolation | Made isolation an actual DB state change. mesa_sim gained real fault behaviors. |
| M3 | Autoencoder anomaly detection | Training-set threshold memorized 30 samples causing entire healthy fleet to be isolated. Fixed with k-fold cross-validation. |
| M4 | GRU trust forecaster, predictive decision path | Min-max normalization caused 0.75->0.47 swing in 3s for healthy agents. Fixed with z-score + logistic squash. Anomaly detector feedback loop fixed by gating on security_score only. |
| M5 | Collusion detection | comms_frequency and bayesian_confidence near-constant for all agents -> 100% behavioral similarity for unrelated pairs. Fixed by dropping those 2 dimensions. |
| M6 | Federated learning with FedAvg + rollback | Verified: new agent with zero local training received functional anomaly scores from federated fallback model. |
| Tuning | Persistence smoothing + Isolation Forest ensemble | Percentile threshold: FPR tripled (6%->14%). Higher sigma multiplier: killed fault detection to 0%. Final: persistence + IsoForest ensemble. Result: 94.4% detection at 5.5% FPR vs autoencoder-alone 0.8%/6.1%. |
| Gap-closure | API Gateway, event-driven mutations, 4 more security rules, final_risk_signal, dashboard recovery ETA | — |

---

## 16. Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|-----------|
| Collusion expected_correlation fixed at 0 (no task graph) | Conservative — may under-flag low-correlation collusion | Most conservative choice available. Real fix needs inter-agent task dependency graph. |
| Single FedAvg cluster, unweighted averaging | Model quality biased by data volume distribution | Needs per-client sample count metadata for weighted FedAvg. |
| GRU (Model B) not federated | Each agent GRU only learns from own history | Full federation implemented for Model C as proof-of-concept. Same pattern documented. |
| 3 security rules unimplemented by design | replay_attack, impossible_transition, leader_override_abuse not detected | These require data this simulation does not produce. Faking them would be worse than omitting. |
| No approve/dismiss UI for collusion review | Flags are queryable but cannot be actioned | GET /reviews exposes all flags and evidence. UI workflow explicitly out-of-scope. |
| API Gateway routing-only | No auth or rate-limiting | Satisfies "single entry point" requirement. Auth is a natural next step. |
| Residual anomaly false-positive rate ~0.5-1% | Some false positives remain | Auto-reinstatement absorbs this — false-positive isolated agents recover automatically within 60 seconds. |

---

## 17. Conclusion

NeuroTrust Mesh demonstrates a complete, end-to-end solution to the multi-agent trust
and security problem, implemented as real production-grade microservices.

### Key Achievements

- 16 independent microservices running in Docker, communicating over a durable event bus
- Three independent ML models — Bayesian, GRU, Autoencoder+IsoForest — each catching
  different failure modes the others miss
- Full explainability — every decision logged with its triggering rule and input values
- Federated learning with validation and rollback — Model C improves across all agents
  without centralizing raw data
- Automatic recovery — no agent is permanently isolated
- Realistic simulation — genuine fault injection (single-agent and coordinated collusion)
- Live dashboard — all signals, decisions, and recovery ETAs in one view
- Comprehensive test suite — all pure-function ML cores covered with statistical tests

### Outcome

The system successfully detects the injected faulty agent (agent-worker-0) within
2-3 minutes of fault onset, redistributes its tasks, and auto-reinstates it once
it recovers. The colluding pair (agent-worker-2, agent-worker-3) is correctly
flagged for human review without auto-isolation, reflecting the higher uncertainty
of cross-agent correlation signals.

---

*Document prepared for project review.*
*For full architecture details and build history, see ARCHITECTURE.md.*
*For quickstart instructions, see README.md.*
