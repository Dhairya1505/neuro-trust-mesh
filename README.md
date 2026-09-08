# NeuroTrust Mesh

A self-evolving Cognitive Digital Twin framework for predictive trust and
autonomous collaboration in hierarchical multi-agent systems — built as
16 real microservices connected over an async event bus, driven by a
Mesa-based multi-agent simulation with genuine fault injection (single-agent
faults *and* coordinated collusion), and a live dashboard over all of it.

For the full system design, event flow, ML model details, and the design
decisions (and dead ends) behind them, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## What it does

Simulated agents heartbeat and work tasks. Their behavior is continuously
scored for trust (Bayesian) and forecast forward (GRU), checked against
their own historical baseline for anomalies (Autoencoder + Isolation
Forest ensemble), matched against rule-based security signals, and
cross-checked against each other for collusion. An Autonomous Decision
Engine fuses all of that into one explicit, inspectable rule table — plus
a combined per-agent `final_risk_signal` for the dashboard — and takes real
corrective action — reassigning a failing leader, redistributing a failing
worker's tasks, isolating a compromised agent (with automatic reinstatement
once it's no longer a risk, and a visible ETA on the dashboard while it
recovers), or flagging a suspected collusion group for human review. Both
the Autoencoder and the GRU forecaster are federated across agents via
FedAvg, with automatic validation and rollback if a merged model degrades.

State-mutating actions (role changes, task moves, isolation, reinstatement)
flow through the event bus rather than direct service-to-service calls;
read-only queries stay REST, a deliberate, documented split. An API Gateway
gives external/operator clients one entry point instead of needing to know
every service's address.

Every decision is logged with the exact rule and evidence that triggered
it — explainability was a first-class design constraint, not an
afterthought.

## Quickstart

```bash
docker compose up --build -d
```

This starts RabbitMQ (management UI at http://localhost:15672, guest/guest),
PostgreSQL, and all 16 application services. The Mesa simulation registers
1 leader + 5 workers and heartbeats every 5 seconds. By default:
- `agent-worker-0` goes faulty after a 60s warm-up (real failures, security
  flags, and occasional silence)
- `agent-worker-2` and `agent-worker-3` start colluding after the same delay
  (synchronized comms bursts, correlated outcomes)

Open **http://localhost:8000** for the live dashboard — every agent's
trust/risk/forecast at a glance (with a recovery ETA once isolated),
collusion signals and any pending human review, federated learning status
for both models, and a recent-decisions feed, all in one place. Give it a
few minutes to train its ML models on real behavioral history before it
gets interesting.

Or hit the APIs directly (through the gateway, or a service's own port):

```bash
# Bayesian trust score decaying for the faulty agent, with a plain-English explanation
curl http://localhost:8004/trust/agent-worker-0
# same call through the API Gateway's single entry point
curl http://localhost:8011/trust_prediction/trust/agent-worker-0

# once it crosses threshold, decision_engine's logs show a "decision made"
# entry and agent_registry reflects the real task/role move
curl http://localhost:8001/agents/agent-worker-0

# fused security risk (rule engine + anomaly ensemble)
curl http://localhost:8006/security/agent-worker-0

# GRU forecast — where trust is headed, with a confidence interval
curl http://localhost:8008/forecast/agent-worker-0

# collusion signals between every currently-tracked pair
curl http://localhost:8009/pairs

# collusion groups flagged for human review (never auto-isolated on this alone)
curl http://localhost:8009/reviews

# federated learning round history for both models (accepted/rolled-back merges)
curl http://localhost:8010/status

# currently-isolated agents and their recovery ETA
curl http://localhost:8012/isolated
```

Bring it down cleanly with `docker compose down` (add `-v` to also drop
the Postgres volume for a truly clean slate on the next run).

## Services

| Service | Port | Responsibility |
|---|---|---|
| `dashboard` | 8000 | Live UI — aggregates every service below into one page |
| `agent_registry` | 8001 | Agent registration, heartbeats, role/status, missed-heartbeat monitor |
| `cdt_service` | 8002 | Cognitive Digital Twin — state mirror, short/long-term memory per agent |
| `behavior_analysis` | 8003 | 7-dim behavioral feature vector, per-agent z-score normalization |
| `trust_prediction` | 8004 | Bayesian (Beta) trust scoring — the reactive, explainable trust signal |
| `decision_engine` | 8005 | Fuses every signal into one explicit rule table and a combined risk signal; the only service that decides actions |
| `security_monitoring` | 8006 | Rule-based security engine (8 rules) + fuses in the anomaly ensemble's risk |
| `anomaly_detection` | 8007 | Autoencoder + Isolation Forest ensemble, per-agent, with federated fallback |
| `trust_forecast` | 8008 | GRU trust forecaster — the predictive, early-warning trust signal, federated |
| `collusion_detection` | 8009 | Cross-agent correlation signals vs. a role/task baseline, persistence-gated, human-review only |
| `federated_learning` | 8010 | FedAvg aggregation of both the anomaly detector's and the GRU's weights, with validation + rollback |
| `api_gateway` | 8011 | Single entry point, reverse-proxies to every service below by name |
| `agent_isolation` | 8012 | Executes isolation, pulls tasks/leadership, auto-reinstates on sustained recovery, exposes recovery ETA |
| `leader_reassignment` | — | Promotes the highest-trust worker when a leader is reassigned |
| `task_redistribution` | — | Moves an unfinished task to another trusted, available worker |
| `mesa_sim` | — | The simulation itself — agents, fault injection, collusion injection |

Services with no port are pure event consumers with no HTTP API of their own.

## Running the unit tests

Covers every pure-function core (Bayesian math, feature normalization, rule
tables, autoencoder/isolation-forest/GRU training and scoring, FedAvg
aggregation for both federated models, collusion signal correlation, fusion
routing, recovery persistence) without needing Docker, Postgres, or
RabbitMQ — including statistical regression tests that pin down
false-positive rates against realistic noise, not just idealized synthetic
data.

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows; use source .venv/bin/activate on Linux/macOS
pip install -r requirements-dev.txt
pip install -r services/trust_prediction/requirements.txt
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r services/anomaly_detection/requirements.txt
pip install -r services/federated_learning/requirements.txt
pytest tests/
```

(The `torch`/`scikit-learn` installs are only needed for the ML-model test
files; the rest of the suite runs on the base dev requirements alone.)

## Project status

All six planned milestones are complete and verified end-to-end, plus a
follow-up tuning pass on the anomaly detector and a later pass closing every
gap found in an audit against the original design docs. See
[ARCHITECTURE.md](ARCHITECTURE.md#build-history--design-decisions) for what
each milestone added, what broke along the way, and how it got fixed.

| Milestone | Adds |
|---|---|
| M1 | Agent core, Cognitive Digital Twin, Bayesian trust, minimal decision loop |
| M2 | Rule-based security engine, real Agent Isolation with auto-reinstatement |
| M3 | Autoencoder anomaly detection, fault-injection warm-up |
| M4 | GRU trust forecaster, predictive decision path |
| M5 | Collusion detection (cross-agent correlation, human-review only) |
| M6 | Federated learning (FedAvg) for the anomaly detector, with rollback |
| Tuning | Persistence smoothing + Isolation Forest ensemble for the anomaly detector |
| Gap-closure | API Gateway; event-driven state mutations; 4 more security rules (replay, spoofed identity, task contradiction, impossible transition); combined `final_risk_signal` fusion layer; GRU federation; role/task-aware collusion baseline; dashboard recovery ETA and pending-review display |

## Known limitations

- **Collusion's `expected_correlation` is a lightweight heuristic, not a real
  task graph**: same-task and same-role pairs get a small baseline
  subtracted (so ordinary collaboration doesn't read as collusion), but this
  is still not the full task-graph/hierarchy-structure model the ML doc
  describes — a real task-dependency graph doesn't exist yet to derive one
  from.
- **Single-cluster federated learning**: all agents are treated as one
  FedAvg cluster for both models; unweighted averaging (no per-client
  sample-count metadata).
- **Residual anomaly-detector false-positive rate**: reduced substantially
  (persistence smoothing + Isolation Forest ensemble) but not eliminated;
  auto-reinstatement absorbs what's left rather than the detector being
  perfectly calibrated. See ARCHITECTURE.md for the full tuning history,
  including two approaches that were tried and rejected with evidence.
- **No approve/dismiss Control Center workflow**: collusion flags and their
  evidence are visible end-to-end (dashboard's "Pending Human Review"
  section, plus `GET /reviews`), but there's still no API or UI action to
  approve/dismiss a flagged group — that stays a read-only surface, the
  actual override workflow is explicitly out of scope per the original
  design docs' own build order (listed last, after federated learning).
- **API Gateway is routing-only**: no auth or rate-limiting yet — it
  satisfies the "single entry point" requirement but isn't a production
  security boundary.
- **Reads still go direct over REST**: only state-mutating writes moved to
  the event bus; read-only queries (agent lookups, federated model
  export/import) deliberately stayed REST rather than becoming
  request/reply events or a locally-cached read-model, a complexity
  tradeoff documented in `eventbus.py`.

## Repo layout

```
libs/neurotrust_common/     shared event bus client, logging, locks, schemas, fusion, DB helper
services/<name>/            one FastAPI microservice per directory
simulation/mesa_sim/        Mesa-based multi-agent simulation, fault + collusion injection
tests/                      unit tests for every service's pure-function core
docker-compose.yml          full 16-service stack + RabbitMQ + Postgres
```
