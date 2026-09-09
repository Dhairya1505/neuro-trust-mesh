import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for service_dir in [
    ROOT / "services" / "trust_prediction",
    ROOT / "services" / "behavior_analysis",
    ROOT / "services" / "decision_engine",
    ROOT / "services" / "anomaly_detection",
    ROOT / "services" / "trust_forecast",
    ROOT / "services" / "agent_isolation",
    ROOT / "services" / "federated_learning",
    ROOT / "libs" / "neurotrust_common",
]:
    sys.path.insert(0, str(service_dir))
