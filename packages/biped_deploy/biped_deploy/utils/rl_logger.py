import json
import numpy as np
from dataclasses import asdict, dataclass


# Save safety checker data
def _json_serializer(obj):
    """Handle numpy types and other non-serializable objects"""
    if isinstance(obj, np.ndarray | np.generic):
        return obj.tolist()
    err_msg = f"Object of type {type(obj)} is not JSON serializable"
    raise TypeError(err_msg)


@dataclass
class RLMetrics:
    action: np.ndarray
    observation: np.ndarray


class RLLogger:
    def __init__(self):
        self.metrics_data: list[RLMetrics] = []

    def record_metrics(self, observations, actions):
        self.metrics_data.append(RLMetrics(action=actions, observation=observations))

    def save_data(self, log_dir):
        # Save metrics data
        metrics_path = log_dir / "rl_metrics.json"
        with metrics_path.open("w") as f:
            json.dump(
                [asdict(m) for m in self.metrics_data],
                f,
                indent=2,
                default=_json_serializer,
            )
        print(f"Saved RL metrics to {metrics_path}")
