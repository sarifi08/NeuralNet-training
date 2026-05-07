"""Input/output normalization for the navigation network.

The simulation returns sensor values in their natural units (m/s, radians,
meters, [0,1] friction). Neural networks train and infer better when every
input is in roughly the same range. The functions in this file are the
*single source of truth* for normalization across the whole project.

If you ever change a normalization constant, retrain. Mixing training-time
and deployment-time normalization is the #1 reason "my net trained fine but
won't drive."
"""
from __future__ import annotations
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────
SPD_MAX = 20.0      # speed clip (m/s)
DIST_MAX = 100.0    # checkpoint-distance clip (m)
RAY_MAX = 50.0      # raycast clip (m); matches RAYCAST_MAX_RANGE in SensorSystem.ts

FEATURE_NAMES = [
    "speed",
    "heading_error",
    "checkpoint_distance",
    "ray_0_front",
    "ray_1_+45",
    "ray_2_+90",
    "ray_3_+135",
    "ray_4_back",
    "ray_5_-135",
    "ray_6_-90",
    "ray_7_-45",
    "ground_friction",
]
ACTION_NAMES = ["throttle", "steering"]
N_FEATURES = 12
N_ACTIONS = 2


def normalize_states(states_raw: np.ndarray) -> np.ndarray:
    """Map raw sensor readings into roughly [-1, 1].

    Args:
        states_raw: shape (N, 12). Columns in FEATURE_NAMES order.

    Returns:
        float32 array of the same shape, scaled to [-1, 1] (or [0, 1] for
        ranges that are physically non-negative).
    """
    s = np.asarray(states_raw, dtype=np.float32).copy()
    s[:, 0] = np.clip(s[:, 0] / SPD_MAX, -1.0, 1.0)         # speed
    s[:, 1] = np.clip(s[:, 1] / np.pi, -1.0, 1.0)           # heading_error
    s[:, 2] = np.clip(s[:, 2] / DIST_MAX, 0.0, 1.0)         # ckpt distance
    s[:, 3:11] = np.clip(s[:, 3:11] / RAY_MAX, 0.0, 1.0)    # 8 rays
    # column 11 (friction) is already in [0, 1]
    return s


def sensors_to_input(sensors: dict) -> np.ndarray:
    """Convert a live sensor dict (from `client.get_sensors()` or the WS
    `state['sensors']`) to the normalized 12-vector the network expects.

    Returns shape (12,), float32.
    """
    raw = np.array(
        [
            sensors["speed"],
            sensors["heading_error"],
            sensors["checkpoint_distance"],
            *sensors["rays"],
            sensors["ground_friction"],
        ],
        dtype=np.float32,
    )
    return normalize_states(raw[None, :])[0]


def clip_action(a: np.ndarray) -> tuple[float, float]:
    """Clamp the network's (throttle, steering) output to the physical [-1, 1]
    range the controller accepts. tanh outputs are already in range, but this
    keeps you safe if you ever swap the output activation.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    throttle = float(np.clip(a[0], -1.0, 1.0))
    steering = float(np.clip(a[1], -1.0, 1.0))
    return throttle, steering
