"""EMA action smoothing — wraps any policy to reduce jerky outputs.

Behavioral cloning often produces high-frequency oscillations in steering
because the network predicts each frame independently. An exponential
moving average over consecutive actions eliminates most of this noise
without adding any learnable parameters.

Usage:
    from drive2win.smooth import make_smooth_policy
    policy = make_smooth_policy(raw_policy, alpha=0.7)
    throttle, steering = policy(state)

alpha controls the trade-off:
    alpha = 1.0  →  no smoothing (original policy)
    alpha = 0.5  →  heavy smoothing (slow to react)
    alpha = 0.7  →  good default: removes oscillations, keeps response
"""
from __future__ import annotations
import numpy as np


def make_smooth_policy(policy_fn, alpha: float = 0.7):
    """Wrap a policy function with exponential moving average smoothing.

    Args:
        policy_fn: callable(state_dict) -> (throttle, steering)
        alpha: EMA weight for current frame. 1-alpha goes to previous output.

    Returns:
        Wrapped callable with the same signature as policy_fn.
        State resets automatically between sessions via reset().
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    prev = np.zeros(2, dtype=np.float64)

    def policy(state):
        nonlocal prev
        raw = policy_fn(state)
        t, s = float(raw[0]), float(raw[1])
        smoothed = alpha * np.array([t, s]) + (1.0 - alpha) * prev
        prev[:] = smoothed
        return float(smoothed[0]), float(smoothed[1])

    def reset():
        nonlocal prev
        prev = np.zeros(2, dtype=np.float64)

    policy.reset = reset
    return policy
