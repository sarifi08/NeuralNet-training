"""Tournament agent — wraps the trained MLP with runtime recovery logic.

This module exposes the `make_policy` hook so the benchmark harness can load
it via --module drive2win.agent without touching benchmark.py.

Usage:
    python 03_benchmark.py --tag v14-final --weights nav_v14.npz --module drive2win.agent
    python -m drive2win.benchmark --weights nav_v14.npz --module drive2win.agent
"""
from __future__ import annotations
import numpy as np

from . import nn as nn_mod
from .normalize import sensors_to_input, clip_action


def make_policy(weights_path: str):
    """Load MLP weights and return a fully-wrapped tournament policy.

    Supports both 12-feature (legacy) and 13-feature (checkpoint-aware) models
    by inspecting W1.shape at load time.
    """
    w = nn_mod.load(weights_path)
    n_in = w["W1"].shape[0]  # 12 = legacy, 13 = checkpoint-aware

    def base_policy(obs):
        x = sensors_to_input(obs["sensors"])  # always 13 features
        if n_in == 12:
            x = x[:12]  # old weights don't have the checkpoint input
        return clip_action(nn_mod.forward(x, w))

    return _with_escape(_with_obstacle_avoidance(_with_checkpoint_precision(base_policy)))


# ── Wrappers ─────────────────────────────────────────────────────────────────

def _with_obstacle_avoidance(base_policy, danger_dist=5.0, warn_dist=12.0,
                             steer_strength=0.6, brake_throttle=0.35):
    """Proactive obstacle avoidance using raycasts.

    Checks front/front-diagonal rays every frame. If an obstacle is within
    danger_dist, overrides steering toward the clearer side and reduces
    throttle. Acts before the bot hits anything, unlike the escape wrapper
    which only fires after the bot is already stuck.

    Ray layout (45° apart, clockwise from front):
        0=front  1=FR  2=R  3=BR  4=back  5=BL  6=L  7=FL
    """
    def policy(obs):
        throttle, steering = base_policy(obs)
        if throttle <= 0:
            return throttle, steering

        sensors = obs.get("sensors") or {}
        rays = sensors.get("rays") or [50.0] * 8

        front = rays[0]
        front_r = rays[1]   # front-right (+45°)
        front_l = rays[7]   # front-left  (-45°)

        # Nothing close — pass through unchanged
        if front >= warn_dist and front_r >= warn_dist and front_l >= warn_dist:
            return throttle, steering

        if front < danger_dist or front_r < danger_dist or front_l < danger_dist:
            # Hard dodge: steer away from whichever side is closer
            if front_r < front_l:
                steering = float(np.clip(steering - steer_strength, -1.0, 1.0))
            else:
                steering = float(np.clip(steering + steer_strength, -1.0, 1.0))
            throttle = min(throttle, brake_throttle)
        else:
            # Soft nudge in the warn zone
            blend = 1.0 - (min(front, front_r, front_l) - danger_dist) / (warn_dist - danger_dist)
            if front_r < front_l:
                steering = float(np.clip(steering - steer_strength * blend * 0.5, -1.0, 1.0))
            else:
                steering = float(np.clip(steering + steer_strength * blend * 0.5, -1.0, 1.0))

        return throttle, steering

    return policy


def _with_escape(base_policy, stuck_threshold=20, escape_frames=30):
    """Reverse out of stuck states. Alternates escape direction on each use."""
    state = {"stuck": 0, "escaping": 0, "escape_dir": 1.0}

    def policy(obs):
        sp = (obs.get("sensors") or {}).get("speed", 0.0)
        if state["escaping"] > 0:
            state["escaping"] -= 1
            if state["escaping"] == 0:
                state["escape_dir"] *= -1.0
            return -1.0, state["escape_dir"] * 0.6
        if sp < 0.3:
            state["stuck"] += 1
            if state["stuck"] >= stuck_threshold:
                state["stuck"] = 0
                state["escaping"] = escape_frames
                return -1.0, state["escape_dir"] * 0.6
        else:
            state["stuck"] = 0
        return base_policy(obs)

    return policy


def _with_checkpoint_precision(base_policy, approach_dist=25.0, heading_k=0.7,
                                speed_dist=12.0, speed_target=2.0, throttle_floor=0.2,
                                near_miss_dist=6.0, near_miss_climb=4.0,
                                near_miss_frames=25):
    """Heading correction + speed control on approach, near-miss recovery."""
    st = {
        "min_dist": 999.0, "prev_dist": 999.0,
        "cp_count": 0, "recovering": 0,
        "retries": 0, "cooldown": 0,
    }

    def policy(obs):
        sensors = obs.get("sensors") or {}
        nav = sensors.get("navigation") or {}
        dist = sensors.get("checkpoint_distance", 999.0)
        cp_count = nav.get("checkpoints_completed", 0) or 0

        if cp_count != st["cp_count"]:
            st["cp_count"] = cp_count
            st["min_dist"] = 999.0
            st["recovering"] = 0
            st["retries"] = 0
            st["cooldown"] = 0

        st["min_dist"] = min(st["min_dist"], dist)
        if st["cooldown"] > 0:
            st["cooldown"] -= 1

        near_miss = (
            st["min_dist"] < near_miss_dist
            and dist > st["min_dist"] + near_miss_climb
            and st["recovering"] == 0
            and st["cooldown"] == 0
            and st["retries"] < 2
        )
        st["prev_dist"] = dist

        if near_miss:
            st["retries"] += 1
            st["recovering"] = near_miss_frames
            st["cooldown"] = near_miss_frames * 3
            st["min_dist"] = 999.0

        if st["recovering"] > 0:
            st["recovering"] -= 1
            heading_err = sensors.get("heading_error", 0.0)
            steer = float(np.clip(heading_err * 0.8, -1.0, 1.0))
            return -0.8, steer

        throttle, steering = base_policy(obs)

        if dist <= approach_dist:
            heading_err = sensors.get("heading_error", 0.0)
            blend = 1.0 - (dist / approach_dist)
            steering = float(np.clip(
                steering - heading_k * blend * heading_err, -1.0, 1.0
            ))
        if dist <= speed_dist:
            speed = sensors.get("speed", 0.0)
            throttle = float(np.clip(
                0.25 * (speed_target - speed), throttle_floor, 0.6
            ))
        return throttle, steering

    return policy
