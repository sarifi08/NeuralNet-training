"""Diagnose training vs inference input mismatch.

This script compares the 12-feature vector used during training
(recording samples) with the 12-feature vector used at inference
(live /sensors mapping). It prints both vectors and their differences.

Run from repo root:
  .venv/bin/python diagnose_inputs.py --seed 42
"""
from __future__ import annotations

import argparse
import time
import numpy as np

from game_client import GameClient

SERVER_URL = "https://ml.ferit.tech"

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


def flatten_sensors(raw: dict) -> dict:
    """Match run_agent.py mapping."""
    nav = raw.get("navigation", {})
    ground = raw.get("ground", {})
    return {
        "speed": raw.get("speed", 0.0),
        "heading_error": nav.get("heading_error", 0.0),
        "checkpoint_distance": nav.get("distance", 0.0),
        "rays": raw.get("rays", []),
        "ground_friction": ground.get("friction", 1.0),
    }


def vec_from_live(flat: dict) -> np.ndarray:
    rays = list(flat.get("rays", []))
    if len(rays) != 8:
        rays = (rays + [0.0] * 8)[:8]
    return np.array(
        [
            flat.get("speed", 0.0),
            flat.get("heading_error", 0.0),
            flat.get("checkpoint_distance", 0.0),
            *rays,
            flat.get("ground_friction", 1.0),
        ],
        dtype=np.float32,
    )


def vec_from_record(sample_state: dict) -> np.ndarray:
    rays = list(sample_state.get("rays", []))
    if len(rays) != 8:
        rays = (rays + [0.0] * 8)[:8]
    return np.array(
        [
            sample_state.get("speed", 0.0),
            sample_state.get("heading_error", 0.0),
            sample_state.get("checkpoint_distance", 0.0),
            *rays,
            sample_state.get("ground_friction", 1.0),
        ],
        dtype=np.float32,
    )


def _retry_stop_recording(client: GameClient, retries: int = 5, delay: float = 0.5) -> bool:
    """Stop recording with retries to handle transient disconnects."""
    for _ in range(retries):
        try:
            client.stop_recording()
            return True
        except Exception:
            time.sleep(delay)
    return False


def _wait_for_sensors(client: GameClient, timeout_s: float = 20.0) -> bool:
    """Wait until /sensors is responding so the sim is fully running."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            s = client.get_sensors()
            if s and "speed" in s and "navigation" in s:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--duration", type=float, default=2.0,
                    help="Seconds to record before fetching samples")
    args = ap.parse_args()

    client = GameClient(SERVER_URL)
    session = client.create_session(
        mode="time_trial",
        player_name="d2w_diag",
        config={"seed": args.seed, "wind_enabled": False},
    )
    print("Open this URL in a NEW TAB and click into it so the sim runs:")
    print(" ", session.get("browser_url"))
    input("Press Enter once the browser tab has focus and the bot is visible. ")

    # Wait until the browser/sim is actually sending sensors
    print("Waiting for sensors to be ready...", end="", flush=True)
    if not _wait_for_sensors(client, timeout_s=30.0):
        print(" timeout")
        print("Sensors never became ready. Make sure the browser tab is open and focused.")
        return
    print(" ready")

    # Grab a live sensors snapshot
    live_raw = client.get_sensors()
    live_flat = flatten_sensors(live_raw)
    live_vec = vec_from_live(live_flat)

    # Record a short clip and grab the first recorded sample
    client.start_recording(sample_rate=20)
    time.sleep(max(0.5, args.duration))
    if not _retry_stop_recording(client):
        print("Failed to stop recording after retries. Try again with the browser focused.")
        return
    rec = client.get_recording()
    samples = rec.get("samples", [])
    if not samples:
        print("No recording samples returned. Is the sim running?")
        return

    sample_state = samples[0].get("state", {})
    rec_vec = vec_from_record(sample_state)

    print("\nFeature order:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {i:02d} {name}")

    print("\nLive (sensors -> vector):")
    print(live_vec)

    print("\nRecorded (sample.state -> vector):")
    print(rec_vec)

    diff = live_vec - rec_vec
    print("\nDiff (live - recorded):")
    print(diff)
    print("\nAbs diff max:", float(np.max(np.abs(diff))))

    try:
        client.delete_session()
    except Exception:
        pass


if __name__ == "__main__":
    main()
