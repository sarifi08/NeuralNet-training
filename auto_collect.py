"""Autonomous data collection via Follow-The-Gap controller.

Algorithm (FGM, used by F1Tenth racing bots):
  For each forward ray, score = clearance * 0.55 + alignment_to_checkpoint * 0.45.
  Pick the highest-scoring ray and steer toward its angle.

This avoids the oscillation of "heading + wall-push" controllers because the
controller commits to the most-open forward direction instead of fighting two
simultaneous corrections.

When stuck, an "informed escape" reads all 8 rays and reverses or pivots
toward the actually-open direction — no hard-coded ±0.9 bias.

Also logs the world-space position of each checkpoint as the bot passes it.

Usage:
    python auto_collect.py --tag v6_s7 --seed 7 --minutes 5
    python auto_collect.py --tag v6_s99 --seed 99 --minutes 5
"""
from __future__ import annotations
import argparse
import json
import threading
import time
import webbrowser

import numpy as np

from game_client import GameClient

SERVER_URL = "https://ml.ferit.tech"
HZ = 20.0

# 8-ray sensor: angle in bot frame (positive = LEFT, negative = RIGHT).
# Indices match the existing code's comments (rays[1,2] = +45,+90; rays[6,7] = -90,-45).
RAY_ANGLES = np.array([
    0.0,           # 0: front
    np.pi / 4,     # 1: front-left  (+45°)
    np.pi / 2,     # 2: left        (+90°)
    3 * np.pi / 4, # 3: rear-left   (+135°)
    np.pi,         # 4: rear        (180°)
   -3 * np.pi / 4, # 5: rear-right  (-135°)
   -np.pi / 2,     # 6: right       (-90°)
   -np.pi / 4,     # 7: front-right (-45°)
])

# Indices considered "forward" for path-finding (in priority order)
FORWARD_RAYS = [0, 1, 7, 2, 6]    # front, ±45°, ±90°
REAR_RAYS    = [4, 3, 5]          # rear, ±135°

CLEARANCE_MIN = 4.5    # m; reject directions with less clearance than this
STUCK_SPEED   = 0.4
STUCK_FRAMES  = 25     # ~1.25 s before triggering escape
ESCAPE_FRAMES = 30     # ~1.5 s of escape behaviour before resuming normal


def follow_the_gap(rays: np.ndarray, target_angle: float) -> tuple[float, float]:
    """Pick the forward ray maximising clearance × alignment with target.

    target_angle: desired direction in bot frame (positive = left).
    Returns (chosen_angle_radians, chosen_clearance_metres).
    """
    best_score     = -1e9
    best_angle     = target_angle
    best_clearance = 0.0

    for i in FORWARD_RAYS:
        a = float(RAY_ANGLES[i])
        d = float(rays[i])
        if d < CLEARANCE_MIN:
            continue
        clearance = min(d / 25.0, 1.0)
        align     = (np.cos(a - target_angle) + 1.0) / 2.0   # 0..1
        score     = 0.55 * clearance + 0.45 * align
        if score > best_score:
            best_score     = score
            best_angle     = a
            best_clearance = d

    return best_angle, best_clearance


def rule_drive(sensors: dict) -> tuple[float, float]:
    """FGM-based controller. No directional bias."""
    nav      = sensors.get("navigation", {})
    he       = float(nav.get("heading_error", 0.0))      # +left, -right
    rays     = np.asarray(sensors.get("rays", [50.0] * 8), dtype=float)
    friction = sensors.get("ground", {}).get("friction", 1.0)

    # Target direction in bot frame, capped to ±90° (only forward arc matters)
    target = float(np.clip(he, -np.pi / 2, np.pi / 2))

    chosen_angle, _ = follow_the_gap(rays, target)

    # Convert bot-frame angle to steering [-1, 1].
    # Convention here: positive steer = right; positive angle (left) ⇒ negative steer.
    steer = float(np.clip(-chosen_angle / (np.pi / 2), -1.0, 1.0))

    # Throttle: only slow down for genuinely close obstacles
    front = float(rays[0])
    if front < 5.0:
        throttle = 0.30
    elif front < 10.0:
        throttle = 0.60
    else:
        throttle = 0.90

    # When the chosen direction is far from the desired direction, the bot is
    # making a tight evasive turn — ease throttle slightly so it doesn't
    # understeer into a wall.
    if abs(chosen_angle - target) > 0.7:
        throttle *= 0.75

    if friction < 0.4:
        throttle *= 0.65

    return float(throttle), float(steer)


def informed_escape(rays: np.ndarray) -> tuple[float, float]:
    """Stuck → look at all rays, head toward the most-open direction.

    If a forward ray has clearance > 5.5 m: drive forward toward that ray.
    Else reverse toward whichever rear ray is most open.
    """
    rays = np.asarray(rays, dtype=float)

    # Best forward option
    fwd_idx     = max(FORWARD_RAYS, key=lambda i: rays[i])
    fwd_clear   = float(rays[fwd_idx])
    if fwd_clear > 5.5:
        a     = float(RAY_ANGLES[fwd_idx])
        steer = float(np.clip(-a / (np.pi / 2), -1.0, 1.0))
        return 0.65, steer

    # Reverse toward the most-open rear ray
    rear_idx   = max(REAR_RAYS, key=lambda i: rays[i])
    rear_angle = float(RAY_ANGLES[rear_idx])
    # When reversing, steer-direction is inverted relative to the desired
    # rear-swing. To make the rear go LEFT (positive angle), use steer = +1.
    if rear_angle > 0.5:        # rear-left is most open
        steer = +1.0
    elif rear_angle < -0.5:     # rear-right is most open
        steer = -1.0
    else:                        # straight back
        steer = 0.0
    return -0.65, steer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag",     default="v6_auto")
    ap.add_argument("--seed",    type=int,   default=42)
    ap.add_argument("--minutes", type=float, default=5.0)
    args = ap.parse_args()

    duration = args.minutes * 60.0

    client  = GameClient(SERVER_URL)
    session = client.create_session(
        mode="time_trial",
        player_name=f"fgm_{args.tag}",
        config={"seed": args.seed, "wind_enabled": False},
    )
    browser_url = session.get(
        "browser_url",
        f"{SERVER_URL}/?session={session['session_id']}",
    )
    print(f"\nOpening browser: {browser_url}")
    webbrowser.open(browser_url)
    print("Waiting 8 s for simulation to load…")
    time.sleep(8)

    print("Checking sensors…", end="", flush=True)
    for _ in range(20):
        try:
            s = client.get_sensors()
            if "speed" in s and "navigation" in s:
                print(" ready!")
                break
        except Exception:
            pass
        time.sleep(0.5)
        print(".", end="", flush=True)
    else:
        print("\nERROR: simulation did not start. Is the browser tab open?")
        client.delete_session()
        return

    client.connect_ws()
    client.start_recording(sample_rate=20)
    print(f"Recording started — driving for {args.minutes:.1f} min on seed {args.seed}…\n")

    sensor_lock = threading.Lock()
    latest: list = [None]
    stop_flag: list = [False]

    def poll():
        while not stop_flag[0]:
            try:
                raw = client.get_sensors()
                with sensor_lock:
                    latest[0] = raw
            except Exception:
                pass

    threading.Thread(target=poll, daemon=True).start()

    interval     = 1.0 / HZ
    start        = time.time()
    steps        = 0
    stuck_streak = 0
    escape_cnt   = 0
    stuck_events = 0

    cp_positions: dict[int, dict] = {}
    last_cp_idx: int | None       = None

    while time.time() - start < duration:
        step_start = time.time()

        with sensor_lock:
            sensors = latest[0]

        if sensors is None:
            time.sleep(interval)
            continue

        sp = float(sensors.get("speed", 0.0))
        if sp < STUCK_SPEED:
            stuck_streak += 1
        else:
            stuck_streak = 0

        # Checkpoint position logging
        nav    = sensors.get("navigation", {})
        cp_idx = nav.get("checkpoint_index", 0)
        pos    = sensors.get("position", {}) or {}
        if last_cp_idx is not None and cp_idx != last_cp_idx:
            if cp_idx not in cp_positions:
                cp_positions[cp_idx] = {
                    "x": float(pos.get("x", 0.0)),
                    "y": float(pos.get("y", 0.0)),
                    "z": float(pos.get("z", 0.0)),
                    "t": time.time() - start,
                }
                print(f"  ★ checkpoint {cp_idx} hit at "
                      f"({pos.get('x', 0):.1f}, {pos.get('z', 0):.1f}) "
                      f"t={time.time()-start:.1f}s")
        last_cp_idx = cp_idx

        rays = np.asarray(sensors.get("rays", [50.0] * 8), dtype=float)

        # Trigger / continue escape
        if escape_cnt == 0 and stuck_streak >= STUCK_FRAMES:
            escape_cnt = ESCAPE_FRAMES
            stuck_events += 1
            stuck_streak = 0

        if escape_cnt > 0:
            throttle, steer = informed_escape(rays)
            escape_cnt -= 1
        else:
            throttle, steer = rule_drive(sensors)

        try:
            client.send_control_ws(throttle, steer)
        except Exception:
            try:
                client.send_control(throttle, steer)
            except Exception:
                pass

        steps += 1
        if steps % 200 == 0:
            elapsed = time.time() - start
            print(f"  t={elapsed:4.0f}s  speed={sp:4.1f}  "
                  f"next_cp={cp_idx}  steps={steps}  "
                  f"stuck_streak={stuck_streak}  escapes={stuck_events}")

        elapsed_step = time.time() - step_start
        sleep_for    = interval - elapsed_step
        if sleep_for > 0:
            time.sleep(sleep_for)

    stop_flag[0] = True
    print("\nStopping recording…")
    info = client.stop_recording()
    print(f"Samples on server: {info.get('sample_count', '?')}")

    states, actions = client.get_recording_as_arrays()
    print(f"Downloaded: states={states.shape}  actions={actions.shape}")

    if len(states) > 0:
        corr = np.corrcoef(states[:, 1], actions[:, 1])[0, 1]
        print(f"heading_error → steering correlation: {corr:.3f}  "
              f"(target: < -0.5)")
        steer_std = float(actions[:, 1].std())
        thr_mean  = float(actions[:, 0].mean())
        print(f"steering std: {steer_std:.3f}  throttle mean: {thr_mean:.3f}")
        print(f"checkpoints reached: {len(cp_positions)}  escapes triggered: {stuck_events}")

    out = f"data_{args.tag}.npz"
    np.savez(out, states=states, actions=actions, seed=args.seed)
    print(f"Saved {out}")

    cp_path = f"checkpoints_seed{args.seed}.json"
    with open(cp_path, "w") as f:
        json.dump({"seed": args.seed, "positions": cp_positions}, f, indent=2)
    print(f"Saved {cp_path}")

    try:
        client.disconnect_ws()
        client.delete_session()
    except Exception:
        pass


if __name__ == "__main__":
    main()
