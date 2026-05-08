"""Autonomous data collection — smooth heading-pursuit + decisive obstacle override.

Lessons from previous iterations:
  - 'heading + bilateral wall-push' (v5) oscillates: in any corridor both
    walls trigger pushes that cancel, controller stops.
  - 'snap-to-ray-angle FGM' (v6) chatters: chosen angle flips between
    discrete options each frame, wheels never settle, bot can't accelerate.

This version:
  1. Continuous heading-pursuit:  steer = -he / 1.2   (gentle, smooth)
  2. Override only when FRONT is blocked: choose ONE side
     (the more open one) and add a strong unilateral correction.
  3. Aggressive throttle (≥0.75 in any half-clear path) so the bot has
     enough force to overcome static friction and actually move.
  4. Long escape (2.5 s) with reverse + hard turn; direction picked from
     ray clearances, no hardcoded bias.
  5. Logs world-space checkpoint positions for later path-planning.

Usage:
    python auto_collect.py --tag v7_s7  --seed 7  --minutes 4
    python auto_collect.py --tag v7_s99 --seed 99 --minutes 4
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

# Ray indices (matches existing convention)
#   0: front       1: front-left (+45°)   2: left (+90°)
#   3: rear-left   4: rear        5: rear-right
#   6: right (-90°)  7: front-right (-45°)

STUCK_SPEED         = 0.4
STUCK_FRAMES        = 30      # ~1.5 s before triggering escape
ESCAPE_REV_FRAMES   = 32      # ~1.6 s of reverse + hard turn
ESCAPE_FWD_FRAMES   = 28      # ~1.4 s of forward push afterwards
POST_ESCAPE_FRAMES  = 100     # ~5.0 s of detour bias — longer commitment

# Professional-driver escape: like a 3-point turn. Reverse with a turn
# (chosen direction), then drive forward with a turn that continues the
# rotation in the SAME direction. If two escapes in a row fail to make
# progress, force-flip direction on the third — don't keep retrying the
# same way.
PROGRESS_DELTA      = 4.0     # m of distance change to count as 'progress'

# Approach scaling — the closer the checkpoint, the tighter we steer.
def steer_scale(distance: float) -> float:
    if distance < 6.0:
        return 0.45
    if distance < 12.0:
        return 0.75
    return 1.2


def rule_drive(sensors: dict, post_escape_dir: float = 0.0,
               post_escape_frames: int = 0) -> tuple[float, float]:
    """Smooth heading-pursuit with decisive front-block override and
    distance-aware aggressiveness."""
    nav      = sensors.get("navigation", {})
    he       = float(nav.get("heading_error", 0.0))                # +left, -right
    distance = float(nav.get("distance", 50.0))
    rays     = np.asarray(sensors.get("rays", [50.0] * 8), dtype=float)
    friction = sensors.get("ground", {}).get("friction", 1.0)

    front       = float(rays[0])
    front_left  = float(rays[1])   # +45°
    left        = float(rays[2])   # +90°
    front_right = float(rays[7])   # -45°
    right       = float(rays[6])   # -90°

    # Primary: smooth heading pursuit, scaled by distance so the bot turns
    # MUCH harder when close to the checkpoint (was too gentle before — it
    # would graze past at 4-5 m without ever fully aligning).
    steer = -he / steer_scale(distance)

    # Post-escape bias: after each escape, push toward the side we just
    # escaped to, decaying linearly. Prevents the bot from immediately
    # swinging back into the wall it just left.
    if post_escape_frames > 0:
        decay = post_escape_frames / POST_ESCAPE_FRAMES
        steer += post_escape_dir * 0.55 * decay

    # Front-blocked override — pick ONE side, the more open one
    if front < 7.0:
        left_score  = front_left  + 0.4 * left
        right_score = front_right + 0.4 * right
        if left_score > right_score + 1.0:
            steer -= 0.85
        elif right_score > left_score + 1.0:
            steer += 0.85
        else:
            steer += -0.85 if he > 0 else 0.85

    steer = float(np.clip(steer, -1.0, 1.0))

    # Throttle — be aggressive in clear paths, but BRAKE when close to a
    # checkpoint and badly off-heading (otherwise we orbit at high speed
    # and can never turn tight enough to actually hit it).
    if distance < 10.0 and abs(he) > 1.0:
        throttle = 0.25                       # slow → tighter turn radius
    elif front < 3.5:
        throttle = 0.45
    elif front < 7.0:
        throttle = 0.75
    elif front < 12.0:
        throttle = 0.9
    else:
        throttle = 1.0

    # Slight ease in very tight turns
    if abs(steer) > 0.75 and distance > 8.0:
        throttle = min(throttle, 0.7)

    if friction < 0.4:
        throttle *= 0.7

    return float(throttle), float(steer)


def pick_escape_dir(rays: np.ndarray) -> float:
    """Decide which way the bot should rotate during escape.

    -1 = rotate LEFT (front swings left, bot ends up facing leftwards)
    +1 = rotate RIGHT
    """
    rays = np.asarray(rays, dtype=float)
    left_score  = float(rays[1] + rays[2] + rays[3])   # +45° + +90° + +135°
    right_score = float(rays[5] + rays[6] + rays[7])   # -135° + -90° + -45°
    return -1.0 if left_score > right_score else +1.0


def escape_reverse(rays: np.ndarray, escape_dir: float) -> tuple[float, float]:
    """Reverse while rotating in the given direction.

    Ackermann under reverse: steer = +1 (right wheel turn) drives the FRONT
    of the bot to swing LEFT. So to rotate LEFT (escape_dir=-1), use steer=+1.
    Formula: steer = -escape_dir.
    """
    return -0.85, float(-escape_dir)


def escape_forward(rays: np.ndarray, escape_dir: float) -> tuple[float, float, float]:
    """Drive forward continuing the same rotation as escape_reverse.

    Forward + steer=+escape_dir continues the same rotation that the reverse
    started. This completes the 3-point-turn: bot ends up facing roughly
    perpendicular to its original heading, ideally on a clear side of the wall.
    """
    rays  = np.asarray(rays, dtype=float)
    front = float(rays[0])
    steer = 0.7 * float(escape_dir)
    throttle = 0.85 if front > 5.0 else 0.55
    return throttle, steer, float(escape_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag",     default="v7_auto")
    ap.add_argument("--seed",    type=int,   default=42)
    ap.add_argument("--minutes", type=float, default=4.0)
    args = ap.parse_args()

    duration = args.minutes * 60.0

    client  = GameClient(SERVER_URL)
    session = client.create_session(
        mode="time_trial",
        player_name=f"hp_{args.tag}",
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
    print(f"Recording — driving for {args.minutes:.1f} min on seed {args.seed}…\n")

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

    interval           = 1.0 / HZ
    start              = time.time()
    steps              = 0
    stuck_streak       = 0
    escape_rev         = 0     # frames remaining of reverse phase
    escape_fwd         = 0     # frames remaining of forward phase
    stuck_events       = 0
    post_escape_frames = 0     # frames remaining of post-escape detour bias
    post_escape_dir    = 0.0   # +1 / -1 / 0 — direction the last escape committed to

    # Force-flip tracking — if last escape made no progress, flip direction
    # next time instead of repeating the same one.
    last_escape_dir       = 0.0     # -1 / +1 — direction of last escape's rotation
    distance_at_escape    = None    # checkpoint distance when last escape triggered
    failed_escape_streak  = 0       # consecutive escapes without progress
    current_escape_dir    = 0.0     # direction of currently-active escape

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

        # Checkpoint position logging — also resets frustration state
        nav      = sensors.get("navigation", {})
        cp_idx   = nav.get("checkpoint_index", 0)
        distance = float(nav.get("distance", 50.0))
        pos      = sensors.get("position", {}) or {}
        if last_cp_idx is not None and cp_idx != last_cp_idx:
            cp_positions[int(cp_idx)] = {
                "x": float(pos.get("x", 0.0)),
                "y": float(pos.get("y", 0.0)),
                "z": float(pos.get("z", 0.0)),
                "t": time.time() - start,
            }
            print(f"  ★ next_cp now {cp_idx} — bot at "
                  f"({pos.get('x', 0):.1f}, {pos.get('z', 0):.1f}) "
                  f"t={time.time()-start:.1f}s")
            # Made checkpoint progress — clear failure tracking
            failed_escape_streak = 0
            distance_at_escape   = None
            last_escape_dir      = 0.0
        last_cp_idx = cp_idx

        rays = np.asarray(sensors.get("rays", [50.0] * 8), dtype=float)

        # Trigger an escape if we're stuck and not already in one
        if escape_rev == 0 and escape_fwd == 0 and stuck_streak >= STUCK_FRAMES:
            # Did the LAST escape make checkpoint progress?
            last_failed = (
                distance_at_escape is not None
                and abs(distance - distance_at_escape) < PROGRESS_DELTA
            )
            if last_failed and last_escape_dir != 0.0:
                # Force-flip: same wall trapped us twice with the same direction
                current_escape_dir   = -last_escape_dir
                failed_escape_streak += 1
                print(f"  ↻ escape force-flipped to "
                      f"{'LEFT' if current_escape_dir<0 else 'RIGHT'} "
                      f"(streak={failed_escape_streak})  d={distance:.1f}m")
            else:
                # Pick by ray clearances
                current_escape_dir   = pick_escape_dir(rays)
                failed_escape_streak = 0

            distance_at_escape = distance
            escape_rev         = ESCAPE_REV_FRAMES
            stuck_events      += 1
            stuck_streak       = 0

        # Drive priority: escape_rev > escape_fwd > normal (with bias)
        if escape_rev > 0:
            throttle, steer = escape_reverse(rays, current_escape_dir)
            escape_rev -= 1
            if escape_rev == 0:
                escape_fwd = ESCAPE_FWD_FRAMES
        elif escape_fwd > 0:
            throttle, steer, chosen_dir = escape_forward(rays, current_escape_dir)
            escape_fwd -= 1
            if escape_fwd == 0:
                post_escape_dir    = chosen_dir
                post_escape_frames = POST_ESCAPE_FRAMES
                last_escape_dir    = chosen_dir
        else:
            throttle, steer = rule_drive(sensors,
                                         post_escape_dir=post_escape_dir,
                                         post_escape_frames=post_escape_frames)
            if post_escape_frames > 0:
                post_escape_frames -= 1

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
            if escape_rev:
                mode = f"ESC-REV{'L' if current_escape_dir<0 else 'R'}"
            elif escape_fwd:
                mode = f"ESC-FWD{'L' if current_escape_dir<0 else 'R'}"
            elif post_escape_frames > 0:
                mode = f"BIAS{int(post_escape_dir):+d}"
            else:
                mode = "drive   "
            print(f"  t={elapsed:4.0f}s  spd={sp:4.1f}  cp={cp_idx}  d={distance:5.1f}m  "
                  f"steps={steps}  esc={stuck_events}  flips={failed_escape_streak}  "
                  f"mode={mode}  thr={throttle:+.2f} str={steer:+.2f}")

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
        print(f"heading_error → steering correlation: {corr:.3f}  (target < -0.5)")
        print(f"steering std: {actions[:,1].std():.3f}  throttle mean: {actions[:,0].mean():.3f}")
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
