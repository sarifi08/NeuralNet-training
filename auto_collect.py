"""Autonomous data collection via rule-based controller.

A hand-coded controller generates clean (sensor, action) pairs with
correct heading_error -> steering correlation — something human visual
driving cannot reliably produce. Just open the browser so the physics
runs; this script drives the bot.

Usage:
    python auto_collect.py --tag v5_auto --seed 42 --minutes 6
    python auto_collect.py --tag v5_s7  --seed 7  --minutes 4
"""
from __future__ import annotations
import argparse
import threading
import time
import webbrowser

import numpy as np

from game_client import GameClient

SERVER_URL = "https://ml.ferit.tech"
HZ = 20.0
STUCK_THRESHOLD = 35    # frames at speed < 0.5 before reversing (~1.75s)
RECOVERY_FRAMES = 22    # frames of reverse (~1.1s)


def rule_drive(sensors: dict) -> tuple[float, float]:
    """Rule-based controller with guaranteed correct sign convention.

    heading_error < 0  ->  checkpoint is to the RIGHT  ->  steer right (+)
    heading_error > 0  ->  checkpoint is to the LEFT   ->  steer left  (-)
    """
    nav    = sensors.get("navigation", {})
    he     = nav.get("heading_error", 0.0)
    rays   = sensors.get("rays", [50.0] * 8)
    ground = sensors.get("ground", {})
    friction = ground.get("friction", 1.0)

    # Primary: proportional heading correction
    steer = float(np.clip(-he / np.pi, -1.0, 1.0))

    # Additive wall-avoidance corrections
    front      = rays[0]
    left_near  = min(rays[1], rays[2])   # ray +45, +90
    right_near = min(rays[6], rays[7])   # ray -90, -45

    if left_near < 12:
        steer += (12 - left_near) / 12 * 0.55   # push right
    if right_near < 12:
        steer -= (12 - right_near) / 12 * 0.55  # push left
    steer = float(np.clip(steer, -1.0, 1.0))

    # Throttle: ease off when badly off-heading or near a wall
    if front < 6:
        throttle = 0.10
    elif abs(he) > 1.5:
        throttle = 0.40
    elif front < 18 or min(left_near, right_near) < 6:
        throttle = 0.55
    else:
        throttle = 0.85

    if friction < 0.4:
        throttle *= 0.75

    return float(throttle), float(steer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag",     default="v5_auto")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--minutes", type=float, default=6.0)
    args = ap.parse_args()

    duration = args.minutes * 60.0

    client = GameClient(SERVER_URL)
    session = client.create_session(
        mode="time_trial",
        player_name=f"rule_{args.tag}",
        config={"seed": args.seed, "wind_enabled": False},
    )
    browser_url = session.get("browser_url",
                              f"{SERVER_URL}/?session={session['session_id']}")
    print(f"\nOpening browser: {browser_url}")
    webbrowser.open(browser_url)
    print("Waiting 8s for simulation to load…")
    time.sleep(8)

    # Confirm sensors are live
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

    # Background sensor polling thread
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

    interval    = 1.0 / HZ
    start       = time.time()
    steps       = 0
    stuck_streak  = 0
    recovery_cnt  = 0
    last_steer    = 0.0

    while time.time() - start < duration:
        step_start = time.time()

        with sensor_lock:
            sensors = latest[0]

        if sensors is None:
            time.sleep(interval)
            continue

        sp = sensors.get("speed", 0.0)
        if sp < 0.5:
            stuck_streak += 1
        else:
            stuck_streak = 0

        if recovery_cnt > 0:
            throttle, steer = -0.7, float(np.clip(-last_steer, -1, 1))
            recovery_cnt -= 1
        elif stuck_streak >= STUCK_THRESHOLD:
            throttle, steer = -0.7, float(np.clip(-last_steer, -1, 1))
            recovery_cnt = RECOVERY_FRAMES
            stuck_streak = 0
        else:
            throttle, steer = rule_drive(sensors)
            last_steer = steer

        try:
            client.send_control_ws(throttle, steer)
        except Exception:
            try:
                client.send_control(throttle, steer)
            except Exception:
                pass

        steps += 1
        if steps % 300 == 0:
            elapsed = time.time() - start
            nav = sensors.get("navigation", {})
            cp  = nav.get("checkpoint_index", "?")
            print(f"  t={elapsed:4.0f}s  speed={sp:4.1f}  next_cp={cp}  "
                  f"steps={steps}  stuck={stuck_streak}")

        elapsed_step = time.time() - step_start
        sleep_for = interval - elapsed_step
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
        print(f"heading_error -> steering correlation: {corr:.3f}  "
              f"(human driving was ~-0.3; rule-based should be < -0.6)")

    out = f"data_{args.tag}.npz"
    np.savez(out, states=states, actions=actions, seed=args.seed)
    print(f"Saved {out}")

    try:
        client.disconnect_ws()
        client.delete_session()
    except Exception:
        pass


if __name__ == "__main__":
    main()
