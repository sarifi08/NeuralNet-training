"""Autonomous benchmark runner — opens the browser automatically for each run.

The simulation runs inside the browser (physics engine is client-side).
Without an open browser tab, the server sends no sensor data and the bot
cannot move. This script auto-opens each session URL, waits for the
simulation to load, then lets the trained policy drive for the allotted time.

Sensor data is polled via REST (GET /sensors); control commands are sent
via WebSocket for low-latency delivery.

Usage:
    python run_agent.py --weights nav_v1.npz --tag v1 --runs 5 --seed 42
    python run_agent.py --weights nav_v7.npz --tag v7 --seeds 42 7 99
    python run_agent.py --weights nav_v8_cnn.pt --tag v8 --module drive2win.cnn --seed 42
"""
from __future__ import annotations
import argparse
import importlib
import json
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np

from game_client import GameClient
from drive2win import nn as nn_mod
from drive2win.eval import score_runs
from drive2win.normalize import sensors_to_input, clip_action
from drive2win.smooth import make_smooth_policy

SERVER_URL = "https://ml.ferit.tech"
TARGET_CHECKPOINTS = 12
BROWSER_LOAD_TIMEOUT = 20   # seconds to wait for browser to load & send first sensor reading


def flatten_sensors(raw: dict) -> dict:
    """Convert nested REST /sensors response to the flat dict sensors_to_input() expects."""
    nav = raw.get("navigation", {})
    ground = raw.get("ground", {})
    return {
        "speed": raw["speed"],
        "heading_error": nav["heading_error"],
        "checkpoint_distance": nav["distance"],
        "rays": raw["rays"],
        "ground_friction": ground.get("friction", 1.0),
        # keep nested fields for checkpoint/crash tracking
        "navigation": nav,
        "position": raw.get("position", {}),
    }


def rest_run_policy(client, policy_fn, duration: float = 60.0, hz: float = 20.0) -> dict:
    """REST-polling control loop with background sensor thread.

    A background thread continuously polls GET /sensors as fast as the network
    allows (~4 Hz over a transatlantic link). The main control loop runs at the
    target Hz using the most recently fetched sensor reading, decoupling control
    frequency from REST latency.
    """
    # ── Background sensor polling thread ─────────────────────────────────
    sensor_lock = threading.Lock()
    latest: list = [None]   # [sensors_dict | None]
    stop_flag: list = [False]

    def poll_loop():
        while not stop_flag[0]:
            try:
                raw = client.get_sensors()
                s = flatten_sensors(raw)
                with sensor_lock:
                    latest[0] = s
            except Exception:
                pass

    poll_thread = threading.Thread(target=poll_loop, daemon=True)
    poll_thread.start()

    # Wait for first reading (already confirmed by run_one's load wait)
    t_wait = time.time()
    while latest[0] is None and time.time() - t_wait < 5.0:
        time.sleep(0.05)

    # ── Control loop ─────────────────────────────────────────────────────
    # Proper 3-point-turn escape (ported from auto_collect.py escape).
    STUCK_THRESHOLD     = 30    # frames at speed<0.3 + wall in front (~1.5s)
    ESCAPE_REV_FRAMES   = 30    # ~1.5s reverse with hard turn
    ESCAPE_FWD_FRAMES   = 22    # ~1.1s forward with continuing turn
    POST_ESCAPE_FRAMES  = 60    # ~3.0s of detour bias added to network steering

    interval = 1.0 / hz
    start = time.time()
    steps = 0
    checkpoints_passed = 0
    last_cp_idx: int | None = None
    crashes = 0
    last_pos = None
    stuck_streak = 0
    max_stuck = 0
    escape_rev = 0          # frames remaining of reverse phase
    escape_fwd = 0          # frames remaining of forward phase
    escape_dir = 0.0        # -1 (rotate left) / +1 (rotate right) for current escape
    post_escape_dir    = 0.0
    post_escape_frames = 0  # frames remaining of post-escape steering bias
    escapes_triggered  = 0
    track = []
    next_log = start

    while time.time() - start < duration:
        step_start = time.time()

        with sensor_lock:
            sensors = latest[0]

        if sensors is None:
            time.sleep(interval)
            continue

        state = {"sensors": sensors, "position": sensors["position"]}

        # Checkpoint tracking via checkpoint_index increments
        nav = sensors["navigation"]
        cp_idx = nav.get("checkpoint_index", 0)
        if last_cp_idx is None:
            last_cp_idx = cp_idx
        elif cp_idx != last_cp_idx:
            checkpoints_passed += (cp_idx - last_cp_idx) % TARGET_CHECKPOINTS
            last_cp_idx = cp_idx

        # Stuck heuristic — REQUIRES a wall in front, not just low speed.
        # Without this check, mud / ice / sand (where speed naturally drops
        # to 0.1-0.3 even when accelerating forward) triggers spurious
        # reverses, sabotaging the trained network's correct mud behaviour.
        sp = sensors["speed"]
        rays = sensors.get("rays", [50.0] * 8)
        front_arc_min = float(min(rays[0], rays[1], rays[7]))
        wall_in_front = front_arc_min < 5.0
        if sp < 0.3 and wall_in_front:
            stuck_streak += 1
        else:
            max_stuck = max(max_stuck, stuck_streak)
            stuck_streak = 0

        # Crash detection — position teleport > 5 m
        pos = sensors["position"]
        if last_pos is not None and pos:
            dx = pos.get("x", 0) - last_pos.get("x", 0)
            dz = pos.get("z", 0) - last_pos.get("z", 0)
            if (dx * dx + dz * dz) > 25.0:
                crashes += 1
        last_pos = pos

        # Trigger a 3-point-turn escape if we're stuck against a wall and not
        # already inside one.
        if (escape_rev == 0 and escape_fwd == 0
                and stuck_streak >= STUCK_THRESHOLD):
            # Pick rotation direction by which forward arc has more clearance.
            # left_score uses rays at +45/+90/+135; right uses -45/-90/-135.
            # Whichever side is more open is the side we want the bot to face
            # AFTER rotating.
            try:
                rs = list(rays)
                left_score  = float(rs[1] + rs[2] + rs[3])
                right_score = float(rs[5] + rs[6] + rs[7])
            except Exception:
                left_score = right_score = 1.0
            escape_dir = -1.0 if left_score > right_score else +1.0
            escape_rev = ESCAPE_REV_FRAMES
            escapes_triggered += 1
            stuck_streak = 0
            if hasattr(policy_fn, "reset"):
                policy_fn.reset()

        if escape_rev > 0:
            # Reverse + hard turn. Ackermann under reverse: steer = -escape_dir
            # makes the FRONT swing toward escape_dir (i.e. rotates the bot
            # toward the more-open side).
            throttle = -0.7
            steering = -float(escape_dir)
            escape_rev -= 1
            if escape_rev == 0:
                escape_fwd = ESCAPE_FWD_FRAMES
        elif escape_fwd > 0:
            # Forward + continuing turn — completes the rotation in same
            # direction so the bot ends up genuinely facing a new heading
            # instead of right back at the wall it came from.
            throttle = 0.7
            steering = 0.7 * float(escape_dir)
            escape_fwd -= 1
            if escape_fwd == 0:
                # Hand off to the network with a 3-second decaying detour
                # bias so heading-pursuit doesn't immediately swing the bot
                # back into the wall it just left.
                post_escape_dir    = float(escape_dir)
                post_escape_frames = POST_ESCAPE_FRAMES
                if hasattr(policy_fn, "reset"):
                    policy_fn.reset()
        else:
            throttle, steering = policy_fn(state)

            # Distance-aware steering boost. The network was trained on data
            # where heading_error → steering correlation is -0.6, but its
            # outputs get gentler as it approaches the checkpoint. Boost
            # when close so we actually hit the cp instead of grazing past.
            distance = float(nav.get("distance", 50.0))
            if distance < 8.0:
                steering = max(-1.0, min(1.0, steering * 1.5))
            elif distance < 14.0:
                steering = max(-1.0, min(1.0, steering * 1.2))

            # Post-escape bias: push toward the side the escape committed to
            # so the bot doesn't immediately swing back into the wall.
            if post_escape_frames > 0:
                decay = post_escape_frames / POST_ESCAPE_FRAMES
                steering = max(-1.0, min(1.0,
                                          steering + post_escape_dir * 0.45 * decay))
                post_escape_frames -= 1

            last_policy_steering = steering

        try:
            client.send_control_ws(throttle, steering)
        except Exception:
            try:
                client.send_control(throttle, steering)
            except Exception:
                pass

        steps += 1

        now = time.time()
        if now >= next_log:
            track.append({"t": now - start, "position": pos, "speed": sp})
            next_log = now + 1.0

        elapsed_step = time.time() - step_start
        sleep_for = interval - elapsed_step
        if sleep_for > 0:
            time.sleep(sleep_for)

    stop_flag[0] = True
    elapsed = time.time() - start
    return {
        "steps": steps,
        "elapsed": elapsed,
        "checkpoints_passed": checkpoints_passed,
        "crashes": crashes,
        "escapes": escapes_triggered,
        "min_speed_streak": max(max_stuck, stuck_streak),
        "track": track,
    }


def make_mlp_policy(weights_path: str, alpha: float = 0.7):
    w = nn_mod.load(weights_path)

    def base(state):
        x = sensors_to_input(state["sensors"])
        return clip_action(nn_mod.forward(x, w))

    return make_smooth_policy(base, alpha=alpha)


def make_module_policy(module_path: str, weights_path: str, alpha: float = 0.7):
    mod = importlib.import_module(module_path)
    raw_policy = mod.make_policy(weights_path)
    return make_smooth_policy(raw_policy, alpha=alpha)


def run_one(policy, seed: int, run_idx: int, total_runs: int,
            duration: float, player_name: str) -> dict:
    client = GameClient(SERVER_URL)
    session = client.create_session(
        mode="time_trial",
        player_name=f"{player_name}_run{run_idx}",
        config={"seed": seed, "wind_enabled": False},
    )
    browser_url = session.get(
        "browser_url",
        f"{SERVER_URL}/?session={session['session_id']}",
    )

    print(f"\n  run {run_idx}/{total_runs}  seed={seed}  session={session['session_id'][:8]}…")
    print(f"  Opening browser: {browser_url}")
    webbrowser.open(browser_url)

    # Connect WS for control sending (sensor data comes via REST)
    client.connect_ws()

    # Wait until the browser loads and REST sensors become available
    print(f"  Waiting for simulation (up to {BROWSER_LOAD_TIMEOUT}s)…", end="", flush=True)
    deadline = time.time() + BROWSER_LOAD_TIMEOUT
    ready = False
    while time.time() < deadline:
        try:
            s = client.get_sensors()
            if s and "speed" in s and "navigation" in s:
                ready = True
                break
        except Exception:
            pass
        time.sleep(0.5)
        print(".", end="", flush=True)

    if not ready:
        print(" TIMEOUT — browser did not start. Is it open?")
        client.disconnect_ws()
        try:
            client.delete_session()
        except Exception:
            pass
        return {"checkpoints_passed": 0, "crashes": 0, "elapsed": duration,
                "steps": 0, "track": [], "min_speed_streak": 0}

    print(" ready!")

    if hasattr(policy, "reset"):
        policy.reset()

    result = rest_run_policy(client, policy, duration=duration, hz=20.0)
    print(f"    checkpoints={result['checkpoints_passed']}/{TARGET_CHECKPOINTS}  "
          f"crashes={result['crashes']}  escapes={result.get('escapes', 0)}  "
          f"steps={result['steps']}")

    client.disconnect_ws()
    try:
        client.delete_session()
    except Exception:
        pass

    return result


def main():
    ap = argparse.ArgumentParser(description="Auto-browser benchmark runner")
    ap.add_argument("--tag",     required=True,  help="Iteration tag e.g. v1, v3-deeper")
    ap.add_argument("--weights", default=None,   help="Weights file. Defaults to nav_<tag>.npz")
    ap.add_argument("--module",  default=None,   help="Custom policy module (e.g. drive2win.cnn)")
    ap.add_argument("--seeds",   type=int, nargs="+", default=[42])
    ap.add_argument("--runs",    type=int, default=5)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--alpha",   type=float, default=0.7,
                    help="EMA smoothing alpha (1.0 = no smoothing)")
    ap.add_argument("--name",    default="bot", help="Player name prefix")
    args = ap.parse_args()

    weights = args.weights or f"nav_{args.tag}.npz"

    if args.module:
        policy = make_module_policy(args.module, weights, alpha=args.alpha)
    else:
        policy = make_mlp_policy(weights, alpha=args.alpha)

    out_dir = Path("benchmarks")
    out_dir.mkdir(exist_ok=True)

    all_seed_results = []
    for seed in args.seeds:
        print(f"\n{'='*56}")
        print(f"  seed={seed}  weights={weights}  runs={args.runs}")
        print(f"{'='*56}")

        runs_out = []
        for i in range(args.runs):
            result = run_one(
                policy=policy,
                seed=seed,
                run_idx=i + 1,
                total_runs=args.runs,
                duration=args.duration,
                player_name=args.name,
            )
            runs_out.append(result)

        summary = score_runs(runs_out, TARGET_CHECKPOINTS)
        all_seed_results.append({"seed": seed, "summary": summary, "runs": runs_out})

        s = summary
        print(f"\n  seed {seed}: complete={int(s['completion_rate']*s['n_runs'])}/{s['n_runs']}  "
              f"median_lap={s['median_lap_time']:.1f}s  crashes={s['mean_crashes']:.1f}  "
              f"max_cp={s['max_checkpoints']}")

    log_path = out_dir / f"{args.tag}.json"
    log = {
        "tag": args.tag, "weights": weights, "module": args.module,
        "runs_per_seed": args.runs, "duration_s": args.duration,
        "seeds": [
            {"seed": r["seed"], "summary": r["summary"], "runs": r["runs"]}
            for r in all_seed_results
        ],
    }
    log_path.write_text(json.dumps(log, indent=2, default=float))
    print(f"\nwrote {log_path}")


if __name__ == "__main__":
    main()
