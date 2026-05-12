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
from typing import IO

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


def rest_run_policy(client, policy_fn, duration: float = 60.0, hz: float = 20.0,
                    frame_log_path: str | None = None,
                    print_status: bool = True) -> dict:
    """REST-polling control loop with background sensor thread.

    A background thread continuously polls GET /sensors as fast as the network
    allows (~4 Hz over a transatlantic link). The main control loop runs at the
    target Hz using the most recently fetched sensor reading, decoupling control
    frequency from REST latency.

    If frame_log_path is provided, every control frame is appended as a JSON
    line capturing sensors, the network's RAW output (before smoothing/boost/
    escape override), the final commanded output, escape state, and an `event`
    field on checkpoint passes / escape triggers / crashes. This is the
    primary diagnostic tool for "what is the bot actually doing?".

    print_status=True prints a 1-second live status line plus inline events
    so you can interpret runs without opening the JSONL.
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
    prev_distance: float | None = None  # nav.distance from PREVIOUS frame, used to
                                        # report close-pass distance on cp events
    crashes = 0
    last_crashes = 0
    last_pos = None
    stuck_streak = 0
    max_stuck = 0
    escape_rev = 0          # frames remaining of reverse phase
    escape_fwd = 0          # frames remaining of forward phase
    escape_dir = 0.0        # -1 (rotate left) / +1 (rotate right) for current escape
    post_escape_dir    = 0.0
    post_escape_frames = 0  # frames remaining of post-escape steering bias
    escapes_triggered  = 0
    # Track whether the CURRENT escape is actually moving the bot, and what
    # the previous escape's outcome was. If the previous escape did not
    # produce ANY motion (max sp during that escape stayed below 0.3), the
    # next escape forces the opposite direction even when the rays say
    # otherwise — because the rays already said this and it didn't work.
    escape_motion_seen   = False  # any sp >= 0.3 during the current escape?
    last_escape_dir      = 0.0    # the direction the previous escape used
    last_escape_succeeded = True  # True until we observe a failed escape
    track = []
    next_log = start
    next_status = start

    log_fp: IO | None = None
    if frame_log_path:
        Path(frame_log_path).parent.mkdir(parents=True, exist_ok=True)
        log_fp = open(frame_log_path, "w")

    try:
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
            current_distance = float(nav.get("distance", -1.0))
            cp_event_dist = None
            if last_cp_idx is None:
                last_cp_idx = cp_idx
            elif cp_idx != last_cp_idx:
                delta = (cp_idx - last_cp_idx) % TARGET_CHECKPOINTS
                checkpoints_passed += delta
                # Use the PREVIOUS frame's distance — current_distance has
                # already updated to point at the next cp. Previous frame is
                # the last reading where the cp we just hit was still target,
                # so its distance is the actual close-pass distance.
                cp_event_dist = (prev_distance if prev_distance is not None
                                 else current_distance)
                if print_status:
                    print(f"  [CP HIT cp={checkpoints_passed} t={time.time()-start:5.1f}s "
                          f"close_pass={cp_event_dist:.2f}m]")
                last_cp_idx = cp_idx

            # Stuck heuristic — REQUIRES a wall in front, not just low speed.
            # Without this check, mud / ice / sand (where speed naturally drops
            # to 0.1-0.3 even when accelerating forward) triggers spurious
            # reverses, sabotaging the trained network's correct mud behaviour.
            #
            # ALSO gated on (escape_rev == 0 and escape_fwd == 0): if we keep
            # incrementing stuck_streak while the escape itself is running,
            # a wedge-pocket where escape can't actually displace the bot
            # leaves stuck_streak >= STUCK_THRESHOLD the moment escape ends,
            # so the trigger fires again on the very next frame and the bot
            # gets pinned in an infinite escape loop. (We saw exactly this
            # for 22s in the seed-42 v12 run.)
            sp = sensors["speed"]
            rays = sensors.get("rays", [50.0] * 8)
            front_arc_min = float(min(rays[0], rays[1], rays[7]))
            wall_in_front = front_arc_min < 5.0
            in_escape = escape_rev > 0 or escape_fwd > 0
            if not in_escape:
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
            crash_event = crashes > last_crashes
            if crash_event and print_status:
                print(f"  [CRASH t={time.time()-start:5.1f}s total={crashes}]")
            last_crashes = crashes

            # Trigger a 3-point-turn escape if we're stuck against a wall and not
            # already inside one.
            escape_event_meta = None
            if (escape_rev == 0 and escape_fwd == 0
                    and stuck_streak >= STUCK_THRESHOLD):
                try:
                    rs = list(rays)
                    left_score  = float(rs[1] + rs[2] + rs[3])
                    right_score = float(rs[5] + rs[6] + rs[7])
                except Exception:
                    left_score = right_score = 1.0
                geom_dir = -1.0 if left_score > right_score else +1.0
                # If the LAST escape didn't move the bot AND the geometry
                # picked the same direction again, force the opposite. The
                # rays haven't changed (we haven't moved), so the choice is
                # the same — but we already know it doesn't work.
                forced_flip = False
                if (not last_escape_succeeded
                        and last_escape_dir != 0.0
                        and geom_dir == last_escape_dir):
                    escape_dir = -last_escape_dir
                    forced_flip = True
                else:
                    escape_dir = geom_dir
                escape_rev = ESCAPE_REV_FRAMES
                escapes_triggered += 1
                stuck_streak = 0
                escape_motion_seen = False
                escape_event_meta = (left_score, right_score, sp, front_arc_min,
                                     forced_flip)
                if print_status:
                    flip_tag = " (FLIP, prev escape failed)" if forced_flip else ""
                    print(f"  [ESCAPE TRIGGER t={time.time()-start:5.1f}s "
                          f"dir={int(escape_dir):+d} L={left_score:.1f} R={right_score:.1f} "
                          f"sp={sp:.2f} front={front_arc_min:.1f}m]{flip_tag}")
                if hasattr(policy_fn, "reset"):
                    policy_fn.reset()

            # Decide control. nn_t/nn_s capture the network's pre-override
            # output (None during escape, since the network isn't queried).
            nn_t: float | None = None
            nn_s: float | None = None
            phase = "drive"
            if escape_rev > 0:
                throttle = -0.7
                steering = -float(escape_dir)
                escape_rev -= 1
                phase = "esc_rev"
                # Track whether the bot is actually moving during this escape.
                # If sp ever exceeds 0.3, the escape is doing something useful;
                # otherwise it's pinned and the next trigger should flip dir.
                if sp >= 0.3:
                    escape_motion_seen = True
                if escape_rev == 0:
                    escape_fwd = ESCAPE_FWD_FRAMES
            elif escape_fwd > 0:
                throttle = 0.7
                steering = 0.7 * float(escape_dir)
                escape_fwd -= 1
                phase = "esc_fwd"
                if sp >= 0.3:
                    escape_motion_seen = True
                if escape_fwd == 0:
                    # Record outcome of this escape for the next-trigger logic.
                    last_escape_dir       = float(escape_dir)
                    last_escape_succeeded = escape_motion_seen
                    post_escape_dir    = float(escape_dir)
                    post_escape_frames = POST_ESCAPE_FRAMES
                    if hasattr(policy_fn, "reset"):
                        policy_fn.reset()
            else:
                throttle, steering = policy_fn(state)
                # Capture raw NN output BEFORE distance boost / post-escape bias.
                # smooth.py exposes last_raw on the smoothed wrapper.
                if hasattr(policy_fn, "last_raw"):
                    nn_t = float(policy_fn.last_raw[0])
                    nn_s = float(policy_fn.last_raw[1])
                else:
                    nn_t, nn_s = float(throttle), float(steering)

                distance = float(nav.get("distance", 50.0))
                if distance < 8.0:
                    steering = max(-1.0, min(1.0, steering * 1.5))
                    phase = "drive_boost1.5"
                elif distance < 14.0:
                    steering = max(-1.0, min(1.0, steering * 1.2))
                    phase = "drive_boost1.2"

                # Safety limiter: slow down when misaligned or very close to obstacles.
                he = float(nav.get("heading_error", 0.0))
                if front_arc_min < 8.0 or abs(he) > 1.2:
                    throttle = min(throttle, 0.4)
                if front_arc_min < 5.0:
                    throttle = min(throttle, 0.2)

                if post_escape_frames > 0:
                    decay = post_escape_frames / POST_ESCAPE_FRAMES
                    steering = max(-1.0, min(1.0,
                                              steering + post_escape_dir * 0.45 * decay))
                    post_escape_frames -= 1
                    phase = phase + "+postesc"

            try:
                client.send_control_ws(throttle, steering)
            except Exception:
                try:
                    client.send_control(throttle, steering)
                except Exception:
                    pass

            steps += 1
            now = time.time()
            t_rel = now - start

            # 1 Hz position track sample (kept for benchmark JSON compatibility)
            if now >= next_log:
                track.append({"t": t_rel, "position": pos, "speed": sp})
                next_log = now + 1.0

            # 1 Hz live status print
            if print_status and now >= next_status:
                nn_str = (f"({nn_t:+.2f},{nn_s:+.2f})"
                          if nn_t is not None else "(esc, esc)")
                he = float(nav.get("heading_error", 0.0))
                d  = float(nav.get("distance", -1.0))
                fr = float(sensors.get("ground_friction", 1.0))
                rays_F = f"[{rays[0]:.1f} {rays[1]:.1f} {rays[7]:.1f}]"
                esc_marker = "-" if phase == "drive" else phase
                print(f"  [t={t_rel:5.1f}s cp={checkpoints_passed}/{TARGET_CHECKPOINTS} "
                      f"nn={nn_str} cmd=({throttle:+.2f},{steering:+.2f}) "
                      f"sp={sp:.1f} he={he:+.2f} d={d:5.1f} fr={fr:.2f} "
                      f"F={rays_F} esc={esc_marker} stuck={stuck_streak}]")
                next_status = now + 1.0

            # Per-frame JSONL log (only when --log-frames was passed)
            if log_fp is not None:
                event = None
                if cp_event_dist is not None:
                    event = f"cp_hit:{checkpoints_passed}:close_pass={cp_event_dist:.2f}m"
                elif escape_event_meta is not None:
                    L, R, esc_sp, esc_front, flipped = escape_event_meta
                    flip_str = ":flip" if flipped else ""
                    event = (f"escape:dir={int(escape_dir):+d}:L={L:.1f}:R={R:.1f}"
                             f":front={esc_front:.1f}{flip_str}")
                elif crash_event:
                    event = f"crash:total={crashes}"
                row = {
                    "t":      round(t_rel, 3),
                    "step":   steps,
                    "sp":     round(float(sp), 3),
                    "he":     round(float(nav.get("heading_error", 0.0)), 4),
                    "d":      round(float(nav.get("distance", -1.0)), 3),
                    "fr":     round(float(sensors.get("ground_friction", 1.0)), 3),
                    "rays":   [round(float(r), 2) for r in rays],
                    "cp":     int(cp_idx),
                    "cp_passed": int(checkpoints_passed),
                    "nn_t":   None if nn_t is None else round(nn_t, 4),
                    "nn_s":   None if nn_s is None else round(nn_s, 4),
                    "cmd_t":  round(float(throttle), 4),
                    "cmd_s":  round(float(steering), 4),
                    "phase":  phase,
                    "stuck":  int(stuck_streak),
                    "esc_rev": int(escape_rev),
                    "esc_fwd": int(escape_fwd),
                    "post_esc": int(post_escape_frames),
                }
                if event is not None:
                    row["event"] = event
                log_fp.write(json.dumps(row) + "\n")

            # Remember this frame's distance so the next-frame cp-hit print
            # can report the close-pass distance, not the distance to the
            # next checkpoint.
            prev_distance = current_distance

            elapsed_step = time.time() - step_start
            sleep_for = interval - elapsed_step
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        if log_fp is not None:
            log_fp.close()

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
            duration: float, player_name: str,
            frame_log_path: str | None = None) -> dict:
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

    result = rest_run_policy(client, policy, duration=duration, hz=20.0,
                             frame_log_path=frame_log_path)
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
    ap.add_argument("--log-frames", action="store_true",
                    help="Write per-frame JSONL diagnostic log to "
                         "benchmarks/<tag>_seed<S>_run<N>_frames.jsonl. "
                         "Captures raw NN output, commanded action, sensors, "
                         "escape state and event tags on cp/escape/crash.")
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
            frame_log_path = None
            if args.log_frames:
                frame_log_path = str(
                    out_dir / f"{args.tag}_seed{seed}_run{i+1}_frames.jsonl"
                )
            result = run_one(
                policy=policy,
                seed=seed,
                run_idx=i + 1,
                total_runs=args.runs,
                duration=args.duration,
                player_name=args.name,
                frame_log_path=frame_log_path,
            )
            if frame_log_path:
                print(f"  wrote frame log: {frame_log_path}")
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
