"""Autonomous benchmark runner — opens the browser automatically for each run.

The simulation runs inside the browser (physics engine is client-side).
Without an open browser tab, the server sends no sensor data and the bot
cannot move. This script auto-opens each session URL, waits for the
simulation to load, then lets the trained policy drive for the allotted time.

Usage:
    python run_agent.py --weights nav_v1.npz --tag v1 --runs 5 --seed 42
    python run_agent.py --weights nav_v7.npz --tag v7 --seeds 42 7 99
    python run_agent.py --weights nav_v8_cnn.pt --tag v8 --module drive2win.cnn --seed 42
"""
from __future__ import annotations
import argparse
import importlib
import json
import time
import webbrowser
from pathlib import Path

import numpy as np

from game_client import GameClient
from drive2win import nn as nn_mod
from drive2win.eval import run_policy, score_runs
from drive2win.normalize import sensors_to_input, clip_action
from drive2win.smooth import make_smooth_policy

SERVER_URL = "https://ml.ferit.tech"
TARGET_CHECKPOINTS = 8
BROWSER_LOAD_TIMEOUT = 15   # seconds to wait for browser to load & send first state


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

    client.connect_ws()

    # Wait until the browser loads and the server starts sending states
    print(f"  Waiting for simulation (up to {BROWSER_LOAD_TIMEOUT}s)…", end="", flush=True)
    deadline = time.time() + BROWSER_LOAD_TIMEOUT
    while time.time() < deadline:
        s = client.get_latest_state()
        if s is not None and "sensors" in s:
            break
        time.sleep(0.1)
        print(".", end="", flush=True)
    else:
        print(" TIMEOUT — no state received. Is the browser open?")
        client.disconnect_ws()
        try:
            client.delete_session()
        except Exception:
            pass
        return {"checkpoints_passed": 0, "crashes": 0, "elapsed": duration,
                "steps": 0, "track": [], "min_speed_streak": 0}

    print(" ready!")

    # Reset EMA state between runs so each run starts fresh
    if hasattr(policy, "reset"):
        policy.reset()

    result = run_policy(client, policy, duration=duration, hz=20.0)
    print(f"    checkpoints={result['checkpoints_passed']}/{TARGET_CHECKPOINTS}  "
          f"crashes={result['crashes']}  steps={result['steps']}")

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

    # Write JSON log (same format as 03_benchmark.py)
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
