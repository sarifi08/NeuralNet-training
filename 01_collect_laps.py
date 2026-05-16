"""Manual data collection — full laps only.

Single-purpose collector for Phase 4 (DAgger / data-coverage fix). Records
your driving for N minutes with ONE instruction: complete the 12-checkpoint
loop. No recovery phase, no deliberate-crash phase — those introduce the
reverse + hard-steer frames we have to filter out anyway.

Usage:
    python 01_collect_laps.py --tag v12_s42 --seed 42 --minutes 4
    python 01_collect_laps.py --tag v12_s7  --seed 7  --minutes 3

How to drive well (gives the network a clean signal to learn from):
  - SMOOTH steering. Don't wiggle the wheel; small, deliberate adjustments.
  - Aim AT the next checkpoint. If you can see it ahead, point the bot at it.
  - Steady throttle. Don't rapid-tap; hold a level you can maintain.
  - Slow before tight corners. Late braking is fine; mid-corner braking
    teaches the network to brake when it should be steering.
  - If you crash, you'll naturally need to back up — that's fine, those
    frames get filtered out at merge time. Don't deliberately drive into
    walls.
"""
from __future__ import annotations
import argparse
import threading
import time
import numpy as np

from game_client import GameClient

SERVER_URL = "https://ml.ferit.tech"


def _poll_positions(client, stop_evt, out, hz=5.0):
    interval = 1.0 / hz
    while not stop_evt.is_set():
        try:
            st  = client.get_latest_state()
            pos = st.get("position") if st else None
            if pos and "x" in pos and "z" in pos:
                out.append((time.time(), pos["x"], pos["z"]))
        except Exception:
            pass
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag",     default="v12_human")
    ap.add_argument("--seed",    type=int,   default=42)
    ap.add_argument("--minutes", type=float, default=4.0)
    args = ap.parse_args()

    duration = args.minutes * 60.0

    client  = GameClient(SERVER_URL)
    session = client.create_session(
        mode="time_trial",
        player_name=f"laps_{args.tag}",
        config={"seed": args.seed, "wind_enabled": False, "obstacles_enabled": True},
    )
    print("\n  Open this URL in a NEW tab and click into it so WASD reach the game:")
    print(f"    {session.get('browser_url')}")
    print()
    print(f"  When ready: drive {args.minutes:.1f} minutes on seed {args.seed}.")
    print(f"  GOAL: complete the 12-checkpoint loop as many times as you can.")
    print(f"  Smooth, deliberate driving. No deliberate crashes.")
    input("\n  Press Enter once the browser tab has focus and you can see the bot.  ")

    client.connect_ws()
    time.sleep(0.5)

    positions: list = []
    stop_evt = threading.Event()
    t = threading.Thread(target=_poll_positions, args=(client, stop_evt, positions),
                         daemon=True)
    t.start()

    client.start_recording(sample_rate=20)
    print(f"\n  Recording {args.minutes:.1f} minutes — drive now.")
    for s in range(int(duration), 0, -10):
        print(f"    ... {s}s remaining")
        time.sleep(min(10, s))

    stop_evt.set()
    info = client.stop_recording()
    print(f"\n  Stopped. Server samples: {info.get('sample_count', '?')}")

    states, actions = client.get_recording_as_arrays()
    print(f"  states  : {states.shape}")
    print(f"  actions : {actions.shape}")

    if len(states) > 0:
        corr = np.corrcoef(states[:, 1], actions[:, 1])[0, 1]
        print(f"  heading→steer correlation: {corr:+.3f}  (target: < -0.4)")
        print(f"  steering std:  {actions[:,1].std():.3f}  (target: > 0.3)")
        print(f"  throttle mean: {actions[:,0].mean():+.3f}")

    pos_arr = np.array([(p[1], p[2]) for p in positions], dtype=np.float32)

    out = f"data_{args.tag}.npz"
    np.savez(out, states=states, actions=actions, positions=pos_arr, seed=args.seed)
    print(f"\n  Saved {out}")

    try:
        client.disconnect_ws()
        client.delete_session()
    except Exception:
        pass


if __name__ == "__main__":
    main()
